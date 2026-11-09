//! Edits are low-rank deltas kept in f32 and applied as overlays (tracks 1 and 2).
//! A rank-1 edit `W += scale · u vᵀ` is exactly a rank-1 LoRA, so kept edits export as LoRA later.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// One rank-1 term. `u` is sparse over output rows (one entry for a single-token logit nudge),
/// `v` is dense over the input dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rank1 {
    pub u: Vec<(u32, f32)>,
    pub v: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    pub id: String,
    /// Target tensor, e.g. "lm_head.weight".
    pub tensor: String,
    /// Sum of rank-1 terms, times `scale`.
    pub terms: Vec<Rank1>,
    pub scale: f32,
    pub origin: Option<NudgeOrigin>,
}

/// Where a logit nudge came from: "at this probe's step, raise `target` over `from`".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NudgeOrigin {
    pub probe_id: String,
    pub step: usize,
    pub from_token: u32,
    pub from_text: String,
    pub target_token: u32,
    pub target_text: String,
    /// Logit gap (from - target) at the site before the edit.
    pub gap_before: f32,
    /// How far past the current leader the target is pushed.
    pub margin: f32,
    pub hidden_norm: f32,
}

impl Delta {
    /// Joey's kept-edit form: `W[k,:] += (δ/‖h‖²) h`. At hidden state `h` this raises logit `k`
    /// by exactly `delta_logit` and leaves every other logit untouched.
    pub fn logit_nudge(tensor: &str, token: u32, h: &[f32], delta_logit: f32) -> Result<Delta> {
        if !delta_logit.is_finite() {
            bail!("logit shift must be finite (got {delta_logit})");
        }
        let norm2: f32 = h.iter().map(|x| x * x).sum();
        if !norm2.is_finite() || norm2 <= 0.0 {
            bail!("hidden state has zero or non-finite norm");
        }
        let v = h.iter().map(|x| x / norm2).collect();
        Ok(Delta {
            id: new_id("d"),
            tensor: tensor.to_string(),
            terms: vec![Rank1 { u: vec![(token, 1.0)], v }],
            scale: delta_logit,
            origin: None,
        })
    }

    /// Add this delta's effect to logits computed at final hidden state `h` (lm_head deltas only).
    pub fn add_to_logits(&self, h: &[f32], logits: &mut [f32]) {
        for t in &self.terms {
            let dot: f32 = t.v.iter().zip(h).map(|(a, b)| a * b).sum();
            let s = self.scale * dot;
            for &(row, w) in &t.u {
                if let Some(l) = logits.get_mut(row as usize) {
                    *l += s * w;
                }
            }
        }
    }

    /// Change to logit `token` at hidden `h` caused by this delta.
    pub fn logit_shift(&self, token: u32, h: &[f32]) -> f32 {
        let mut out = 0.0;
        for t in &self.terms {
            let w: f32 = t.u.iter().filter(|(r, _)| *r == token).map(|(_, w)| w).sum();
            if w != 0.0 {
                let dot: f32 = t.v.iter().zip(h).map(|(a, b)| a * b).sum();
                out += self.scale * dot * w;
            }
        }
        out
    }

    /// Rows of the target tensor this delta touches.
    pub fn rows(&self) -> Vec<u32> {
        let mut r: Vec<u32> = self.terms.iter().flat_map(|t| t.u.iter().map(|(i, _)| *i)).collect();
        r.sort_unstable();
        r.dedup();
        r
    }

    /// Unit-length direction of the first term's `v` (for cosine features).
    pub fn direction(&self) -> Vec<f32> {
        let Some(t) = self.terms.first() else { return vec![] };
        let n = t.v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n == 0.0 {
            return t.v.clone();
        }
        t.v.iter().map(|x| x / n).collect()
    }

    /// Upper bound on the largest absolute element of the dense delta (for the fp16/bf16 magnitude
    /// gate, cf. the paper's 1e5 lesson). Terms are summed per row, so overlapping terms are covered.
    /// Any non-finite value yields infinity, which fails the gate.
    pub fn max_abs(&self) -> f32 {
        if !self.is_finite() {
            return f32::INFINITY;
        }
        let mut per_row: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for t in &self.terms {
            let vmax = t.v.iter().fold(0f32, |m, x| m.max(x.abs()));
            for (r, w) in &t.u {
                *per_row.entry(*r).or_default() += (w * vmax * self.scale).abs();
            }
        }
        per_row.values().copied().fold(0f32, f32::max)
    }

    /// True when the scale and every vector entry are finite.
    pub fn is_finite(&self) -> bool {
        self.scale.is_finite()
            && self.terms.iter().all(|t| t.v.iter().all(|x| x.is_finite()) && t.u.iter().all(|(_, w)| w.is_finite()))
    }

    /// A copy without the dense vectors, for JSON rows. Vectors go to the ledger blob column.
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "tensor": self.tensor,
            "rank": self.terms.len(),
            "rows": self.rows(),
            "scale": self.scale,
            "max_abs": self.max_abs(),
            "origin": self.origin,
        })
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

pub fn new_id(prefix: &str) -> String {
    let u = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}_{}", &u[..12])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nudge_shifts_only_target_by_exact_amount() {
        let h = vec![0.5, -1.0, 2.0, 0.25];
        let d = Delta::logit_nudge("lm_head.weight", 2, &h, 3.0).unwrap();
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        d.add_to_logits(&h, &mut logits);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert!((logits[2] - 6.0).abs() < 1e-5);
        assert_eq!(logits[3], 4.0);
        assert!((d.logit_shift(2, &h) - 3.0).abs() < 1e-5);
        assert_eq!(d.logit_shift(1, &h), 0.0);
    }

    #[test]
    fn non_finite_deltas_are_rejected_or_flagged() {
        let h = vec![1.0, 2.0];
        assert!(Delta::logit_nudge("w", 0, &h, f32::NAN).is_err());
        let mut d = Delta::logit_nudge("w", 0, &h, 1.0).unwrap();
        d.scale = f32::NAN;
        assert!(!d.is_finite());
        assert_eq!(d.max_abs(), f32::INFINITY);
    }
}
