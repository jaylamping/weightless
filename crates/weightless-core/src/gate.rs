//! Gates decide keep/reject for a trial. Every gate's pass/fail and metrics are logged, with enum
//! reason codes, so the navigator can learn from rejects (most of the signal).

use crate::backend::{Trace, TraceRequest};
use crate::delta::{cosine, Delta};
use crate::spec::{Objective, ProbeRole, Spec};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReasonCode {
    /// A control's output changed.
    SpillControl,
    /// Pre-flight predicted a control would change (rejected without decoding).
    SpillControlPredicted,
    /// A required answer disappeared from a completion.
    AnswerLost,
    /// A probe got longer under the Shorter objective.
    ProbeLonger,
    /// No probe improved.
    NoGain,
    /// The edited site did not take the target token.
    SiteNotFlipped,
    /// Delta magnitude unsafe for half-precision weights.
    MagnitudeUnsafe,
    /// The backend failed while running the trial.
    BackendError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub name: String,
    pub pass: bool,
    pub metrics: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub keep: bool,
    pub reasons: Vec<ReasonCode>,
    pub gates: Vec<GateResult>,
    /// "preflight" if rejected before decoding, else "decode".
    pub stage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub probe_id: String,
    pub role: ProbeRole,
    pub base_len: usize,
    pub new_len: Option<usize>,
    pub first_divergence: Option<usize>,
    pub tokens_changed: Option<usize>,
    pub identical: Option<bool>,
    pub answer_ok: Option<bool>,
    /// Pre-flight's exact prediction of the first greedy step this delta flips (lm_head deltas).
    pub predicted_first_flip: Option<usize>,
    pub new_text: Option<String>,
}

/// Features of the edit site, computed before any decoding. Navigator inputs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SiteFeatures {
    pub gap_before: Option<f32>,
    pub hidden_norm: Option<f32>,
    /// Max cosine between the edit direction and any hidden state in the controls / other probes.
    /// Joey's lesson: > ~0.98 means the edit will spill.
    pub max_cos_controls: f32,
    pub max_cos_other_probes: f32,
    pub delta_max_abs: f32,
}

pub fn request_for(spec: &Spec, probe_idx: usize, capture_hidden: bool) -> TraceRequest {
    TraceRequest {
        prompt: spec.probes[probe_idx].prompt.clone(),
        max_new_tokens: spec.max_new_tokens,
        top_k: spec.top_k,
        capture_hidden,
        decode: Default::default(),
    }
}

pub fn site_features(spec: &Spec, baseline: &[Trace], delta: &Delta) -> SiteFeatures {
    let dir = delta.direction();
    let site_probe = delta.origin.as_ref().map(|o| o.probe_id.as_str());
    let mut f = SiteFeatures {
        gap_before: delta.origin.as_ref().map(|o| o.gap_before),
        hidden_norm: delta.origin.as_ref().map(|o| o.hidden_norm),
        delta_max_abs: delta.max_abs(),
        ..Default::default()
    };
    for (p, t) in spec.probes.iter().zip(baseline) {
        if Some(p.id.as_str()) == site_probe {
            continue;
        }
        let m = t
            .steps
            .iter()
            .filter_map(|s| s.hidden.as_deref())
            .map(|h| cosine(&dir, h))
            .fold(f32::MIN, f32::max)
            .max(0.0);
        match p.role {
            ProbeRole::Control => f.max_cos_controls = f.max_cos_controls.max(m),
            ProbeRole::Probe => f.max_cos_other_probes = f.max_cos_other_probes.max(m),
        }
    }
    f
}

/// First step at which a greedy decode of `trace` would change under `delta`, given base logits of
/// the delta's target token at each step (`target_logits[i]`). Exact for single-row lm_head nudges,
/// because only the target logit moves.
pub fn predicted_first_flip(trace: &Trace, delta: &Delta, target: u32, target_logits: &[f32]) -> Option<usize> {
    for (i, s) in trace.steps.iter().enumerate() {
        let h = s.hidden.as_deref()?;
        let shift = delta.logit_shift(target, h);
        let leader = s.top.first().map(|t| t.logit).unwrap_or(f32::MIN);
        if s.token_id == target {
            // Already chosen; a negative shift could dethrone it.
            let runner = s.top.get(1).map(|t| t.logit).unwrap_or(f32::MIN);
            if leader + shift < runner {
                return Some(i);
            }
        } else if target_logits[i] + shift > leader {
            return Some(i);
        }
    }
    None
}

pub fn compare(spec: &Spec, probe_idx: usize, base: &Trace, new: &Trace) -> ProbeResult {
    let p = &spec.probes[probe_idx];
    let a = base.token_ids();
    let b = new.token_ids();
    let first_div = a.iter().zip(&b).position(|(x, y)| x != y).or(if a.len() != b.len() {
        Some(a.len().min(b.len()))
    } else {
        None
    });
    let changed = a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
    ProbeResult {
        probe_id: p.id.clone(),
        role: p.role,
        base_len: a.len(),
        new_len: Some(b.len()),
        first_divergence: first_div,
        tokens_changed: Some(changed),
        identical: Some(first_div.is_none()),
        answer_ok: p.required_answer.as_ref().map(|ans| new.text.contains(ans.as_str())),
        predicted_first_flip: None,
        new_text: Some(new.text.clone()),
    }
}

/// Decode-stage verdict over all probe results.
pub fn decide(spec: &Spec, delta: &Delta, results: &[ProbeResult], site_flipped: Option<bool>) -> Verdict {
    let mut gates = Vec::new();
    let mut reasons = Vec::new();

    let spilled: Vec<&str> = results
        .iter()
        .filter(|r| r.role == ProbeRole::Control && r.identical == Some(false))
        .map(|r| r.probe_id.as_str())
        .collect();
    gates.push(GateResult {
        name: "controls_identical".into(),
        pass: spilled.is_empty(),
        metrics: serde_json::json!({ "spilled": spilled }),
    });
    if !spilled.is_empty() {
        reasons.push(ReasonCode::SpillControl);
    }

    let lost: Vec<&str> =
        results.iter().filter(|r| r.answer_ok == Some(false)).map(|r| r.probe_id.as_str()).collect();
    gates.push(GateResult {
        name: "answers_kept".into(),
        pass: lost.is_empty(),
        metrics: serde_json::json!({ "lost": lost }),
    });
    if !lost.is_empty() {
        reasons.push(ReasonCode::AnswerLost);
    }

    if let Some(flipped) = site_flipped {
        gates.push(GateResult { name: "site_flipped".into(), pass: flipped, metrics: serde_json::json!({}) });
        if !flipped {
            reasons.push(ReasonCode::SiteNotFlipped);
        }
    }

    if spec.objective == Objective::Shorter {
        let probes: Vec<&ProbeResult> = results.iter().filter(|r| r.role == ProbeRole::Probe).collect();
        let deltas: Vec<i64> =
            probes.iter().map(|r| r.new_len.unwrap_or(r.base_len) as i64 - r.base_len as i64).collect();
        let longer = deltas.iter().any(|d| *d > 0);
        let shorter = deltas.iter().any(|d| *d < 0);
        gates.push(GateResult {
            name: "objective_shorter".into(),
            pass: shorter && !longer,
            metrics: serde_json::json!({ "len_deltas": deltas, "total": deltas.iter().sum::<i64>() }),
        });
        if longer {
            reasons.push(ReasonCode::ProbeLonger);
        }
        if !shorter {
            reasons.push(ReasonCode::NoGain);
        }
    }

    let mag_ok = delta.max_abs() <= spec.max_abs_delta;
    gates.push(GateResult {
        name: "magnitude".into(),
        pass: mag_ok,
        metrics: serde_json::json!({ "max_abs": delta.max_abs(), "limit": spec.max_abs_delta }),
    });
    if !mag_ok {
        reasons.push(ReasonCode::MagnitudeUnsafe);
    }

    Verdict { keep: reasons.is_empty(), reasons, gates, stage: "decode".into() }
}
