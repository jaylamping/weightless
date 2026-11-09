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
    /// False only if the base completion had the required answer and the new one lost it.
    pub answer_ok: Option<bool>,
    /// Whether the base completion contained the required answer.
    #[serde(default)]
    pub base_has_answer: Option<bool>,
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

/// True if (logit `a`, id `a_id`) ranks above (`b`, `b_id`) under the decoder's ordering: higher
/// logit first, ties to the lower token id.
fn ranks_above(a: f32, a_id: u32, b: f32, b_id: u32) -> bool {
    a > b || (a == b && a_id < b_id)
}

/// First step at which a greedy decode of `trace` would change under `delta`, given current logits
/// of the delta's target token at each step (`target_logits[i]`). Exact for single-row lm_head nudges,
/// because only the target logit moves; ties follow the decoder (lower token id wins).
pub fn predicted_first_flip(trace: &Trace, delta: &Delta, target: u32, target_logits: &[f32]) -> Option<usize> {
    for (i, s) in trace.steps.iter().enumerate() {
        let h = s.hidden.as_deref()?;
        let shift = delta.logit_shift(target, h);
        let Some(leader) = s.top.first() else { continue };
        if s.token_id == target {
            // Already chosen; a negative shift could dethrone it.
            if let Some(runner) = s.top.get(1) {
                if !ranks_above(leader.logit + shift, leader.id, runner.logit, runner.id) {
                    return Some(i);
                }
            }
        } else if ranks_above(target_logits[i] + shift, target, leader.logit, leader.id) {
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
        answer_ok: p
            .required_answer
            .as_ref()
            .map(|ans| !base.text.contains(ans.as_str()) || new.text.contains(ans.as_str())),
        base_has_answer: p.required_answer.as_ref().map(|ans| base.text.contains(ans.as_str())),
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

    let mag_ok = delta.max_abs().is_finite() && delta.max_abs() <= spec.max_abs_delta;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Prompt, PromptFormat, Step, StopReason, TokenLogit, Trace};
    use crate::spec::Probe;

    fn spec(objective: Objective) -> Spec {
        let p = |id: &str, role, ans: Option<&str>| Probe {
            id: id.into(),
            role,
            prompt: Prompt { text: id.into(), format: PromptFormat::Raw, thinking: false },
            required_answer: ans.map(Into::into),
        };
        Spec {
            goal: "t".into(),
            objective,
            probes: vec![p("p", ProbeRole::Probe, None), p("c", ProbeRole::Control, Some("56"))],
            max_new_tokens: 8,
            top_k: 4,
            max_abs_delta: 10.0,
        }
    }

    fn trace(tokens: &[u32], text: &str) -> Trace {
        Trace {
            prompt_tokens: 1,
            steps: tokens
                .iter()
                .map(|&t| Step {
                    token_id: t,
                    text: String::new(),
                    top: vec![TokenLogit { id: t, text: String::new(), logit: 1.0 }],
                    gap: 1.0,
                    hidden_norm: 1.0,
                    hidden: None,
                })
                .collect(),
            text: text.into(),
            stop: StopReason::MaxTokens,
            wall_ms: 0,
        }
    }

    fn delta(scale: f32) -> Delta {
        Delta::logit_nudge("w", 0, &[1.0, 0.0], scale).unwrap()
    }

    fn verdict(s: &Spec, d: &Delta, probe: (&[u32], &[u32]), control: (&str, &str), flipped: Option<bool>) -> Verdict {
        let base_c = trace(&[9, 9], control.0);
        let new_c = trace(if control.0 == control.1 { &[9, 9] } else { &[9, 8] }, control.1);
        let r = vec![
            compare(s, 0, &trace(probe.0, ""), &trace(probe.1, "")),
            compare(s, 1, &base_c, &new_c),
        ];
        decide(s, d, &r, flipped)
    }

    #[test]
    fn decide_covers_every_reason() {
        let s = spec(Objective::Shorter);
        let d = delta(1.0);
        let ok = ("56", "56");
        let pass = verdict(&s, &d, (&[1, 2, 3], &[1, 2]), ok, Some(true));
        assert!(pass.keep, "{:?}", pass.reasons);

        let r = |v: Verdict| v.reasons;
        assert_eq!(r(verdict(&s, &d, (&[1, 2], &[1, 2, 3]), ok, Some(true))), vec![ReasonCode::ProbeLonger, ReasonCode::NoGain]);
        assert_eq!(r(verdict(&s, &d, (&[1, 2], &[1, 2]), ok, Some(true))), vec![ReasonCode::NoGain]);
        assert_eq!(r(verdict(&s, &d, (&[1, 2, 3], &[1, 2]), ok, Some(false))), vec![ReasonCode::SiteNotFlipped]);
        assert_eq!(
            r(verdict(&s, &d, (&[1, 2, 3], &[1, 2]), ("56", "57"), Some(true))),
            vec![ReasonCode::SpillControl, ReasonCode::AnswerLost]
        );
        assert_eq!(r(verdict(&s, &delta(100.0), (&[1, 2, 3], &[1, 2]), ok, Some(true))), vec![ReasonCode::MagnitudeUnsafe]);
        let mut nan = delta(1.0);
        nan.scale = f32::NAN;
        assert_eq!(r(verdict(&s, &nan, (&[1, 2, 3], &[1, 2]), ok, Some(true))), vec![ReasonCode::MagnitudeUnsafe]);

        // ControlsOnly ignores length.
        let s2 = spec(Objective::ControlsOnly);
        assert!(verdict(&s2, &d, (&[1, 2], &[1, 2, 3]), ok, Some(true)).keep);
    }

    #[test]
    fn answer_lost_only_when_base_had_it() {
        let s = spec(Objective::ControlsOnly);
        let r = compare(&s, 1, &trace(&[9], "fifty-six"), &trace(&[8], "something else"));
        assert_eq!(r.answer_ok, Some(true));
        assert_eq!(r.base_has_answer, Some(false));
        let r = compare(&s, 1, &trace(&[9], "56"), &trace(&[8], "57"));
        assert_eq!(r.answer_ok, Some(false));
    }

    fn step(token: u32, top: &[(u32, f32)], h: Vec<f32>) -> Step {
        Step {
            token_id: token,
            text: String::new(),
            top: top.iter().map(|&(id, logit)| TokenLogit { id, text: String::new(), logit }).collect(),
            gap: top[0].1 - top[1].1,
            hidden_norm: 1.0,
            hidden: Some(h),
        }
    }

    #[test]
    fn first_flip_follows_decoder_tie_break() {
        // Leader id 5 at 2.0. Target id 3 at 1.0, nudged by exactly +1.0 to tie: the lower id wins.
        let t = Trace {
            prompt_tokens: 1,
            steps: vec![step(5, &[(5, 2.0), (3, 1.0)], vec![1.0, 0.0])],
            text: String::new(),
            stop: StopReason::MaxTokens,
            wall_ms: 0,
        };
        let d = Delta::logit_nudge("w", 3, &[1.0, 0.0], 1.0).unwrap();
        assert_eq!(predicted_first_flip(&t, &d, 3, &[1.0]), Some(0));
        // Same tie with a higher target id does not flip.
        let d = Delta::logit_nudge("w", 7, &[1.0, 0.0], 1.0).unwrap();
        assert_eq!(predicted_first_flip(&t, &d, 7, &[1.0]), None);
        // A negative shift on the chosen token dethrones it once it drops to the runner-up's level
        // and the runner-up has the lower id.
        let t2 = Trace { steps: vec![step(5, &[(5, 2.0), (3, 1.5)], vec![1.0, 0.0])], ..t.clone() };
        let d = Delta::logit_nudge("w", 5, &[1.0, 0.0], -0.5).unwrap();
        assert_eq!(predicted_first_flip(&t2, &d, 5, &[2.0]), Some(0));
        let d = Delta::logit_nudge("w", 5, &[1.0, 0.0], -0.25).unwrap();
        assert_eq!(predicted_first_flip(&t2, &d, 5, &[2.0]), None);
    }
}
