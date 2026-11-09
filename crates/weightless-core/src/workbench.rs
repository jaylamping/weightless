//! The edit loop: baseline → propose → pre-flight → decode trial → gate → ledger → keep.
//! Backend-agnostic; the desktop app, the CLI and (later) the agent API all drive this.

use crate::backend::{Backend, Trace, TraceRequest};
use crate::delta::{new_id, Delta, NudgeOrigin};
use crate::gate::{self, GateResult, ProbeResult, ReasonCode, Verdict};
use crate::ledger::{now, Ledger, Prediction, TrialRecord};
use crate::spec::{ProbeRole, Spec};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Snapshot {
    /// None = the base model.
    pub id: Option<String>,
    pub deltas: Vec<Delta>,
    /// Trial ids in keep order.
    pub trials: Vec<String>,
    /// Snapshot id before each keep (the parent), so unkeeping the last edit returns to a
    /// snapshot that exists in the ledger.
    pub parents: Vec<Option<String>>,
}

/// One candidate edit to try.
pub struct Candidate {
    pub delta: Delta,
    pub proposer: String,
    pub rationale: Option<String>,
    pub decision_id: String,
    pub prediction: Option<Prediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreOptions {
    pub budget: usize,
    /// Only edit steps whose top-2 gap is below this (near-ties, per Joey's lessons).
    pub max_gap: f32,
    /// Push the target this far past the current leader.
    pub margin: f32,
    /// Keep every trial that passes the gates (synthetic campaigns). Off by default.
    pub auto_keep: bool,
}

impl Default for ExploreOptions {
    fn default() -> Self {
        Self { budget: 20, max_gap: 2.0, margin: 0.5, auto_keep: false }
    }
}

pub struct Workbench {
    backend: Box<dyn Backend>,
    spec: Spec,
    ledger: Ledger,
    session_id: String,
    baseline: Option<Vec<Trace>>,
    snapshot: Snapshot,
    request_id: Option<String>,
}

impl Workbench {
    pub fn new(backend: Box<dyn Backend>, ledger: Ledger, spec: Spec) -> Result<Self> {
        spec.validate()?;
        let mut wb = Self {
            backend,
            spec,
            ledger,
            session_id: String::new(),
            baseline: None,
            snapshot: Snapshot::default(),
            request_id: None,
        };
        wb.start_session()?;
        Ok(wb)
    }

    fn start_session(&mut self) -> Result<()> {
        self.session_id = new_id("s");
        self.ledger.start_session(&self.session_id, &self.spec, self.backend.info(), Default::default())
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }
    pub fn spec(&self) -> &Spec {
        &self.spec
    }
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// A new spec starts a new ledger session from the base model: kept edits are reverted, so
    /// rows never mix specs and every new trial is gated against a model state its session knows.
    pub fn set_spec(&mut self, spec: Spec) -> Result<()> {
        spec.validate()?;
        for d in self.snapshot.deltas.iter().rev() {
            self.backend.revert(&d.id)?;
        }
        self.snapshot = Snapshot::default();
        self.spec = spec;
        self.baseline = None;
        self.request_id = None;
        self.start_session()
    }

    /// Set the active dial request; later trials link to it. Returns its id.
    pub fn set_dial_request(&mut self, user_words: &str, request: &serde_json::Value) -> Result<String> {
        let id = new_id("req");
        self.ledger.insert_dial_request(&id, &self.session_id, user_words, request)?;
        self.request_id = Some(id.clone());
        Ok(id)
    }

    /// Greedy traces of every probe under the current snapshot, with hidden states.
    pub fn baseline(&mut self) -> Result<&[Trace]> {
        if self.baseline.is_none() {
            let mut out = Vec::with_capacity(self.spec.probes.len());
            for i in 0..self.spec.probes.len() {
                out.push(self.backend.trace(&gate::request_for(&self.spec, i, true))?);
            }
            self.baseline = Some(out);
        }
        Ok(self.baseline.as_deref().unwrap())
    }

    pub fn trace(&mut self, req: &TraceRequest) -> Result<Trace> {
        self.backend.trace(req)
    }

    fn probe_index(&self, probe_id: &str) -> Result<usize> {
        self.spec.probes.iter().position(|p| p.id == probe_id).ok_or_else(|| anyhow!("no probe {probe_id}"))
    }

    /// Current logit of `token` at hidden states, including kept deltas.
    fn current_token_logits(&self, token: u32, hiddens: &[&[f32]]) -> Result<Vec<f32>> {
        let mut l = self.backend.token_logits(token, hiddens)?;
        for (x, h) in l.iter_mut().zip(hiddens) {
            for d in &self.snapshot.deltas {
                *x += d.logit_shift(token, h);
            }
        }
        Ok(l)
    }

    /// Build the rank-1 lm_head edit that makes `target` win at `step` of `probe_id` by `margin`.
    pub fn propose_nudge(&mut self, probe_id: &str, step: usize, target: u32, margin: f32) -> Result<Delta> {
        let idx = self.probe_index(probe_id)?;
        let tensor = self.backend.info().lm_head_tensor.clone();
        self.baseline()?;
        let s = self.baseline.as_ref().unwrap()[idx]
            .steps
            .get(step)
            .with_context(|| format!("probe {probe_id} has no step {step}"))?
            .clone();
        let h = s.hidden.as_deref().context("baseline has no hidden states")?;
        if s.token_id == target {
            bail!("step {step} already emits that token");
        }
        let leader = s.top.first().map(|t| t.logit).context("empty step")?;
        let target_logit = self.current_token_logits(target, &[h])?[0];
        let gap = leader - target_logit;
        let mut d = Delta::logit_nudge(&tensor, target, h, gap + margin)?;
        d.origin = Some(NudgeOrigin {
            probe_id: probe_id.into(),
            step,
            from_token: s.token_id,
            from_text: s.text.clone(),
            target_token: target,
            target_text: self.backend.token_text(target),
            gap_before: gap,
            margin,
            hidden_norm: s.hidden_norm,
        });
        Ok(d)
    }

    /// Run one candidate through pre-flight, decode and gates, and write it to the ledger.
    pub fn run_trial(&mut self, c: Candidate) -> Result<TrialRecord> {
        let start = Instant::now();
        self.baseline()?;
        let base = self.baseline.clone().unwrap();
        let site = gate::site_features(&self.spec, &base, &c.delta);

        // Pre-flight: for single-row lm_head nudges the first flipped step is exactly predictable.
        let rows = c.delta.rows();
        let mut results: Vec<ProbeResult> = Vec::new();
        let mut predicted = vec![None; base.len()];
        if rows.len() == 1 && c.delta.tensor == self.backend.info().lm_head_tensor {
            for (i, t) in base.iter().enumerate() {
                let hs: Vec<&[f32]> = t.steps.iter().filter_map(|s| s.hidden.as_deref()).collect();
                let tl = self.current_token_logits(rows[0], &hs)?;
                predicted[i] = gate::predicted_first_flip(t, &c.delta, rows[0], &tl);
            }
        }
        let control_spill: Vec<String> = self
            .spec
            .probes
            .iter()
            .zip(&predicted)
            .filter(|(p, f)| p.role == ProbeRole::Control && f.is_some())
            .map(|(p, _)| p.id.clone())
            .collect();
        let mag_ok = c.delta.max_abs().is_finite() && c.delta.max_abs() <= self.spec.max_abs_delta;

        let verdict = if !control_spill.is_empty() || !mag_ok {
            for (i, p) in self.spec.probes.iter().enumerate() {
                results.push(ProbeResult {
                    probe_id: p.id.clone(),
                    role: p.role,
                    base_len: base[i].steps.len(),
                    new_len: None,
                    first_divergence: None,
                    tokens_changed: None,
                    identical: None,
                    answer_ok: None,
                    base_has_answer: None,
                    predicted_first_flip: predicted[i],
                    new_text: None,
                });
            }
            let mut reasons = Vec::new();
            if !control_spill.is_empty() {
                reasons.push(ReasonCode::SpillControlPredicted);
            }
            if !mag_ok {
                reasons.push(ReasonCode::MagnitudeUnsafe);
            }
            Verdict {
                keep: false,
                reasons,
                gates: vec![
                    GateResult {
                        name: "preflight_controls".into(),
                        pass: control_spill.is_empty(),
                        metrics: serde_json::json!({ "predicted_spill": control_spill }),
                    },
                    GateResult {
                        name: "magnitude".into(),
                        pass: mag_ok,
                        metrics: serde_json::json!({ "max_abs": c.delta.max_abs(), "limit": self.spec.max_abs_delta }),
                    },
                ],
                stage: "preflight".into(),
            }
        } else {
            self.backend.apply(&c.delta)?;
            let traces: Result<Vec<Trace>> = (0..self.spec.probes.len())
                .map(|i| self.backend.trace(&gate::request_for(&self.spec, i, false)))
                .collect();
            self.backend.revert(&c.delta.id)?;
            match traces {
                Ok(traces) => {
                    for (i, t) in traces.iter().enumerate() {
                        let mut r = gate::compare(&self.spec, i, &base[i], t);
                        r.predicted_first_flip = predicted[i];
                        results.push(r);
                    }
                    let site_flipped = c.delta.origin.as_ref().and_then(|o| {
                        let i = self.probe_index(&o.probe_id).ok()?;
                        let t = &traces[i];
                        let same_prefix = base[i].steps.iter().zip(&t.steps).take(o.step).all(|(a, b)| a.token_id == b.token_id);
                        Some(same_prefix && t.steps.get(o.step).map(|s| s.token_id) == Some(o.target_token))
                    });
                    gate::decide(&self.spec, &c.delta, &results, site_flipped)
                }
                Err(e) => Verdict {
                    keep: false,
                    reasons: vec![ReasonCode::BackendError],
                    gates: vec![GateResult { name: "decode".into(), pass: false, metrics: serde_json::json!({ "error": e.to_string() }) }],
                    stage: "decode".into(),
                },
            }
        };

        let rec = TrialRecord {
            id: new_id("t"),
            session_id: self.session_id.clone(),
            decision_id: c.decision_id,
            request_id: self.request_id.clone(),
            snapshot_id: self.snapshot.id.clone(),
            created_at: now(),
            proposer: c.proposer,
            rationale: c.rationale,
            delta: c.delta.summary(),
            site,
            prediction: c.prediction,
            results,
            verdict,
            wall_ms: start.elapsed().as_millis() as u64,
            kept: false,
        };
        self.ledger.insert_trial(&rec, &c.delta)?;
        Ok(rec)
    }

    /// Make a trial's delta part of the working model. Snapshot = parent + this delta.
    ///
    /// Only a trial that passed its gates against the *current* model state can be kept, so every
    /// snapshot in the ledger is a stack that was gated as a whole. Re-test stale trials first.
    pub fn keep(&mut self, trial_id: &str) -> Result<Snapshot> {
        let rec = self.ledger.trial(trial_id)?.with_context(|| format!("no trial {trial_id}"))?;
        if !rec.verdict.keep {
            bail!("trial {trial_id} did not pass its gates");
        }
        if rec.session_id != self.session_id || rec.snapshot_id != self.snapshot.id {
            bail!("trial {trial_id} was gated against a different model state; re-test it on the current one first");
        }
        let delta = self.ledger.load_delta(trial_id)?;
        if self.snapshot.deltas.iter().any(|d| d.id == delta.id) {
            bail!("trial {trial_id} is already kept");
        }
        self.backend.apply(&delta)?;
        self.baseline = None;
        let parent = self.snapshot.id.clone();
        let snap_id = new_id("snap");
        let mut ids: Vec<String> = self.snapshot.deltas.iter().map(|d| d.id.clone()).collect();
        ids.push(delta.id.clone());
        if let Err(e) = self.ledger.mark_kept(trial_id, &snap_id, parent.as_deref(), &ids, &self.session_id) {
            self.backend.revert(&delta.id)?;
            return Err(e);
        }
        self.snapshot.deltas.push(delta);
        self.snapshot.trials.push(trial_id.to_string());
        self.snapshot.parents.push(parent);
        self.snapshot.id = Some(snap_id);
        Ok(self.snapshot.clone())
    }

    /// Take the most recently kept delta back out, returning to its parent snapshot. Only the last
    /// keep can be undone: removing one from the middle would leave a stack that was never gated.
    pub fn unkeep(&mut self, trial_id: &str) -> Result<Snapshot> {
        if self.snapshot.trials.last().map(String::as_str) != Some(trial_id) {
            bail!("only the most recently kept trial can be unkept");
        }
        let d = self.snapshot.deltas.last().unwrap().clone();
        self.backend.revert(&d.id)?;
        self.baseline = None;
        if let Err(e) = self.ledger.mark_unkept(trial_id) {
            self.backend.apply(&d)?;
            return Err(e);
        }
        self.snapshot.deltas.pop();
        self.snapshot.trials.pop();
        self.snapshot.id = self.snapshot.parents.pop().flatten();
        Ok(self.snapshot.clone())
    }

    /// Run a logged trial's delta again on the current model state, as a new sibling row.
    pub fn retest(&mut self, trial_id: &str) -> Result<TrialRecord> {
        let rec = self.ledger.trial(trial_id)?.with_context(|| format!("no trial {trial_id}"))?;
        let mut delta = self.ledger.load_delta(trial_id)?;
        delta.id = new_id("d");
        if let Some(o) = &delta.origin {
            self.probe_index(&o.probe_id)?;
        }
        self.run_trial(Candidate {
            delta,
            proposer: rec.proposer,
            rationale: Some(format!("retest of {trial_id}")),
            decision_id: rec.decision_id,
            prediction: None,
        })
    }

    /// Heuristic proposer: at every near-tie step of every probe, try promoting the runner-up, and
    /// (for concision) any end-of-turn token in the top-k. Each site is one decision, so its
    /// candidates are siblings in the ledger. This is how the app generates navigator data on its own.
    pub fn explore(&mut self, opts: &ExploreOptions, on_trial: &mut dyn FnMut(&TrialRecord)) -> Result<Vec<TrialRecord>> {
        self.baseline()?;
        let eos = self.backend_eos();
        let base = self.baseline.clone().unwrap();
        // (gap, probe_id, step, targets)
        let mut sites: Vec<(f32, String, usize, Vec<u32>)> = Vec::new();
        for (p, t) in self.spec.probes.iter().zip(&base) {
            if p.role != ProbeRole::Probe {
                continue;
            }
            for (i, s) in t.steps.iter().enumerate() {
                let mut targets = Vec::new();
                if s.top.len() >= 2 && s.gap < opts.max_gap {
                    targets.push(s.top[1].id);
                }
                for c in s.top.iter().skip(1) {
                    if eos.contains(&c.id) && s.top[0].logit - c.logit < opts.max_gap * 2.0 && !targets.contains(&c.id) {
                        targets.push(c.id);
                    }
                }
                if !targets.is_empty() {
                    sites.push((s.gap, p.id.clone(), i, targets));
                }
            }
        }
        sites.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut out = Vec::new();
        'sites: for (gap, probe_id, step, targets) in sites {
            let decision_id = new_id("dec");
            for target in targets {
                if out.len() >= opts.budget {
                    break 'sites;
                }
                let delta = self
                    .propose_nudge(&probe_id, step, target, opts.margin)
                    .with_context(|| format!("proposing {probe_id}[{step}] -> token {target}"))?;
                let rec = self.run_trial(Candidate {
                    delta,
                    proposer: "heuristic:near_tie".into(),
                    rationale: Some(format!("gap {gap:.3} at {probe_id}[{step}]")),
                    decision_id: decision_id.clone(),
                    prediction: None,
                })?;
                on_trial(&rec);
                let keep = rec.verdict.keep;
                let id = rec.id.clone();
                out.push(rec);
                if keep && opts.auto_keep {
                    self.keep(&id)?;
                    // Baseline changed; remaining sites are stale. Stop and let the caller re-run.
                    break 'sites;
                }
            }
        }
        Ok(out)
    }

    fn backend_eos(&self) -> Vec<u32> {
        ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]
            .iter()
            .filter_map(|t| self.backend.token_id(t))
            .collect()
    }
}
