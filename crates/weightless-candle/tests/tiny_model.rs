//! End-to-end tests on a tiny random Qwen3 written to a temp dir: trace, overlay parity, revert,
//! exact pre-flight prediction, and a full explore → ledger → keep → export loop.

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::qwen3;
use std::path::Path;
use weightless_candle::{CandleBackend, DeviceChoice, LoadOptions};
use weightless_core::backend::*;
use weightless_core::delta::Delta;
use weightless_core::spec::{Objective, Probe, ProbeRole, Spec};
use weightless_core::{ExploreOptions, Ledger, Workbench};

const VOCAB: usize = 64;
const SEED: u64 = 7;

/// Deterministic weights (candle's CPU RNG can't be seeded): every run builds the same model.
fn fill_deterministic(vm: &VarMap) {
    let data = vm.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    let mut state = SEED;
    for name in names {
        let var = &data[name];
        if name.ends_with("norm.weight") {
            continue; // RMSNorm weights stay at 1.
        }
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let a = if name.contains("embed_tokens") { 1.0 } else { 1.0 / (*dims.last().unwrap() as f32).sqrt() };
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
            })
            .collect();
        var.set(&candle_core::Tensor::from_vec(vals, dims, &Device::Cpu).unwrap()).unwrap();
    }
}

fn write_tiny_model(dir: &Path) {
    let cfg = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": VOCAB, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "head_dim": 8, "attention_bias": false, "num_key_value_heads": 2,
        "max_position_embeddings": 256, "sliding_window": null, "max_window_layers": 2,
        "tie_word_embeddings": false, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
        "use_sliding_window": false, "hidden_act": "silu"
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let c: qwen3::Config = serde_json::from_value(cfg).unwrap();
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    qwen3::ModelForCausalLM::new(&c, vb).unwrap();
    fill_deterministic(&vm);
    vm.save(dir.join("model.safetensors")).unwrap();

    let vocab: serde_json::Map<String, serde_json::Value> =
        (0..VOCAB).map(|i| (format!("w{i}"), serde_json::json!(i))).collect();
    let tok = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "w0"}
    });
    std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
}

fn load(dir: &Path) -> CandleBackend {
    CandleBackend::load(dir, &LoadOptions { device: DeviceChoice::Cpu, dtype: None }).unwrap()
}

fn raw(text: &str) -> Prompt {
    Prompt { text: text.into(), format: PromptFormat::Raw, thinking: false }
}

fn req(text: &str, hidden: bool) -> TraceRequest {
    TraceRequest { prompt: raw(text), max_new_tokens: 12, top_k: 5, capture_hidden: hidden, decode: Default::default() }
}

#[test]
fn trace_overlay_and_revert() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_model(dir.path());
    let mut b = load(dir.path());
    assert_eq!(b.info().lm_head_tensor, "lm_head.weight");
    assert!(b.tensor_index().iter().any(|t| t.name == "lm_head.weight" && t.shape == vec![VOCAB, 32]));

    let base = b.trace(&req("w1 w2 w3", true)).unwrap();
    assert_eq!(base.steps.len(), 12);
    assert_eq!(b.trace(&req("w1 w2 w3", false)).unwrap().token_ids(), base.token_ids(), "deterministic");

    // Promote the runner-up at step 0 by its gap + 0.5.
    let s0 = &base.steps[0];
    let target = s0.top[1].id;
    let h = s0.hidden.as_ref().unwrap();
    let d = Delta::logit_nudge("lm_head.weight", target, h, s0.gap + 0.5).unwrap();
    b.apply(&d).unwrap();
    let edited = b.trace(&req("w1 w2 w3", false)).unwrap();
    assert_eq!(edited.steps[0].token_id, target);
    // Overlay parity: the target's logit is exactly base + gap + 0.5 = leader + 0.5.
    assert!((edited.steps[0].top[0].logit - (s0.top[0].logit + 0.5)).abs() < 1e-4);
    // token_logits agrees with the traced logits.
    let tl = b.token_logits(target, &[h]).unwrap()[0];
    assert!((tl - s0.top[1].logit).abs() < 1e-4);

    b.revert(&d.id).unwrap();
    assert_eq!(b.trace(&req("w1 w2 w3", false)).unwrap().token_ids(), base.token_ids(), "revert is exact");
    assert!(b.revert(&d.id).is_err());
}

#[test]
fn explore_writes_ledger_and_preflight_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_model(dir.path());
    let b = load(dir.path());
    let p = |id: &str, role, text: &str| Probe { id: id.into(), role, prompt: raw(text), required_answer: None };
    let spec = Spec {
        goal: "test: change probes without touching controls".into(),
        objective: Objective::ControlsOnly,
        probes: vec![
            p("a", ProbeRole::Probe, "w1 w2 w3"),
            p("b", ProbeRole::Probe, "w9 w8"),
            p("c", ProbeRole::Control, "w20 w21 w22"),
        ],
        max_new_tokens: 10,
        top_k: 5,
        max_abs_delta: 1e4,
    };
    let ledger_path = dir.path().join("ledger.db");
    let mut wb = Workbench::new(Box::new(b), Ledger::open(&ledger_path).unwrap(), spec).unwrap();
    let req_id = wb
        .set_dial_request("make it different", &serde_json::json!({"behaviour": "test", "direction": "any"}))
        .unwrap();
    let opts = ExploreOptions { budget: 12, max_gap: 1e9, margin: 0.5, auto_keep: false };
    let mut seen = 0;
    let recs = wb.explore(&opts, &mut |_| seen += 1).unwrap();
    assert_eq!(recs.len(), 12);
    assert_eq!(seen, 12);

    assert!(recs.iter().any(|r| r.verdict.stage == "decode"), "no trial reached decode");
    // Pre-flight's predicted first flip must equal the decoded first divergence, on every probe.
    for r in &recs {
        if r.verdict.stage == "decode" {
            for pr in &r.results {
                assert_eq!(pr.predicted_first_flip, pr.first_divergence, "probe {} in {}", pr.probe_id, r.id);
            }
        } else {
            assert!(r.verdict.reasons.iter().any(|c| format!("{c:?}") == "SpillControlPredicted"));
        }
    }

    // Keep/unkeep lineage. The seeded model always yields passing trials, so this always runs.
    let passing: Vec<_> = recs.iter().filter(|r| r.verdict.keep).collect();
    assert!(passing.len() >= 2, "seeded model should give at least 2 passing trials, got {}", passing.len());
    let failing = recs.iter().find(|r| !r.verdict.keep).expect("some trial should fail");
    assert!(wb.keep(&failing.id).unwrap_err().to_string().contains("did not pass"));

    let (g1, g2) = (passing[0], passing[1]);
    let site = g1.delta["origin"]["probe_id"].as_str().unwrap().to_string();
    let i = wb.spec().probes.iter().position(|p| p.id == site).unwrap();
    let before = wb.baseline().unwrap().to_vec();
    let s1 = wb.keep(&g1.id).unwrap();
    assert_eq!(s1.trials, vec![g1.id.clone()]);
    let after = wb.baseline().unwrap().to_vec();
    assert_ne!(before[i].token_ids(), after[i].token_ids());
    let snaps = wb.ledger().snapshots(wb.session_id()).unwrap();
    assert_eq!(snaps.len(), 1);
    assert_eq!((snaps[0].id.as_str(), snaps[0].parent_id.as_deref()), (s1.id.as_deref().unwrap(), None));
    assert_eq!(wb.ledger().counts(wb.session_id()).unwrap().2, 1);
    assert!(wb.ledger().trial(&g1.id).unwrap().unwrap().kept);

    // g2 was gated on the base model, not on s1: keeping it must be refused until re-tested.
    assert!(wb.keep(&g2.id).unwrap_err().to_string().contains("different model state"));
    let re = wb.retest(&g2.id).unwrap();
    assert_eq!(re.snapshot_id, s1.id);
    assert_eq!(re.decision_id, g2.decision_id);
    // Pre-flight stays exact with a kept delta active.
    if re.verdict.stage == "decode" {
        for pr in &re.results {
            assert_eq!(pr.predicted_first_flip, pr.first_divergence, "retest probe {}", pr.probe_id);
        }
    }
    let more = wb.explore(&ExploreOptions { budget: 6, ..opts.clone() }, &mut |_| {}).unwrap();
    for r in more.iter().filter(|r| r.verdict.stage == "decode") {
        assert_eq!(r.snapshot_id, s1.id);
        for pr in &r.results {
            assert_eq!(pr.predicted_first_flip, pr.first_divergence, "stacked probe {} in {}", pr.probe_id, r.id);
        }
    }
    let stacked = std::iter::once(&re).chain(more.iter()).find(|r| r.verdict.keep);
    let g = stacked.expect("seeded model should give a passing trial on top of the first keep");
    {
        let s2 = wb.keep(&g.id).unwrap();
        let snaps = wb.ledger().snapshots(wb.session_id()).unwrap();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[1].parent_id, s1.id, "second snapshot's parent is the first");
        assert_eq!(snaps[1].delta_ids.len(), 2);
        // Only the most recent keep can be undone.
        assert!(wb.unkeep(&g1.id).is_err());
        let back = wb.unkeep(&g.id).unwrap();
        assert_eq!(back.id, s1.id, "unkeep returns to the parent snapshot, which exists in the ledger");
        assert_ne!(back.id, s2.id);
    }
    let base_again = wb.unkeep(&g1.id).unwrap();
    assert_eq!(base_again.id, None);
    assert!(!wb.ledger().trial(&g1.id).unwrap().unwrap().kept);
    assert_eq!(wb.ledger().counts(wb.session_id()).unwrap().2, 0);
    assert_eq!(wb.baseline().unwrap()[i].token_ids(), before[i].token_ids());

    // A new spec starts from the base model.
    wb.keep(&g1.id).unwrap();
    assert_eq!(wb.backend().active_deltas().len(), 1);
    let session = wb.session_id().to_string();
    wb.set_spec(wb.spec().clone()).unwrap();
    assert!(wb.backend().active_deltas().is_empty());
    assert_eq!(wb.snapshot().id, None);
    assert_ne!(wb.session_id(), session);

    let mut out = Vec::new();
    let n = wb.ledger().export_jsonl(&mut out, false).unwrap();
    let (total, _, _) = wb.ledger().counts(&session).unwrap();
    assert_eq!(n as i64, total);
    assert!(total >= 13);
    let first: serde_json::Value = serde_json::from_slice(out.split(|c| *c == b'\n').next().unwrap()).unwrap();
    assert_eq!(first["schema_version"], 1);
    assert_eq!(first["proposer"], "heuristic:near_tie");
    assert_eq!(first["session"]["goal_text"], "test: change probes without touching controls");
    assert!(first["session"]["model"]["fingerprint"].as_str().unwrap().len() == 64);
    assert!(first.get("delta_full").is_none());
    assert_eq!(first["request"]["id"], req_id.as_str());
    assert_eq!(first["request"]["user_words"], "make it different");
}
