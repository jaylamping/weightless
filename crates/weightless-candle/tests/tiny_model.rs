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

    // Keep the first passing trial (if any), and check the baseline moved and lineage was written.
    if let Some(good) = recs.iter().find(|r| r.verdict.keep) {
        let site = good.delta["origin"]["probe_id"].as_str().unwrap().to_string();
        let before = wb.baseline().unwrap().to_vec();
        let snap = wb.keep(&good.id).unwrap();
        assert_eq!(snap.trials, vec![good.id.clone()]);
        let after = wb.baseline().unwrap().to_vec();
        let i = wb.spec().probes.iter().position(|p| p.id == site).unwrap();
        assert_ne!(before[i].token_ids(), after[i].token_ids());
        wb.unkeep(&good.id).unwrap();
        assert_eq!(wb.baseline().unwrap()[i].token_ids(), before[i].token_ids());
    }

    let mut out = Vec::new();
    let n = wb.ledger().export_jsonl(&mut out, false).unwrap();
    assert_eq!(n, 12);
    let first: serde_json::Value = serde_json::from_slice(out.split(|c| *c == b'\n').next().unwrap()).unwrap();
    assert_eq!(first["schema_version"], 1);
    assert_eq!(first["proposer"], "heuristic:near_tie");
    assert_eq!(first["session"]["goal_text"], "test: change probes without touching controls");
    assert!(first["session"]["model"]["fingerprint"].as_str().unwrap().len() == 64);
    assert!(first.get("delta_full").is_none());
    assert_eq!(first["request"]["id"], req_id.as_str());
    assert_eq!(first["request"]["user_words"], "make it different");
    let (total, _, _) = wb.ledger().counts(wb.session_id()).unwrap();
    assert_eq!(total, 12);
}
