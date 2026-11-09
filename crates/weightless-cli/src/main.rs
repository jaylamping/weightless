//! `weightless`: the engine without the desktop app. Handy for scripted data generation on a
//! headless box and for checking a model loads before opening it in the app.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use weightless_candle::{CandleBackend, DeviceChoice, LoadOptions};
use weightless_core::backend::{Backend, Prompt, PromptFormat, TraceRequest};
use weightless_core::{ExploreOptions, Ledger, Spec, Workbench};

#[derive(Parser)]
#[command(name = "weightless", version, about = "Weightless edit engine (headless)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args)]
struct ModelArgs {
    /// Local model directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "auto")]
    device: Dev,
    /// f32, bf16 or f16 (default: bf16 on GPU, f32 on CPU).
    #[arg(long)]
    dtype: Option<String>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Dev {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print model info, capabilities and tensor count.
    Info(ModelArgs),
    /// Greedy trace of one prompt with per-step top-k and gaps.
    Trace {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long)]
        raw: bool,
    },
    /// Run near-tie trials against a spec and write them to the ledger.
    Explore {
        #[command(flatten)]
        m: ModelArgs,
        /// Spec JSON (goal, probes, controls). Default: the built-in concision spec.
        #[arg(long)]
        spec: Option<PathBuf>,
        #[arg(long, default_value = "weightless-ledger.db")]
        ledger: PathBuf,
        #[arg(long, default_value_t = 20)]
        budget: usize,
        #[arg(long, default_value_t = 2.0)]
        max_gap: f32,
        #[arg(long, default_value_t = 0.5)]
        margin: f32,
        /// Keep trials that pass every gate (then stop; re-run to continue from the new snapshot).
        #[arg(long)]
        auto_keep: bool,
    },
    /// Write the ledger as JSONL (one trial per line) for navigator training.
    Export {
        #[arg(long, default_value = "weightless-ledger.db")]
        ledger: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Include full delta vectors.
        #[arg(long)]
        vectors: bool,
    },
    /// Print the built-in spec as JSON, as a starting point for your own.
    DefaultSpec,
}

fn load(m: &ModelArgs) -> Result<CandleBackend> {
    let device = match m.device {
        Dev::Auto => DeviceChoice::Auto,
        Dev::Cpu => DeviceChoice::Cpu,
        Dev::Cuda => DeviceChoice::Cuda,
        Dev::Metal => DeviceChoice::Metal,
    };
    CandleBackend::load(&m.model, &LoadOptions { device, dtype: m.dtype.clone() })
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Info(m) => {
            let b = load(&m)?;
            println!("{}", serde_json::to_string_pretty(b.info())?);
            println!("{}", serde_json::to_string_pretty(b.capabilities())?);
            println!("{} tensors", b.tensor_index().len());
        }
        Cmd::Trace { m, prompt, max_new_tokens, top_k, raw } => {
            let mut b = load(&m)?;
            let format = if raw { PromptFormat::Raw } else { PromptFormat::Chat };
            let t = b.trace(&TraceRequest {
                prompt: Prompt { text: prompt, format, thinking: false },
                max_new_tokens,
                top_k,
                capture_hidden: false,
                decode: Default::default(),
            })?;
            for (i, s) in t.steps.iter().enumerate() {
                let alts: Vec<String> = s.top.iter().skip(1).map(|c| format!("{:?} {:.2}", c.text, c.logit)).collect();
                println!("{i:>4} {:<16} gap {:>6.2}  | {}", format!("{:?}", s.text), s.gap, alts.join("  "));
            }
            println!("\n{}\n({} tokens, {:?}, {} ms)", t.text, t.steps.len(), t.stop, t.wall_ms);
        }
        Cmd::Explore { m, spec, ledger, budget, max_gap, margin, auto_keep } => {
            let spec: Spec = match spec {
                Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
                None => Spec::default(),
            };
            let b = load(&m)?;
            let mut wb = Workbench::new(Box::new(b), Ledger::open(&ledger)?, spec)?;
            eprintln!("session {} — tracing baseline…", wb.session_id());
            wb.baseline()?;
            let opts = ExploreOptions { budget, max_gap, margin, auto_keep };
            wb.explore(&opts, &mut |r| {
                let site = &r.delta["origin"];
                println!(
                    "{} {:<5} {:<9} {}[{}] {:?}->{:?} gap {:.2} {:?} {}ms",
                    r.id,
                    if r.verdict.keep { "PASS" } else { "fail" },
                    r.verdict.stage,
                    site["probe_id"].as_str().unwrap_or("?"),
                    site["step"],
                    site["from_text"].as_str().unwrap_or(""),
                    site["target_text"].as_str().unwrap_or(""),
                    site["gap_before"].as_f64().unwrap_or(0.0),
                    r.verdict.reasons,
                    r.wall_ms
                );
            })?;
            let (n, pass, kept) = wb.ledger().counts(wb.session_id())?;
            eprintln!("{n} trials, {pass} passed gates, {kept} kept → {}", ledger.display());
        }
        Cmd::Export { ledger, out, vectors } => {
            let l = Ledger::open(&ledger)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
            let n = l.export_jsonl(&mut f, vectors)?;
            eprintln!("wrote {n} trials to {}", out.display());
        }
        Cmd::DefaultSpec => println!("{}", serde_json::to_string_pretty(&Spec::default())?),
    }
    Ok(())
}
