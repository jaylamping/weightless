//! Primary Weightless backend on the candle stack.
//!
//! Track 1 picked mistral.rs's model code and kernels (candle-based) for the primary backend. This
//! scaffold starts from candle-transformers' Qwen3 dense model, which has the same shape of API, so
//! the whole loop (load, trace with hidden taps, f32 overlay edits, scoring, ledger) runs today on
//! Qwen3 checkpoints. Porting Qwen3.5 (GDN hybrid) from mistral.rs is the next backend step.
//!
//! Edits are never merged into the weights: `lm_head` deltas are added to the logits in f32 on the
//! host, from the captured final hidden state. That is exact for rank-1 edits, reverts are free, and
//! it works the same on a quantized base later.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;
use weightless_core::backend::*;
use weightless_core::delta::Delta;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceChoice {
    /// CUDA if compiled in and present, else Metal, else CPU.
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LoadOptions {
    #[serde(default)]
    pub device: DeviceChoice,
    /// "f32", "bf16" or "f16". Default: bf16 on GPU, f32 on CPU.
    #[serde(default)]
    pub dtype: Option<String>,
}

pub struct CandleBackend {
    /// Pristine model; cloned per trace so every trace starts with an empty KV cache.
    model: qwen3::Model,
    /// Output projection [vocab, hidden] in the model dtype.
    lm_head: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    info: ModelInfo,
    caps: Capabilities,
    tensors: Vec<TensorInfo>,
    deltas: Vec<Delta>,
    eos: Vec<u32>,
    chatml: bool,
}

fn pick_device(c: DeviceChoice) -> Result<Device> {
    Ok(match c {
        DeviceChoice::Cpu => Device::Cpu,
        DeviceChoice::Cuda => Device::new_cuda(0)?,
        DeviceChoice::Metal => Device::new_metal(0)?,
        DeviceChoice::Auto => {
            if candle_core::utils::cuda_is_available() {
                Device::new_cuda(0)?
            } else if candle_core::utils::metal_is_available() {
                Device::new_metal(0)?
            } else {
                Device::Cpu
            }
        }
    })
}

fn device_name(d: &Device) -> String {
    match d {
        Device::Cpu => "cpu".into(),
        Device::Cuda(_) => "cuda:0".into(),
        Device::Metal(_) => "metal:0".into(),
    }
}

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "f32" => DType::F32,
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        _ => bail!("unsupported dtype {s}"),
    })
}

fn dtype_name(d: DType) -> String {
    format!("{d:?}").to_lowercase()
}

fn safetensor_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no .safetensors files in {}", dir.display());
    }
    Ok(files)
}

/// sha256 over config.json and each shard's safetensors header. Cheap (no weight bytes read) and
/// changes whenever names, shapes, dtypes or offsets change.
fn fingerprint(config: &[u8], files: &[PathBuf]) -> Result<String> {
    let mut h = Sha256::new();
    h.update(config);
    for f in files {
        let mut file = std::fs::File::open(f)?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len).min(100 << 20) as usize;
        let mut header = vec![0u8; n];
        file.read_exact(&mut header)?;
        h.update(len);
        h.update(&header);
    }
    Ok(hex::encode(h.finalize()))
}

fn read_eos(dir: &Path, config: &serde_json::Value, tok: &Tokenizer) -> Vec<u32> {
    let mut out = Vec::new();
    let mut push = |v: &serde_json::Value| match v {
        serde_json::Value::Number(n) => out.extend(n.as_u64().map(|x| x as u32)),
        serde_json::Value::Array(a) => out.extend(a.iter().filter_map(|x| x.as_u64().map(|x| x as u32))),
        _ => {}
    };
    if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
        if let Ok(g) = serde_json::from_str::<serde_json::Value>(&s) {
            push(&g["eos_token_id"]);
        }
    }
    push(&config["eos_token_id"]);
    out.extend(tok.token_to_id("<|im_end|>"));
    out.sort_unstable();
    out.dedup();
    out
}

impl CandleBackend {
    pub fn load(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self> {
        let dir = dir.as_ref();
        let config_bytes = std::fs::read(dir.join("config.json")).context("reading config.json")?;
        let config_json: serde_json::Value = serde_json::from_slice(&config_bytes)?;
        let arch = config_json["architectures"][0].as_str().unwrap_or("unknown").to_string();
        if arch != "Qwen3ForCausalLM" {
            bail!(
                "{arch} is not supported yet. This backend runs Qwen3 dense models (e.g. Qwen/Qwen3-0.6B); \
                 Qwen3.5 is next, ported from mistral.rs"
            );
        }
        let cfg: qwen3::Config = serde_json::from_value(config_json.clone()).context("parsing Qwen3 config")?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;

        let device = pick_device(opts.device)?;
        let dtype = match &opts.dtype {
            Some(s) => parse_dtype(s)?,
            None if device.is_cpu() => DType::F32,
            None => DType::BF16,
        };
        let files = safetensor_files(dir)?;
        let fp = fingerprint(&config_bytes, &files)?;

        let tensors = {
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files)? };
            let mut t: Vec<TensorInfo> = st
                .tensors()
                .into_iter()
                .map(|(name, v)| TensorInfo {
                    name,
                    shape: v.shape().to_vec(),
                    dtype: format!("{:?}", v.dtype()).to_lowercase(),
                })
                .collect();
            t.sort_by(|a, b| a.name.cmp(&b.name));
            t
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
        let model = qwen3::Model::new(&cfg, vb.clone())?;
        let lm_head_tensor = if cfg.tie_word_embeddings { "model.embed_tokens.weight" } else { "lm_head.weight" };
        let lm_head = vb.get((cfg.vocab_size, cfg.hidden_size), lm_head_tensor)?;

        let eos = read_eos(dir, &config_json, &tokenizer);
        let chatml = tokenizer.token_to_id("<|im_start|>").is_some();
        let id = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();

        let info = ModelInfo {
            id,
            path: dir.display().to_string(),
            architecture: arch,
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_hidden_layers,
            tied_embeddings: cfg.tie_word_embeddings,
            lm_head_tensor: lm_head_tensor.into(),
            fingerprint: fp,
            backend: "weightless-candle".into(),
            backend_version: format!("{} (candle-transformers qwen3)", env!("CARGO_PKG_VERSION")),
            dtype: dtype_name(dtype),
            device: device_name(&device),
            quantization: "none".into(),
        };
        let caps = Capabilities {
            run: true,
            full_logits: true,
            hidden_taps: TapLevel::FinalOnly,
            weight_read: false,
            delta_edits: DeltaSupport::LmHeadOnly,
            per_request_deltas: false,
            batched_trials: false,
            dtypes: vec!["f32".into(), "bf16".into(), "f16".into()],
            devices: vec!["cpu".into(), "cuda".into(), "metal".into()],
        };
        Ok(Self { model, lm_head, tokenizer, device, dtype, info, caps, tensors, deltas: vec![], eos, chatml })
    }

    fn format_prompt(&self, p: &Prompt) -> String {
        match p.format {
            PromptFormat::Raw => p.text.clone(),
            PromptFormat::Chat if self.chatml => {
                let mut s = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", p.text);
                if !p.thinking {
                    s.push_str("<think>\n\n</think>\n\n");
                }
                s
            }
            PromptFormat::Chat => p.text.clone(),
        }
    }

    fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tokenizer.decode(ids, skip_special).unwrap_or_default()
    }
}

impl Backend for CandleBackend {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tensor_index(&self) -> Vec<TensorInfo> {
        self.tensors.clone()
    }

    fn trace(&mut self, req: &TraceRequest) -> Result<Trace> {
        let start = Instant::now();
        let text = self.format_prompt(&req.prompt);
        let enc = self.tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
        let prompt_ids = enc.get_ids().to_vec();
        if prompt_ids.is_empty() {
            bail!("prompt tokenized to nothing");
        }
        let mut model = self.model.clone();
        let mut input = prompt_ids.clone();
        let mut offset = 0;
        let mut steps = Vec::new();
        let mut generated = Vec::new();
        let mut stop = StopReason::MaxTokens;

        for _ in 0..req.max_new_tokens {
            let x = Tensor::new(input.as_slice(), &self.device)?.unsqueeze(0)?;
            let h = model.forward(&x, offset)?;
            offset += input.len();
            let last = h.i((0, h.dim(1)? - 1))?;
            let mut logits: Vec<f32> =
                self.lm_head.matmul(&last.unsqueeze(1)?)?.squeeze(1)?.to_dtype(DType::F32)?.to_vec1()?;
            let hv: Vec<f32> = last.to_dtype(DType::F32)?.to_vec1()?;
            for d in &self.deltas {
                d.add_to_logits(&hv, &mut logits);
            }

            let k = req.top_k.max(2).min(logits.len());
            let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
            idx.select_nth_unstable_by(k - 1, |a, b| {
                logits[*b as usize].total_cmp(&logits[*a as usize]).then(a.cmp(b))
            });
            idx.truncate(k);
            idx.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]).then(a.cmp(b)));
            let token = idx[0];
            let top: Vec<TokenLogit> = idx
                .iter()
                .map(|&i| TokenLogit { id: i, text: self.decode(&[i], false), logit: logits[i as usize] })
                .collect();
            let hidden_norm = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
            steps.push(Step {
                token_id: token,
                text: top[0].text.clone(),
                gap: top[0].logit - top[1].logit,
                top,
                hidden_norm,
                hidden: req.capture_hidden.then_some(hv),
            });
            generated.push(token);
            if self.eos.contains(&token) {
                stop = StopReason::Eos;
                break;
            }
            input = vec![token];
        }

        Ok(Trace {
            prompt_tokens: prompt_ids.len(),
            text: self.decode(&generated, true),
            steps,
            stop,
            wall_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn token_logits(&self, token: u32, hiddens: &[&[f32]]) -> Result<Vec<f32>> {
        if hiddens.is_empty() {
            return Ok(vec![]);
        }
        let row = self.lm_head.i(token as usize)?.unsqueeze(1)?; // [hidden, 1]
        let flat: Vec<f32> = hiddens.iter().flat_map(|h| h.iter().copied()).collect();
        let hs = Tensor::from_vec(flat, (hiddens.len(), self.info.hidden_size), &self.device)?.to_dtype(self.dtype)?;
        Ok(hs.matmul(&row)?.squeeze(1)?.to_dtype(DType::F32)?.to_vec1()?)
    }

    fn apply(&mut self, delta: &Delta) -> Result<()> {
        if delta.tensor != self.info.lm_head_tensor {
            bail!("this backend only edits {} so far (got {})", self.info.lm_head_tensor, delta.tensor);
        }
        if delta.terms.iter().any(|t| t.v.len() != self.info.hidden_size) {
            bail!("delta input dim does not match hidden size {}", self.info.hidden_size);
        }
        if self.deltas.iter().any(|d| d.id == delta.id) {
            bail!("delta {} already applied", delta.id);
        }
        self.deltas.push(delta.clone());
        Ok(())
    }

    fn revert(&mut self, delta_id: &str) -> Result<()> {
        let n = self.deltas.len();
        self.deltas.retain(|d| d.id != delta_id);
        if self.deltas.len() == n {
            bail!("delta {delta_id} is not applied");
        }
        Ok(())
    }

    fn active_deltas(&self) -> Vec<String> {
        self.deltas.iter().map(|d| d.id.clone()).collect()
    }

    fn token_text(&self, id: u32) -> String {
        self.decode(&[id], false)
    }

    fn token_id(&self, text: &str) -> Option<u32> {
        self.tokenizer.token_to_id(text)
    }
}
