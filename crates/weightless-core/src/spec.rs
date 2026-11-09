//! A session's spec: the goal, the probes an edit should improve, and the controls it must not touch.
//! Lessons from Joey's Qwen campaign: judge per prompt with controls, never by total length alone.

use crate::backend::{Prompt, PromptFormat};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeRole {
    /// The edit should improve this prompt toward the objective.
    Probe,
    /// The edit must leave this prompt's output unchanged.
    Control,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub id: String,
    pub role: ProbeRole,
    pub prompt: Prompt,
    /// Text the completion must still contain (e.g. "56" for 8x7).
    #[serde(default)]
    pub required_answer: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Concision: at least one probe gets shorter, none gets longer.
    Shorter,
    /// No objective: gates check controls only. Useful for free exploration.
    ControlsOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spec {
    /// The goal in plain words, as the user or agent stated it. Stored on the session for the navigator.
    pub goal: String,
    pub objective: Objective,
    pub probes: Vec<Probe>,
    pub max_new_tokens: usize,
    pub top_k: usize,
    /// Edits whose largest element exceeds this are rejected (fp16/bf16 safety, cf. arXiv 2611.05341).
    pub max_abs_delta: f32,
}

impl Spec {
    /// Probe ids must be unique and non-empty, and there must be at least one probe.
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.probes.iter().any(|p| p.role == ProbeRole::Probe) {
            anyhow::bail!("spec needs at least one probe (role \"probe\")");
        }
        let mut seen = std::collections::HashSet::new();
        for p in &self.probes {
            if p.id.trim().is_empty() {
                anyhow::bail!("probe ids must not be empty");
            }
            if !seen.insert(p.id.as_str()) {
                anyhow::bail!("duplicate probe id {:?}", p.id);
            }
        }
        if self.max_new_tokens == 0 || self.top_k < 2 {
            anyhow::bail!("max_new_tokens must be > 0 and top_k >= 2");
        }
        if !self.max_abs_delta.is_finite() || self.max_abs_delta <= 0.0 {
            anyhow::bail!("max_abs_delta must be a positive finite number");
        }
        Ok(())
    }

    pub fn probe(&self, id: &str) -> Option<&Probe> {
        self.probes.iter().find(|p| p.id == id)
    }
}

fn chat(text: &str) -> Prompt {
    Prompt { text: text.into(), format: PromptFormat::Chat, thinking: false }
}

impl Default for Spec {
    /// A small concision campaign modelled on Joey's Qwen3.5 runs.
    fn default() -> Self {
        let p = |id: &str, role, text: &str, ans: Option<&str>| Probe {
            id: id.into(),
            role,
            prompt: chat(text),
            required_answer: ans.map(Into::into),
        };
        Spec {
            goal: "Answer more concisely without changing the control answers.".into(),
            objective: Objective::Shorter,
            probes: vec![
                p("rectangle", ProbeRole::Probe, "What is a rectangle? Answer briefly.", None),
                p("photosynthesis", ProbeRole::Probe, "Explain photosynthesis in one or two sentences.", None),
                p("capital", ProbeRole::Probe, "What is the capital of France?", Some("Paris")),
                p("mul", ProbeRole::Control, "What is 8x7?", Some("56")),
                p("square", ProbeRole::Control, "A square has side 4. What is its area?", Some("16")),
            ],
            max_new_tokens: 96,
            top_k: 8,
            max_abs_delta: 1.0e4,
        }
    }
}
