//! The ledger: every trial (kept or rejected, including pre-flight rejects) as one row, in SQLite.
//! It is the navigator model's training data, so rows carry the session context (goal, spec, model,
//! backend, dtype, decode path), the proposer, predictions made before the gate, per-gate results
//! with reason codes, and cost. Field list agreed with docs/research/08-navigator-model.md §4.
//!
//! Dense delta vectors live in a blob column and are left out of the JSONL export by default.

use crate::backend::{DecodePath, ModelInfo};
use crate::delta::Delta;
use crate::gate::{ProbeResult, SiteFeatures, Verdict};
use crate::spec::Spec;
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

pub const SCHEMA_VERSION: i64 = 1;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    goal_text TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    model_json TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    backend TEXT NOT NULL,
    dtype TEXT NOT NULL,
    quantization TEXT NOT NULL,
    device TEXT NOT NULL,
    decode_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    parent_id TEXT,
    trial_id TEXT,
    created_at TEXT NOT NULL,
    delta_ids_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    snapshot_id TEXT,
    created_at TEXT NOT NULL,
    proposer TEXT NOT NULL,
    rationale TEXT,
    delta_json TEXT NOT NULL,
    delta_blob BLOB NOT NULL,
    site_json TEXT NOT NULL,
    prediction_json TEXT,
    results_json TEXT NOT NULL,
    keep INTEGER NOT NULL,
    stage TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    gates_json TEXT NOT NULL,
    wall_ms INTEGER NOT NULL,
    kept INTEGER NOT NULL DEFAULT 0,
    hidden_ref TEXT,
    request_id TEXT
);
CREATE TABLE IF NOT EXISTS dial_requests (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    user_words TEXT NOT NULL,
    request_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trials_session ON trials(session_id, created_at);
CREATE INDEX IF NOT EXISTS trials_decision ON trials(decision_id);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    detail_json TEXT
);
CREATE TABLE IF NOT EXISTS agent_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls_json TEXT,
    tool_result_ref TEXT,
    created_at TEXT NOT NULL
);
"#;

/// Who proposed a trial: "user", "heuristic:near_tie", "agent:<model>", "navigator:<version>".
pub type Proposer = String;

/// A scorer's guess written before the gate runs, so the navigator can be graded against outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub predictor: String,
    pub p_keep: Option<f32>,
    pub margin: Option<f32>,
}

/// Follow-up facts about a trial learned later.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Kept,
    RevertedLater,
    StackRegression,
    ExportFlip,
    Reconfirmed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialRecord {
    pub id: String,
    pub session_id: String,
    /// Groups every candidate considered at the same edit site, so kept vs rejected siblings
    /// become preference pairs.
    pub decision_id: String,
    /// The dial request (driver's typed version of the user's words) that caused this trial.
    pub request_id: Option<String>,
    pub snapshot_id: Option<String>,
    pub created_at: String,
    pub proposer: Proposer,
    pub rationale: Option<String>,
    pub delta: serde_json::Value,
    pub site: SiteFeatures,
    pub prediction: Option<Prediction>,
    pub results: Vec<ProbeResult>,
    pub verdict: Verdict,
    pub wall_ms: u64,
    pub kept: bool,
}

pub struct Ledger {
    conn: Connection,
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

impl Ledger {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("opening ledger {}", path.as_ref().display()))?;
        Self::init(conn)
    }

    pub fn in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch(SCHEMA)?;
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;
        Ok(Self { conn })
    }

    pub fn start_session(&self, id: &str, spec: &Spec, model: &ModelInfo, decode: DecodePath) -> Result<()> {
        self.conn.execute(
            "INSERT INTO sessions (id, schema_version, started_at, goal_text, spec_json, model_json,
                model_fingerprint, backend, dtype, quantization, device, decode_path)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                id,
                SCHEMA_VERSION,
                now(),
                spec.goal,
                serde_json::to_string(spec)?,
                serde_json::to_string(model)?,
                model.fingerprint,
                format!("{} {}", model.backend, model.backend_version),
                model.dtype,
                model.quantization,
                model.device,
                serde_json::to_string(&decode)?,
            ],
        )?;
        Ok(())
    }

    pub fn insert_trial(&self, t: &TrialRecord, delta: &Delta) -> Result<()> {
        self.conn.execute(
            "INSERT INTO trials (id, session_id, decision_id, snapshot_id, created_at, proposer, rationale,
                delta_json, delta_blob, site_json, prediction_json, results_json, keep, stage,
                reasons_json, gates_json, wall_ms, kept, request_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
            params![
                t.id,
                t.session_id,
                t.decision_id,
                t.snapshot_id,
                t.created_at,
                t.proposer,
                t.rationale,
                serde_json::to_string(&t.delta)?,
                serde_json::to_vec(delta)?,
                serde_json::to_string(&t.site)?,
                t.prediction.as_ref().map(serde_json::to_string).transpose()?,
                serde_json::to_string(&t.results)?,
                t.verdict.keep,
                t.verdict.stage,
                serde_json::to_string(&t.verdict.reasons)?,
                serde_json::to_string(&t.verdict.gates)?,
                t.wall_ms as i64,
                t.kept,
                t.request_id,
            ],
        )?;
        Ok(())
    }

    /// Record a dial request: the user's plain words and the driver's typed request
    /// (docs/research/08-navigator-model.md §1a). Trials point back to it by `request_id`.
    pub fn insert_dial_request(&self, id: &str, session_id: &str, user_words: &str, request: &serde_json::Value) -> Result<()> {
        self.conn.execute(
            "INSERT INTO dial_requests (id, session_id, created_at, user_words, request_json) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, session_id, now(), user_words, request.to_string()],
        )?;
        Ok(())
    }

    pub fn load_delta(&self, trial_id: &str) -> Result<Delta> {
        let blob: Vec<u8> = self
            .conn
            .query_row("SELECT delta_blob FROM trials WHERE id = ?1", params![trial_id], |r| r.get(0))
            .with_context(|| format!("no trial {trial_id}"))?;
        Ok(serde_json::from_slice(&blob)?)
    }

    pub fn trial(&self, trial_id: &str) -> Result<Option<TrialRecord>> {
        self.conn
            .query_row(&format!("{TRIAL_SELECT} WHERE id = ?1"), params![trial_id], row_to_trial)
            .optional()?
            .transpose()
    }

    pub fn mark_kept(&self, trial_id: &str, snapshot_id: &str, parent: Option<&str>, delta_ids: &[String], session_id: &str) -> Result<()> {
        self.conn.execute("UPDATE trials SET kept = 1 WHERE id = ?1", params![trial_id])?;
        self.conn.execute(
            "INSERT INTO snapshots (id, session_id, parent_id, trial_id, created_at, delta_ids_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![snapshot_id, session_id, parent, trial_id, now(), serde_json::to_string(delta_ids)?],
        )?;
        self.event(trial_id, EventKind::Kept, None)
    }

    pub fn event(&self, trial_id: &str, kind: EventKind, detail: Option<serde_json::Value>) -> Result<()> {
        self.conn.execute(
            "INSERT INTO events (trial_id, kind, created_at, detail_json) VALUES (?1, ?2, ?3, ?4)",
            params![
                trial_id,
                serde_json::to_value(kind)?.as_str().unwrap_or_default(),
                now(),
                detail.map(|d| d.to_string())
            ],
        )?;
        Ok(())
    }

    pub fn agent_turn(
        &self,
        session_id: &str,
        turn: i64,
        role: &str,
        content: &str,
        tool_calls: Option<&serde_json::Value>,
        tool_result_ref: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO agent_turns (session_id, turn, role, content, tool_calls_json, tool_result_ref, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![session_id, turn, role, content, tool_calls.map(|v| v.to_string()), tool_result_ref, now()],
        )?;
        Ok(())
    }

    /// Most recent trials first.
    pub fn recent_trials(&self, session_id: Option<&str>, limit: usize) -> Result<Vec<TrialRecord>> {
        let sql = format!(
            "{TRIAL_SELECT} WHERE (?1 IS NULL OR session_id = ?1) ORDER BY created_at DESC LIMIT ?2"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![session_id, limit as i64], row_to_trial)?;
        rows.map(|r| r?).collect()
    }

    pub fn counts(&self, session_id: &str) -> Result<(i64, i64, i64)> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(keep), 0), COALESCE(SUM(kept), 0) FROM trials WHERE session_id = ?1",
            params![session_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?)
    }

    /// One JSON object per trial, joined with its session (goal, spec, model, backend, decode path)
    /// and follow-up events. This is the navigator's training file.
    pub fn export_jsonl(&self, out: &mut dyn Write, include_vectors: bool) -> Result<usize> {
        let mut stmt = self.conn.prepare(
            "SELECT t.id, t.session_id, t.decision_id, t.snapshot_id, t.created_at, t.proposer, t.rationale,
                    t.delta_json, t.delta_blob, t.site_json, t.prediction_json, t.results_json, t.keep, t.stage,
                    t.reasons_json, t.gates_json, t.wall_ms, t.kept,
                    s.schema_version, s.goal_text, s.spec_json, s.model_json, s.backend, s.dtype,
                    s.quantization, s.device, s.decode_path,
                    t.request_id, d.user_words, d.request_json,
                    (SELECT json_group_array(json_object('kind', kind, 'at', created_at, 'detail', detail_json))
                       FROM events e WHERE e.trial_id = t.id)
             FROM trials t JOIN sessions s ON s.id = t.session_id
             LEFT JOIN dial_requests d ON d.id = t.request_id ORDER BY t.created_at",
        )?;
        let mut rows = stmt.query([])?;
        let mut n = 0;
        while let Some(r) = rows.next()? {
            let j = |i: usize| -> Result<serde_json::Value> {
                let s: Option<String> = r.get(i)?;
                Ok(s.map(|s| serde_json::from_str(&s)).transpose()?.unwrap_or(serde_json::Value::Null))
            };
            let mut obj = serde_json::json!({
                "schema_version": r.get::<_, i64>(18)?,
                "trial_id": r.get::<_, String>(0)?,
                "session_id": r.get::<_, String>(1)?,
                "decision_id": r.get::<_, String>(2)?,
                "snapshot_id": r.get::<_, Option<String>>(3)?,
                "created_at": r.get::<_, String>(4)?,
                "proposer": r.get::<_, String>(5)?,
                "rationale": r.get::<_, Option<String>>(6)?,
                "delta": j(7)?,
                "site": j(9)?,
                "prediction": j(10)?,
                "results": j(11)?,
                "keep": r.get::<_, bool>(12)?,
                "stage": r.get::<_, String>(13)?,
                "reasons": j(14)?,
                "gates": j(15)?,
                "wall_ms": r.get::<_, i64>(16)?,
                "kept": r.get::<_, bool>(17)?,
                "session": {
                    "goal_text": r.get::<_, String>(19)?,
                    "spec": j(20)?,
                    "model": j(21)?,
                    "backend": r.get::<_, String>(22)?,
                    "dtype": r.get::<_, String>(23)?,
                    "quantization": r.get::<_, String>(24)?,
                    "device": r.get::<_, String>(25)?,
                    "decode_path": j(26)?,
                },
                "request": {
                    "id": r.get::<_, Option<String>>(27)?,
                    "user_words": r.get::<_, Option<String>>(28)?,
                    "request": j(29)?,
                },
                "events": j(30)?,
            });
            if include_vectors {
                let blob: Vec<u8> = r.get(8)?;
                obj["delta_full"] = serde_json::from_slice(&blob)?;
            }
            serde_json::to_writer(&mut *out, &obj)?;
            out.write_all(b"\n")?;
            n += 1;
        }
        Ok(n)
    }
}

const TRIAL_SELECT: &str = "SELECT id, session_id, decision_id, snapshot_id, created_at, proposer, rationale,
    delta_json, site_json, prediction_json, results_json, keep, stage, reasons_json, gates_json, wall_ms, kept,
    request_id FROM trials";

fn row_to_trial(r: &rusqlite::Row) -> rusqlite::Result<Result<TrialRecord>> {
    let parse = |i: usize| -> rusqlite::Result<String> { r.get(i) };
    let build = || -> Result<TrialRecord> {
        let prediction: Option<String> = r.get(9)?;
        Ok(TrialRecord {
            id: r.get(0)?,
            session_id: r.get(1)?,
            decision_id: r.get(2)?,
            snapshot_id: r.get(3)?,
            created_at: r.get(4)?,
            proposer: r.get(5)?,
            rationale: r.get(6)?,
            delta: serde_json::from_str(&parse(7)?)?,
            site: serde_json::from_str(&parse(8)?)?,
            prediction: prediction.map(|p| serde_json::from_str(&p)).transpose()?,
            results: serde_json::from_str(&parse(10)?)?,
            verdict: Verdict {
                keep: r.get(11)?,
                stage: r.get(12)?,
                reasons: serde_json::from_str(&parse(13)?)?,
                gates: serde_json::from_str(&parse(14)?)?,
            },
            wall_ms: r.get::<_, i64>(15)? as u64,
            kept: r.get(16)?,
            request_id: r.get(17)?,
        })
    };
    Ok(build())
}
