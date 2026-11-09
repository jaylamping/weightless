//! Tauri shell: exposes the Workbench to the React dashboard as commands. Engine work runs on
//! blocking threads so the UI stays responsive; explore streams each trial as a "trial" event.

use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use weightless_candle::{CandleBackend, LoadOptions};
use weightless_core::backend::{Capabilities, ModelInfo, Prompt, TensorInfo, Trace, TraceRequest};
use weightless_core::ledger::TrialRecord;
use weightless_core::workbench::Snapshot;
use weightless_core::{Candidate, ExploreOptions, Ledger, Spec, Workbench};

struct AppState {
    wb: Arc<Mutex<Option<Workbench>>>,
    ledger_path: PathBuf,
}

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    format!("{e:#}")
}

/// Run `f` against the loaded workbench on a blocking thread.
async fn with_wb<T: Send + 'static>(
    state: &State<'_, AppState>,
    f: impl FnOnce(&mut Workbench) -> anyhow::Result<T> + Send + 'static,
) -> CmdResult<T> {
    let wb = state.wb.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = wb.lock().map_err(err)?;
        let wb = guard.as_mut().ok_or_else(|| "no model loaded".to_string())?;
        f(wb).map_err(err)
    })
    .await
    .map_err(err)?
}

#[derive(Serialize)]
struct Status {
    info: ModelInfo,
    capabilities: Capabilities,
    session_id: String,
    spec: Spec,
    snapshot: Snapshot,
    trials: i64,
    passed: i64,
    kept: i64,
    ledger_path: String,
}

fn status_of(wb: &Workbench, ledger_path: &str) -> anyhow::Result<Status> {
    let (trials, passed, kept) = wb.ledger().counts(wb.session_id())?;
    Ok(Status {
        info: wb.backend().info().clone(),
        capabilities: wb.backend().capabilities().clone(),
        session_id: wb.session_id().to_string(),
        spec: wb.spec().clone(),
        snapshot: wb.snapshot().clone(),
        trials,
        passed,
        kept,
        ledger_path: ledger_path.to_string(),
    })
}

#[tauri::command]
async fn load_model(state: State<'_, AppState>, path: String, options: LoadOptions) -> CmdResult<Status> {
    let wb = state.wb.clone();
    let ledger_path = state.ledger_path.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let backend = CandleBackend::load(&path, &options).map_err(err)?;
        let ledger = Ledger::open(&ledger_path).map_err(err)?;
        let new = Workbench::new(Box::new(backend), ledger, Spec::default()).map_err(err)?;
        let s = status_of(&new, &ledger_path.display().to_string()).map_err(err)?;
        *wb.lock().map_err(err)? = Some(new);
        Ok(s)
    })
    .await
    .map_err(err)?
}

#[tauri::command]
async fn status(state: State<'_, AppState>) -> CmdResult<Option<Status>> {
    let p = state.ledger_path.display().to_string();
    let guard = state.wb.lock().map_err(err)?;
    guard.as_ref().map(|wb| status_of(wb, &p)).transpose().map_err(err)
}

#[tauri::command]
async fn tensors(state: State<'_, AppState>) -> CmdResult<Vec<TensorInfo>> {
    with_wb(&state, |wb| Ok(wb.backend().tensor_index())).await
}

#[tauri::command]
async fn set_spec(state: State<'_, AppState>, spec: Spec) -> CmdResult<()> {
    with_wb(&state, move |wb| wb.set_spec(spec)).await
}

/// Record the driver's typed dial request for the user's words; later trials link to it.
#[tauri::command]
async fn set_dial_request(state: State<'_, AppState>, user_words: String, request: serde_json::Value) -> CmdResult<String> {
    with_wb(&state, move |wb| wb.set_dial_request(&user_words, &request)).await
}

#[tauri::command]
async fn baseline(state: State<'_, AppState>) -> CmdResult<Vec<Trace>> {
    with_wb(&state, |wb| Ok(wb.baseline()?.iter().map(Trace::without_hidden).collect())).await
}

#[tauri::command]
async fn trace_prompt(state: State<'_, AppState>, prompt: Prompt, max_new_tokens: usize, top_k: usize) -> CmdResult<Trace> {
    with_wb(&state, move |wb| {
        wb.trace(&TraceRequest { prompt, max_new_tokens, top_k, capture_hidden: false, decode: Default::default() })
    })
    .await
}

/// Try making `target` win at `probe_id[step]`. Logged with proposer "user".
#[tauri::command]
async fn try_nudge(
    state: State<'_, AppState>,
    probe_id: String,
    step: usize,
    target: u32,
    margin: f32,
) -> CmdResult<TrialRecord> {
    with_wb(&state, move |wb| {
        let delta = wb.propose_nudge(&probe_id, step, target, margin)?;
        wb.run_trial(Candidate {
            delta,
            proposer: "user".into(),
            rationale: None,
            decision_id: weightless_core::delta::new_id("dec"),
            prediction: None,
        })
    })
    .await
}

#[tauri::command]
async fn explore(app: AppHandle, state: State<'_, AppState>, options: ExploreOptions) -> CmdResult<Vec<TrialRecord>> {
    with_wb(&state, move |wb| {
        wb.explore(&options, &mut |r| {
            let _ = app.emit("trial", r);
        })
    })
    .await
}

#[tauri::command]
async fn keep(state: State<'_, AppState>, trial_id: String) -> CmdResult<Snapshot> {
    with_wb(&state, move |wb| wb.keep(&trial_id)).await
}

#[tauri::command]
async fn unkeep(state: State<'_, AppState>, trial_id: String) -> CmdResult<Snapshot> {
    with_wb(&state, move |wb| wb.unkeep(&trial_id)).await
}

#[tauri::command]
async fn trials(state: State<'_, AppState>, limit: usize, all_sessions: bool) -> CmdResult<Vec<TrialRecord>> {
    with_wb(&state, move |wb| {
        let sid = (!all_sessions).then(|| wb.session_id().to_string());
        wb.ledger().recent_trials(sid.as_deref(), limit)
    })
    .await
}

#[tauri::command]
async fn export_ledger(state: State<'_, AppState>, path: String, vectors: bool) -> CmdResult<usize> {
    with_wb(&state, move |wb| {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
        wb.ledger().export_jsonl(&mut f, vectors)
    })
    .await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&dir)?;
            app.manage(AppState { wb: Arc::new(Mutex::new(None)), ledger_path: dir.join("ledger.db") });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_model,
            status,
            tensors,
            set_spec,
            set_dial_request,
            baseline,
            trace_prompt,
            try_nudge,
            explore,
            keep,
            unkeep,
            trials,
            export_ledger
        ])
        .run(tauri::generate_context!())
        .expect("error while running Weightless");
}
