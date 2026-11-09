//! Weightless core: the backend-agnostic edit engine.
//!
//! - [`backend`]: the adapter contract every runtime implements, with capability flags.
//! - [`delta`]: rank-1 f32 deltas applied as overlays (never merged into base weights).
//! - [`spec`]: goal, probes and controls for a session.
//! - [`gate`]: pre-flight and decode gates with reason codes.
//! - [`ledger`]: SQLite trial ledger, the navigator model's training data.
//! - [`workbench`]: the edit loop that ties them together.

pub mod backend;
pub mod delta;
pub mod gate;
pub mod ledger;
pub mod spec;
pub mod workbench;

pub use backend::*;
pub use delta::Delta;
pub use ledger::Ledger;
pub use spec::Spec;
pub use workbench::{Candidate, ExploreOptions, Workbench};
