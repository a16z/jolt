//! Two-phase advice claim-reduction symbolic relations (cycle -> address), split
//! into per-kind types so each claims struct carries a single non-`Option` slot.

mod address_phase;
mod cycle_phase;

pub use address_phase::*;
pub use cycle_phase::*;
