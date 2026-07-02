//! Two-phase program-image (initial RAM) claim-reduction symbolic relations.

mod address_phase;
mod cycle_phase;

pub use address_phase::*;
pub use cycle_phase::*;
