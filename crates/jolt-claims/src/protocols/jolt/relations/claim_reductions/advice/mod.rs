//! Two-phase advice claim-reduction symbolic relations (cycle -> address).

use crate::protocols::jolt::{JoltAdviceKind, PrecommittedReductionDimensions};

/// `(advice kind, two-phase dimensions)` shape shared by the advice cycle- and
/// address-phase reductions.
pub type AdviceReductionShape = (JoltAdviceKind, PrecommittedReductionDimensions);

mod address_phase;
mod cycle_phase;

pub use address_phase::*;
pub use cycle_phase::*;
