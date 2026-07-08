//! Two-phase committed-bytecode claim-reduction symbolic relations.

use crate::protocols::jolt::PrecommittedReductionDimensions;

/// `(two-phase dimensions, chunk count)` shape shared by the committed-bytecode
/// cycle- and address-phase reductions.
pub type BytecodeReductionShape = (PrecommittedReductionDimensions, usize);

mod address_phase;
mod cycle_phase;

pub use address_phase::*;
pub use cycle_phase::*;
