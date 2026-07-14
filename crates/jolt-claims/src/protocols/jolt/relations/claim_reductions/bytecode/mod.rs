//! Two-phase committed-bytecode claim-reduction symbolic relations.

use crate::protocols::jolt::PrecommittedReductionDimensions;

/// `(two-phase dimensions, chunk count)` shape shared by the committed-bytecode
/// cycle- and address-phase reductions.
pub type BytecodeReductionShape = (PrecommittedReductionDimensions, usize);

/// The cycle-phase shape: [`BytecodeReductionShape`] plus the staged-val count
/// its eta fold spans — five in base mode, six in lattice mode (the store
/// stage). The address phase stays stage-count-agnostic (it consumes the
/// single cycle-phase intermediate opening).
pub type BytecodeReductionCycleShape = (PrecommittedReductionDimensions, usize, usize);

mod address_phase;
mod cycle_phase;

pub use address_phase::*;
pub use cycle_phase::*;
