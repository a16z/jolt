use crate::protocols::jolt::PrecommittedReductionDimensions;

pub type BytecodeReductionShape = PrecommittedReductionDimensions;

mod address_phase;
mod cycle_phase;

pub use address_phase::*;
pub use cycle_phase::*;
