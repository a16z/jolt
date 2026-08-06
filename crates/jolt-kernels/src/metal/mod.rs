//! Apple Metal compute kernels.
//!
//! Reusable field arithmetic and hybrid sumcheck kernels for Apple GPUs.

mod booleanity;
mod bytecode_read_raf;
mod hamming_weight_claim_reduction;
mod instruction_input;
mod instruction_ra_virtualization;
mod instruction_read_raf;
pub mod solinas;
mod spartan_outer;

pub use booleanity::{BooleanityAddressMetalConfig, BooleanityMetalConfig};
pub use bytecode_read_raf::{BytecodeReadRafMetalConfig, BytecodeReadRafResidentRows};
pub use hamming_weight_claim_reduction::HammingWeightMetalConfig;
pub use instruction_input::InstructionInputMetalConfig;
pub use instruction_ra_virtualization::InstructionRaVirtualizationMetalConfig;
#[cfg(test)]
pub(crate) use instruction_read_raf::MetalInstructionReadRafKernel;
pub use instruction_read_raf::{InstructionReadRafMetalConfig, MetalBackend, MetalConfig};
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use spartan_outer::{
    OuterRemainderEvalError, OuterRemainderEvalFixture, OuterRemainderEvalResult,
    OuterRemainderEvalSample,
};
pub use spartan_outer::{SpartanOuterRemainderMetalConfig, SpartanOuterUniskipMetalConfig};
