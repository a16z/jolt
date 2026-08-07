//! Apple Metal compute kernels.
//!
//! Reusable field arithmetic and hybrid sumcheck kernels for Apple GPUs.

mod backend;
mod booleanity;
mod bytecode_read_raf;
mod hamming_weight_claim_reduction;
mod instruction_claim_reduction;
mod instruction_input;
mod instruction_ra_virtualization;
mod instruction_read_raf;
mod ram_raf_evaluation;
mod ram_val_check;
mod registers_claim_reduction;
mod registers_val_evaluation;
pub mod solinas;
mod spartan_dense;
mod spartan_outer;
mod spartan_product;
mod spartan_shift;

pub use backend::{MetalBackend, MetalConfig};
pub use booleanity::{
    BooleanityAddressImplementation, BooleanityAddressMetalConfig, BooleanityMetalConfig,
};
pub use bytecode_read_raf::{
    BytecodeReadRafAddressImplementation, BytecodeReadRafAddressMetalConfig,
    BytecodeReadRafMetalConfig, BytecodeReadRafResidentRows,
};
pub use hamming_weight_claim_reduction::{HammingWeightImplementation, HammingWeightMetalConfig};
pub use instruction_claim_reduction::InstructionClaimReductionMetalConfig;
pub use instruction_input::InstructionInputMetalConfig;
pub use instruction_ra_virtualization::InstructionRaVirtualizationMetalConfig;
pub use instruction_read_raf::InstructionReadRafMetalConfig;
#[cfg(test)]
pub(crate) use instruction_read_raf::MetalInstructionReadRafKernel;
pub use ram_raf_evaluation::RamRafEvaluationMetalConfig;
pub use ram_val_check::RamValCheckMetalConfig;
pub use registers_claim_reduction::{
    RegistersClaimReductionImplementation, RegistersClaimReductionMetalConfig,
};
pub use registers_val_evaluation::RegistersValEvaluationMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use spartan_outer::{
    OuterRemainderEvalError, OuterRemainderEvalFixture, OuterRemainderEvalResult,
    OuterRemainderEvalSample, OuterRemainderGpuActiveBreakdown,
};
pub use spartan_outer::{SpartanOuterRemainderMetalConfig, SpartanOuterUniskipMetalConfig};
pub use spartan_product::SpartanProductRemainderMetalConfig;
pub use spartan_shift::SpartanShiftMetalConfig;
