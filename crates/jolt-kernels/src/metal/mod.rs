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
mod ram_cycle_family;
mod ram_hamming_booleanity;
mod ram_ra_claim_reduction;
mod ram_ra_virtualization;
mod ram_raf_evaluation;
mod ram_read_write;
mod ram_val_check;
mod registers_claim_reduction;
mod registers_read_write;
mod registers_val_evaluation;
pub mod solinas;
mod spartan_dense;
mod spartan_outer;
mod spartan_product;
mod spartan_shift;

pub use backend::{MetalBackend, MetalConfig};
pub use booleanity::{BooleanityAddressMetalConfig, BooleanityMetalConfig};
pub use bytecode_read_raf::{
    BytecodeReadRafAddressImplementation, BytecodeReadRafAddressMetalConfig,
    BytecodeReadRafMetalConfig, BytecodeReadRafResidentRows,
};
pub use hamming_weight_claim_reduction::HammingWeightMetalConfig;
pub use instruction_claim_reduction::InstructionClaimReductionMetalConfig;
pub use instruction_input::{InstructionInputDenseStorageMode, InstructionInputMetalConfig};
pub use instruction_ra_virtualization::InstructionRaVirtualizationMetalConfig;
#[cfg(test)]
pub(crate) use instruction_read_raf::MetalInstructionReadRafKernel;
pub use instruction_read_raf::{
    InstructionReadRafAddressImplementation, InstructionReadRafMetalConfig,
};
pub use ram_hamming_booleanity::RamHammingBooleanityMetalConfig;
pub use ram_ra_virtualization::RamRaVirtualizationMetalConfig;
pub use ram_raf_evaluation::RamRafEvaluationMetalConfig;
pub use ram_val_check::RamValCheckMetalConfig;
pub use registers_claim_reduction::{
    RegistersClaimReductionImplementation, RegistersClaimReductionMetalConfig,
};
pub use registers_val_evaluation::{
    RegistersValEvaluationMetalConfig, RegistersValEvaluationSource,
};
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use spartan_outer::OuterRemainderGpuActiveBreakdown;
pub use spartan_outer::{SpartanOuterRemainderMetalConfig, SpartanOuterUniskipMetalConfig};
pub use spartan_product::{SpartanProductRemainderMetalConfig, SpartanProductWitnessSource};
pub use spartan_shift::SpartanShiftMetalConfig;
