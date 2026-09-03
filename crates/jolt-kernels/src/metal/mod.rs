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
pub(crate) mod ram_records;
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
    BytecodeReadRafMetalConfig,
};
pub use hamming_weight_claim_reduction::HammingWeightMetalConfig;
pub use instruction_claim_reduction::InstructionClaimReductionMetalConfig;
pub use instruction_input::{InstructionInputDenseStorageMode, InstructionInputMetalConfig};
pub use instruction_ra_virtualization::InstructionRaVirtualizationMetalConfig;
pub use instruction_read_raf::InstructionReadRafMetalConfig;
pub use ram_hamming_booleanity::RamHammingBooleanityMetalConfig;
#[cfg(feature = "test-utils")]
pub use ram_hamming_booleanity::{
    RamHammingBooleanityCpuEvalFixture, RamHammingBooleanityEvalError,
    RamHammingBooleanityEvalResult, RamHammingBooleanityEvalSample,
    RamHammingBooleanityRoundTiming, RamHammingBooleanityShapeSnapshot,
};
pub use ram_ra_claim_reduction::RamRaClaimReductionMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use ram_ra_claim_reduction::{
    RamRaClaimReductionCpuMetalEvalFixture, RamRaClaimReductionEvalError,
    RamRaClaimReductionEvalResult, RamRaClaimReductionEvalSample,
    RamRaClaimReductionMetalEvalSample, RamRaClaimReductionRoundTiming,
    RamRaClaimReductionShapeSnapshot,
};
pub use ram_ra_virtualization::RamRaVirtualizationMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use ram_ra_virtualization::{
    RamRaVirtualizationCpuEvalSample, RamRaVirtualizationCpuMetalEvalFixture,
    RamRaVirtualizationEvalError, RamRaVirtualizationEvalResult, RamRaVirtualizationRoundTiming,
    RamRaVirtualizationShapeSnapshot,
};
pub use ram_raf_evaluation::RamRafEvaluationMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use ram_raf_evaluation::{
    RamRafEvaluationCpuEvalSample, RamRafEvaluationCpuMetalEvalFixture, RamRafEvaluationEvalError,
    RamRafEvaluationEvalResult, RamRafEvaluationMetalEvalSample, RamRafEvaluationRoundTiming,
    RamRafEvaluationShapeSnapshot,
};
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use ram_read_write::{
    RamReadWriteBucketSnapshot, RamReadWriteCpuEvalSample, RamReadWriteCpuMetalEvalFixture,
    RamReadWriteDispatchSnapshot, RamReadWriteEvalError, RamReadWriteEvalResult,
    RamReadWriteMetalEvalSample, RamReadWritePreparationSnapshot, RamReadWriteRoundTiming,
};
pub use ram_val_check::RamValCheckMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use ram_val_check::{
    RamValCheckCpuEvalSample, RamValCheckCpuMetalEvalFixture, RamValCheckEvalError,
    RamValCheckEvalResult, RamValCheckRoundTiming, RamValCheckShapeSnapshot,
};
pub use registers_claim_reduction::{
    RegistersClaimReductionImplementation, RegistersClaimReductionMetalConfig,
};
pub use registers_read_write::RegistersReadWriteMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use registers_read_write::{
    RegistersReadWriteCpuEvalSample, RegistersReadWriteCpuMetalEvalFixture,
    RegistersReadWriteEvalError, RegistersReadWriteEvalResult, RegistersReadWriteMetalEvalSample,
    RegistersReadWriteRoundTiming, RegistersReadWriteShapeSnapshot,
};
pub use registers_val_evaluation::{
    RegistersValEvaluationMetalConfig, RegistersValEvaluationSource,
};
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use spartan_outer::{
    OuterRemainderCpuEvalSample, OuterRemainderCpuMetalEvalFixture, OuterRemainderEvalError,
    OuterRemainderEvalResult, OuterRemainderGpuActiveBreakdown, OuterRemainderMetalEvalSample,
    OuterRemainderPipelineSnapshot, OuterRemainderThreadSnapshot,
};
pub use spartan_outer::{SpartanOuterRemainderMetalConfig, SpartanOuterUniskipMetalConfig};
pub use spartan_product::SpartanProductRemainderMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use spartan_product::{
    ProductRemainderCpuEvalSample, ProductRemainderCpuMetalEvalFixture, ProductRemainderEvalError,
    ProductRemainderEvalResult, ProductRemainderMetalEvalSample,
    ProductRemainderNumericWidthSnapshot, ProductRemainderRoundTiming,
    ProductRemainderShapeSnapshot,
};
pub use spartan_shift::SpartanShiftMetalConfig;
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub use spartan_shift::{
    SpartanShiftCpuMetalEvalFixture, SpartanShiftEvalError, SpartanShiftEvalResult,
    SpartanShiftEvalSample, SpartanShiftRoundTiming, SpartanShiftShapeSnapshot,
};
