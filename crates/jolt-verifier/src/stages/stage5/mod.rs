#[cfg(feature = "field-inline")]
pub mod field_inline;
#[cfg(feature = "field-inline")]
pub mod field_registers_val_evaluation;
pub mod instruction_read_raf;
pub mod outputs;
pub mod ram_ra_claim_reduction;
pub mod registers_val_evaluation;
mod verify;

#[cfg(feature = "field-inline")]
pub use field_registers_val_evaluation::FieldRegistersValEvaluationOutputClaims;
pub use instruction_read_raf::{InstructionReadRaf, InstructionReadRafOutputClaims};
pub use outputs::{Stage5Output, Stage5OutputClaims, Stage5OutputPoints, Stage5ZkOutput};
pub use ram_ra_claim_reduction::RamRaClaimReductionOutputClaims;
pub use registers_val_evaluation::{RegistersValEvaluation, RegistersValEvaluationOutputClaims};
pub use verify::{stage5_input_points_from_upstream, stage5_input_values_from_upstream, verify};
