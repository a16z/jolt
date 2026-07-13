pub mod instruction_read_raf;
pub mod outputs;
pub mod ram_ra_claim_reduction;
pub mod registers_val_evaluation;
mod verify;

pub use instruction_read_raf::{InstructionReadRaf, InstructionReadRafOutputClaims};
pub use outputs::{Stage5Output, Stage5OutputClaims, Stage5OutputPoints, Stage5ZkOutput};
pub use ram_ra_claim_reduction::RamRaClaimReductionOutputClaims;
pub use registers_val_evaluation::{RegistersValEvaluation, RegistersValEvaluationOutputClaims};
pub use verify::verify;
