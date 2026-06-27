//! Stage 2 product uni-skip and five-instance batch verifier.

pub mod instruction_claim_reduction;
pub mod outputs;
pub mod product_remainder;
pub mod product_uniskip;
pub mod ram_output_check;
pub mod ram_raf_evaluation;
pub mod ram_read_write_checking;
mod verify;

pub use instruction_claim_reduction::{
    InstructionClaimReduction, InstructionClaimReductionInputClaims,
    InstructionClaimReductionOutputClaims,
};
pub use outputs::{
    Stage2BatchOutputClaims, Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2ZkOutput,
    VerifiedProductUniSkip,
};
pub use product_remainder::{
    ProductRemainder, ProductRemainderInputClaims, ProductRemainderOutputClaims,
};
pub use product_uniskip::{ProductUniskip, ProductUniskipInputClaims, ProductUniskipOutputClaims};
pub use ram_output_check::{RamOutputCheck, RamOutputCheckInputClaims, RamOutputCheckOutputClaims};
pub use ram_raf_evaluation::{
    RamRafEvaluation, RamRafEvaluationInputClaims, RamRafEvaluationOutputClaims,
};
pub use ram_read_write_checking::{
    RamReadWriteChecking, RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
pub use verify::{stage2_batch_output_claims_with_points, stage2_expected_final_claim, verify};
