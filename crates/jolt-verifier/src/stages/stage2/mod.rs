//! Stage 2 product uni-skip and five-instance batch verifier.

pub mod inputs;
pub mod outputs;
pub mod ram_read_write_checking;
mod verify;

pub use inputs::{deps, product_uniskip_input_claim, Deps, Stage2ProductUniSkipInputValues};
pub use outputs::{
    Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2ZkOutput, VerifiedProductUniSkip,
    VerifiedStage2Batch, VerifiedStage2Sumcheck,
};
pub use ram_read_write_checking::{
    RamReadWriteChecking, RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
pub use verify::{
    stage2_batch_input_claims, stage2_batch_opening_points, stage2_clear_output,
    stage2_expected_final_claim, stage2_expected_outputs, verify, Stage2BatchExpectedOutputClaims,
    Stage2BatchInputClaimRequest, Stage2BatchInputClaims, Stage2BatchOpeningPoints,
    Stage2BatchPointRequest, Stage2ClearOutputRequest, Stage2ExpectedOutputRequest,
    Stage2ProductUniSkipClearRequest, Stage2ProductUniSkipOutputClaimData,
    Stage2RegularBatchClearRequest,
};
