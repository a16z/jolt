//! Stage 2 product uni-skip and five-instance batch verifier.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, product_uniskip_input_claim, Deps, Stage2ProductUniSkipInputValues};
pub use outputs::{
    Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2ZkOutput, VerifiedProductUniSkip,
    VerifiedStage2Batch, VerifiedStage2Sumcheck,
};
pub use verify::{
    append_stage2_opening_claims, stage2_batch_input_claims, stage2_batch_opening_points,
    stage2_clear_output, stage2_expected_final_claim, stage2_expected_outputs,
    stage2_output_claim_values, verify, Stage2BatchExpectedOutputClaims,
    Stage2BatchInputClaimRequest, Stage2BatchInputClaims, Stage2BatchOpeningPoints,
    Stage2BatchPointRequest, Stage2ClearOutputRequest, Stage2ExpectedOutputRequest,
    Stage2ProductUniSkipClearRequest, Stage2ProductUniSkipOutputClaimData,
    Stage2RegularBatchClearRequest,
};
