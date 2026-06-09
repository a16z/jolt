//! Stage 3 verifier: Spartan shift, instruction input, and register reduction.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps, Stage3Claims};
pub use outputs::{Stage3ClearOutput, Stage3Output, Stage3PublicOutput, Stage3ZkOutput};
pub use verify::{
    append_stage3_opening_claims, stage3_expected_final_claim, stage3_expected_output_claims,
    stage3_input_claims, stage3_output_claim_values, verify, Stage3ExpectedOutputClaims,
    Stage3ExpectedOutputRequest, Stage3InputClaimRequest, Stage3InputClaims,
};
