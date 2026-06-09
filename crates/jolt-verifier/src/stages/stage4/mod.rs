//! Stage 4 verifier entry point.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps, Stage4Claims};
pub use outputs::{Stage4ClearOutput, Stage4Output, Stage4ZkOutput};
pub use verify::{
    append_ram_val_check_gamma_domain_separator, append_stage4_opening_claims,
    stage4_expected_final_claim, stage4_expected_output_claims, stage4_input_claims,
    stage4_opening_points, stage4_output_claim_values, verify, Stage4ExpectedOutputClaims,
    Stage4ExpectedOutputRequest, Stage4InputClaimRequest, Stage4InputClaims,
    Stage4OpeningPointRequest, Stage4OpeningPoints,
};
