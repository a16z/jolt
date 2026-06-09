//! Stage 7 verifier entry point.

pub mod inputs;
pub mod outputs;
pub mod verify;

pub use inputs::{deps, Deps};
pub use outputs::{Stage7ClearOutput, Stage7Output, Stage7ZkOutput};
pub use verify::{
    stage7_advice_address_output, stage7_clear_output, stage7_expected_final_claim,
    stage7_expected_output_claim_values, stage7_expected_outputs, stage7_hamming_opening_point,
    stage7_hamming_opening_points, stage7_hamming_output_claim,
    stage7_hamming_output_opening_claims, stage7_hamming_sumcheck_point,
    stage7_hamming_virtualization_address_points, stage7_input_claims, stage7_output_claim_values,
    stage7_output_claims, verify, Stage7AdviceAddressOutput, Stage7AdviceAddressOutputRequest,
    Stage7BatchExpectedOutputClaims, Stage7ClearOutputRequest, Stage7ExpectedOutputsRequest,
    Stage7HammingOpeningPointRequest, Stage7HammingOpeningPoints, Stage7HammingOutputClaimRequest,
    Stage7HammingOutputOpeningClaimsRequest, Stage7HammingSumcheckPointRequest,
    Stage7InputClaimRequest, Stage7InputClaims, Stage7OutputClaimValuesRequest,
    Stage7OutputClaimsRequest,
};
