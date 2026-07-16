//! Stage 7 verifier entry point.

pub mod advice_address_phase;
pub mod committed_reduction_address_phase;
pub mod hamming_weight_claim_reduction;
pub mod outputs;
pub mod verify;

pub use outputs::{
    Stage7Challenges, Stage7ClearOutput, Stage7InputClaims, Stage7InputPoints, Stage7Output,
    Stage7OutputClaims, Stage7OutputPoints, Stage7ZkOutput,
};
pub use verify::{build_stage7_sumchecks, stage7_input_values_from_upstream, verify};
