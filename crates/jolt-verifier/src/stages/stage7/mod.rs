//! Stage 7 verifier entry point.

pub mod advice_address_phase;
pub mod chunk_reconstruction;
pub mod committed_reduction_address_phase;
pub mod hamming_weight_claim_reduction;
pub mod outputs;
pub mod verify;

pub use outputs::{Stage7ClearOutput, Stage7Output, Stage7OutputClaims, Stage7ZkOutput};
pub use verify::{
    stage7_hamming_virtualization_address_points, verify, Stage7InstancePoints, Stage7Layouts,
    Stage7Relations,
};
