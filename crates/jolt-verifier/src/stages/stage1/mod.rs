//! Stage 1 Spartan outer verifier.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{spartan_outer_claims_from_r1cs_inputs, stage1_claims_from_r1cs_inputs};
pub use outputs::{
    stage1_clear_output, Stage1ClearOutput, Stage1Output, Stage1PublicOutput, Stage1ZkOutput,
    VerifiedSpartanOuterSumcheck,
};
pub use verify::verify;
