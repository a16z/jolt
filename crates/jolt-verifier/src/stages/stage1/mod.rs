//! Stage 1 Spartan outer verifier.

pub mod outer_remainder;
pub mod outer_uniskip;
pub mod outputs;
mod verify;

pub use outer_remainder::{OuterRemainder, OuterRemainderInputClaims, OuterRemainderOutputClaims};
pub use outer_uniskip::{OuterUniskip, OuterUniskipInputClaims, OuterUniskipOutputClaims};
pub use outputs::{
    spartan_outer_claims_from_r1cs_inputs, stage1_challenges, stage1_claims_from_r1cs_inputs,
    stage1_clear_output, Stage1Challenges, Stage1ClearOutput, Stage1Output, Stage1ZkOutput,
    VerifiedSpartanOuterSumcheck,
};
pub use verify::verify;
