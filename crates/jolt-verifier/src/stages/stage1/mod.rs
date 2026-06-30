//! Stage 1 Spartan outer verifier.

pub mod outer_remainder;
pub mod outer_uniskip;
pub mod outputs;
mod verify;

pub use outer_remainder::{OuterRemainder, OuterRemainderInputClaims, OuterRemainderOutputClaims};
pub use outer_uniskip::{OuterUniskip, OuterUniskipInputClaims, OuterUniskipOutputClaims};
pub use outputs::{
    outer_remainder_outputs_from_r1cs_inputs, Stage1BatchOutputClaims, Stage1Challenges,
    Stage1ClearOutput, Stage1Output, Stage1ZkOutput,
};
pub use verify::verify;
