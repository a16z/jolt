//! Stage 1 Spartan outer verifier.

pub mod inputs;
pub mod outputs;
mod verify;

pub use outputs::{
    Stage1ClearOutput, Stage1Output, Stage1PublicOutput, Stage1ZkOutput,
    VerifiedSpartanOuterSumcheck,
};
pub use verify::verify;
