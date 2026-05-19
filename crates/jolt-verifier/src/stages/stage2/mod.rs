//! Stage 2 product uni-skip and five-instance batch verifier.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps};
pub use outputs::{
    Stage2Output, VerifiedProductUniSkip, VerifiedStage2Batch, VerifiedStage2Sumcheck,
};
pub use verify::verify;
