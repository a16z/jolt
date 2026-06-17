//! Stage 2 product uni-skip and five-instance batch verifier.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps};
pub use outputs::{
    Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2ZkOutput, VerifiedProductUniSkip,
    VerifiedStage2Batch, VerifiedStage2Sumcheck,
};
pub use verify::verify;
