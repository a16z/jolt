//! Stage 6 verifier entry point.

pub mod inputs;
pub mod outputs;
pub mod verify;

pub use inputs::{deps, Deps};
pub use outputs::{Stage6ClearOutput, Stage6Output, Stage6ZkOutput};
pub use verify::verify;
