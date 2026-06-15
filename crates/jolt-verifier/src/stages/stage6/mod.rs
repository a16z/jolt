//! Stage 6 verifier entry point.

pub mod inputs;
pub mod inputs_a;
pub mod inputs_b;
pub mod outputs;
pub mod outputs_a;
pub mod outputs_b;
pub mod verify;
mod verify_a;
pub(crate) mod verify_b;

pub use inputs::{deps, Deps};
pub use outputs::{Stage6ClearOutput, Stage6Output, Stage6ZkOutput};
pub use verify::verify;
