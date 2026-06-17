//! Stage 2 copy-constraint verifier.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{Stage2Inputs, Stage2Proof};
pub use outputs::Stage2Output;
pub use verify::verify;
