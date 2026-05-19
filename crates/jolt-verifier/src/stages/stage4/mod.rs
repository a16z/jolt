//! Stage 4 verifier entry point.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps, Stage4Claims};
pub use outputs::Stage4Output;
pub use verify::verify;
