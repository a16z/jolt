//! Stage 7 verifier entry point.

pub mod inputs;
pub mod outputs;
pub mod verify;

pub use inputs::{deps, Deps};
pub use outputs::Stage7Output;
pub use verify::verify;
