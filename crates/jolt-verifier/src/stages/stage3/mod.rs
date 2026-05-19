//! Stage 3 verifier: Spartan shift, instruction input, and register reduction.

pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps, Stage3Claims};
pub use outputs::Stage3Output;
pub use verify::verify;
