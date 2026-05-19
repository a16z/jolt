//! Stage 1 Spartan outer verifier.

pub mod inputs;
pub mod outputs;
mod verify;

pub use outputs::{Stage1Output, VerifiedSpartanOuterSumcheck};
pub use verify::verify;
