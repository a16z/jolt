//! Stage 1 Spartan outer verifier.

pub mod outer_remainder;
pub mod outputs;
mod verify;

pub use outer_remainder::OuterRemainderOutputClaims;
pub use outputs::{Stage1BatchOutputClaims, Stage1ClearOutput, Stage1Output, Stage1ZkOutput};
pub use verify::verify;
