//! Stage 3 verifier: Spartan shift, instruction input, and register reduction.

pub mod instruction_input;
pub mod outputs;
pub mod registers_claim_reduction;
pub mod spartan_shift;
mod verify;

pub use outputs::{Stage3Output, Stage3OutputClaims, Stage3OutputPoints, Stage3ZkOutput};
pub use verify::verify;
