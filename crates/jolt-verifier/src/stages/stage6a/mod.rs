//! Stage 6a (address-phase) verifier entry point.

pub mod booleanity;
pub mod bytecode_read_raf;
pub mod outputs;
pub mod verify;

pub use outputs::{Stage6aClearOutput, Stage6aOutput, Stage6aZkOutput};
pub use verify::verify;
