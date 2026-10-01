//! Stage 6a (address-phase) verifier entry point.

pub mod batch;
pub mod booleanity;
pub mod bytecode_read_raf;
#[cfg(feature = "field-inline")]
pub mod field_inline;
pub mod outputs;
pub mod verify;

pub use outputs::{Stage6aClearOutput, Stage6aOutput, Stage6aZkOutput};
pub use verify::verify;
