//! Virtual instructions used internally by the Jolt VM.
//!
//! These do not correspond directly to RISC-V ISA instructions but are
//! needed by the proving system for constraint checking and arithmetic helpers.

pub mod arithmetic;
pub mod assert;
pub mod bitwise;
