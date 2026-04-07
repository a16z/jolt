//! Virtual instructions used internally by the Jolt VM.
//!
//! These do not correspond directly to RISC-V ISA instructions but are
//! needed by the proving system for constraint checking, arithmetic helpers,
//! and instruction decompositions.

pub mod advice;
pub mod arithmetic;
pub mod assert;
pub mod bitwise;
pub mod byte;
pub mod division;
pub mod extension;
pub mod shift;
pub mod xor_rotate;
