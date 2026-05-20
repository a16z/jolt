//! RV64 program-image decoding.
//!
//! This module owns the architecture gate for the program pipeline. ELF32 and
//! RV32 inputs are unsupported.

pub mod decode;
pub mod elf;

pub use elf::{decode_elf, Rv64ProgramImage};
