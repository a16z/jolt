//! RV64 program-image decoding.
//!
//! This module owns the architecture gate for the new program pipeline. ELF32
//! and RV32 inputs are unsupported here even if tracer keeps historical RV32
//! execution branches internally.

pub mod decode;
pub mod elf;

pub use elf::{decode_elf, Rv64ProgramImage};
