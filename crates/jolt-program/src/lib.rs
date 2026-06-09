//! Program image, bytecode expansion, and preprocessing pipeline for Jolt.
//!
//! This crate's program-construction pipeline is RV64-only. ELF32/RV32 inputs
//! are rejected at the image boundary.

pub mod error;
pub mod execution;
pub mod expand;
#[cfg(feature = "field-inline")]
pub mod field_inline;
#[cfg(feature = "image")]
pub mod image;
pub mod lookup;
pub mod preprocess;

pub use error::ProgramError;
pub use execution::{build_jolt_program, build_jolt_program_with_inline_provider, JoltProgram};
