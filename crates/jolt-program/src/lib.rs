//! Program image, bytecode expansion, and preprocessing pipeline for Jolt.
//!
//! This crate's program-construction pipeline is RV64-only. ELF32/RV32 inputs
//! are rejected at the image boundary; historical RV32 execution code may remain
//! in `tracer`, but it is not part of the verifier-facing `jolt-program` path.

pub mod error;
pub mod execution;
pub mod expand;
#[cfg(feature = "image")]
pub mod image;
pub mod preprocess;

pub use error::ProgramError;
pub use execution::{build_executable, ExecutableProgram};
