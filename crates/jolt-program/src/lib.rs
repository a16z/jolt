//! Program image, bytecode expansion, and preprocessing pipeline for Jolt.
//!
//! This crate's program-construction pipeline is RV64-only. ELF32/RV32 inputs
//! are rejected at the image boundary.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

pub mod error;
pub mod execution;
pub mod expand;
#[cfg(feature = "field-inline")]
pub mod field_inline;
#[cfg(feature = "image")]
pub mod image;
pub mod preprocess;

pub use error::ProgramError;
pub use execution::{build_jolt_program, build_jolt_program_with_inline_provider, JoltProgram};
