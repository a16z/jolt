//! Program image, bytecode expansion, and preprocessing pipeline for Jolt.

pub mod error;
pub mod expand;
#[cfg(feature = "image")]
pub mod image;
pub mod preprocess;

pub use error::ProgramError;
