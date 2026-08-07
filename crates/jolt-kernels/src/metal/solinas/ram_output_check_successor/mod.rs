//! Isolated successor design for the modular RAM output-check member.
//!
//! This module is not registered with the backend or Metal source assembler.
//! `WIRING.md` rejects standalone submission and defines the batch-coalesced
//! experiment boundary.

mod abi;
pub mod model;

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;

pub use abi::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked fixed fixtures")]
mod tests;
