//! Isolated successor design for the InstructionInput Metal backend.
//!
//! This module is deliberately not registered with the production backend.

mod abi;
pub mod model;

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;

pub use abi::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid design fixtures")]
mod tests;
