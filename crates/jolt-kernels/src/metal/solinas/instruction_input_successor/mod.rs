//! Experimental successor kernels for the InstructionInput Metal backend.
//!
//! The split phases are available to tests and evaluator builds. The
//! production backend still selects the existing fused transition.

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
