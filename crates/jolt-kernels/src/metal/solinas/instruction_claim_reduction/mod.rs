//! Isolated instruction claim-reduction Metal design.

mod abi;
mod model;
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub use abi::*;
pub use model::*;

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use fixed valid shapes")]
mod tests;
