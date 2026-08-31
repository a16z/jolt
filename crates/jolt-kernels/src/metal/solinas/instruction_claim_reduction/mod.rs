//! Isolated instruction claim-reduction Metal design.

mod abi;
mod model;
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;
mod runtime;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub use abi::*;
pub use model::*;
pub use runtime::*;

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests use fixed valid shapes"
)]
mod tests;
