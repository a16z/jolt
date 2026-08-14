//! Test-only packed Booleanity-address implementation.
//!
//! Production uses the resident-row implementation in [`super::booleanity_address`].

mod abi;
pub mod model;
mod runtime;

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;

pub use abi::*;
pub use runtime::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests;
