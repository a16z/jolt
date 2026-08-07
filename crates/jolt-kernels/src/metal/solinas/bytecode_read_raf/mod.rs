//! Static successor packet for the packed bytecode read/RAF address phase.

mod abi;
mod model;
mod oracle;
#[cfg(feature = "test-utils")]
mod runtime;
mod slice;

pub use abi::*;
pub use model::*;
pub use oracle::*;
#[cfg(feature = "test-utils")]
pub use runtime::*;
pub use slice::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests;
