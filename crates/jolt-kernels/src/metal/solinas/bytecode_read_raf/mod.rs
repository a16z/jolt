//! Static successor packet for the packed bytecode read/RAF address phase.

mod abi;
mod handoff;
mod model;
mod oracle;
mod runtime;
mod slice;

pub use abi::*;
pub use handoff::*;
pub use model::*;
pub use oracle::*;
pub use runtime::*;
pub use slice::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests;
