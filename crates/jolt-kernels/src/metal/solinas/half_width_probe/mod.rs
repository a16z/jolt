//! Isolated contract for Solinas `Fp128` by 64-bit scalar multiplication.

mod abi;
mod model;
mod oracle;
mod runtime;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub use abi::*;
pub use model::*;
pub use oracle::*;
pub use runtime::*;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid fixtures")]
mod tests;
