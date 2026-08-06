//! Isolated late-cycle registers read/write Metal design.

mod abi;
mod runtime;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub use abi::*;
pub use runtime::*;
