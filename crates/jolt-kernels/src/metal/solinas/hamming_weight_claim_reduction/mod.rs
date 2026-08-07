//! Isolated first-principles successor for `HammingWeightClaimReduction`.
//!
//! This module is not registered with the production Metal backend.

mod abi;
mod compile_probe;
pub mod model;
mod slice;

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;

pub use abi::*;
pub use compile_probe::*;
pub use slice::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");
