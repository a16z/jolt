//! Isolated first-principles successor for `HammingWeightClaimReduction`.
//!
//! This module is not registered with the production Metal backend.

mod abi;
#[cfg(feature = "test-utils")]
mod compile_probe;
pub mod model;
#[cfg(feature = "test-utils")]
mod runtime;
mod slice;

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;

pub use abi::*;
#[cfg(feature = "test-utils")]
pub use compile_probe::*;
#[cfg(feature = "test-utils")]
pub use runtime::*;
pub use slice::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");
