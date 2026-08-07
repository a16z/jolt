//! Isolated first slice for the next BooleanityAddress Metal backend.
//!
//! The module is deliberately not registered with the production backend or
//! Metal source assembler. `WIRING.md` defines the integration and evidence
//! gates that must be satisfied before either registration is appropriate.

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
