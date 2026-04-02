//! Re-exports [`PolynomialId`] from `jolt-compiler`.
//!
//! The canonical definition lives in `jolt-compiler` so the protocol module
//! can use typed IDs directly. This re-export keeps the existing import
//! paths working for downstream crates.

pub use jolt_compiler::PolynomialId;
