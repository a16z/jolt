//! Compatibility boundary for existing `jolt-core` proof artifacts.

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
mod audit;
pub mod claims;
mod codec;
pub mod config;
#[cfg(feature = "jolt-core-compat")]
pub mod convert;
pub mod ids;

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
pub use audit::{audit_zk_blindfold_protocol_shape, ZkBlindFoldProtocolShape};
