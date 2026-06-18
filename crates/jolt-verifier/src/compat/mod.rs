//! Compatibility boundary for existing `jolt-core` proof artifacts.

#[cfg(feature = "zk")]
mod audit;
pub mod claims;
mod codec;
pub mod config;
pub mod ids;

#[cfg(feature = "zk")]
pub use audit::{audit_zk_blindfold_protocol_shape, ZkBlindFoldProtocolShape};
