//! Compatibility boundary for existing `jolt-core` proof artifacts.

pub mod claims;
mod codec;
pub mod config;
#[cfg(feature = "jolt-core-compat")]
pub mod convert;
pub mod ids;
