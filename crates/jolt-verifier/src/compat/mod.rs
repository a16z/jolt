//! Compatibility boundary for existing `jolt-core` proof artifacts.

mod codec;
pub mod config;
#[cfg(feature = "jolt-core-compat")]
pub mod convert;
pub mod ids;
