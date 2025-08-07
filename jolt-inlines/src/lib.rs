//! Inline implementations for Jolt VM
//! 
//! This crate provides various inline implementations that can be
//! conditionally compiled based on feature flags.

#![cfg_attr(not(feature = "host"), no_std)]

// Conditionally include modules based on feature flags
#[cfg(feature = "sha256")]
pub mod sha256;

// Re-export modules at the crate root when their features are enabled
#[cfg(feature = "sha256")]
pub use sha256::*;