//! BLAKE2 inline implementation module

#![cfg_attr(not(feature = "host"), no_std)]

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod exec;
#[cfg(feature = "host")]
pub mod trace_generator;

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::*;

// Test modules and utilities
#[cfg(all(test, feature = "host"))]
pub mod test_utils;
#[cfg(all(test, feature = "host"))]
mod tests;
