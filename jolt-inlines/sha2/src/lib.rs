//! SHA256 inline implementation module

#![cfg_attr(not(feature = "host"), no_std)]

pub const INLINE_OPCODE: u32 = 0x0B;

pub const SHA256_FUNCT3: u32 = 0x00;
pub const SHA256_FUNCT7: u32 = 0x00;
pub const SHA256_NAME: &str = "SHA256_INLINE";

pub const SHA256_INIT_FUNCT3: u32 = 0x01;
pub const SHA256_INIT_FUNCT7: u32 = 0x00;
pub const SHA256_INIT_NAME: &str = "SHA256_INIT_INLINE";

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod exec;
#[cfg(feature = "host")]
pub mod sequence_builder;

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::*;

// Test modules and constants
#[cfg(all(test, feature = "host"))]
pub mod test_constants;
#[cfg(all(test, feature = "host"))]
pub mod test_utils;
#[cfg(all(test, feature = "host"))]
mod tests;
