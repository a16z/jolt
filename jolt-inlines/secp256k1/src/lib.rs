//! Elliptic curves inline implementation module

#![cfg_attr(not(feature = "host"), no_std)]

pub const INLINE_OPCODE: u32 = 0x0B;
pub const VIRTUAL_INSTRUCTION_TYPE_I_OPCODE: u32 = 0x5B;
pub const SECP256K1_FUNCT7: u32 = 0x05;

// base field division (pure non-deterministic advice, no checks)
pub const SECP256K1_DIVQ_ADV_FUNCT3: u32 = 0x00;
pub const SECP256K1_DIVQ_ADV_NAME: &str = "SECP256K1_DIVQ_ADV";

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod sequence_builder;

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::*;

#[cfg(all(test, feature = "host"))]
mod tests;
