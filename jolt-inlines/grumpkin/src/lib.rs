//! grumpkin inline implementation module
//! Contains 2 inlines accessible via
//! a wrapper around ark-grumpkin types:
//! the inlines are for
//! 0x00: base field division
//! 0x01: scalar field division

#![cfg_attr(not(feature = "host"), no_std)]

pub const INLINE_OPCODE: u32 = 0x0B;
pub const GRUMPKIN_FUNCT7: u32 = 0x06;

// base field (q) division (pure non-deterministic advice, no checks)
// that is, given a and b in Fq, compute c = a / b
pub const GRUMPKIN_DIVQ_ADV_FUNCT3: u32 = 0x00;
pub const GRUMPKIN_DIVQ_ADV_NAME: &str = "GRUMPKIN_DIVQ_ADV";

// scalar field (r) division (pure non-deterministic advice, no checks)
// that is, given a and b in Fr, compute c = a / b
pub const GRUMPKIN_DIVR_ADV_FUNCT3: u32 = 0x01;
pub const GRUMPKIN_DIVR_ADV_NAME: &str = "GRUMPKIN_DIVR_ADV";

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
