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

// scalar field (r) glv decomposition (pure non-deterministic advice, no checks)
// this is, given k in Fr, compute k1, k2 such that k = k1 + k2 * lambda (mod r)
// and |k1|, |k2| <= 2^128
// returns (s1, |k1|, s2, |k2|) where s1, s2 are the signs of k1, k2 respectively as 64-bit integers (0 for positive, 1 for negative)
pub const GRUMPKIN_GLVR_ADV_FUNCT3: u32 = 0x02;
pub const GRUMPKIN_GLVR_ADV_NAME: &str = "GRUMPKIN_GLVR_ADV";

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
