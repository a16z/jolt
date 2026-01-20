//! BLAKE3 inline implementation module
#![cfg_attr(not(feature = "host"), no_std)]

pub const INLINE_OPCODE: u32 = 0x0B;

pub const BLAKE3_FUNCT3: u32 = 0x00;
pub const BLAKE3_KEYED64_FUNCT3: u32 = 0x01; // Keyed64: hash(left || right) with IV from rd, PARENT flag
pub const BLAKE3_FUNCT7: u32 = 0x03;
pub const BLAKE3_NAME: &str = "BLAKE3_INLINE";
pub const BLAKE3_KEYED64_NAME: &str = "BLAKE3_KEYED64_INLINE";

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

#[cfg(all(test, feature = "host"))]
pub mod test_utils;

/// BLAKE3 initialization vector (IV)
#[rustfmt::skip]
pub const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// BLAKE3 message scheduling constants for each round
#[rustfmt::skip]
pub const MSG_SCHEDULE: [[usize; 16]; NUM_ROUNDS as usize] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

pub const NUM_ROUNDS: u8 = 7;
pub const CHAINING_VALUE_LEN: usize = 8;
pub const MSG_BLOCK_LEN: usize = 16;
pub const COUNTER_LEN: usize = 2;
pub const BLOCK_INPUT_SIZE_IN_BYTES: usize = 64;
pub const OUTPUT_SIZE_IN_BYTES: usize = 32;
pub const WORD_SIZE: usize = 32;

// BLAKE3 flags
pub const FLAG_CHUNK_START: u32 = 1;
pub const FLAG_CHUNK_END: u32 = 2;
pub const FLAG_PARENT: u32 = 4; // Used for Merkle tree parent nodes
pub const FLAG_ROOT: u32 = 8;
pub const FLAG_KEYED_HASH: u32 = 16; // Used in tests for generic compress
