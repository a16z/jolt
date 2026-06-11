// SPDX-License-Identifier: MIT

//! Goldilocks Poseidon2 inline for the Jolt zkVM.
//!
//! This crate implements the canonical Plonky3-compatible 8-wide
//! Poseidon2 permutation over the Goldilocks field. The guest API emits
//! one custom inline instruction that permutes an eight-limb state
//! in-place; the host registration expands that instruction into a
//! deterministic virtual-instruction sequence for Jolt tracing.
//!
//! ## Inline opcode encoding
//!
//! Custom RISC-V instruction:
//!
//! ```text
//! .insn r INLINE_OPCODE, POSEIDON2_GOLDILOCKS_FUNCT3, POSEIDON2_GOLDILOCKS_FUNCT7, x0, rs1, x0
//! ```
//!
//! - `rs1` points to a 64-byte (8 × u64), 8-byte-aligned state buffer
//!   that is permuted in place.
//! - Round constants are embedded in the inline expansion as virtual
//!   immediates; `rs2` is unused.
//!
//! `INLINE_OPCODE` (0x0B) is shared with the upstream
//! `jolt-inlines-*` crates; this crate owns the `funct7 = 0x08`
//! namespace, with `funct3` enumerating its operations (`0x00` = the
//! 8-wide permutation).

#![cfg_attr(not(feature = "host"), no_std)]

/// Shared custom inline opcode space. Same value used by all
/// `jolt-inlines-*` crates upstream.
pub const INLINE_OPCODE: u32 = 0x0B;

/// `funct3` for the Goldilocks Poseidon2 permutation opcode.
///
pub const POSEIDON2_GOLDILOCKS_FUNCT3: u32 = 0x00;

/// `funct7` for the Goldilocks Poseidon2 permutation opcode.
pub const POSEIDON2_GOLDILOCKS_FUNCT7: u32 = 0x08;

/// Human-readable inline name. Used in trace-file headers and
/// upstream registration.
pub const POSEIDON2_GOLDILOCKS_NAME: &str = "POSEIDON2_GOLDILOCKS_INLINE";

/// State width for our Poseidon2 instance. Hard-coded to 8; v0 is
/// not generic over width.
pub const STATE_WIDTH: usize = 8;

/// Convenience: an 8-element Goldilocks state.
pub type Poseidon2GoldilocksState = [u64; STATE_WIDTH];

/// Goldilocks field modulus `p = 2^64 - 2^32 + 1`.
pub const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// Goldilocks field modular addition.
///
/// Lives in `lib.rs` (not `exec.rs`) because the SDK's
/// `poseidon2_hash_pair` absorbs inputs via `add_mod` in BOTH host and
/// no_std/guest builds. The guest path can't see `exec` (host-only).
#[inline]
pub fn add_mod(a: u64, b: u64) -> u64 {
    let (mut sum, overflow) = a.overflowing_add(b);
    if overflow {
        sum = sum.wrapping_sub(GOLDILOCKS_MODULUS);
    }
    if sum >= GOLDILOCKS_MODULUS {
        sum -= GOLDILOCKS_MODULUS;
    }
    sum
}

/// 86 Poseidon2 round constants for the Goldilocks 8-wide instance.
///
/// Layout: 32 external initial (4 rounds × 8 elements) + 22 internal
/// (state[0] only) + 32 external final (4 rounds × 8 elements).
///
/// Kept in the crate root so both the host reference and sequence
/// builder share the same table.
#[rustfmt::skip]
pub static POSEIDON2_ROUND_CONSTANTS_GOLDILOCKS_8: [u64; 86] = [
    // External initial: 4 rounds × 8 elements
    0xdd5743e7f2a5a5d9, 0xcb3a864e58ada44b, 0xffa2449ed32f8cdc, 0x42025f65d6bd13ee,
    0x7889175e25506323, 0x34b98bb03d24b737, 0xbdcc535ecc4faa2a, 0x5b20ad869fc0d033,
    0xf1dda5b9259dfcb4, 0x27515210be112d59, 0x4227d1718c766c3f, 0x26d333161a5bd794,
    0x49b938957bf4b026, 0x4a56b5938b213669, 0x1120426b48c8353d, 0x6b323c3f10a56cad,
    0xce57d6245ddca6b2, 0xb1fc8d402bba1eb1, 0xb5c5096ca959bd04, 0x6db55cd306d31f7f,
    0xc49d293a81cb9641, 0x1ce55a4fe979719f, 0xa92e60a9d178a4d1, 0x002cc64973bcfd8c,
    0xcea721cce82fb11b, 0xe5b55eb8098ece81, 0x4e30525c6f1ddd66, 0x43c6702827070987,
    0xaca68430a7b5762a, 0x3674238634df9c93, 0x88cee1c825e33433, 0xde99ae8d74b57176,
    // Internal: 22 scalars (state[0] only)
    0x488897d85ff51f56, 0x1140737ccb162218, 0xa7eeb9215866ed35, 0x9bd2976fee49fcc9,
    0xc0c8f0de580a3fcc, 0x4fb2dae6ee8fc793, 0x343a89f35f37395b, 0x223b525a77ca72c8,
    0x56ccb62574aaa918, 0xc4d507d8027af9ed, 0xa080673cf0b7e95c, 0xf0184884eb70dcf8,
    0x044f10b0cb3d5c69, 0xe9e3f7993938f186, 0x1b761c80e772f459, 0x606cec607a1b5fac,
    0x14a0c2e1d45f03cd, 0x4eace8855398574f, 0xf905ca7103eff3e6, 0xf8c8f8d20862c059,
    0xb524fe8bdd678e5a, 0xfbb7865901a1ec41,
    // External final: 4 rounds × 8 elements
    0x014ef1197d341346, 0x9725e20825d07394, 0xfdb25aef2c5bae3b, 0xbe5402dc598c971e,
    0x93a5711f04cdca3d, 0xc45a9a5b2f8fb97b, 0xfe8946a924933545, 0x2af997a27369091c,
    0xaa62c88e0b294011, 0x058eb9d810ce9f74, 0xb3cb23eced349ae4, 0xa3648177a77b4a84,
    0x43153d905992d95d, 0xf4e2a97cda44aa4b, 0x5baa2702b908682f, 0x082923bdf4f750d1,
    0x98ae09a325893803, 0xf8a6475077968838, 0xceb0735bf00b2c5f, 0x0a1a5d953888e072,
    0x2fcb190489f94475, 0xb5be06270dec69fc, 0x739cb934b09acf8b, 0x537750b75ec7f25b,
    0xe9dd318bae1f3961, 0xf7462137299efe1a, 0xb1f6b8eee9adb940, 0xbdebcc8a809dfe6b,
    0x40fc1f791b178113, 0x3ac1c3362d014864, 0x9a016184bdb8aeba, 0x95f2394459fbc25e,
];

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod exec;

#[cfg(feature = "host")]
pub mod sequence_builder;

#[cfg(feature = "host")]
pub mod host;

#[cfg(all(test, feature = "host"))]
mod test_constants;

#[cfg(all(test, feature = "host"))]
mod tests;
