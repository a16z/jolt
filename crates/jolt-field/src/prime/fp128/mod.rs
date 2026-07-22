//! 128-bit prime field for primes of the form `p = 2^128 − c` with `c < 2^32`.
//!
//! Uses Solinas-style two-fold reduction: no Montgomery form, ~23 cycles/mul
//! on both AArch64 and x86-64.  The offset `c` is computed at compile time
//! from the const-generic modulus `P`.
//!
//! ## Built-in primes
//!
//! Two built-in protocol primes are exposed:
//!
//! - `Prime128OffsetA7F7` (`p = 2^128 − 2^32 + 22537`, `C = 0xFFFFA7F7`),
//!   whose multiplicative group has a smooth subgroup of order
//!   `2^3 · 3^7 = 17 496` (with a clean radix-3 substructure of order
//!   `3^7 = 2187`). This is the default protocol prime.
//! - `Prime128Offset2355` (`p = 2^128 − 2355`), with smooth subgroup
//!   `2² · 3 · 5² · 7² = 14 700`, supported as a peer prime.
//!
//! A secondary split-NTT-only prime `Prime128Offset159`
//! (`p = 2^128 − 159`, `p ≡ 33 mod 64`) is kept for the algebra benchmark/test
//! path that only needs 32-way roots of unity.

mod add_sub;
mod core;
mod mul;
mod primes;
mod reduce;
#[cfg(test)]
mod tests;
mod traits;
mod wide;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use ::core::arch::asm;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{FieldCore, FromPrimitiveInt};
use rand_core::RngCore;

use crate::{CanonicalField, HalvingField, PseudoMersenneField};

use super::util::{is_pow2_u64, log2_pow2_u64, mul64_wide};

pub use self::core::Fp128;
pub use primes::{Prime128Offset159, Prime128Offset2355, Prime128Offset275, Prime128OffsetA7F7};

/// Pack two u64 limbs into `[lo, hi]`.
#[inline(always)]
pub(super) const fn pack(lo: u64, hi: u64) -> [u64; 2] {
    [lo, hi]
}

/// Convert `u128` → `[u64; 2]`.
#[inline(always)]
pub(super) const fn from_u128(x: u128) -> [u64; 2] {
    [x as u64, (x >> 64) as u64]
}

/// Convert `[u64; 2]` → `u128`.
#[inline(always)]
pub(super) const fn to_u128(x: [u64; 2]) -> u128 {
    x[0] as u128 | (x[1] as u128) << 64
}
