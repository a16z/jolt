//! Solinas backend: pseudo-Mersenne prime fields `p = 2^k − c`.
//!
//! `word.rs` stamps the `u32`- and `u64`-backed field types from one fold
//! algebra; `fp128.rs` is the hand-written two-limb field; this module holds
//! the family trait, the `2^k − offset` registry, and shared helpers.

mod ext;
mod fp128;
mod word;

pub use ext::{
    canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas, Ext2,
    FpExt2, FpExt4, FpExt8,
};
pub use fp128::Fp128;
pub use word::{Fp32, Fp64};

use crate::Ring;

/// Maximum supported offset in the `2^k − offset` specialization.
pub const PRIME_OFFSET_MAX: u128 = 1 << 16;

/// Current active bit-size bound for concrete field aliases.
pub const PRIME_OFFSET_IMPLEMENTED_MAX_BITS: u32 = 128;

/// Metadata describing a registered `2^k − offset` pseudo-Mersenne modulus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimeOffsetSpec {
    /// `k` in `2^k − offset`.
    pub bits: u32,
    /// `offset` in `2^k − offset`.
    pub offset: u16,
    /// Modulus value.
    pub modulus: u128,
}

/// Compute `2^k − offset` for `k <= 128`.
pub const fn pseudo_mersenne_modulus(bits: u32, offset: u128) -> Option<u128> {
    if bits == 0 || bits > 128 || offset == 0 {
        return None;
    }
    if bits == 128 {
        Some(u128::MAX - (offset - 1))
    } else {
        Some((1u128 << bits) - offset)
    }
}

/// `2^k − offset` as the storage word for a registered alias; fails at
/// compile time on invalid parameters.
#[expect(
    clippy::panic,
    reason = "CTFE-only: all call sites are const registry entries"
)]
const fn pm(bits: u32, offset: u128) -> u128 {
    match pseudo_mersenne_modulus(bits, offset) {
        Some(m) => m,
        None => panic!("invalid pseudo-Mersenne parameters"),
    }
}

const fn spec(bits: u32, offset: u16) -> PrimeOffsetSpec {
    PrimeOffsetSpec {
        bits,
        offset,
        modulus: pm(bits, offset as u128),
    }
}

/// `2^k − offset` profiles currently enabled in-code.
pub const PRIME_OFFSET_SPECS: [PrimeOffsetSpec; 9] = [
    spec(24, 3),
    spec(30, 35),
    spec(31, 19),
    spec(32, 99),
    spec(40, 195),
    spec(48, 59),
    spec(56, 27),
    spec(64, 59),
    spec(128, 275),
];

/// Return the registered prime spec for exactly `(bits, offset)`.
pub const fn registered_prime_offset_spec(bits: u32, offset: u128) -> Option<PrimeOffsetSpec> {
    let mut i = 0;
    while i < PRIME_OFFSET_SPECS.len() {
        if PRIME_OFFSET_SPECS[i].bits == bits && (PRIME_OFFSET_SPECS[i].offset as u128) == offset {
            return Some(PRIME_OFFSET_SPECS[i]);
        }
        i += 1;
    }
    None
}

/// Check whether `(k, offset)` is an explicitly registered `2^k − offset` prime.
pub const fn is_registered_prime_offset(bits: u32, offset: u128) -> bool {
    offset <= PRIME_OFFSET_MAX
        && bits <= PRIME_OFFSET_IMPLEMENTED_MAX_BITS
        && registered_prime_offset_spec(bits, offset).is_some()
}

/// Prime field for `2^24 - 3`.
pub type Prime24Offset3 = Fp32<{ pm(24, 3) as u32 }>;
/// Prime field for `2^30 - 35`.
pub type Prime30Offset35 = Fp32<{ pm(30, 35) as u32 }>;
/// Prime field for `2^31 - 19`.
pub type Prime31Offset19 = Fp32<{ pm(31, 19) as u32 }>;
/// Prime field for `2^32 - 99`.
pub type Prime32Offset99 = Fp32<{ pm(32, 99) as u32 }>;
/// Prime field for `2^40 - 195`.
pub type Prime40Offset195 = Fp64<{ pm(40, 195) as u64 }>;
/// Prime field for `2^48 - 59`.
pub type Prime48Offset59 = Fp64<{ pm(48, 59) as u64 }>;
/// Prime field for `2^56 - 27`.
pub type Prime56Offset27 = Fp64<{ pm(56, 27) as u64 }>;
/// Prime field for `2^64 - 59`.
pub type Prime64Offset59 = Fp64<{ pm(64, 59) as u64 }>;
/// Prime field for `2^128 − 275`.
pub type Prime128Offset275 = Fp128<{ pm(128, 275) }>;
/// Prime field for `2^128 − 159`. Split-NTT-only helper prime.
pub type Prime128Offset159 = Fp128<{ pm(128, 159) }>;
/// Prime field for `2^128 − 2355` (`p ≡ 5 mod 8`): smooth multiplicative
/// subgroup of order `14700 = 2² · 3 · 5² · 7²` for mixed-radix FFT.
pub type Prime128Offset2355 = Fp128<{ pm(128, 2355) }>;
/// Prime field for `2^128 − 2^32 + 22537` (`C = 0xFFFF_A7F7`): smooth
/// multiplicative subgroup of order `2^3 · 3^7 = 17496` (pure radix-3
/// subgroup `3^7 = 2187`). The default protocol prime.
pub type Prime128OffsetA7F7 = Fp128<{ pm(128, 0xFFFF_A7F7) }>;

/// Builds the balanced signed-digit table for `1 <= log_basis <= 6`.
pub fn balanced_digit_lut<F: Ring>(log_basis: u32) -> [F; 64] {
    debug_assert!(log_basis > 0 && log_basis <= 6);
    let basis = 1usize << log_basis;
    let half_basis = (basis >> 1) as i64;
    std::array::from_fn(|i| {
        if i < basis {
            F::from_i64(i as i64 - half_basis)
        } else {
            F::zero()
        }
    })
}

/// Horner reduction of arbitrary-length little-endian bytes modulo the field
/// order (the >16-byte path of
/// [`from_bytes_le_reduced`](crate::CanonicalEncoding::from_bytes_le_reduced)).
#[inline(always)]
pub(crate) fn reduce_le_bytes_mod_order<F: Ring>(bytes: &[u8]) -> F {
    let base = F::from_u64(256);
    bytes.iter().rev().fold(F::zero(), |acc, &byte| {
        acc * base + F::from_u64(byte as u64)
    })
}
