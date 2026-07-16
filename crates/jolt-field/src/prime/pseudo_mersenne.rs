//! `2^k - offset` pseudo-Mersenne registry and field aliases.
//!
//! Concrete aliases include both coordinates of `q = 2^k - offset` so adding
//! another prime at the same bit width does not create an implicit canonical
//! choice.

use super::{Fp32, Fp64};

/// Maximum supported offset in this `2^k - offset` specialization.
pub const PRIME_OFFSET_MAX: u128 = 1u128 << 16;

/// Current active bit-size bound for concrete field aliases in this phase.
pub const PRIME_OFFSET_IMPLEMENTED_MAX_BITS: u32 = 128;

/// Metadata describing a `2^k - offset` pseudo-Mersenne modulus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimeOffsetSpec {
    /// `k` in `2^k - offset`.
    pub bits: u32,
    /// `offset` in `2^k - offset`.
    pub offset: u16,
    /// Modulus value.
    pub modulus: u128,
}

/// Compute `2^k - offset` for `k <= 128`.
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

/// Return the registered prime spec for exactly `(bits, offset)`.
pub const fn registered_prime_offset_spec(bits: u32, offset: u128) -> Option<PrimeOffsetSpec> {
    let mut i = 0;
    while i < PRIME_OFFSET_SPECS.len() {
        let spec = PRIME_OFFSET_SPECS[i];
        if spec.bits == bits && (spec.offset as u128) == offset {
            return Some(spec);
        }
        i += 1;
    }
    None
}

/// Check whether `(k, offset)` is an explicitly registered `2^k - offset` prime.
pub const fn is_registered_prime_offset(bits: u32, offset: u128) -> bool {
    if bits > PRIME_OFFSET_IMPLEMENTED_MAX_BITS || offset > PRIME_OFFSET_MAX {
        return false;
    }
    registered_prime_offset_spec(bits, offset).is_some()
}

/// `offset` for `k = 24`.
pub(crate) const PRIME24_OFFSET3_OFFSET: u16 = 3;
/// `offset` for `k = 30`.
pub(crate) const PRIME30_OFFSET35_OFFSET: u16 = 35;
/// `offset` for `k = 31`.
pub(crate) const PRIME31_OFFSET19_OFFSET: u16 = 19;
/// `offset` for `k = 32`.
pub(crate) const PRIME32_OFFSET99_OFFSET: u16 = 99;
/// `offset` for `k = 40`.
pub(crate) const PRIME40_OFFSET195_OFFSET: u16 = 195;
/// `offset` for `k = 48`.
pub(crate) const PRIME48_OFFSET59_OFFSET: u16 = 59;
/// `offset` for `k = 56`.
pub(crate) const PRIME56_OFFSET27_OFFSET: u16 = 27;
/// `offset` for `k = 64`.
pub(crate) const PRIME64_OFFSET59_OFFSET: u16 = 59;
/// `offset` for `k = 128`.
pub(crate) const PRIME128_OFFSET275_OFFSET: u16 = 275;

/// `2^24 - 3`.
pub(crate) const PRIME24_OFFSET3_MODULUS: u32 =
    ((1u128 << 24) - (PRIME24_OFFSET3_OFFSET as u128)) as u32;
/// `2^30 - 35`.
pub(crate) const PRIME30_OFFSET35_MODULUS: u32 =
    ((1u128 << 30) - (PRIME30_OFFSET35_OFFSET as u128)) as u32;
/// `2^31 - 19`.
pub(crate) const PRIME31_OFFSET19_MODULUS: u32 =
    ((1u128 << 31) - (PRIME31_OFFSET19_OFFSET as u128)) as u32;
/// `2^32 - 99`.
pub(crate) const PRIME32_OFFSET99_MODULUS: u32 =
    ((1u128 << 32) - (PRIME32_OFFSET99_OFFSET as u128)) as u32;
/// `2^40 - 195`.
pub(crate) const PRIME40_OFFSET195_MODULUS: u64 =
    ((1u128 << 40) - (PRIME40_OFFSET195_OFFSET as u128)) as u64;
/// `2^48 - 59`.
pub(crate) const PRIME48_OFFSET59_MODULUS: u64 =
    ((1u128 << 48) - (PRIME48_OFFSET59_OFFSET as u128)) as u64;
/// `2^56 - 27`.
pub(crate) const PRIME56_OFFSET27_MODULUS: u64 =
    ((1u128 << 56) - (PRIME56_OFFSET27_OFFSET as u128)) as u64;
/// `2^64 - 59`.
pub(crate) const PRIME64_OFFSET59_MODULUS: u64 = u64::MAX - ((PRIME64_OFFSET59_OFFSET as u64) - 1);
/// `2^128 - 275`.
pub(crate) const PRIME128_OFFSET275_MODULUS: u128 =
    u128::MAX - (PRIME128_OFFSET275_OFFSET as u128 - 1);

/// Prime field for `2^24 - 3`.
pub type Prime24Offset3 = Fp32<PRIME24_OFFSET3_MODULUS>;
/// Prime field for `2^30 - 35`.
pub type Prime30Offset35 = Fp32<PRIME30_OFFSET35_MODULUS>;
/// Prime field for `2^31 - 19`.
pub type Prime31Offset19 = Fp32<PRIME31_OFFSET19_MODULUS>;
/// Prime field for `2^32 - 99`.
pub type Prime32Offset99 = Fp32<PRIME32_OFFSET99_MODULUS>;
/// Prime field for `2^40 - 195`.
pub type Prime40Offset195 = Fp64<PRIME40_OFFSET195_MODULUS>;
/// Prime field for `2^48 - 59`.
pub type Prime48Offset59 = Fp64<PRIME48_OFFSET59_MODULUS>;
/// Prime field for `2^56 - 27`.
pub type Prime56Offset27 = Fp64<PRIME56_OFFSET27_MODULUS>;
/// Prime field for `2^64 - 59`.
pub type Prime64Offset59 = Fp64<PRIME64_OFFSET59_MODULUS>;

/// `2^k - offset` profiles currently enabled in-code.
///
/// Every enabled entry satisfies the current in-code `2^k - offset` policy.
pub const PRIME_OFFSET_SPECS: [PrimeOffsetSpec; 9] = [
    PrimeOffsetSpec {
        bits: 24,
        offset: PRIME24_OFFSET3_OFFSET,
        modulus: PRIME24_OFFSET3_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 30,
        offset: PRIME30_OFFSET35_OFFSET,
        modulus: PRIME30_OFFSET35_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 31,
        offset: PRIME31_OFFSET19_OFFSET,
        modulus: PRIME31_OFFSET19_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 32,
        offset: PRIME32_OFFSET99_OFFSET,
        modulus: PRIME32_OFFSET99_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 40,
        offset: PRIME40_OFFSET195_OFFSET,
        modulus: PRIME40_OFFSET195_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 48,
        offset: PRIME48_OFFSET59_OFFSET,
        modulus: PRIME48_OFFSET59_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 56,
        offset: PRIME56_OFFSET27_OFFSET,
        modulus: PRIME56_OFFSET27_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 64,
        offset: PRIME64_OFFSET59_OFFSET,
        modulus: PRIME64_OFFSET59_MODULUS as u128,
    },
    PrimeOffsetSpec {
        bits: 128,
        offset: PRIME128_OFFSET275_OFFSET,
        modulus: PRIME128_OFFSET275_MODULUS,
    },
];

// All PseudoMersenneField impls for Fp32/Fp64/Fp128 are blanket impls in
// their respective modules (fp32.rs, fp64.rs, fp128.rs).
