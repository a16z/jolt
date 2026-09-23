//! Solinas backend: pseudo-Mersenne prime fields `p = 2^k − c`.
//!
//! `word.rs` stamps the `u32`- and `u64`-backed field types from one fold
//! algebra; `fp128.rs` is the hand-written two-limb field; this module holds
//! the family trait, the `2^k − offset` registry, and shared helpers.

mod ext;
mod fp128;
mod packed;
pub mod parallel;
mod unreduced;
mod word;

pub use ext::{
    canonical_extension_basis, solve_frobenius_moore, validate_canonical_frobenius_thetas, Ext2,
    FpExt2, FpExt4, FpExt8,
};
pub use fp128::Fp128;
pub use packed::{
    Fp128Packing, Fp32Packing, Fp64Packing, PackedFpExt2, PackedFpExt4, PackedFpExt8,
};
pub use unreduced::{
    AccumPair, FoldMatrixFp32, FoldMatrixFp64, Fp128MulU64Accum, Fp128ProductAccum, Fp128x8i32,
    Fp32ProductAccum, Fp32x2i32, Fp64ProductAccum, Fp64x4i32, FpExt2Fp64ProductAccum,
    FpExt4Fp32ProductAccum,
};
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

/// Sample uniformly from `[0, modulus)` with canonical byte consumption.
///
/// `modulus_bits` is the significant bit length of `modulus`. Each attempt
/// reads exactly `ceil(modulus_bits / 8)` little-endian bytes, clears unused
/// high bits, and rejects candidates greater than or equal to `modulus`. This
/// byte-consumption contract is deterministic for a fixed
/// [`rand_core::RngCore`] stream.
#[inline]
pub(crate) fn sample_uniform_below<R: rand_core::RngCore>(
    rng: &mut R,
    modulus: u128,
    modulus_bits: u32,
) -> u128 {
    debug_assert!(modulus > 0);
    debug_assert_eq!(modulus_bits, u128::BITS - modulus.leading_zeros());
    let byte_len = modulus_bits.div_ceil(8) as usize;
    let mask = if modulus_bits == u128::BITS {
        u128::MAX
    } else {
        (1u128 << modulus_bits) - 1
    };
    loop {
        let mut bytes = [0u8; 16];
        rng.fill_bytes(&mut bytes[..byte_len]);
        let candidate = u128::from_le_bytes(bytes) & mask;
        if candidate < modulus {
            return candidate;
        }
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

#[cfg(test)]
mod sampling_tests {
    use super::{pm, sample_uniform_below, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
    use crate::Field;
    use rand_core::{Error, RngCore};

    struct ScriptedRng {
        bytes: Vec<u8>,
        cursor: usize,
    }

    impl ScriptedRng {
        fn new(bytes: Vec<u8>) -> Self {
            Self { bytes, cursor: 0 }
        }
    }

    impl RngCore for ScriptedRng {
        fn next_u32(&mut self) -> u32 {
            let mut bytes = [0u8; 4];
            self.fill_bytes(&mut bytes);
            u32::from_le_bytes(bytes)
        }

        fn next_u64(&mut self) -> u64 {
            let mut bytes = [0u8; 8];
            self.fill_bytes(&mut bytes);
            u64::from_le_bytes(bytes)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            let end = self.cursor + dest.len();
            dest.copy_from_slice(&self.bytes[self.cursor..end]);
            self.cursor = end;
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    #[test]
    fn rejection_consumes_candidates_and_resumes_at_the_cursor() {
        let mut rng = ScriptedRng::new(vec![251, 7, 250]);
        assert_eq!(sample_uniform_below(&mut rng, 251, 8), 7);
        assert_eq!(rng.cursor, 2);
        assert_eq!(sample_uniform_below(&mut rng, 251, 8), 250);
        assert_eq!(rng.cursor, 3);
    }

    #[test]
    fn non_byte_aligned_modulus_masks_unused_high_bits() {
        // 0xff_ff_ff_ff becomes 0x3f_ff_ff_ff at a 30-bit modulus width, then
        // is rejected. The following little-endian candidate 42 is accepted.
        let mut rng = ScriptedRng::new(vec![0xff, 0xff, 0xff, 0xff, 42, 0, 0, 0]);
        assert_eq!(sample_uniform_below(&mut rng, (1u128 << 30) - 35, 30), 42);
        assert_eq!(rng.cursor, 8);
    }

    #[test]
    fn sub_word_modulus_reads_only_its_canonical_byte_width() {
        let mut rng = ScriptedRng::new(vec![42, 0, 0, 99]);
        assert_eq!(sample_uniform_below(&mut rng, (1u128 << 24) - 3, 24), 42);
        assert_eq!(rng.cursor, 3);
    }

    #[test]
    fn prime_fields_share_canonical_rejection_and_byte_consumption() {
        let fp32_modulus = pm(32, 99) as u32;
        let mut fp32_bytes = Vec::from(fp32_modulus.to_le_bytes());
        fp32_bytes.extend_from_slice(&42u32.to_le_bytes());
        let mut fp32_rng = ScriptedRng::new(fp32_bytes);
        assert_eq!(
            Prime32Offset99::random(&mut fp32_rng),
            Prime32Offset99::from_canonical_u32(42)
        );
        assert_eq!(fp32_rng.cursor, 8);

        let fp64_modulus = pm(64, 59) as u64;
        let mut fp64_bytes = Vec::from(fp64_modulus.to_le_bytes());
        fp64_bytes.extend_from_slice(&42u64.to_le_bytes());
        let mut fp64_rng = ScriptedRng::new(fp64_bytes);
        assert_eq!(
            Prime64Offset59::random(&mut fp64_rng),
            Prime64Offset59::from_canonical_u64(42)
        );
        assert_eq!(fp64_rng.cursor, 16);

        let fp128_modulus = pm(128, 0xFFFF_A7F7);
        let mut fp128_bytes = Vec::from(fp128_modulus.to_le_bytes());
        fp128_bytes.extend_from_slice(&42u128.to_le_bytes());
        let mut fp128_rng = ScriptedRng::new(fp128_bytes);
        // SAFETY: 42 is below the field modulus.
        assert_eq!(Prime128OffsetA7F7::random(&mut fp128_rng), unsafe {
            Prime128OffsetA7F7::from_canonical_u128(42)
        });
        assert_eq!(fp128_rng.cursor, 32);
    }
}
