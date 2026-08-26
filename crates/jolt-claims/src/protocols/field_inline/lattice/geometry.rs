//! Pure packed-mode geometry of the field-register extension: the u64 limb
//! decomposition of `FieldRdInc`'s canonical representative and the single
//! linear recomposition identity over `F`
//! (`specs/field-inline-portability.md`, Axis 1).
//!
//! Each limb is `RdInc`-shaped (`|limb| < 2^64`) and rides the shared
//! balanced-digit machinery ([`crate::lattice`]) verbatim: per limb,
//! `chunk_count` centered digit columns plus one signed carry column, every
//! column a `(digit-value ‖ cycle)` one-hot in the `Ra` variable-count class.
//! Digit smallness enforcement doubles as the limb range check
//! (`limb < 2^64`), and a zero `FieldRdInc` sits entirely on the digit-zero
//! rows the packed commitment omits — non-FR cycles are free.

use jolt_field::{CanonicalBytes, Ring};
use thiserror::Error;

use crate::lattice::{BalancedChunkingError, BalancedIncChunking, BALANCED_INC_BITS};
use crate::protocols::field_inline::FieldInlineCommittedPolynomial;

/// Bit width of one committed limb column group: the shared balanced-digit
/// window, so every limb rides the digit machinery unchanged.
pub const FIELD_INC_LIMB_BITS: usize = BALANCED_INC_BITS;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FieldIncLimbGeometryError {
    #[error(transparent)]
    Chunking(#[from] BalancedChunkingError),
    #[error(transparent)]
    PrefixLayout(#[from] jolt_openings::OpeningsError),
    #[error("the field-inc limb decomposition requires at least one limb")]
    ZeroLimbCount,
}

/// Number of u64 limbs of `F`'s canonical representative — the committed
/// limb-column groups of the packed `FieldRdInc` treatment (4 for a 254-bit
/// field, 2 for a 128-bit field).
pub fn field_inc_limb_count<F: CanonicalBytes>() -> usize {
    F::NUM_BYTES.div_ceil(FIELD_INC_LIMB_BITS / 8)
}

/// The little-endian u64 limbs of `value`'s canonical representative. Exactly
/// [`field_inc_limb_count`] limbs; a short final byte chunk zero-extends.
pub fn canonical_limbs<F: CanonicalBytes>(value: &F) -> Vec<u64> {
    let bytes = value.to_bytes_le_vec();
    bytes
        .chunks(FIELD_INC_LIMB_BITS / 8)
        .map(|chunk| {
            let mut limb = [0u8; 8];
            // `chunks(8)` yields 1..=8 bytes, so the prefix slice is in range.
            limb[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(limb)
        })
        .collect()
}

/// The radix weight of limb `limb` in the recomposition identity
/// `FieldRdInc = Σ_i 2^(64·i) · limb_i` — exact over `F` because the
/// canonical representative is `< p`, so the limbs recompose with no carries
/// and no modular wraparound.
pub fn limb_place_value<F: Ring>(limb: usize) -> F {
    F::pow2(FIELD_INC_LIMB_BITS * limb)
}

/// The coefficient of one committed limb column in the single linear
/// recomposition of `FieldRdInc` from its balanced digit/carry columns:
/// digit `(i, j) ↦ 2^(64·i) · 2^(w·j)` and carry `i ↦ 2^(64·i) · 2^64`, so
/// that `FieldRdInc(cycle) = Σ_column coefficient · value(hot row)` under the
/// shared centered value map ([`crate::lattice::balanced_inc_value`]).
/// `None` for ids outside the limb decomposition or digits past the
/// chunking's window.
pub fn recomposition_coefficient<F: Ring>(
    chunking: BalancedIncChunking,
    polynomial: FieldInlineCommittedPolynomial,
) -> Option<F> {
    match polynomial {
        FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb, index } => (index
            < chunking.chunk_count())
        .then(|| limb_place_value::<F>(limb) * chunking.place_value::<F>(index)),
        FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb } => {
            Some(limb_place_value::<F>(limb) * F::pow2(FIELD_INC_LIMB_BITS))
        }
        FieldInlineCommittedPolynomial::FieldRdInc => None,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::lattice::{balanced_carry_row, balanced_digit_row, balanced_inc_value};
    use crate::protocols::field_inline::lattice::packing::{
        field_inc_limb_columns, FieldIncLimbShape,
    };
    use jolt_field::{Field, Fr, Ring};
    use jolt_poly::boolean_point_msb;

    #[test]
    fn limb_count_covers_the_canonical_byte_width() {
        // BN254 Fr: 32 canonical bytes -> 4 u64 limbs. (The 128-bit packed
        // field's count of 2 is pinned where that field is visible —
        // jolt-akita's norm-budget test.)
        assert_eq!(field_inc_limb_count::<Fr>(), 4);
    }

    /// The honest packed encoding of `value`: every limb column's selected
    /// row through the shared verbatim encoder, decoded by
    /// [`recomposition_coefficient`] and the centered value map.
    fn recompose(value: &Fr, log_k_chunk: usize) -> Fr {
        let chunking = BalancedIncChunking::new(log_k_chunk).unwrap();
        let shape = FieldIncLimbShape {
            limbs: field_inc_limb_count::<Fr>(),
            log_t: 1,
            log_k_chunk,
        };
        let limbs = canonical_limbs(value);
        assert_eq!(limbs.len(), shape.limbs);
        field_inc_limb_columns(&shape)
            .unwrap()
            .into_iter()
            .map(|column| {
                let row = match column {
                    FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb, index } => {
                        balanced_digit_row(limbs[limb] as i128, log_k_chunk, index)
                    }
                    FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb } => {
                        balanced_carry_row(limbs[limb] as i128, log_k_chunk)
                    }
                    FieldInlineCommittedPolynomial::FieldRdInc => unreachable!("not a limb column"),
                };
                assert!(row < 1 << log_k_chunk);
                recomposition_coefficient::<Fr>(chunking, column).unwrap()
                    * balanced_inc_value(&boolean_point_msb::<Fr>(log_k_chunk, row))
            })
            .sum()
    }

    /// The spec's Axis 1 identity, end to end through the verbatim digit
    /// machinery: `FieldRdInc = Σ_i 2^(64·i)·limb_i` is exact for every
    /// canonical representative — the digit machinery hosts full-range u64
    /// limbs (carry included) with no wraparound.
    #[test]
    fn recomposition_is_exact_over_the_canonical_representative() {
        let one = Fr::from_u64(1);
        let values = [
            Fr::from_u64(0),
            one,
            Fr::from_u64(0) - one,
            Fr::from_u64(u64::MAX),
            Fr::pow2(64),
            Fr::pow2(127) - one,
            Fr::pow2(200) + Fr::from_u64(0x0123_4567_89ab_cdef),
            Fr::from_u64(2).inverse().unwrap(),
        ];
        for log_k_chunk in [4usize, 8] {
            for value in &values {
                assert_eq!(recompose(value, log_k_chunk), *value);
            }
        }
    }

    /// A zero `FieldRdInc` selects the digit-zero row in every limb column —
    /// non-FR and padding cycles put nothing in the packed commitment.
    #[test]
    fn zero_field_rd_inc_sits_on_the_omitted_digit_zero_rows() {
        for log_k_chunk in [4usize, 8] {
            for limb in canonical_limbs(&Fr::from_u64(0)) {
                assert_eq!(limb, 0);
                assert_eq!(balanced_carry_row(limb as i128, log_k_chunk), 0);
                for index in 0..BalancedIncChunking::new(log_k_chunk).unwrap().chunk_count() {
                    assert_eq!(balanced_digit_row(limb as i128, log_k_chunk, index), 0);
                }
            }
        }
    }

    #[test]
    fn recomposition_coefficient_is_none_off_the_limb_columns() {
        let chunking = BalancedIncChunking::new(8).unwrap();
        assert_eq!(
            recomposition_coefficient::<Fr>(chunking, FieldInlineCommittedPolynomial::FieldRdInc),
            None
        );
        assert_eq!(
            recomposition_coefficient::<Fr>(
                chunking,
                FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb: 0, index: 8 },
            ),
            None
        );
    }
}
