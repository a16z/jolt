//! Id-free balanced-digit algebra shared by the packed (lattice) protocol
//! families: the 64-bit balanced radix window, its chunking, the centered
//! row-value MLE, and the honest row encoder.
//!
//! Like [`crate::twist`], this module carries no protocol ids — each protocol
//! family instantiates it with its own committed-polynomial vocabulary
//! (pinned by the `protocol_modules_are_import_disjoint` boundary test).

use jolt_field::Ring;
use thiserror::Error;

/// Bit width one balanced-digit numeral covers: the digits decompose a 64-bit
/// window and the signed carry column carries place value `2^64`.
pub const BALANCED_INC_BITS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BalancedChunkingError {
    #[error("increment digit width must be nonzero")]
    ZeroChunkWidth,
    #[error("increment digit width {chunk_width} must divide {BALANCED_INC_BITS}")]
    ChunkWidthMisaligned { chunk_width: usize },
    #[error("increment digit width {chunk_width} does not fit the address domain")]
    ChunkWidthTooLarge { chunk_width: usize },
}

/// The balanced radix-`2^chunk_width` decomposition of one 64-bit increment
/// window: `BALANCED_INC_BITS / chunk_width` digit columns plus a signed
/// carry, every digit centered in `[-2^(chunk_width-1), 2^(chunk_width-1))`
/// so that `Σ_j 2^(chunk_width·j)·digit_j + 2^BALANCED_INC_BITS·carry` is the
/// signed value itself — no unsigned shift. See [`balanced_inc_value`] for
/// the value map, which sends digit zero to zero and therefore lets a zero
/// value sit entirely on the row the commitment omits.
///
/// The chunk width is fixed to the shared one-hot chunk size (`log_k_chunk`)
/// so the digit polynomials sit in the `Ra` families' variable-count class
/// and can share their final packed point (see `specs/lattice-claims.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BalancedIncChunking {
    chunk_width: usize,
}

impl BalancedIncChunking {
    pub const fn new(chunk_width: usize) -> Result<Self, BalancedChunkingError> {
        if chunk_width == 0 {
            return Err(BalancedChunkingError::ZeroChunkWidth);
        }
        if chunk_width >= usize::BITS as usize {
            return Err(BalancedChunkingError::ChunkWidthTooLarge { chunk_width });
        }
        if !BALANCED_INC_BITS.is_multiple_of(chunk_width) {
            return Err(BalancedChunkingError::ChunkWidthMisaligned { chunk_width });
        }
        Ok(Self { chunk_width })
    }

    pub const fn chunk_width(self) -> usize {
        self.chunk_width
    }

    pub const fn chunk_count(self) -> usize {
        BALANCED_INC_BITS / self.chunk_width
    }

    /// The place value `2^(chunk_width * index)` weighting chunk `index` in
    /// the little-endian reconstruction of the low 64 bits.
    pub fn place_value<F: Ring>(self, index: usize) -> F {
        F::pow2(self.chunk_width * index)
    }
}

/// MLE of the centered row value used by balanced increment digits: the
/// identity MLE `Σ_i 2^(n−1−i)·point_i` recentered by `2^n` times the sign
/// (msb) coordinate.
pub fn balanced_inc_value<F: Ring>(address_point: &[F]) -> F {
    let bits = address_point.len();
    let unsigned = address_point
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (position, coordinate)| {
            acc + F::pow2(bits - 1 - position) * *coordinate
        });
    let msb = address_point.first().copied().unwrap_or_else(F::zero);
    unsigned - F::pow2(bits) * msb
}

/// Bias whose radix-`2^width` digits recenter every digit of
/// `value + bias` into the balanced alphabet: `(K/2) · Σ_j K^j` over the
/// window's digit places (`K = 2^width`), per digit width dividing
/// [`BALANCED_INC_BITS`] (0 elsewhere). Precomputed because the closed
/// form's i128 division lowers to a `__udivti3` libcall, and the encoders
/// read the bias per cycle per balanced column.
const BALANCED_BIASES: [i128; BALANCED_INC_BITS + 1] = {
    let mut table = [0i128; BALANCED_INC_BITS + 1];
    let mut width = 1;
    while width <= BALANCED_INC_BITS {
        if BALANCED_INC_BITS.is_multiple_of(width) {
            let radix = 1i128 << width;
            table[width] = (radix / 2) * (((1i128 << BALANCED_INC_BITS) - 1) / (radix - 1));
        }
        width += 1;
    }
    table
};

fn balanced_bias(width: usize) -> i128 {
    debug_assert!(width > 0 && BALANCED_INC_BITS.is_multiple_of(width));
    BALANCED_BIASES[width]
}

fn biased_for_balanced_digits(value: i128, width: usize) -> i128 {
    debug_assert!(value.unsigned_abs() < 1u128 << BALANCED_INC_BITS);
    value + balanced_bias(width)
}

/// The honest encoder of one centered digit: the one-hot row selected by
/// digit `index` of `value` (`|value| < 2^BALANCED_INC_BITS`), decoding to
/// the centered row value under [`balanced_inc_value`].
pub fn balanced_digit_row(value: i128, width: usize, index: usize) -> usize {
    let radix = 1i128 << width;
    let standard = (biased_for_balanced_digits(value, width) >> (width * index)) & (radix - 1);
    ((standard + radix / 2) & (radix - 1)) as usize
}

/// The honest encoder of the signed carry above bit 63: the one-hot row
/// selected by the carry of `value`, decoding to the carry value under
/// [`balanced_inc_value`].
pub fn balanced_carry_row(value: i128, width: usize) -> usize {
    let radix = 1i128 << width;
    let carry = biased_for_balanced_digits(value, width) >> BALANCED_INC_BITS;
    debug_assert!((-1..=1).contains(&carry));
    carry.rem_euclid(radix) as usize
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use jolt_poly::boolean_point_msb;

    #[test]
    fn chunking_requires_divisor_widths() {
        assert_eq!(
            BalancedIncChunking::new(0),
            Err(BalancedChunkingError::ZeroChunkWidth)
        );
        assert_eq!(
            BalancedIncChunking::new(7),
            Err(BalancedChunkingError::ChunkWidthMisaligned { chunk_width: 7 })
        );

        let chunking = BalancedIncChunking::new(8).unwrap();
        assert_eq!(chunking.chunk_width(), 8);
        assert_eq!(chunking.chunk_count(), 8);
    }

    #[test]
    fn place_values_reconstruct_little_endian_chunks() {
        let chunking = BalancedIncChunking::new(16).unwrap();
        assert_eq!(chunking.chunk_count(), 4);

        let value: u64 = 0x0123_4567_89ab_cdef;
        let reconstructed = (0..chunking.chunk_count()).fold(Fr::from_u64(0), |acc, index| {
            let chunk = (value >> (16 * index)) & 0xffff;
            acc + chunking.place_value::<Fr>(index) * Fr::from_u64(chunk)
        });
        assert_eq!(reconstructed, Fr::from_u64(value));
    }

    #[test]
    fn balanced_inc_value_matches_centered_boolean_rows() {
        for width in [4, 8] {
            let radix = 1usize << width;
            for row in 0..radix {
                let expected = if row < radix / 2 {
                    row as i128
                } else {
                    row as i128 - radix as i128
                };
                assert_eq!(
                    balanced_inc_value(&boolean_point_msb::<Fr>(width, row)),
                    Fr::from_i128(expected)
                );
            }
        }
    }

    /// The encoder rows decode back to the encoded value through the centered
    /// value map and the chunking's place values — the balanced numeral is a
    /// faithful signed encoding over the whole `|value| < 2^64` window.
    #[test]
    fn encoder_rows_decode_to_the_encoded_value() {
        for width in [4usize, 8] {
            let chunking = BalancedIncChunking::new(width).unwrap();
            for value in [
                0i128,
                1,
                -1,
                7,
                -3,
                (1 << 40) + 5,
                -(1 << 63),
                (1i128 << 64) - 1,
                -((1i128 << 64) - 1),
            ] {
                let mut decoded = Fr::from_i128(0);
                for index in 0..chunking.chunk_count() {
                    let row = balanced_digit_row(value, width, index);
                    assert!(row < 1 << width);
                    decoded += chunking.place_value::<Fr>(index)
                        * balanced_inc_value(&boolean_point_msb::<Fr>(width, row));
                }
                let carry = balanced_carry_row(value, width);
                decoded += Fr::pow2(BALANCED_INC_BITS)
                    * balanced_inc_value(&boolean_point_msb::<Fr>(width, carry));
                assert_eq!(decoded, Fr::from_i128(value), "value {value}");
            }
        }
    }
}
