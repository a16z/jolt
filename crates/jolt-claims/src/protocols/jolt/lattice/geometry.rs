//! Pure lattice geometry: the inc chunking, the byte one-hot variable
//! counts, and the decode-weight point algebra defining the reconstruction
//! relations' deriveds. Vocabulary is inherited — see the
//! [module doc](super).

use jolt_field::{Field, RingCore};
use jolt_poly::math::Math;
use jolt_poly::{eq_index_msb, IdentityPolynomial, MultilinearEvaluation};
use thiserror::Error;

use super::super::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;

/// Bit width of the unsigned fused increment (`FusedInc + 2^64` fits in 65
/// bits: the chunk polynomials carry the low 64,
/// [`UnsignedIncMsb`](crate::protocols::jolt::JoltCommittedPolynomial::UnsignedIncMsb)
/// the top bit).
pub const UNSIGNED_INC_BITS: usize = 64;

/// Bytecode read-raf val stages in lattice mode: the base stages plus one
/// carrying the `OpFlags(Store)` opening that `IncVirtualization` consumes as
/// its destination selector.
pub const LATTICE_BYTECODE_VAL_STAGES: usize = NUM_BYTECODE_VAL_STAGES + 1;

/// Bits of one byte value — the hot-value bits of a byte one-hot polynomial.
pub const BYTE_BITS: usize = 8;

/// Bytes of one 64-bit word.
pub const WORD_BYTES: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LatticeGeometryError {
    #[error(transparent)]
    PackingRegistration(#[from] jolt_openings::OpeningsError),
    #[error("unsigned inc chunk width must be nonzero")]
    ZeroChunkWidth,
    #[error("unsigned inc chunk width {chunk_width} must divide {UNSIGNED_INC_BITS}")]
    ChunkWidthMisaligned { chunk_width: usize },
    #[error("unsigned inc chunk width {chunk_width} does not fit the address domain")]
    ChunkWidthTooLarge { chunk_width: usize },
    #[error("byte one-hot byte count must be nonzero")]
    ZeroByteCount,
}

/// The base-`2^chunk_width` one-hot decomposition of the low
/// [`UNSIGNED_INC_BITS`] bits of the fused unsigned increment.
///
/// The chunk width is fixed to the shared one-hot chunk size (`log_k_chunk`)
/// so the chunk polynomials sit in the `Ra` families' variable-count class
/// and can share their final packed point (see `specs/lattice-claims.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsignedIncChunking {
    chunk_width: usize,
}

impl UnsignedIncChunking {
    pub const fn new(chunk_width: usize) -> Result<Self, LatticeGeometryError> {
        if chunk_width == 0 {
            return Err(LatticeGeometryError::ZeroChunkWidth);
        }
        if chunk_width >= usize::BITS as usize {
            return Err(LatticeGeometryError::ChunkWidthTooLarge { chunk_width });
        }
        if !UNSIGNED_INC_BITS.is_multiple_of(chunk_width) {
            return Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width });
        }
        Ok(Self { chunk_width })
    }

    pub const fn chunk_width(self) -> usize {
        self.chunk_width
    }

    pub const fn chunk_count(self) -> usize {
        UNSIGNED_INC_BITS / self.chunk_width
    }

    /// The place value `2^(chunk_width * index)` weighting chunk `index` in
    /// the little-endian reconstruction of the low 64 bits.
    pub fn place_value<F: RingCore>(self, index: usize) -> F {
        F::pow2(self.chunk_width * index)
    }
}

/// Number of variables of a byte one-hot polynomial carrying `bytes` byte
/// places per instance: `(byte ‖ place ‖ instance)` Boolean indices,
/// msb-first. Non-power-of-two byte counts round the place dimension up;
/// dummy place entries are zero by convention.
pub fn byte_num_vars(bytes: usize, log_rows: usize) -> Result<usize, LatticeGeometryError> {
    if bytes == 0 {
        return Err(LatticeGeometryError::ZeroByteCount);
    }
    Ok(byte_num_vars_unchecked(bytes, log_rows))
}

fn byte_num_vars_unchecked(bytes: usize, log_rows: usize) -> usize {
    BYTE_BITS + bytes.log_2() + log_rows
}

/// Number of `(byte ‖ place)` variables of a word byte one-hot — the round
/// count of the reconstruction sumchecks that bind only those (trusted
/// advice, program image; the word point is fixed by the incoming claim).
pub fn byte_place_vars() -> usize {
    byte_num_vars_unchecked(WORD_BYTES, 0)
}

/// Number of variables of the byte one-hot polynomial of 64-bit words — the
/// advice and program-image encoding, and the round count of the untrusted
/// advice reconstruction sumcheck (relation rounds and slot variable count
/// are the same number by construction). The infallible `WORD_BYTES` case of
/// [`byte_num_vars`].
pub fn word_byte_num_vars(log_words: usize) -> usize {
    byte_num_vars_unchecked(WORD_BYTES, log_words)
}

/// The multilinear extension of `place ↦ 256^place` at `point`, msb-first:
/// `Π_position ((256^(2^(bits − 1 − position)) − 1) · point[position] + 1)`.
/// The radix half of the byte decode; the value half is `jolt-poly`'s
/// `IdentityPolynomial`.
pub fn place_value_weight<F: RingCore>(point: &[F]) -> F {
    let bits = point.len();
    point
        .iter()
        .enumerate()
        .map(|(position, coordinate)| {
            (F::pow2(BYTE_BITS << (bits - 1 - position)) - F::one()) * *coordinate + F::one()
        })
        .product()
}

/// The byte-decode weight MLE `value(byte) · 256^place` at the bound
/// `(byte ‖ place)` coordinates — the weight rebuilding a little-endian word
/// from its byte one-hot entries: `word(instance) = Σ_{byte, place}
/// decode(byte, place) · Bytes(byte ‖ place ‖ instance)`. This is the
/// semantic definition of the `ByteDecode` deriveds of the reconstruction
/// relations.
pub fn byte_decode_weight<F: Field>(byte_point: &[F], place_point: &[F]) -> F {
    IdentityPolynomial::new(byte_point.len()).evaluate(byte_point) * place_value_weight(place_point)
}

/// The weight MLE of a selector block at the bound hot-value coordinates:
/// `Σ_{value < values} eq(lane_point, block_start + value) · eq(value_point,
/// value)`. The semantic definition of the `RegisterSelectorWeight` /
/// `LookupSelectorWeight` deriveds (a one-hot selector's lane-eq weights, one
/// lane per register / table index); the `LaneWeight(lane)` deriveds of the
/// direct 0/1 flag lanes are plain `eq_index_msb(lane_point, lane)`.
pub fn selector_block_weight<F: Field>(
    lane_point: &[F],
    block_start: usize,
    value_point: &[F],
    values: usize,
) -> F {
    (0..values)
        .map(|value| {
            eq_index_msb::<F>(lane_point, (block_start + value) as u128)
                * eq_index_msb::<F>(value_point, value as u128)
        })
        .sum()
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{boolean_point_msb, EqPolynomial};

    #[test]
    fn chunking_requires_divisor_widths() {
        assert_eq!(
            UnsignedIncChunking::new(0),
            Err(LatticeGeometryError::ZeroChunkWidth)
        );
        assert_eq!(
            UnsignedIncChunking::new(7),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
        );

        let chunking = UnsignedIncChunking::new(8).unwrap();
        assert_eq!(chunking.chunk_width(), 8);
        assert_eq!(chunking.chunk_count(), 8);
    }

    #[test]
    fn place_values_reconstruct_little_endian_chunks() {
        let chunking = UnsignedIncChunking::new(16).unwrap();
        assert_eq!(chunking.chunk_count(), 4);

        let value: u64 = 0x0123_4567_89ab_cdef;
        let reconstructed = (0..chunking.chunk_count()).fold(Fr::from_u64(0), |acc, index| {
            let chunk = (value >> (16 * index)) & 0xffff;
            acc + chunking.place_value::<Fr>(index) * Fr::from_u64(chunk)
        });
        assert_eq!(reconstructed, Fr::from_u64(value));
    }

    #[test]
    fn byte_num_vars_round_byte_counts_to_powers_of_two() {
        assert_eq!(byte_num_vars(8, 5), Ok(8 + 3 + 5));
        assert_eq!(byte_num_vars(1, 5), Ok(8 + 5));
        // 32-byte field elements: place dimension is already a power of two.
        assert_eq!(byte_num_vars(32, 4), Ok(8 + 5 + 4));
        // Non-power-of-two byte counts round up (dummy places are zero).
        assert_eq!(byte_num_vars(6, 4), Ok(8 + 3 + 4));
        assert_eq!(
            byte_num_vars(0, 4),
            Err(LatticeGeometryError::ZeroByteCount)
        );
        assert_eq!(word_byte_num_vars(5), byte_num_vars(8, 5).unwrap());
    }

    fn random_point(seed: u64, len: usize) -> Vec<Fr> {
        (0..len)
            .map(|index| Fr::from_u64(seed + 3 * index as u64 + 1))
            .collect()
    }

    /// At Boolean points the decode weight is exactly `byte · 256^place`, so
    /// the weighted sum over a word's hot entries reproduces the word value.
    #[test]
    fn byte_decode_weight_decodes_boolean_indices() {
        let word: u64 = 0x0123_4567_89ab_cdef;
        let decoded = (0..WORD_BYTES).fold(Fr::from_u64(0), |acc, place| {
            let byte = ((word >> (8 * place)) & 0xff) as usize;
            acc + byte_decode_weight(
                &boolean_point_msb::<Fr>(BYTE_BITS, byte),
                &boolean_point_msb::<Fr>(WORD_BYTES.log_2(), place),
            )
        });
        assert_eq!(decoded, Fr::from_u64(word));
    }

    /// The decode weight is the MLE of `(byte, place) ↦ byte · 256^place`:
    /// pin the closed form against the brute-force eq-weighted sum at a
    /// non-Boolean point.
    #[test]
    fn byte_decode_weight_matches_brute_force_mle() {
        let byte_point = random_point(7, BYTE_BITS);
        let place_point = random_point(11, WORD_BYTES.log_2());

        let mut expected = Fr::from_u64(0);
        for byte in 0..(1u128 << BYTE_BITS) {
            for place in 0..WORD_BYTES {
                let value = Fr::from_u64(byte as u64) * Fr::pow2(8 * place);
                expected += eq_index_msb::<Fr>(&byte_point, byte)
                    * eq_index_msb::<Fr>(&place_point, place as u128)
                    * value;
            }
        }
        assert_eq!(byte_decode_weight(&byte_point, &place_point), expected);
    }

    #[test]
    fn selector_block_weight_matches_brute_force_mle() {
        let lane_point = random_point(13, 7);
        let value_point = random_point(17, 5);
        let block_start = 32;
        let values = 32;

        let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);
        let mut expected = Fr::from_u64(0);
        for value in 0..values {
            expected +=
                lane_weights[block_start + value] * eq_index_msb::<Fr>(&value_point, value as u128);
        }
        assert_eq!(
            selector_block_weight(&lane_point, block_start, &value_point, values),
            expected
        );
    }
}
