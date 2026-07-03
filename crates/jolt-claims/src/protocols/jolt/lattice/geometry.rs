use jolt_field::RingCore;
use jolt_poly::math::Math;
use thiserror::Error;

use super::super::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;

/// Bit width of the unsigned fused increment (`FusedInc + 2^64` fits in 65
/// bits: the chunk columns carry the low 64, [`UnsignedIncMsb`]
/// (`crate::protocols::jolt::JoltCommittedPolynomial::UnsignedIncMsb`) the top
/// bit).
pub const UNSIGNED_INC_BITS: usize = 64;

/// Bytecode read-raf val stages in lattice mode: the base stages plus one
/// carrying the `OpFlags(Store)` opening that `IncVirtualization` consumes as
/// its destination selector.
pub const LATTICE_BYTECODE_VAL_STAGES: usize = NUM_BYTECODE_VAL_STAGES + 1;

/// Symbol bits of a byte one-hot column.
pub const BYTE_SYMBOL_BITS: usize = 8;

/// Byte limbs of one 64-bit word.
pub const WORD_BYTE_LIMBS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LatticeGeometryError {
    #[error(transparent)]
    PackingRegistration(#[from] jolt_openings::OpeningsError),
    #[error("unsigned inc chunk width must be nonzero")]
    ZeroChunkWidth,
    #[error("unsigned inc chunk width {chunk_width} must divide {UNSIGNED_INC_BITS}")]
    ChunkWidthMisaligned { chunk_width: usize },
    #[error("unsigned inc chunk width {chunk_width} does not fit the symbol domain")]
    ChunkWidthTooLarge { chunk_width: usize },
    #[error("byte column limb count must be nonzero")]
    ZeroLimbCount,
}

/// The base-`2^chunk_width` one-hot decomposition of the low
/// [`UNSIGNED_INC_BITS`] bits of the fused unsigned increment.
///
/// The chunk width is fixed to the shared one-hot chunk size (`log_k_chunk`)
/// so the chunk columns sit in the `Ra` columns' arity class and can share
/// their final packed point (see `specs/lattice-claims.md`, shared-final-point
/// invariant).
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

/// MLE of the symbol-value identity over a one-hot column's address bits,
/// msb-first (the `(symbol ...)` cell order): `Σ_i 2^(w-1-i) · r[i]`.
/// Decodes a one-hot chunk opening to its symbol value; the single owner of
/// the bit-order convention for both the prover instance and the verifier's
/// `IdentityAtAddress` public.
pub fn identity_mle<F: RingCore>(address_point: &[F]) -> F {
    address_point
        .iter()
        .fold(F::zero(), |acc, coordinate| acc + acc + *coordinate)
}

/// Arity of a one-hot column over `symbol_bits` symbols, `limbs` limbs, and a
/// `2^log_rows` row domain, under the `(symbol ‖ limb ‖ row)` cell order.
/// Limb counts round up to a power of two; dummy limb cells are zero by
/// convention.
pub fn one_hot_column_vars(
    symbol_bits: usize,
    limbs: usize,
    log_rows: usize,
) -> Result<usize, LatticeGeometryError> {
    if limbs == 0 {
        return Err(LatticeGeometryError::ZeroLimbCount);
    }
    Ok(symbol_bits + limbs.log_2() + log_rows)
}

/// Arity of a byte one-hot column carrying `limbs` bytes per row.
pub fn byte_column_vars(limbs: usize, log_rows: usize) -> Result<usize, LatticeGeometryError> {
    one_hot_column_vars(BYTE_SYMBOL_BITS, limbs, log_rows)
}

/// Arity of the byte one-hot column of 64-bit words — the advice and
/// program-image encoding, and the cell-variable count of the advice
/// byte-validity sumcheck (relation rounds and packed arity are the same
/// number by construction).
pub const fn word_byte_column_vars(log_words: usize) -> usize {
    BYTE_SYMBOL_BITS + WORD_BYTE_LIMBS.ilog2() as usize + log_words
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

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
    fn column_vars_round_limbs_to_powers_of_two() {
        assert_eq!(byte_column_vars(8, 5), Ok(8 + 3 + 5));
        assert_eq!(byte_column_vars(1, 5), Ok(8 + 5));
        // 32-byte field elements: limb dimension is already a power of two.
        assert_eq!(byte_column_vars(32, 4), Ok(8 + 5 + 4));
        // Non-power-of-two limb counts round up (dummy limbs are zero).
        assert_eq!(byte_column_vars(6, 4), Ok(8 + 3 + 4));
        assert_eq!(
            byte_column_vars(0, 4),
            Err(LatticeGeometryError::ZeroLimbCount)
        );
        assert_eq!(word_byte_column_vars(5), byte_column_vars(8, 5).unwrap());
    }

    #[test]
    fn identity_mle_decodes_boolean_symbols_msb_first() {
        use jolt_field::{Fr, FromPrimitiveInt};
        for symbol in 0..8u64 {
            let point: Vec<Fr> = (0..3)
                .rev()
                .map(|bit| Fr::from_u64((symbol >> bit) & 1))
                .collect();
            assert_eq!(identity_mle(&point), Fr::from_u64(symbol));
        }
    }
}
