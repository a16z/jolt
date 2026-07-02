use jolt_field::RingCore;
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

/// Byte limbs of one 64-bit word and their index bits.
pub const WORD_BYTE_LIMBS: usize = 8;
pub const WORD_BYTE_LIMB_BITS: usize = 3;

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
    chunk_count: usize,
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
        Ok(Self {
            chunk_width,
            chunk_count: UNSIGNED_INC_BITS / chunk_width,
        })
    }

    pub const fn chunk_width(self) -> usize {
        self.chunk_width
    }

    pub const fn chunk_count(self) -> usize {
        self.chunk_count
    }

    pub const fn alphabet_size(self) -> usize {
        1 << self.chunk_width
    }

    /// The place value `2^(chunk_width * index)` weighting chunk `index` in
    /// the little-endian reconstruction of the low 64 bits.
    pub fn place_value<F: RingCore>(self, index: usize) -> F {
        F::pow2(self.chunk_width * index)
    }
}

/// `ceil(log2(value))` for column-arity math; `0` for `value <= 1`.
pub const fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

/// Arity of a one-hot column over `symbol_bits` symbols, `limbs` limbs, and a
/// `2^log_rows` row domain, under the `(symbol ‖ limb ‖ row)` cell order.
/// Limb counts round up to a power of two; dummy limb cells are zero by
/// convention.
pub const fn one_hot_column_vars(
    symbol_bits: usize,
    limbs: usize,
    log_rows: usize,
) -> Result<usize, LatticeGeometryError> {
    if limbs == 0 {
        return Err(LatticeGeometryError::ZeroLimbCount);
    }
    Ok(symbol_bits + ceil_log2(limbs.next_power_of_two()) + log_rows)
}

/// Arity of a byte one-hot column carrying `limbs` bytes per row.
pub const fn byte_column_vars(
    limbs: usize,
    log_rows: usize,
) -> Result<usize, LatticeGeometryError> {
    one_hot_column_vars(BYTE_SYMBOL_BITS, limbs, log_rows)
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
        assert_eq!(chunking.alphabet_size(), 256);
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
    }
}
