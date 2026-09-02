//! Pure lattice geometry for balanced increment chunking.

use jolt_field::{JoltField, Ring};
use jolt_poly::{IdentityPolynomial, MultilinearEvaluation};
use thiserror::Error;

use super::super::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;

/// Bit width the balanced fused-increment digits cover, and hence the place
/// value of [`BalancedIncCarry`](crate::protocols::jolt::JoltCommittedPolynomial::BalancedIncCarry).
pub const FUSED_INC_BITS: usize = 64;

/// Bytecode read-raf val stages in lattice mode: the base stages plus one
/// carrying the `OpFlags(Store)` opening that `IncVirtualization` consumes as
/// its destination selector. Always six; in an akita build the active
/// `NUM_BYTECODE_VAL_STAGES` already folds the store stage, so it equals this.
#[cfg(not(feature = "akita"))]
pub const LATTICE_BYTECODE_VAL_STAGES: usize = NUM_BYTECODE_VAL_STAGES + 1;
#[cfg(feature = "akita")]
pub const LATTICE_BYTECODE_VAL_STAGES: usize = NUM_BYTECODE_VAL_STAGES;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LatticeGeometryError {
    #[error(transparent)]
    PrefixLayout(#[from] jolt_openings::OpeningsError),
    #[error("increment digit width must be nonzero")]
    ZeroChunkWidth,
    #[error("increment digit width {chunk_width} must divide {FUSED_INC_BITS}")]
    ChunkWidthMisaligned { chunk_width: usize },
    #[error("increment digit width {chunk_width} does not fit the address domain")]
    ChunkWidthTooLarge { chunk_width: usize },
    #[error("OneHotTrace supports only 4-bit or 8-bit one-hot chunks, got {chunk_width}")]
    UnsupportedOneHotTraceChunkWidth { chunk_width: usize },
    #[error(
        "OneHotTrace at K=2^{chunk_width} requires {expected} instruction columns, got {actual}"
    )]
    UnexpectedOneHotTraceInstructionColumns {
        chunk_width: usize,
        actual: usize,
        expected: usize,
    },
    #[error(
        "OneHotTrace has {actual} columns, exceeding the K=2^{chunk_width} packed capacity {capacity}"
    )]
    TooManyOneHotTraceColumns {
        chunk_width: usize,
        actual: usize,
        capacity: usize,
    },
}

/// The balanced radix-`2^chunk_width` decomposition of the fused increment:
/// `FUSED_INC_BITS / chunk_width` digit columns plus the signed
/// [`BalancedIncCarry`](crate::protocols::jolt::JoltCommittedPolynomial::BalancedIncCarry),
/// every digit centered in `[-2^(chunk_width-1), 2^(chunk_width-1))` so that
/// `Σ_j 2^(chunk_width·j)·digit_j + 2^FUSED_INC_BITS·carry` is the signed
/// increment itself — no unsigned shift. See [`balanced_inc_value`] for the
/// value map, which sends digit zero to zero and therefore lets a
/// zero increment sit entirely on the row the commitment omits.
///
/// The chunk width is fixed to the shared one-hot chunk size (`log_k_chunk`)
/// so the digit polynomials sit in the `Ra` families' variable-count class
/// and can share their final packed point (see `specs/lattice-claims.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BalancedIncChunking {
    chunk_width: usize,
}

impl BalancedIncChunking {
    pub const fn new(chunk_width: usize) -> Result<Self, LatticeGeometryError> {
        if chunk_width == 0 {
            return Err(LatticeGeometryError::ZeroChunkWidth);
        }
        if chunk_width >= usize::BITS as usize {
            return Err(LatticeGeometryError::ChunkWidthTooLarge { chunk_width });
        }
        if !FUSED_INC_BITS.is_multiple_of(chunk_width) {
            return Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width });
        }
        Ok(Self { chunk_width })
    }

    pub const fn chunk_width(self) -> usize {
        self.chunk_width
    }

    pub const fn chunk_count(self) -> usize {
        FUSED_INC_BITS / self.chunk_width
    }

    /// The place value `2^(chunk_width * index)` weighting chunk `index` in
    /// the little-endian reconstruction of the low 64 bits.
    pub fn place_value<F: Ring>(self, index: usize) -> F {
        F::pow2(self.chunk_width * index)
    }
}

/// MLE of the centered row value used by balanced increment digits.
pub fn balanced_inc_value<F: JoltField>(address_point: &[F]) -> F {
    let unsigned = IdentityPolynomial::new(address_point.len()).evaluate(address_point);
    let msb = address_point.first().copied().unwrap_or_else(F::zero);
    unsigned - F::pow2(address_point.len()) * msb
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
            Err(LatticeGeometryError::ZeroChunkWidth)
        );
        assert_eq!(
            BalancedIncChunking::new(7),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
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
}
