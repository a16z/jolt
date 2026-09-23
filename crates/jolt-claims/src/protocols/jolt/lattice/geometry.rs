//! Pure lattice geometry for balanced increment chunking.

use thiserror::Error;

use super::super::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;

use crate::lattice::BalancedChunkingError;
/// Bit width the balanced fused-increment digits cover, and hence the place
/// value of [`BalancedIncCarry`](crate::protocols::jolt::JoltCommittedPolynomial::BalancedIncCarry):
/// the shared balanced-digit window.
pub use crate::lattice::BALANCED_INC_BITS as FUSED_INC_BITS;
/// The shared balanced-digit algebra ([`crate::lattice`]), re-exported at
/// its historical home (here the digits decompose the fused increment).
pub use crate::lattice::{balanced_inc_value, BalancedIncChunking};

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

impl From<BalancedChunkingError> for LatticeGeometryError {
    fn from(error: BalancedChunkingError) -> Self {
        match error {
            BalancedChunkingError::ZeroChunkWidth => Self::ZeroChunkWidth,
            BalancedChunkingError::ChunkWidthMisaligned { chunk_width } => {
                Self::ChunkWidthMisaligned { chunk_width }
            }
            BalancedChunkingError::ChunkWidthTooLarge { chunk_width } => {
                Self::ChunkWidthTooLarge { chunk_width }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared chunking's errors surface through the jolt-side error type
    /// unchanged (the `From` mapping is variant-for-variant).
    #[test]
    fn chunking_errors_map_into_the_lattice_error() {
        assert_eq!(
            BalancedIncChunking::new(7).map_err(LatticeGeometryError::from),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
        );
    }
}
