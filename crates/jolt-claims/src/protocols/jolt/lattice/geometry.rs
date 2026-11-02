//! Pure lattice geometry.

use thiserror::Error;

use super::super::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;

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
    #[error("OneHotTrace has {actual} columns, above the {capacity}-column row mask limit")]
    TooManyOneHotTraceColumns { actual: usize, capacity: usize },
}
