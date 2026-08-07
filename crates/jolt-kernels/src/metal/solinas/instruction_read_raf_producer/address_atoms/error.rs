use thiserror::Error;

use super::super::ProducerLayoutError;
use super::accounting::AddressAtomPlaneRole;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AddressAtomError {
    #[error(transparent)]
    Producer(#[from] ProducerLayoutError),
    #[error("address atom source plane {name} has {got} elements, expected {expected}")]
    SourcePlaneElements {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("address atom sorted cycle list has {got} elements, expected {expected}")]
    SortedCycleLength { expected: usize, got: usize },
    #[error("address atom cycle {cycle} at position {position} is outside 0..{rows}")]
    CycleOutOfRange {
        position: usize,
        cycle: usize,
        rows: usize,
    },
    #[error("address atom cycle {cycle} appears more than once")]
    DuplicateCycle { cycle: usize },
    #[error("address atom key decreases at sorted position {position}")]
    NonMonotoneKey { position: usize },
    #[error("address atom topology has {atoms} atoms for {rows} rows")]
    InvalidAtomCount { rows: usize, atoms: usize },
    #[error("address atom topology {name} has {got} elements, expected {expected}")]
    TopologyLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid address atom topology: {0}")]
    InvalidTopology(&'static str),
    #[error("address atom topology belongs to a different producer shard")]
    ShardMismatch,
    #[error("address atom cycle weights have {got} elements, expected {expected}")]
    CycleWeightLength { expected: usize, got: usize },
    #[error(
        "address atom split equality shape {e_out}x{e_in} does not cover {total_rows} global rows"
    )]
    SplitEqualityShape {
        total_rows: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error("address atom {0} size overflowed")]
    SizeOverflow(&'static str),
    #[error("address atom {role:?} requires {bytes} bytes, exceeding the per-buffer limit")]
    BufferTooLarge {
        role: AddressAtomPlaneRole,
        bytes: usize,
    },
    #[error("address atom {name} identity must be nonzero")]
    MissingIdentity { name: &'static str },
    #[error("address atom allocation identity {identity:#x} is aliased")]
    AliasedAllocation { identity: usize },
    #[error("address atom producer status for shard {shard} is {status:#010x}")]
    NonzeroStatus { shard: usize, status: u32 },
    #[error("address atom topology completion {got} precedes source completion {minimum}")]
    IncompleteTopology { minimum: u64, got: u64 },
    #[error(
        "address atom {role:?} shape is {got_elements} elements/{got_bytes} bytes, expected {expected_elements}/{expected_bytes}"
    )]
    PlaneShape {
        role: AddressAtomPlaneRole,
        expected_elements: usize,
        got_elements: usize,
        expected_bytes: u64,
        got_bytes: u64,
    },
    #[error("address atom {role:?} belongs to a different Metal device")]
    DeviceMismatch { role: AddressAtomPlaneRole },
    #[error("address atom {role:?} generation is {got}, expected source generation {expected}")]
    GenerationMismatch {
        role: AddressAtomPlaneRole,
        expected: u64,
        got: u64,
    },
    #[error("address atom {role:?} completion is {got}, expected topology completion {expected}")]
    PlaneCompletionMismatch {
        role: AddressAtomPlaneRole,
        expected: u64,
        got: u64,
    },
    #[error("address atom batch has {got} shard receipts, expected {expected}")]
    ReceiptShardCount { expected: usize, got: usize },
    #[error("address atom batch receipt {index} is not the expected producer shard")]
    ReceiptShard { index: usize },
    #[error("address atom batch receipt {index} has different producer provenance")]
    BatchProvenanceMismatch { index: usize },
    #[error(
        "address atom mass plane is {got_elements} elements/{got_bytes} bytes, expected {expected_elements}/{expected_bytes}"
    )]
    MassPlaneShape {
        expected_elements: usize,
        got_elements: usize,
        expected_bytes: u64,
        got_bytes: u64,
    },
    #[error("address atom masses belong to a different Metal device")]
    MassDeviceMismatch,
    #[error("address atom mass generation is {got}, expected {expected}")]
    MassGenerationMismatch { expected: u64, got: u64 },
    #[error("address atom mass completion {got} precedes topology completion {minimum}")]
    IncompleteMasses { minimum: u64, got: u64 },
}

pub type AddressAtomResult<T> = std::result::Result<T, AddressAtomError>;

pub(super) fn validate_atom_count(rows: usize, atoms: usize) -> AddressAtomResult<()> {
    if atoms == 0 || atoms > rows {
        return Err(AddressAtomError::InvalidAtomCount { rows, atoms });
    }
    Ok(())
}
