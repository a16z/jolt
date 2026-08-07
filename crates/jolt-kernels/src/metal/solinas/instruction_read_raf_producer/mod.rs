//! Sharded producer layout for instruction read+RAF kernels.
//!
//! Cycle-order lookup limbs and claims remain resident because cycle-round
//! consumers need the original ordering. The scatter creates only per-shard,
//! segment-grouped lookup limbs and a cycle-to-grouped-local inverse. Segment
//! order is fixed; order within one chunk and segment is intentionally not.
//!
//! Chunk counts are expected to be accumulated while the producer creates the
//! cycle claim plane. Prefixing those counts supplies the scatter metadata, so
//! the device scatter needs one payload pass and no count pass.

mod abi;
mod binding;
mod oracle;
mod runtime;
#[cfg(test)]
mod tests;

pub use abi::{
    BufferShape, ChunkSegmentCounts, PlaneRole, ProducerGeometry, ProducerShardPlan,
    ScatterDispatchPlan, ScatterLayout, ScatterParams, ScatterTraffic, SCATTER_BUFFER_ROLES,
};
pub use oracle::{decode_claim, HostScatter, ProducerSelector};
pub use runtime::{
    CompletedInstructionReadRafProducer, CompletedProducerShard, ProducerCompletionReceipt,
    ProducerExecutionTiming, ProducerPlaneInitialization, ProducerPlaneReceipt,
    ProducerPreparationTiming, ProducerRuntimeError, ProducerShardInput, ProducerSourceReceipt,
    ResidentInstructionReadRafProducer, ResidentProducerPlane, ResidentProducerSourceShard,
};

use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use thiserror::Error;

pub const LOOKUP_TABLES: usize = LookupTableKind::<RISCV_XLEN>::COUNT;
pub const TABLE_SELECTOR_VALUES: usize = LOOKUP_TABLES + 1;
pub const RAF_SELECTOR_VALUES: usize = 2;
pub const GROUPED_SEGMENTS: usize = TABLE_SELECTOR_VALUES * RAF_SELECTOR_VALUES;
pub const GROUPED_SEGMENT_OFFSETS: usize = GROUPED_SEGMENTS + 1;
pub const PRODUCER_CHUNK_ROWS: usize = 4096;
pub const PRODUCER_THREADS_PER_GROUP: usize = 1024;
pub const MAX_TOTAL_ROWS: usize = 1 << 28;
pub const MAX_SHARD_ROWS: usize = 1 << 26;
pub const MAX_BUFFER_BYTES: usize = 2 * 1024 * 1024 * 1024;

pub const PRODUCER_INPUT_BYTES_PER_ROW: usize = 8 + 8 + 1;
pub const PRODUCER_OUTPUT_BYTES_PER_ROW: usize = 8 + 8 + 4;
/// Two lookup-limb loads, one claim load, two lookup-limb writes, and one inverse write.
pub const PRODUCER_PAYLOAD_BYTES_PER_ROW: usize =
    PRODUCER_INPUT_BYTES_PER_ROW + PRODUCER_OUTPUT_BYTES_PER_ROW;
pub const PRODUCER_THREADGROUP_BYTES: usize = GROUPED_SEGMENTS * size_of::<u32>();

pub const SCATTER_STATUS_INVALID_GEOMETRY: u32 = 1 << 0;
pub const SCATTER_STATUS_INVALID_SELECTOR: u32 = 1 << 1;
pub const SCATTER_STATUS_INVALID_LAYOUT: u32 = 1 << 2;
pub const SCATTER_STATUS_OUT_OF_BOUNDS: u32 = 1 << 3;
pub const SCATTER_STATUS_COUNT_MISMATCH: u32 = 1 << 4;

/// Isolated source for the sharded producer scatter pass.
pub const METAL_SOURCE: &str = include_str!("shader.metal");

const _: () = assert!(LOOKUP_TABLES == 40);
const _: () = assert!(GROUPED_SEGMENTS == 82);
const _: () = assert!(GROUPED_SEGMENT_OFFSETS == 83);
const _: () = assert!(PRODUCER_INPUT_BYTES_PER_ROW == 17);
const _: () = assert!(PRODUCER_OUTPUT_BYTES_PER_ROW == 20);
const _: () = assert!(PRODUCER_PAYLOAD_BYTES_PER_ROW == 37);
const _: () = assert!(PRODUCER_THREADGROUP_BYTES == 328);

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProducerLayoutError {
    #[error("row count must be a nonzero power of two no larger than 2^28, got {0}")]
    InvalidRowCount(usize),
    #[error("shard index {index} is outside 0..{shards}")]
    InvalidShardIndex { index: usize, shards: usize },
    #[error("lookup table index {0} is outside the producer table range")]
    InvalidTableIndex(usize),
    #[error("claim byte {0:#04x} has an invalid table selector")]
    InvalidClaim(u8),
    #[error("{0} size overflowed")]
    SizeOverflow(&'static str),
    #[error("{plane:?} requires {bytes} bytes, exceeding the per-buffer limit")]
    BufferTooLarge { plane: PlaneRole, bytes: usize },
    #[error("{plane:?} element count mismatch: expected {expected}, got {got}")]
    PlaneElements {
        plane: PlaneRole,
        expected: usize,
        got: usize,
    },
    #[error("scatter layout belongs to a different shard")]
    ShardMismatch,
    #[error("invalid scatter layout: {0}")]
    InvalidLayout(&'static str),
    #[error("host scatter invariant failed: {0}")]
    ScatterInvariant(&'static str),
}

pub type Result<T> = std::result::Result<T, ProducerLayoutError>;

fn validate_rows(rows: usize) -> Result<()> {
    if rows == 0 || !rows.is_power_of_two() || rows > MAX_TOTAL_ROWS {
        return Err(ProducerLayoutError::InvalidRowCount(rows));
    }
    Ok(())
}

const fn size_of<T>() -> usize {
    std::mem::size_of::<T>()
}
