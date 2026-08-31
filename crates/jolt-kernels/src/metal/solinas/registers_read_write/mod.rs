#[cfg(feature = "test-utils")]
pub(crate) mod runtime;
mod sequence;
mod stage1;

pub(super) const SOURCE: &str = concat!(
    include_str!("shader.metal"),
    "\n",
    include_str!("sequence.metal"),
    "\n",
    include_str!("fused_sequence.metal")
);

pub(crate) use sequence::{
    PendingRegistersReadWriteStage1Pipelines, RegistersReadWriteCycleFinish,
    RegistersReadWriteCycleSequence,
};
pub(crate) use stage1::{
    RegistersReadWriteStage1ChunkWriter, RegistersReadWriteStage1Source,
    RegistersReadWriteStage1Storage,
};

pub const REGISTERS_READ_WRITE_FIRST_MESSAGE_PIPELINE: &str =
    "solinas_registers_read_write_first_message";
pub const REGISTERS_READ_WRITE_FIRST_MESSAGE_INTERSECTION_PIPELINE: &str =
    "solinas_registers_read_write_first_message_intersection";
pub const REGISTERS_READ_WRITE_BOOTSTRAP_PIPELINE: &str =
    "solinas_registers_read_write_bootstrap_fused";
pub const REGISTERS_READ_WRITE_STATELESS_BOOTSTRAP_PIPELINE: &str =
    "solinas_registers_read_write_stateless_bootstrap_message";
pub const REGISTERS_READ_WRITE_STATELESS_REPLAY_BOOTSTRAP_PIPELINE: &str =
    "solinas_registers_read_write_stateless_replay_bootstrap_message";
pub const REGISTERS_READ_WRITE_REPLAY_BOOTSTRAP_PIPELINE: &str =
    "solinas_registers_read_write_replay_bootstrap_fused";
pub const REGISTERS_READ_WRITE_REPLAY_THREE_BOOTSTRAP_PIPELINE: &str =
    "solinas_registers_read_write_replay_three_bootstrap_fused";
pub const REGISTERS_READ_WRITE_REPLAY_THREE_MATERIALIZE_PIPELINE: &str =
    "solinas_registers_read_write_replay_three_materialize";
pub const REGISTERS_READ_WRITE_INDEXED_STATE_MESSAGE_PIPELINE: &str =
    "solinas_registers_read_write_indexed_state_message";
pub const REGISTERS_READ_WRITE_INDEXED_STATE_GEOMETRY_PIPELINE: &str =
    "solinas_registers_read_write_indexed_state_geometry";
pub const REGISTERS_READ_WRITE_INDEXED_BIND_MESSAGE_PIPELINE: &str =
    "solinas_registers_read_write_indexed_bind_message_fused";
pub const REGISTERS_READ_WRITE_INDEXED_COOPERATIVE_PIPELINE: &str =
    "solinas_registers_read_write_indexed_bind_message_cooperative";
pub const REGISTERS_READ_WRITE_WIDE_INDEXED_COOPERATIVE_PIPELINE: &str =
    "solinas_registers_read_write_wide_indexed_bind_message_cooperative";
pub const REGISTERS_READ_WRITE_TRANSITION_BIND_MESSAGE_PIPELINE: &str =
    "solinas_registers_read_write_transition_bind_message_fused";
pub const REGISTERS_READ_WRITE_TRANSITION_COOPERATIVE_PIPELINE: &str =
    "solinas_registers_read_write_transition_bind_message_cooperative";
pub const REGISTERS_READ_WRITE_WIDE_TRANSITION_COOPERATIVE_PIPELINE: &str =
    "solinas_registers_read_write_wide_transition_bind_message_cooperative";
pub const REGISTERS_READ_WRITE_DIRECT_BIND_MESSAGE_PIPELINE: &str =
    "solinas_registers_read_write_direct_bind_message_fused";
pub const REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE: &str =
    "solinas_registers_read_write_direct_bind_message_cooperative";
pub const REGISTERS_READ_WRITE_DIRECT_GEOMETRY_PIPELINE: &str =
    "solinas_registers_read_write_direct_geometry";
pub const REGISTERS_READ_WRITE_OPERAND_CLAIMS_PIPELINE: &str =
    "solinas_registers_read_write_operand_claims";
pub const REGISTERS_READ_WRITE_COMPACT_RS1_CLAIM_PIPELINE: &str =
    "solinas_registers_read_write_compact_rs1_claim";
pub const REGISTERS_READ_WRITE_DERIVE_RD_PRE_CHUNKS_PIPELINE: &str =
    "solinas_registers_read_write_derive_rd_pre_chunks";
pub const REGISTERS_READ_WRITE_FIXUP_RD_PRE_PIPELINE: &str =
    "solinas_registers_read_write_fixup_rd_pre";
pub const REGISTERS_READ_WRITE_SOURCE_PRIMER_PIPELINE: &str =
    "solinas_registers_read_write_source_primer";
pub const REGISTERS_READ_WRITE_REDUCTION_PIPELINE: &str = "solinas_ram_read_write_reduce";
pub const REGISTERS_READ_WRITE_THREADS: usize = 256;
#[cfg(feature = "test-utils")]
pub const REGISTERS_READ_WRITE_PAIRS_PER_GROUP: usize = REGISTERS_READ_WRITE_THREADS;
pub const REGISTERS_READ_WRITE_SIMD_WIDTH: usize = 32;
pub const REGISTERS_READ_WRITE_THREADGROUP_BYTES_MAX: u64 = 16 * 1024;
pub const REGISTERS_READ_WRITE_RD_PRE_CHUNK_ROWS: usize = 1 << 12;
pub const REGISTERS_READ_WRITE_RD_PRE_REGISTERS: usize = 64;
pub const REGISTERS_READ_WRITE_RD_PRE_DERIVE_THREADS: usize = 32;
pub const REGISTERS_READ_WRITE_RD_PRE_FIXUP_THREADS: usize = 256;
pub const REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS: usize = 32;
pub const REGISTERS_READ_WRITE_RD_PRE_FIXUP_REGISTERS_PER_GROUP: usize =
    REGISTERS_READ_WRITE_RD_PRE_FIXUP_THREADS / REGISTERS_READ_WRITE_RD_PRE_FIXUP_PARTITIONS;
pub const REGISTERS_READ_WRITE_RD_PRE_FIXUP_GROUPS: usize =
    REGISTERS_READ_WRITE_RD_PRE_REGISTERS / REGISTERS_READ_WRITE_RD_PRE_FIXUP_REGISTERS_PER_GROUP;
