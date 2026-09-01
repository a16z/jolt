use std::{
    ffi::c_void,
    mem::size_of,
    slice,
    sync::Arc,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::{Field as _, One as _, Zero as _};
use jolt_poly::EqPolynomial;
#[cfg(any(test, feature = "test-utils"))]
use metal::{foreign_types::ForeignType, CommandBuffer};
use metal::{
    objc::rc::autoreleasepool, Buffer, CommandQueue, ComputePipelineState, FunctionConstantValues,
    MTLDataType, MTLResourceOptions, MTLSize, NSRange,
};

use super::{
    REGISTERS_READ_WRITE_BOOTSTRAP_PIPELINE, REGISTERS_READ_WRITE_COMPACT_RS1_CLAIM_PIPELINE,
    REGISTERS_READ_WRITE_DIRECT_BIND_MESSAGE_PIPELINE,
    REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
    REGISTERS_READ_WRITE_DIRECT_GEOMETRY_PIPELINE,
    REGISTERS_READ_WRITE_FIRST_MESSAGE_INTERSECTION_PIPELINE,
    REGISTERS_READ_WRITE_FIRST_MESSAGE_PIPELINE,
    REGISTERS_READ_WRITE_INDEXED_BIND_MESSAGE_PIPELINE,
    REGISTERS_READ_WRITE_INDEXED_COOPERATIVE_PIPELINE,
    REGISTERS_READ_WRITE_INDEXED_STATE_GEOMETRY_PIPELINE,
    REGISTERS_READ_WRITE_INDEXED_STATE_MESSAGE_PIPELINE,
    REGISTERS_READ_WRITE_OPERAND_CLAIMS_PIPELINE, REGISTERS_READ_WRITE_REDUCTION_PIPELINE,
    REGISTERS_READ_WRITE_REPLAY_BOOTSTRAP_PIPELINE,
    REGISTERS_READ_WRITE_REPLAY_THREE_BOOTSTRAP_PIPELINE,
    REGISTERS_READ_WRITE_REPLAY_THREE_MATERIALIZE_PIPELINE, REGISTERS_READ_WRITE_SIMD_WIDTH,
    REGISTERS_READ_WRITE_SOURCE_PRIMER_PIPELINE, REGISTERS_READ_WRITE_STATELESS_BOOTSTRAP_PIPELINE,
    REGISTERS_READ_WRITE_STATELESS_REPLAY_BOOTSTRAP_PIPELINE,
    REGISTERS_READ_WRITE_THREADGROUP_BYTES_MAX, REGISTERS_READ_WRITE_THREADS,
    REGISTERS_READ_WRITE_TRANSITION_BIND_MESSAGE_PIPELINE,
    REGISTERS_READ_WRITE_TRANSITION_COOPERATIVE_PIPELINE,
    REGISTERS_READ_WRITE_WIDE_INDEXED_COOPERATIVE_PIPELINE,
    REGISTERS_READ_WRITE_WIDE_TRANSITION_COOPERATIVE_PIPELINE,
};
use crate::metal::solinas::registers_claim_reduction::RegistersClaimResidentRdPlane;
use crate::metal::solinas::runtime::PooledPrivateBuffer;
#[cfg(feature = "test-utils")]
use crate::metal::solinas::PipelineLimits;
use crate::metal::solinas::{
    completed_command_gpu_time, encode_column_reductions, set_inline_bytes, Fp128, MetalError,
    RegistersReadWriteStage1Source, SolinasMetal,
};
use crate::optimized::registers_read_write::{
    AlignedCompactRegisterIndices, AlignedPackedRegisterRows, BoundRegisterCycleRoot,
    PackedRegisterCycleRow, PACKED_REGISTER_ROWS_ALIGNMENT,
};

const MAX_REGISTER_BLOCK_CAPACITY: usize = 64;
const CROSS_REPRESENTATION_REUSE_LOG_T_MIN: usize = 25;
const COMPACT_RS1_SOURCE_LOG_T_MIN: usize = 28;
const ASYNC_SOURCE_RETIREMENT_LOG_T_MIN: usize = 28;
const DIRECT_TRANSITION_BOUND_ROUNDS: usize = 4;
const WIDE_INDEXED_BOUND_ROUND: usize = 4;
const SPARSE_COOPERATIVE_WORK_ITEMS_MIN: usize = 65_536;
const SPARSE_COOPERATIVE_WORK_ITEMS_PER_GROUP: usize = REGISTERS_READ_WRITE_THREADS / 4;
const DIRECT_COOPERATIVE_WORK_ITEMS_MIN: usize = 32_768;
const DIRECT_COOPERATIVE_HIGH_ROUTE_WORK_ITEMS_MIN: usize = 256;
const DIRECT_COOPERATIVE_WORK_ITEMS_PER_GROUP: usize = REGISTERS_READ_WRITE_THREADS / 8;
const STATE_TILE_BLOCKS: usize = REGISTERS_READ_WRITE_THREADS;
const PRIVATE_PAYLOAD_POOL_THRESHOLD_BYTES: u64 = 1;
const PRIVATE_PAYLOAD_POOL_CAP_BYTES: u64 = 32 * 1024 * 1024 * 1024;
#[cfg(any(test, feature = "test-utils"))]
const SOURCE_PRIMER_PAGE_BYTES: usize = 16 * 1024;
#[cfg(any(test, feature = "test-utils"))]
const SOURCE_PRIMER_THREADS_PER_THREADGROUP: usize = 256;
#[cfg(any(test, feature = "test-utils"))]
const SOURCE_PRIMER_THREADGROUPS: usize = 256;
#[cfg(any(test, feature = "test-utils"))]
const SOURCE_PRIMER_THREADS: usize =
    SOURCE_PRIMER_THREADS_PER_THREADGROUP * SOURCE_PRIMER_THREADGROUPS;

const fn direct_cooperative_work_items_min(log_t: usize) -> usize {
    if log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN {
        DIRECT_COOPERATIVE_HIGH_ROUTE_WORK_ITEMS_MIN
    } else {
        DIRECT_COOPERATIVE_WORK_ITEMS_MIN
    }
}

const fn uses_operand_carry(log_t: usize, stage1_source: bool) -> bool {
    log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN
        && (stage1_source || log_t < COMPACT_RS1_SOURCE_LOG_T_MIN)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FirstMessageParams {
    row_count: u32,
    pair_count: u32,
    output_stride: u32,
    e_in_length: u32,
    source_stride: u32,
}

const _: [(); 20] = [(); size_of::<FirstMessageParams>()];

#[repr(C)]
#[derive(Clone, Copy)]
struct SequenceParams {
    row_count: u32,
    input_blocks: u32,
    output_blocks: u32,
    input_capacity: u32,
    output_capacity: u32,
    work_items: u32,
    output_stride: u32,
    e_in_length: u32,
    ra_lut_bits: u32,
    wa_lut_bits: u32,
    emit_message: u32,
    reserved: u32,
    source_stride: u32,
}

const _: [(); 52] = [(); size_of::<SequenceParams>()];

#[repr(C)]
#[derive(Clone, Copy)]
struct OperandClaimsParams {
    row_count: u32,
    cycles_per_high_block: u32,
    address_bits: u32,
    output_stride: u32,
    remap_indices: u32,
}

const _: [(); 20] = [(); size_of::<OperandClaimsParams>()];

#[repr(C)]
#[derive(Clone, Copy)]
struct SourcePrimerParams {
    word_counts: [u64; 3],
    page_words: u32,
    total_threads: u32,
}

const _: [(); 32] = [(); size_of::<SourcePrimerParams>()];

struct CommonStateBuffers {
    lengths: PooledPrivateBuffer,
    offsets: Buffer,
    tile_bases: Buffer,
    columns: PooledPrivateBuffer,
    previous: PooledPrivateBuffer,
    next: PooledPrivateBuffer,
    values: PooledPrivateBuffer,
    increments: PooledPrivateBuffer,
    masks: Buffer,
    blocks: usize,
    slots: usize,
    bytes: usize,
}

struct IndexedStateBuffers {
    common: CommonStateBuffers,
    ra: PooledPrivateBuffer,
    wa: PooledPrivateBuffer,
    bytes: usize,
}

struct WideIndexedStateBuffers {
    common: CommonStateBuffers,
    ra: PooledPrivateBuffer,
    wa: PooledPrivateBuffer,
    bytes: usize,
}

struct DirectStateBuffers {
    common: CommonStateBuffers,
    ra: PooledPrivateBuffer,
    wa: PooledPrivateBuffer,
    operand: Option<PooledPrivateBuffer>,
    bytes: usize,
}

struct DirectPlaneBuffers {
    geometry: StateGeometry,
    ra: PooledPrivateBuffer,
    wa: PooledPrivateBuffer,
    operand: Option<PooledPrivateBuffer>,
    bytes: usize,
}

enum CycleSequenceSource {
    Packed(Arc<AlignedPackedRegisterRows>),
    Stage1 {
        source: RegistersReadWriteStage1Source,
        rd_post: RegistersClaimResidentRdPlane,
    },
}

struct Stage1SourceBuffers {
    instruction_read_raf: Buffer,
    instruction_read_raf_fused_offset: u64,
    rd_post: Buffer,
}

#[derive(Clone, Copy)]
enum PrefillTarget<'a> {
    Wide(&'a WideIndexedStateBuffers),
    Direct(&'a DirectPlaneBuffers),
}

impl PrefillTarget<'_> {
    fn needs_prefill(self) -> bool {
        match self {
            Self::Wide(state) => [
                &state.common.lengths,
                &state.common.columns,
                &state.common.previous,
                &state.common.next,
                &state.common.values,
                &state.common.increments,
                &state.ra,
                &state.wa,
            ]
            .into_iter()
            .any(|buffer| !buffer.was_reused()),
            Self::Direct(state) => {
                [&state.ra, &state.wa]
                    .into_iter()
                    .any(|buffer| !buffer.was_reused())
                    || state
                        .operand
                        .as_ref()
                        .is_some_and(|buffer| !buffer.was_reused())
            }
        }
    }

    fn encode(self, encoder: &metal::BlitCommandEncoderRef) {
        match self {
            Self::Wide(state) => {
                for buffer in [
                    &state.common.lengths,
                    &state.common.columns,
                    &state.common.previous,
                    &state.common.next,
                    &state.common.values,
                    &state.common.increments,
                    &state.ra,
                    &state.wa,
                ] {
                    if !buffer.was_reused() {
                        encoder.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
                    }
                }
            }
            Self::Direct(state) => {
                for buffer in [&state.ra, &state.wa] {
                    if !buffer.was_reused() {
                        encoder.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
                    }
                }
                if let Some(operand) = state.operand.as_ref().filter(|buffer| !buffer.was_reused())
                {
                    encoder.fill_buffer(operand, NSRange::new(0, operand.length()), 0);
                }
            }
        }
    }
}

enum CycleState {
    Indexed(IndexedStateBuffers),
    WideIndexed(WideIndexedStateBuffers),
    Direct(DirectStateBuffers),
}

impl CycleState {
    fn bytes(&self) -> usize {
        match self {
            Self::Indexed(state) => state.bytes,
            Self::WideIndexed(state) => state.bytes,
            Self::Direct(state) => state.bytes,
        }
    }

    fn blocks(&self) -> usize {
        match self {
            Self::Indexed(state) => state.common.blocks,
            Self::WideIndexed(state) => state.common.blocks,
            Self::Direct(state) => state.common.blocks,
        }
    }
}

struct StateGeometry {
    blocks: usize,
    slots: usize,
    tile_bases: Vec<u32>,
    offsets: Buffer,
    masks: Buffer,
}

impl StateGeometry {
    fn bytes(&self) -> usize {
        self.offsets.length() as usize
            + self.masks.length() as usize
            + self.tile_bases.capacity() * size_of::<u32>()
    }
}

struct SequencePipelines {
    first_message: ComputePipelineState,
    first_message_intersection: ComputePipelineState,
    bootstrap: ComputePipelineState,
    stateless_bootstrap: ComputePipelineState,
    stateless_replay_bootstrap: ComputePipelineState,
    replay_bootstrap: ComputePipelineState,
    replay_three_bootstrap: ComputePipelineState,
    replay_three_materialize: ComputePipelineState,
    indexed_state_geometry: ComputePipelineState,
    indexed_state_message: ComputePipelineState,
    indexed: ComputePipelineState,
    indexed_cooperative: ComputePipelineState,
    wide_indexed_cooperative: ComputePipelineState,
    transition: ComputePipelineState,
    transition_cooperative: ComputePipelineState,
    wide_transition_cooperative: ComputePipelineState,
    direct: [Option<ComputePipelineState>; 5],
    direct_geometry: ComputePipelineState,
    direct_cooperative: [Option<ComputePipelineState>; 5],
    operand_claims: ComputePipelineState,
    compact_rs1_claim: ComputePipelineState,
    reduction: ComputePipelineState,
}

pub(crate) struct PendingRegistersReadWriteStage1Pipelines {
    handle: Option<JoinHandle<Result<(), MetalError>>>,
}

#[derive(Clone, Copy, Debug)]
#[cfg(any(test, feature = "test-utils"))]
pub(crate) struct RegistersReadWriteSourcePrimerObservation {
    pub(crate) pages: usize,
    pub(crate) read_bytes: usize,
    #[cfg(feature = "test-utils")]
    pub(crate) output_bytes: usize,
    #[cfg(feature = "test-utils")]
    pub(crate) total_wall: Duration,
    #[cfg(feature = "test-utils")]
    pub(crate) join_wall: Duration,
    #[cfg(feature = "test-utils")]
    pub(crate) gpu_active: Duration,
}

#[must_use = "the resident-source primer must be joined before the source is consumed"]
#[cfg(any(test, feature = "test-utils"))]
pub(crate) struct PendingRegistersReadWriteSourcePrimer {
    sources: [Buffer; 3],
    source_identities: [usize; 3],
    command: Option<CommandBuffer>,
    checksums: Buffer,
    _queue: CommandQueue,
    pages: usize,
    read_bytes: usize,
    #[cfg(feature = "test-utils")]
    started: Instant,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingRegistersReadWriteStage1Pipelines {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

#[cfg(all(feature = "allocative", any(test, feature = "test-utils")))]
impl allocative::Allocative for PendingRegistersReadWriteSourcePrimer {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.enter_self_sized::<Self>().exit();
    }
}

impl Drop for PendingRegistersReadWriteStage1Pipelines {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl Drop for PendingRegistersReadWriteSourcePrimer {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.wait_until_completed();
        }
    }
}

impl PendingRegistersReadWriteStage1Pipelines {
    pub(crate) fn join(mut self) -> Result<Duration, MetalError> {
        let started = Instant::now();
        let handle = self
            .handle
            .take()
            .ok_or(MetalError::InvalidRegistersReadWriteState(
                "registers read-write pipeline warm-up was already consumed",
            ))?;
        handle.join().map_err(|_| {
            MetalError::InvalidRegistersReadWriteState(
                "registers read-write pipeline warm-up panicked",
            )
        })??;
        Ok(started.elapsed())
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl PendingRegistersReadWriteSourcePrimer {
    #[cfg(feature = "test-utils")]
    pub(crate) fn join(mut self) -> Result<RegistersReadWriteSourcePrimerObservation, MetalError> {
        self.complete()
    }

    fn complete(&mut self) -> Result<RegistersReadWriteSourcePrimerObservation, MetalError> {
        let identities = self
            .sources
            .each_ref()
            .map(|buffer| buffer.as_ptr() as usize);
        if identities != self.source_identities
            || self.checksums.length() as usize != SOURCE_PRIMER_THREADS * size_of::<u32>()
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write source-primer resources changed before completion",
            ));
        }
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidRegistersReadWriteState(
                "registers read-write source-primer command was already joined",
            ))?;
        #[cfg(feature = "test-utils")]
        let join_started = Instant::now();
        command.wait_until_completed();
        #[cfg(feature = "test-utils")]
        let join_wall = join_started.elapsed();
        #[cfg(feature = "test-utils")]
        let total_wall = self.started.elapsed();
        #[cfg(feature = "test-utils")]
        let gpu_active = completed_command_gpu_time(&command)?;
        Ok(RegistersReadWriteSourcePrimerObservation {
            pages: self.pages,
            read_bytes: self.read_bytes,
            #[cfg(feature = "test-utils")]
            output_bytes: self.checksums.length() as usize,
            #[cfg(feature = "test-utils")]
            total_wall,
            #[cfg(feature = "test-utils")]
            join_wall,
            #[cfg(feature = "test-utils")]
            gpu_active,
        })
    }

    #[cfg(test)]
    fn join_with_checksums(
        mut self,
    ) -> Result<(RegistersReadWriteSourcePrimerObservation, Vec<u32>), MetalError> {
        let observation = self.complete()?;
        Ok((
            observation,
            buffer_slice::<u32>(&self.checksums, SOURCE_PRIMER_THREADS).to_vec(),
        ))
    }
}

struct SequenceScratch {
    e_in: Buffer,
    e_out: Buffer,
    ra_lut: Buffer,
    wa_lut: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
    geometry_counts: Buffer,
    bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistersReadWriteCycleObservation {
    pub(crate) quadratic: [AkitaField; 2],
    pub(crate) wall: Duration,
    pub(crate) gpu_active: Duration,
    pub(crate) prefill_gpu_active: Duration,
    pub(crate) allocation: Duration,
    #[cfg(feature = "test-utils")]
    pub(crate) live_entries: usize,
    pub(crate) resident_bytes: usize,
    pub(crate) peak_transition_bytes: usize,
    pub(crate) retired_source_bytes: usize,
}

pub(crate) struct RegistersReadWriteCycleFinish {
    pub(crate) roots: Vec<BoundRegisterCycleRoot<AkitaField>>,
    pub(crate) increment: AkitaField,
    pub(crate) wall: Duration,
    pub(crate) gpu_active: Duration,
    pub(crate) allocation: Duration,
    pub(crate) resident_bytes: usize,
    pub(crate) peak_transition_bytes: usize,
}

struct PendingSourceRetirement {
    handle: Option<JoinHandle<()>>,
}

struct RetiredSourceResources {
    _source: Buffer,
    _source_owner: Option<Arc<AlignedPackedRegisterRows>>,
    _stage1_source_buffers: Option<Stage1SourceBuffers>,
    _stage1_source_owner: Option<RegistersReadWriteStage1Source>,
}

impl PendingSourceRetirement {
    fn join(mut self) -> Result<Duration, MetalError> {
        let started = Instant::now();
        let handle = self
            .handle
            .take()
            .ok_or(MetalError::InvalidRegistersReadWriteState(
                "registers read-write source retirement was already joined",
            ))?;
        handle.join().map_err(|_| {
            MetalError::InvalidRegistersReadWriteState(
                "registers read-write source retirement panicked",
            )
        })?;
        Ok(started.elapsed())
    }
}

impl Drop for PendingSourceRetirement {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegistersReadWriteOperandClaimsObservation {
    pub(crate) claims: [AkitaField; 2],
    pub(crate) prepare: Duration,
    pub(crate) wall: Duration,
    pub(crate) gpu_active: Duration,
}

pub(crate) struct RegistersReadWriteCycleSequence {
    context: SolinasMetal,
    pipelines: SequencePipelines,
    source: Option<Buffer>,
    source_owner: Option<Arc<AlignedPackedRegisterRows>>,
    stage1_source_owner: Option<RegistersReadWriteStage1Source>,
    stage1_source_buffers: Option<Stage1SourceBuffers>,
    retired_source_resources: Option<RetiredSourceResources>,
    source_retirement: Option<PendingSourceRetirement>,
    compact_rs1_source: Option<Buffer>,
    _compact_rs1_owner: Option<Arc<AlignedCompactRegisterIndices>>,
    register_unmap: [u8; 64],
    register_map: Buffer,
    remap_registers: bool,
    stage1_source: bool,
    scratch: SequenceScratch,
    state: Option<CycleState>,
    spare_indexed: Option<IndexedStateBuffers>,
    spare_common: Option<CommonStateBuffers>,
    spare_direct: Option<DirectStateBuffers>,
    pending_geometry: Option<StateGeometry>,
    anticipated_geometry: Option<StateGeometry>,
    anticipated_indexed: Option<IndexedStateBuffers>,
    anticipated_wide_geometry: Option<StateGeometry>,
    anticipated_wide_indexed: Option<WideIndexedStateBuffers>,
    anticipated_direct_geometry: Option<StateGeometry>,
    anticipated_direct_next_geometry: Option<StateGeometry>,
    anticipated_direct: Option<DirectPlaneBuffers>,
    anticipated_direct_next: Option<DirectPlaneBuffers>,
    prefill_queue: Option<CommandQueue>,
    source_bytes: usize,
    compact_rs1_indices_bytes: usize,
    physical_rows: usize,
    cycles: usize,
    log_t: usize,
    rounds_bound: usize,
    initial_message_emitted: bool,
    deferred_bootstrap_challenge: Option<AkitaField>,
    deferred_replay_challenge: Option<AkitaField>,
    deferred_wide_challenge: Option<AkitaField>,
    ra_lut: Vec<AkitaField>,
    wa_lut: Vec<AkitaField>,
    rs1_weights: Option<Vec<AkitaField>>,
    gamma: AkitaField,
    threads: usize,
    #[cfg(feature = "test-utils")]
    limits: PipelineLimits,
    private_buffer_pool_epoch: u64,
}

impl SolinasMetal {
    pub(super) fn compile_registers_read_write_source_pipeline(
        &self,
        name: &'static str,
        remap_registers: bool,
        stage1_source: bool,
    ) -> Result<ComputePipelineState, MetalError> {
        let key = (
            name,
            Some(0x100 | u32::from(remap_registers) | (u32::from(stage1_source) << 1)),
        );
        {
            let cache = self
                .pipeline_cache
                .lock()
                .map_err(|_| MetalError::PipelineCachePoisoned)?;
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline.clone());
            }
        }
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            std::ptr::from_ref(&remap_registers).cast::<c_void>(),
            MTLDataType::Bool,
            2,
        );
        constants.set_constant_value_at_index(
            std::ptr::from_ref(&stage1_source).cast::<c_void>(),
            MTLDataType::Bool,
            3,
        );
        let function = self
            .library
            .get_function(name, Some(constants))
            .map_err(|message| MetalError::FunctionLookup { name, message })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation { name, message })?;
        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| MetalError::PipelineCachePoisoned)?;
        Ok(cache.entry(key).or_insert(pipeline).clone())
    }

    fn compile_registers_read_write_direct_pipeline(
        &self,
        name: &'static str,
        operand_carry_kind: u32,
    ) -> Result<ComputePipelineState, MetalError> {
        let key = (name, Some(operand_carry_kind));
        {
            let cache = self
                .pipeline_cache
                .lock()
                .map_err(|_| MetalError::PipelineCachePoisoned)?;
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline.clone());
            }
        }
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            std::ptr::from_ref(&operand_carry_kind).cast::<c_void>(),
            MTLDataType::UInt,
            1,
        );
        let function = self
            .library
            .get_function(name, Some(constants))
            .map_err(|message| MetalError::FunctionLookup { name, message })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation { name, message })?;
        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| MetalError::PipelineCachePoisoned)?;
        Ok(cache.entry(key).or_insert(pipeline).clone())
    }

    pub(crate) fn submit_registers_read_write_stage1_pipeline_warmup(
        &self,
        source: &RegistersReadWriteStage1Source,
    ) -> Result<PendingRegistersReadWriteStage1Pipelines, MetalError> {
        let view = source.device_view();
        let log_t = view.cycles.ilog2() as usize;
        if 1usize.checked_shl(log_t as u32) != Some(view.cycles) {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write pipeline warm-up requires a power-of-two domain",
            ));
        }
        let remap_registers = view.remap_registers;
        let context = self.clone();
        let handle = std::thread::Builder::new()
            .name("jolt-registers-read-write-pipelines".to_owned())
            .spawn(move || {
                context.warm_registers_read_write_stage1_pipelines(remap_registers, log_t)
            })
            .map_err(|_| {
                MetalError::InvalidRegistersReadWriteState(
                    "registers read-write pipeline warm-up thread could not start",
                )
            })?;
        Ok(PendingRegistersReadWriteStage1Pipelines {
            handle: Some(handle),
        })
    }

    fn warm_registers_read_write_stage1_pipelines(
        &self,
        remap_registers: bool,
        log_t: usize,
    ) -> Result<(), MetalError> {
        for name in [
            REGISTERS_READ_WRITE_FIRST_MESSAGE_PIPELINE,
            REGISTERS_READ_WRITE_FIRST_MESSAGE_INTERSECTION_PIPELINE,
            REGISTERS_READ_WRITE_BOOTSTRAP_PIPELINE,
            REGISTERS_READ_WRITE_STATELESS_BOOTSTRAP_PIPELINE,
            REGISTERS_READ_WRITE_STATELESS_REPLAY_BOOTSTRAP_PIPELINE,
            REGISTERS_READ_WRITE_REPLAY_BOOTSTRAP_PIPELINE,
            REGISTERS_READ_WRITE_REPLAY_THREE_BOOTSTRAP_PIPELINE,
            REGISTERS_READ_WRITE_REPLAY_THREE_MATERIALIZE_PIPELINE,
        ] {
            let _ =
                self.compile_registers_read_write_source_pipeline(name, remap_registers, true)?;
        }
        if uses_operand_carry(log_t, true) {
            let _ = self.compile_registers_read_write_direct_pipeline(
                REGISTERS_READ_WRITE_DIRECT_BIND_MESSAGE_PIPELINE,
                4,
            )?;
            for kind in 1..=4 {
                let _ = self.compile_registers_read_write_direct_pipeline(
                    REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
                    kind,
                )?;
            }
        } else {
            let _ = self.compile_registers_read_write_direct_pipeline(
                REGISTERS_READ_WRITE_DIRECT_BIND_MESSAGE_PIPELINE,
                0,
            )?;
            if log_t >= 21 {
                let _ = self.compile_registers_read_write_direct_pipeline(
                    REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
                    0,
                )?;
            }
        }
        let _ = self.compile_named_pipeline(REGISTERS_READ_WRITE_SOURCE_PRIMER_PIPELINE)?;
        Ok(())
    }

    #[cfg(feature = "test-utils")]
    pub(crate) fn submit_registers_read_write_source_primer(
        &self,
        source: &RegistersReadWriteStage1Source,
        rd_post: &RegistersClaimResidentRdPlane,
    ) -> Result<PendingRegistersReadWriteSourcePrimer, MetalError> {
        let view = source.device_view();
        if view.cycles == 0
            || view.device_registry_id != self.device_registry_id()
            || rd_post.geometry().rows() != view.cycles
            || rd_post.device_registry_id() != self.device_registry_id()
            || rd_post.allocation_identity() == 0
            || rd_post.source_generation() == 0
            || rd_post.completion_serial() == 0
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write source primer received mismatched resident owners",
            ));
        }
        self.submit_registers_read_write_source_primer_buffers(
            [
                view.instruction_input.clone(),
                view.instruction_read_raf.clone(),
                rd_post.buffer().clone(),
            ],
            [false, true, false],
        )
    }

    #[cfg(any(test, feature = "test-utils"))]
    fn submit_registers_read_write_source_primer_buffers(
        &self,
        sources: [Buffer; 3],
        active_sources: [bool; 3],
    ) -> Result<PendingRegistersReadWriteSourcePrimer, MetalError> {
        let mut word_counts = [0u64; 3];
        for (index, source) in sources.iter().enumerate() {
            let bytes = source.length();
            if bytes == 0
                || !bytes.is_multiple_of(size_of::<u64>() as u64)
                || source.device().registry_id() != self.device_registry_id()
            {
                return Err(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write source primer received an invalid source buffer",
                ));
            }
            if active_sources[index] {
                word_counts[index] = bytes / size_of::<u64>() as u64;
            }
        }
        let page_words = (SOURCE_PRIMER_PAGE_BYTES / size_of::<u64>()) as u64;
        let pages = word_counts.into_iter().try_fold(0u64, |total, words| {
            total.checked_add(words.div_ceil(page_words))
        });
        let pages = pages
            .and_then(|pages| usize::try_from(pages).ok())
            .ok_or(MetalError::InputTooLong(usize::MAX))?;
        let read_bytes = pages
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::InputTooLong(pages))?;
        let params = SourcePrimerParams {
            word_counts,
            page_words: u32::try_from(page_words)
                .map_err(|_| MetalError::InputTooLong(SOURCE_PRIMER_PAGE_BYTES))?,
            total_threads: SOURCE_PRIMER_THREADS as u32,
        };
        let pipeline = self.compile_named_pipeline(REGISTERS_READ_WRITE_SOURCE_PRIMER_PIPELINE)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != REGISTERS_READ_WRITE_SIMD_WIDTH
            || limits.max_total_threads_per_threadgroup < SOURCE_PRIMER_THREADS_PER_THREADGROUP
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write source-primer pipeline limits changed",
            ));
        }
        let checksum_bytes = SOURCE_PRIMER_THREADS
            .checked_mul(size_of::<u32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(SOURCE_PRIMER_THREADS))?;
        self.validate_additional_working_set(checksum_bytes)?;
        let checksums = self
            .device
            .new_buffer(checksum_bytes, MTLResourceOptions::StorageModeShared);
        let queue = self.device.new_command_queue();
        let command = queue.new_command_buffer().to_owned();
        let _started = autoreleasepool(|| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, source) in sources.iter().enumerate() {
                encoder.set_buffer(index as u64, Some(source), 0);
            }
            encoder.set_buffer(3, Some(&checksums), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: SOURCE_PRIMER_THREADGROUPS as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SOURCE_PRIMER_THREADS_PER_THREADGROUP as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            let started = Instant::now();
            command.commit();
            started
        });
        let source_identities = sources.each_ref().map(|buffer| buffer.as_ptr() as usize);
        Ok(PendingRegistersReadWriteSourcePrimer {
            sources,
            source_identities,
            command: Some(command),
            checksums,
            _queue: queue,
            pages,
            read_bytes,
            #[cfg(feature = "test-utils")]
            started: _started,
        })
    }

    pub(crate) fn prepare_registers_read_write_cycle_sequence(
        &self,
        source_owner: Arc<AlignedPackedRegisterRows>,
        log_t: usize,
        gamma: AkitaField,
    ) -> Result<RegistersReadWriteCycleSequence, MetalError> {
        self.prepare_registers_read_write_cycle_sequence_from_source(
            CycleSequenceSource::Packed(source_owner),
            log_t,
            gamma,
        )
    }

    pub(crate) fn prepare_registers_read_write_cycle_sequence_from_stage1(
        &self,
        source_owner: RegistersReadWriteStage1Source,
        rd_post: RegistersClaimResidentRdPlane,
        log_t: usize,
        gamma: AkitaField,
    ) -> Result<RegistersReadWriteCycleSequence, MetalError> {
        self.prepare_registers_read_write_cycle_sequence_from_source(
            CycleSequenceSource::Stage1 {
                source: source_owner,
                rd_post,
            },
            log_t,
            gamma,
        )
    }

    fn prepare_registers_read_write_cycle_sequence_from_source(
        &self,
        source_owner: CycleSequenceSource,
        log_t: usize,
        gamma: AkitaField,
    ) -> Result<RegistersReadWriteCycleSequence, MetalError> {
        if !(4..=28).contains(&log_t) {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write sequence geometry is unsupported",
            ));
        }
        let cycles = 1usize << log_t;
        let (
            physical_rows,
            active_registers,
            remap_registers,
            register_map,
            register_unmap,
            source_bytes,
            compact_rs1_indices_bytes,
            stage1_source,
        ) = match &source_owner {
            CycleSequenceSource::Packed(owner) => {
                let rows = owner.device_view();
                let compact_rs1_source = rows.compact_rs1_source();
                let source_bytes = rows.allocation_bytes();
                if rows.rows() == 0
                    || rows.rows() > cycles
                    || !rows
                        .as_ptr()
                        .addr()
                        .is_multiple_of(PACKED_REGISTER_ROWS_ALIGNMENT)
                    || !source_bytes.is_multiple_of(PACKED_REGISTER_ROWS_ALIGNMENT)
                    || source_bytes
                        < rows
                            .rows()
                            .checked_mul(size_of::<PackedRegisterCycleRow>())
                            .ok_or(MetalError::InputTooLong(rows.rows()))?
                    || (log_t >= COMPACT_RS1_SOURCE_LOG_T_MIN
                        && compact_rs1_source.is_none_or(|(ptr, bytes)| {
                            !ptr.addr().is_multiple_of(PACKED_REGISTER_ROWS_ALIGNMENT)
                                || !bytes.is_multiple_of(PACKED_REGISTER_ROWS_ALIGNMENT)
                                || bytes < rows.rows()
                        }))
                {
                    return Err(MetalError::InvalidRegistersReadWriteState(
                        "packed source does not satisfy the cycle-sequence contract",
                    ));
                }
                let mut register_map = [0u8; 128];
                for (index, mapped) in register_map.iter_mut().enumerate() {
                    *mapped = index as u8;
                }
                (
                    rows.rows(),
                    rows.active_registers(),
                    rows.remaps_registers(),
                    register_map,
                    rows.register_unmap(),
                    source_bytes,
                    compact_rs1_source.map_or(0, |(_, bytes)| bytes),
                    false,
                )
            }
            CycleSequenceSource::Stage1 { source, rd_post } => {
                let source = source.device_view();
                let expected_instruction_bytes = cycles
                    .checked_mul(6 * size_of::<u64>())
                    .ok_or(MetalError::InputTooLong(cycles))?;
                let expected_instruction_read_raf_bytes = cycles
                    .checked_mul(4 * size_of::<u64>())
                    .ok_or(MetalError::InputTooLong(cycles))?;
                let expected_rd_index_bytes = cycles;
                let expected_rd_post_bytes = cycles
                    .checked_mul(size_of::<u64>())
                    .ok_or(MetalError::InputTooLong(cycles))?;
                if source.physical_rows == 0
                    || source.physical_rows > cycles
                    || source.cycles != cycles
                    || source.device_registry_id != self.device_registry_id()
                    || source.instruction_input.length() as usize != expected_instruction_bytes
                    || source.instruction_read_raf.length() as usize
                        != expected_instruction_read_raf_bytes
                    || source.rd_indices.length() as usize != expected_rd_index_bytes
                    || rd_post.geometry().rows() != cycles
                    || rd_post.device_registry_id() != self.device_registry_id()
                    || rd_post.resident_bytes() as usize != expected_rd_post_bytes
                    || rd_post.buffer().length() as usize != expected_rd_post_bytes
                    || rd_post.allocation_identity() == 0
                    || rd_post.source_generation() == 0
                    || rd_post.completion_serial() == 0
                {
                    return Err(MetalError::InvalidRegistersReadWriteState(
                        "Stage-1 register source does not satisfy the cycle-sequence contract",
                    ));
                }
                let source_bytes = expected_instruction_bytes
                    .checked_add(expected_instruction_read_raf_bytes)
                    .and_then(|bytes| bytes.checked_add(expected_rd_index_bytes))
                    .and_then(|bytes| bytes.checked_add(expected_rd_post_bytes))
                    .ok_or(MetalError::InputTooLong(cycles))?;
                (
                    source.physical_rows,
                    source.active_registers,
                    source.remap_registers,
                    source.register_map,
                    source.register_unmap,
                    source_bytes,
                    cycles,
                    true,
                )
            }
        };
        if active_registers > MAX_REGISTER_BLOCK_CAPACITY {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write Metal state has more than 64 active registers",
            ));
        }
        let private_buffer_pool_epoch = self.begin_private_buffer_pool_epoch(
            (physical_rows, log_t),
            PRIVATE_PAYLOAD_POOL_CAP_BYTES,
        )?;

        let (direct, direct_cooperative) = if uses_operand_carry(log_t, stage1_source) {
            (
                [
                    None,
                    None,
                    None,
                    None,
                    Some(self.compile_registers_read_write_direct_pipeline(
                        REGISTERS_READ_WRITE_DIRECT_BIND_MESSAGE_PIPELINE,
                        4,
                    )?),
                ],
                [
                    None,
                    Some(self.compile_registers_read_write_direct_pipeline(
                        REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
                        1,
                    )?),
                    Some(self.compile_registers_read_write_direct_pipeline(
                        REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
                        2,
                    )?),
                    Some(self.compile_registers_read_write_direct_pipeline(
                        REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
                        3,
                    )?),
                    Some(self.compile_registers_read_write_direct_pipeline(
                        REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
                        4,
                    )?),
                ],
            )
        } else {
            (
                [
                    Some(self.compile_registers_read_write_direct_pipeline(
                        REGISTERS_READ_WRITE_DIRECT_BIND_MESSAGE_PIPELINE,
                        0,
                    )?),
                    None,
                    None,
                    None,
                    None,
                ],
                [
                    (log_t >= 21)
                        .then(|| {
                            self.compile_registers_read_write_direct_pipeline(
                                REGISTERS_READ_WRITE_DIRECT_COOPERATIVE_PIPELINE,
                                0,
                            )
                        })
                        .transpose()?,
                    None,
                    None,
                    None,
                    None,
                ],
            )
        };
        let pipelines = SequencePipelines {
            first_message: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_FIRST_MESSAGE_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            first_message_intersection: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_FIRST_MESSAGE_INTERSECTION_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            bootstrap: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_BOOTSTRAP_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            stateless_bootstrap: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_STATELESS_BOOTSTRAP_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            stateless_replay_bootstrap: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_STATELESS_REPLAY_BOOTSTRAP_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            replay_bootstrap: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_REPLAY_BOOTSTRAP_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            replay_three_bootstrap: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_REPLAY_THREE_BOOTSTRAP_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            replay_three_materialize: self.compile_registers_read_write_source_pipeline(
                REGISTERS_READ_WRITE_REPLAY_THREE_MATERIALIZE_PIPELINE,
                remap_registers,
                stage1_source,
            )?,
            indexed_state_geometry: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_INDEXED_STATE_GEOMETRY_PIPELINE)?,
            indexed_state_message: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_INDEXED_STATE_MESSAGE_PIPELINE)?,
            indexed: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_INDEXED_BIND_MESSAGE_PIPELINE)?,
            indexed_cooperative: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_INDEXED_COOPERATIVE_PIPELINE)?,
            wide_indexed_cooperative: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_WIDE_INDEXED_COOPERATIVE_PIPELINE)?,
            transition: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_TRANSITION_BIND_MESSAGE_PIPELINE)?,
            transition_cooperative: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_TRANSITION_COOPERATIVE_PIPELINE)?,
            wide_transition_cooperative: self.compile_named_pipeline(
                REGISTERS_READ_WRITE_WIDE_TRANSITION_COOPERATIVE_PIPELINE,
            )?,
            direct,
            direct_geometry: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_DIRECT_GEOMETRY_PIPELINE)?,
            direct_cooperative,
            operand_claims: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_OPERAND_CLAIMS_PIPELINE)?,
            compact_rs1_claim: self
                .compile_named_pipeline(REGISTERS_READ_WRITE_COMPACT_RS1_CLAIM_PIPELINE)?,
            reduction: self.compile_named_pipeline(REGISTERS_READ_WRITE_REDUCTION_PIPELINE)?,
        };
        let limits = pipelines
            .direct
            .iter()
            .chain(pipelines.direct_cooperative.iter())
            .flatten()
            .next()
            .map(Self::limits)
            .ok_or(MetalError::InvalidRegistersReadWriteState(
                "registers read-write sequence has no reachable direct pipeline",
            ))?;
        let mut all_pipelines = vec![
            &pipelines.first_message,
            &pipelines.first_message_intersection,
            &pipelines.bootstrap,
            &pipelines.stateless_bootstrap,
            &pipelines.stateless_replay_bootstrap,
            &pipelines.replay_bootstrap,
            &pipelines.replay_three_bootstrap,
            &pipelines.replay_three_materialize,
            &pipelines.indexed_state_geometry,
            &pipelines.indexed_state_message,
            &pipelines.indexed,
            &pipelines.indexed_cooperative,
            &pipelines.wide_indexed_cooperative,
            &pipelines.transition,
            &pipelines.transition_cooperative,
            &pipelines.wide_transition_cooperative,
            &pipelines.direct_geometry,
            &pipelines.operand_claims,
            &pipelines.compact_rs1_claim,
            &pipelines.reduction,
        ];
        all_pipelines.extend(pipelines.direct.iter().flatten());
        all_pipelines.extend(pipelines.direct_cooperative.iter().flatten());
        if all_pipelines.iter().any(|pipeline| {
            Self::limits(pipeline).thread_execution_width != REGISTERS_READ_WRITE_SIMD_WIDTH
        }) {
            return Err(MetalError::UnsupportedRegistersReadWriteExecutionWidth {
                expected: REGISTERS_READ_WRITE_SIMD_WIDTH,
                got: all_pipelines
                    .iter()
                    .map(|pipeline| Self::limits(pipeline).thread_execution_width)
                    .min()
                    .unwrap_or(0),
            });
        }
        let threads = Self::resolve_threadgroup_width(Some(REGISTERS_READ_WRITE_THREADS), limits)?;
        let static_threadgroup_bytes = all_pipelines
            .iter()
            .map(|pipeline| Self::limits(pipeline).static_threadgroup_memory_length)
            .max()
            .unwrap_or(0);
        if threads != REGISTERS_READ_WRITE_THREADS
            || static_threadgroup_bytes > REGISTERS_READ_WRITE_THREADGROUP_BYTES_MAX
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write sequence pipeline exceeds its checked schedule",
            ));
        }

        let first_groups = (cycles / 2).div_ceil(threads);
        let maximum_cycle_eq_fields = 1usize << log_t.div_ceil(2);
        let compact_claim_source = log_t >= COMPACT_RS1_SOURCE_LOG_T_MIN || stage1_source;
        let (maximum_e_in_fields, maximum_e_out_fields, maximum_partial_fields) =
            if compact_claim_source {
                let joint_bits = log_t + REGISTER_ADDRESS_BITS;
                let high_bits = core::cmp::min(log_t, joint_bits.div_ceil(2));
                let low_cycle_bits = joint_bits - high_bits - REGISTER_ADDRESS_BITS;
                (
                    core::cmp::max(maximum_cycle_eq_fields, 1usize << high_bits),
                    core::cmp::max(maximum_cycle_eq_fields, 1usize << (joint_bits - high_bits)),
                    first_groups.max(physical_rows.div_ceil(1usize << low_cycle_bits)),
                )
            } else {
                (
                    maximum_cycle_eq_fields,
                    maximum_cycle_eq_fields,
                    first_groups,
                )
            };
        let scratch = SequenceScratch::new(
            self,
            maximum_partial_fields,
            maximum_e_in_fields,
            maximum_e_out_fields,
        )?;
        if !stage1_source {
            self.validate_buffer_length(
                u64::try_from(source_bytes).map_err(|_| MetalError::InputTooLong(source_bytes))?,
            )?;
        }
        if compact_rs1_indices_bytes != 0 {
            self.validate_buffer_length(
                u64::try_from(compact_rs1_indices_bytes)
                    .map_err(|_| MetalError::InputTooLong(compact_rs1_indices_bytes))?,
            )?;
        }
        let base_resident_bytes = source_bytes
            .checked_add(scratch.bytes)
            .and_then(|bytes| bytes.checked_add(compact_rs1_indices_bytes))
            .ok_or(MetalError::InputTooLong(source_bytes))?;
        self.validate_additional_working_set(
            u64::try_from(base_resident_bytes)
                .map_err(|_| MetalError::InputTooLong(source_bytes))?,
        )?;
        let (
            source,
            packed_source_owner,
            stage1_source_owner,
            stage1_source_buffers,
            compact_rs1_source,
            compact_rs1_owner,
        ) = match source_owner {
            CycleSequenceSource::Packed(owner) => {
                let rows = owner.device_view();
                let compact_owner = owner.compact_rs1_owner();
                let source = self.device.new_buffer_with_bytes_no_copy(
                    rows.as_ptr().cast_mut().cast::<c_void>(),
                    source_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                );
                let compact = compact_owner.as_ref().map(|owner| {
                    self.device.new_buffer_with_bytes_no_copy(
                        owner.as_ptr().cast_mut().cast::<c_void>(),
                        owner.allocation_bytes() as u64,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
                });
                (source, Some(owner), None, None, compact, compact_owner)
            }
            CycleSequenceSource::Stage1 { source, rd_post } => {
                let source_view = source.device_view();
                let instruction_buffer = source_view.instruction_input.clone();
                let instruction_read_raf_fused_offset = cycles
                    .checked_mul(2 * size_of::<u64>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or(MetalError::InputTooLong(cycles))?;
                let buffers = Stage1SourceBuffers {
                    instruction_read_raf: source_view.instruction_read_raf.clone(),
                    instruction_read_raf_fused_offset,
                    rd_post: rd_post.buffer().clone(),
                };
                let compact = Some(new_buffer::<u8>(
                    self,
                    cycles,
                    MTLResourceOptions::StorageModePrivate,
                )?);
                (
                    instruction_buffer,
                    None,
                    Some(source),
                    Some(buffers),
                    compact,
                    None,
                )
            }
        };
        let register_map_buffer = new_buffer::<u8>(
            self,
            register_map.len(),
            MTLResourceOptions::StorageModeShared,
        )?;
        buffer_slice_mut::<u8>(&register_map_buffer, register_map.len())
            .copy_from_slice(&register_map);
        let gamma_sq = gamma * gamma;
        Ok(RegistersReadWriteCycleSequence {
            context: self.clone(),
            pipelines,
            source: Some(source),
            source_owner: packed_source_owner,
            stage1_source_owner,
            stage1_source_buffers,
            retired_source_resources: None,
            source_retirement: None,
            compact_rs1_source,
            _compact_rs1_owner: compact_rs1_owner,
            register_unmap,
            register_map: register_map_buffer,
            remap_registers,
            stage1_source,
            scratch,
            state: None,
            spare_indexed: None,
            spare_common: None,
            spare_direct: None,
            pending_geometry: None,
            anticipated_geometry: None,
            anticipated_indexed: None,
            anticipated_wide_geometry: None,
            anticipated_wide_indexed: None,
            anticipated_direct_geometry: None,
            anticipated_direct_next_geometry: None,
            anticipated_direct: None,
            anticipated_direct_next: None,
            prefill_queue: (log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN)
                .then(|| self.device.new_command_queue()),
            source_bytes,
            compact_rs1_indices_bytes,
            physical_rows,
            cycles,
            log_t,
            rounds_bound: 0,
            initial_message_emitted: false,
            deferred_bootstrap_challenge: None,
            deferred_replay_challenge: None,
            deferred_wide_challenge: None,
            ra_lut: vec![AkitaField::zero(), gamma, gamma_sq, gamma + gamma_sq],
            wa_lut: vec![AkitaField::zero(), AkitaField::one()],
            rs1_weights: uses_operand_carry(log_t, stage1_source).then(|| vec![AkitaField::one()]),
            gamma,
            threads,
            #[cfg(feature = "test-utils")]
            limits,
            private_buffer_pool_epoch,
        })
    }
}

impl SequenceScratch {
    fn new(
        context: &SolinasMetal,
        partial_count: usize,
        maximum_e_in_fields: usize,
        maximum_e_out_fields: usize,
    ) -> Result<Self, MetalError> {
        let e_in = new_buffer::<Fp128>(
            context,
            maximum_e_in_fields,
            MTLResourceOptions::StorageModeShared,
        )?;
        let e_out = new_buffer::<Fp128>(
            context,
            maximum_e_out_fields,
            MTLResourceOptions::StorageModeShared,
        )?;
        let ra_lut = new_buffer::<Fp128>(context, 1 << 16, MTLResourceOptions::StorageModeShared)?;
        let wa_lut = new_buffer::<Fp128>(context, 1 << 8, MTLResourceOptions::StorageModeShared)?;
        let partial_a = new_buffer::<Fp128>(
            context,
            2 * partial_count.max(1),
            MTLResourceOptions::StorageModeShared,
        )?;
        let partial_b = new_buffer::<Fp128>(
            context,
            2 * partial_count.max(1),
            MTLResourceOptions::StorageModeShared,
        )?;
        let geometry_counts = new_buffer::<u32>(
            context,
            partial_count.max(1),
            MTLResourceOptions::StorageModeShared,
        )?;
        let bytes = [
            e_in.length(),
            e_out.length(),
            ra_lut.length(),
            wa_lut.length(),
            partial_a.length(),
            partial_b.length(),
            geometry_counts.length(),
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes as usize)
                .ok_or(MetalError::InputTooLong(total))
        })?;
        Ok(Self {
            e_in,
            e_out,
            ra_lut,
            wa_lut,
            partial_a,
            partial_b,
            geometry_counts,
            bytes,
        })
    }
}

impl RegistersReadWriteCycleSequence {
    fn source(&self) -> Result<&Buffer, MetalError> {
        self.source
            .as_ref()
            .ok_or(MetalError::InvalidRegistersReadWriteState(
                "registers read-write packed source was retired before its last use",
            ))
    }

    fn bind_source(&self, encoder: &metal::ComputeCommandEncoderRef) -> Result<(), MetalError> {
        encoder.set_buffer(0, Some(self.source()?), 0);
        if self.stage1_source {
            let source = self.stage1_source_buffers.as_ref().ok_or(
                MetalError::InvalidRegistersReadWriteState(
                    "registers read-write lost its Stage-1 source buffers",
                ),
            )?;
            encoder.set_buffer(23, Some(self.source()?), 0);
            encoder.set_buffer(
                24,
                Some(&source.instruction_read_raf),
                source.instruction_read_raf_fused_offset,
            );
            encoder.set_buffer(25, Some(&source.rd_post), 0);
            encoder.set_buffer(26, Some(&self.register_map), 0);
        }
        Ok(())
    }

    fn retire_source(&mut self) -> Result<usize, MetalError> {
        if self.log_t < CROSS_REPRESENTATION_REUSE_LOG_T_MIN || self.source.is_none() {
            return Ok(0);
        }
        if self.retired_source_resources.is_some() || self.source_retirement.is_some() {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write source retirement was scheduled twice",
            ));
        }
        let retired = self.source_bytes;
        let resources = RetiredSourceResources {
            _source: self
                .source
                .take()
                .ok_or(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write source disappeared before retirement",
                ))?,
            _source_owner: self.source_owner.take(),
            _stage1_source_buffers: self.stage1_source_buffers.take(),
            _stage1_source_owner: self.stage1_source_owner.take(),
        };
        if self.log_t >= ASYNC_SOURCE_RETIREMENT_LOG_T_MIN {
            self.retired_source_resources = Some(resources);
        } else {
            drop(resources);
        }
        Ok(retired)
    }

    fn start_source_retirement(&mut self) -> Result<(), MetalError> {
        let resources = self.retired_source_resources.take().ok_or(
            MetalError::InvalidRegistersReadWriteState(
                "registers read-write source retirement has no resources",
            ),
        )?;
        let handle = std::thread::Builder::new()
            .name("jolt-register-source-retirement".to_owned())
            .spawn(move || drop(resources))
            .map_err(|_| {
                MetalError::InvalidRegistersReadWriteState(
                    "failed to spawn registers read-write source retirement",
                )
            })?;
        self.source_retirement = Some(PendingSourceRetirement {
            handle: Some(handle),
        });
        Ok(())
    }

    pub(crate) fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        gamma: AkitaField,
    ) -> Result<RegistersReadWriteCycleObservation, MetalError> {
        if self.initial_message_emitted || self.state.is_some() || self.rounds_bound != 0 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write initial message was emitted twice",
            ));
        }
        self.write_weights(0, e_in, e_out)?;
        let geometry_free = self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN;
        let pair_count = if geometry_free {
            self.physical_rows.div_ceil(2)
        } else {
            self.cycles / 2
        };
        let groups = pair_count.div_ceil(self.threads);
        let allocation_started = Instant::now();
        let geometry = if geometry_free {
            None
        } else {
            Some((
                new_buffer::<u16>(
                    &self.context,
                    pair_count,
                    MTLResourceOptions::StorageModePrivate,
                )?,
                new_buffer::<u64>(
                    &self.context,
                    pair_count,
                    MTLResourceOptions::StorageModePrivate,
                )?,
            ))
        };
        let allocation = allocation_started.elapsed();
        let params = FirstMessageParams {
            row_count: checked_u32(self.physical_rows)?,
            pair_count: checked_u32(pair_count)?,
            output_stride: checked_u32(groups)?,
            e_in_length: checked_u32(e_in.len())?,
            source_stride: checked_u32(self.cycles)?,
        };
        let gamma_sq = Fp128::from_jolt_field(&(gamma * gamma));
        let gamma = Fp128::from_jolt_field(&gamma);
        let (quadratic, wall, gpu_active) = autoreleasepool(|| {
            let command = self.context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            let pipeline = if geometry_free {
                &self.pipelines.first_message_intersection
            } else {
                &self.pipelines.first_message
            };
            encoder.set_compute_pipeline_state(pipeline);
            self.bind_source(encoder)?;
            if self.stage1_source {
                let compact_rs1 = self.compact_rs1_source.as_ref().ok_or(
                    MetalError::InvalidRegistersReadWriteState(
                        "registers read-write lost its compact rs1 output",
                    ),
                )?;
                encoder.set_buffer(27, Some(compact_rs1), 0);
            }
            encoder.set_buffer(1, Some(&self.scratch.e_in), 0);
            encoder.set_buffer(2, Some(&self.scratch.e_out), 0);
            encoder.set_buffer(3, Some(&self.scratch.partial_a), 0);
            set_inline_bytes(encoder, 4, &params);
            set_inline_bytes(encoder, 5, &gamma);
            set_inline_bytes(encoder, 6, &gamma_sq);
            if let Some((geometry_offsets, geometry_masks)) = geometry.as_ref() {
                encoder.set_buffer(7, Some(&self.scratch.geometry_counts), 0);
                encoder.set_buffer(8, Some(geometry_offsets), 0);
                encoder.set_buffer(9, Some(geometry_masks), 0);
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.scratch.partial_a,
                &self.scratch.partial_b,
                groups,
                2,
                REGISTERS_READ_WRITE_SIMD_WIDTH,
            )?;
            encoder.end_encoding();
            let started = Instant::now();
            command.commit();
            command.wait_until_completed();
            let wall = started.elapsed();
            let gpu_active = completed_command_gpu_time(command)?;
            let output = if final_in_a {
                &self.scratch.partial_a
            } else {
                &self.scratch.partial_b
            };
            Ok::<_, MetalError>((read_quadratic(&self.context, output)?, wall, gpu_active))
        })?;
        if self.stage1_source {
            tracing::info!(
                target: "jolt::metal",
                predecessor_scan_dispatches = 0usize,
                algorithm = "signed_fused_inc_v1",
                "reconstructed registers read-write rd-pre from Stage-1 deltas"
            );
        }
        self.pending_geometry = match geometry {
            Some((geometry_offsets, geometry_masks)) => {
                Some(self.read_geometry(pair_count, groups, geometry_offsets, geometry_masks)?)
            }
            None => None,
        };
        self.initial_message_emitted = true;
        let resident_bytes = self.resident_bytes();
        Ok(RegistersReadWriteCycleObservation {
            quadratic,
            wall,
            gpu_active,
            prefill_gpu_active: Duration::ZERO,
            allocation,
            #[cfg(feature = "test-utils")]
            live_entries: 0,
            resident_bytes,
            peak_transition_bytes: resident_bytes,
            retired_source_bytes: 0,
        })
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<RegistersReadWriteCycleObservation, MetalError> {
        if !self.initial_message_emitted
            || self.rounds_bound + 1 >= self.log_t
            || e_in.is_empty()
            || e_out.is_empty()
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write bind/message is outside the cycle phase",
            ));
        }
        let new_rounds_bound = self.rounds_bound + 1;
        self.write_weights(new_rounds_bound, e_in, e_out)?;
        let input_blocks = self.cycles >> self.rounds_bound;
        let output_blocks = input_blocks / 2;
        let work_items = output_blocks / 2;
        let groups = work_items.div_ceil(self.threads);
        let mut ra_lut_bits = 0;
        let mut wa_lut_bits = 0;
        if new_rounds_bound <= 3 {
            ra_lut_bits = self.ra_lut.len().trailing_zeros();
            wa_lut_bits = self.wa_lut.len().trailing_zeros();
            bind_lut(&mut self.ra_lut, challenge)?;
            bind_lut(&mut self.wa_lut, challenge)?;
        }
        if let Some(rs1_weights) = self.rs1_weights.as_mut() {
            if new_rounds_bound <= 8 {
                let low_scale = AkitaField::one() - challenge;
                let old = rs1_weights.as_slice();
                let mut next = Vec::with_capacity(2 * old.len());
                next.extend(old.iter().map(|weight| *weight * low_scale));
                next.extend(old.iter().map(|weight| *weight * challenge));
                *rs1_weights = next;
            }
            if new_rounds_bound == 8 {
                write_fields(&self.scratch.ra_lut, rs1_weights)?;
            }
        }
        if new_rounds_bound <= direct_transition_bound(self.log_t) {
            write_fields(&self.scratch.ra_lut, &self.ra_lut)?;
            write_fields(&self.scratch.wa_lut, &self.wa_lut)?;
        }
        let allocation_started = Instant::now();
        let anticipated_output = if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN {
            match new_rounds_bound {
                3 => Some(CycleState::Indexed(self.anticipated_indexed.take().ok_or(
                    MetalError::InvalidRegistersReadWriteState(
                        "registers read-write sequence is missing its anticipated indexed state",
                    ),
                )?)),
                4 => Some(CycleState::WideIndexed(
                    self.anticipated_wide_indexed.take().ok_or(
                        MetalError::InvalidRegistersReadWriteState(
                            "registers read-write sequence is missing its anticipated wide state",
                        ),
                    )?,
                )),
                5 => {
                    let planes =
                        self.anticipated_direct
                            .take()
                            .ok_or(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write sequence is missing anticipated direct planes",
                        ))?;
                    let common = self
                        .spare_common
                        .take()
                        .or_else(|| self.spare_indexed.take().map(|state| state.common))
                        .ok_or(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write sequence is missing the direct common donor",
                        ))?;
                    self.spare_indexed = None;
                    Some(CycleState::Direct(
                        planes.into_state(&self.context, common)?,
                    ))
                }
                6 => {
                    let planes = self.anticipated_direct_next.take().ok_or(
                        MetalError::InvalidRegistersReadWriteState(
                            "registers read-write sequence is missing next direct planes",
                        ),
                    )?;
                    let common = self
                        .spare_common
                        .take()
                        .or_else(|| self.spare_indexed.take().map(|state| state.common))
                        .ok_or(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write sequence is missing the next common donor",
                        ))?;
                    self.spare_indexed = None;
                    Some(CycleState::Direct(
                        planes.into_state(&self.context, common)?,
                    ))
                }
                _ => None,
            }
        } else {
            None
        };
        let geometry =
            if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN && new_rounds_bound == 1 {
                self.pending_geometry.take()
            } else if anticipated_output.is_some() {
                if let Some(redundant) = self.pending_geometry.take() {
                    if redundant.blocks != output_blocks {
                        return Err(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write redundant geometry has the wrong block count",
                        ));
                    }
                }
                if self.anticipated_geometry.is_some() {
                    return Err(MetalError::InvalidRegistersReadWriteState(
                        "registers read-write sequence retained stale anticipated geometry",
                    ));
                }
                None
            } else {
                Some(self.pending_geometry.take().ok_or(
                    MetalError::InvalidRegistersReadWriteState(
                        "registers read-write sequence is missing anticipated geometry",
                    ),
                )?)
            };
        if geometry
            .as_ref()
            .is_some_and(|geometry| geometry.blocks != output_blocks)
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write anticipated geometry has the wrong block count",
            ));
        }
        if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN && new_rounds_bound == 1 {
            let active_work_items = self.physical_rows.div_ceil(4);
            let active_groups = active_work_items.div_ceil(self.threads);
            let geometry_groups = work_items.div_ceil(self.threads);
            let next_offsets = new_buffer::<u16>(
                &self.context,
                work_items,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let next_masks = new_buffer::<u64>(
                &self.context,
                work_items,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let future_blocks = work_items / 2;
            let future_groups = future_blocks.div_ceil(self.threads);
            let future_counts = new_buffer::<u32>(
                &self.context,
                future_groups,
                MTLResourceOptions::StorageModeShared,
            )?;
            let future_offsets = new_buffer::<u16>(
                &self.context,
                future_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let future_masks = new_buffer::<u64>(
                &self.context,
                future_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let wide_future_blocks = future_blocks / 2;
            let wide_future_groups = wide_future_blocks.div_ceil(self.threads);
            let wide_future_counts = new_buffer::<u32>(
                &self.context,
                wide_future_groups,
                MTLResourceOptions::StorageModeShared,
            )?;
            let wide_future_offsets = new_buffer::<u16>(
                &self.context,
                wide_future_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let wide_future_masks = new_buffer::<u64>(
                &self.context,
                wide_future_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let direct_future_blocks = wide_future_blocks / 2;
            let direct_future_groups = direct_future_blocks.div_ceil(self.threads);
            let direct_future_counts = new_buffer::<u32>(
                &self.context,
                direct_future_groups,
                MTLResourceOptions::StorageModeShared,
            )?;
            let direct_future_offsets = new_buffer::<u16>(
                &self.context,
                direct_future_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let direct_future_masks = new_buffer::<u64>(
                &self.context,
                direct_future_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let direct_next_blocks = direct_future_blocks / 2;
            let direct_next_groups = direct_next_blocks.div_ceil(self.threads);
            let direct_next_counts = new_buffer::<u32>(
                &self.context,
                direct_next_groups,
                MTLResourceOptions::StorageModeShared,
            )?;
            let direct_next_offsets = new_buffer::<u16>(
                &self.context,
                direct_next_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let direct_next_masks = new_buffer::<u64>(
                &self.context,
                direct_next_blocks,
                MTLResourceOptions::StorageModePrivate,
            )?;
            let allocation = allocation_started.elapsed();
            let peak_transition_bytes = self
                .base_resident_bytes()
                .checked_add(geometry.as_ref().map_or(0, StateGeometry::bytes))
                .and_then(|bytes| bytes.checked_add(next_offsets.length() as usize))
                .and_then(|bytes| bytes.checked_add(next_masks.length() as usize))
                .and_then(|bytes| bytes.checked_add(future_counts.length() as usize))
                .and_then(|bytes| bytes.checked_add(future_offsets.length() as usize))
                .and_then(|bytes| bytes.checked_add(future_masks.length() as usize))
                .and_then(|bytes| bytes.checked_add(wide_future_counts.length() as usize))
                .and_then(|bytes| bytes.checked_add(wide_future_offsets.length() as usize))
                .and_then(|bytes| bytes.checked_add(wide_future_masks.length() as usize))
                .and_then(|bytes| bytes.checked_add(direct_future_counts.length() as usize))
                .and_then(|bytes| bytes.checked_add(direct_future_offsets.length() as usize))
                .and_then(|bytes| bytes.checked_add(direct_future_masks.length() as usize))
                .and_then(|bytes| bytes.checked_add(direct_next_counts.length() as usize))
                .and_then(|bytes| bytes.checked_add(direct_next_offsets.length() as usize))
                .and_then(|bytes| bytes.checked_add(direct_next_masks.length() as usize))
                .ok_or(MetalError::InputTooLong(work_items))?;
            self.context.validate_additional_working_set(
                u64::try_from(peak_transition_bytes)
                    .map_err(|_| MetalError::InputTooLong(peak_transition_bytes))?,
            )?;
            let params = SequenceParams {
                row_count: checked_u32(self.physical_rows)?,
                input_blocks: checked_u32(input_blocks)?,
                output_blocks: checked_u32(output_blocks)?,
                input_capacity: 0,
                output_capacity: 0,
                work_items: checked_u32(active_work_items)?,
                output_stride: checked_u32(active_groups)?,
                e_in_length: checked_u32(e_in.len())?,
                ra_lut_bits,
                wa_lut_bits,
                emit_message: 1,
                reserved: 0,
                source_stride: checked_u32(self.cycles)?,
            };
            let future_params = SequenceParams {
                work_items: checked_u32(future_blocks)?,
                output_stride: checked_u32(future_groups)?,
                ..params
            };
            let wide_future_params = SequenceParams {
                work_items: checked_u32(wide_future_blocks)?,
                output_stride: checked_u32(wide_future_groups)?,
                ..params
            };
            let direct_future_params = SequenceParams {
                work_items: checked_u32(direct_future_blocks)?,
                output_stride: checked_u32(direct_future_groups)?,
                ..params
            };
            let direct_next_params = SequenceParams {
                work_items: checked_u32(direct_next_blocks)?,
                output_stride: checked_u32(direct_next_groups)?,
                ..params
            };
            let deferred_challenge = challenge;
            let (quadratic, wall, gpu_active) = self.submit_stateless_bootstrap(
                challenge,
                params,
                active_groups,
                &next_offsets,
                &next_masks,
                future_params,
                future_groups,
                &future_counts,
                &future_offsets,
                &future_masks,
                wide_future_params,
                wide_future_groups,
                &wide_future_counts,
                &wide_future_offsets,
                &wide_future_masks,
                direct_future_params,
                direct_future_groups,
                &direct_future_counts,
                &direct_future_offsets,
                &direct_future_masks,
                direct_next_params,
                direct_next_groups,
                &direct_next_counts,
                &direct_next_offsets,
                &direct_next_masks,
            )?;
            self.pending_geometry =
                Some(self.read_geometry(work_items, geometry_groups, next_offsets, next_masks)?);
            self.anticipated_geometry = Some(self.read_geometry_with_counts(
                future_blocks,
                future_groups,
                future_offsets,
                future_masks,
                &future_counts,
            )?);
            self.anticipated_wide_geometry = Some(self.read_geometry_with_counts(
                wide_future_blocks,
                wide_future_groups,
                wide_future_offsets,
                wide_future_masks,
                &wide_future_counts,
            )?);
            self.anticipated_direct_geometry = Some(self.read_geometry_with_counts(
                direct_future_blocks,
                direct_future_groups,
                direct_future_offsets,
                direct_future_masks,
                &direct_future_counts,
            )?);
            self.anticipated_direct_next_geometry = Some(self.read_geometry_with_counts(
                direct_next_blocks,
                direct_next_groups,
                direct_next_offsets,
                direct_next_masks,
                &direct_next_counts,
            )?);
            self.deferred_bootstrap_challenge = Some(deferred_challenge);
            self.rounds_bound = new_rounds_bound;
            let resident_bytes = self.resident_bytes();
            return Ok(RegistersReadWriteCycleObservation {
                quadratic,
                wall,
                gpu_active,
                prefill_gpu_active: Duration::ZERO,
                allocation,
                #[cfg(feature = "test-utils")]
                live_entries: 0,
                resident_bytes,
                peak_transition_bytes: peak_transition_bytes.max(resident_bytes),
                retired_source_bytes: 0,
            });
        }
        if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN && new_rounds_bound == 2 {
            let active_work_items = self.physical_rows.div_ceil(8);
            let active_groups = active_work_items.div_ceil(self.threads);
            let geometry = geometry
                .as_ref()
                .ok_or(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write stateless replay is missing geometry",
                ))?;
            let future_geometry = self.anticipated_geometry.take().ok_or(
                MetalError::InvalidRegistersReadWriteState(
                    "registers read-write stateless replay is missing future geometry",
                ),
            )?;
            if future_geometry.blocks != work_items {
                return Err(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write future geometry has the wrong block count",
                ));
            }
            let future_state = IndexedStateBuffers::new(
                &self.context,
                future_geometry,
                MTLResourceOptions::StorageModePrivate,
                self.private_buffer_pool_epoch,
            )?;
            let allocation = allocation_started.elapsed();
            let peak_transition_bytes = self
                .base_resident_bytes()
                .checked_add(geometry.bytes())
                .and_then(|bytes| bytes.checked_add(future_state.bytes))
                .ok_or(MetalError::InputTooLong(work_items))?;
            self.context.validate_additional_working_set(
                u64::try_from(peak_transition_bytes)
                    .map_err(|_| MetalError::InputTooLong(peak_transition_bytes))?,
            )?;
            let params = SequenceParams {
                row_count: checked_u32(self.physical_rows)?,
                input_blocks: checked_u32(input_blocks)?,
                output_blocks: checked_u32(output_blocks)?,
                input_capacity: 0,
                output_capacity: 0,
                work_items: checked_u32(active_work_items)?,
                output_stride: checked_u32(active_groups)?,
                e_in_length: checked_u32(e_in.len())?,
                ra_lut_bits,
                wa_lut_bits,
                emit_message: 1,
                reserved: 1,
                source_stride: checked_u32(self.cycles)?,
            };
            let deferred_challenge = challenge;
            let (quadratic, wall, gpu_active, prefill_gpu_active) = self
                .submit_stateless_replay_bootstrap(
                    challenge,
                    params,
                    active_groups,
                    &future_state,
                )?;
            self.anticipated_indexed = Some(future_state);
            self.deferred_replay_challenge = Some(deferred_challenge);
            self.rounds_bound = new_rounds_bound;
            let resident_bytes = self.resident_bytes();
            return Ok(RegistersReadWriteCycleObservation {
                quadratic,
                wall,
                gpu_active,
                prefill_gpu_active,
                allocation,
                #[cfg(feature = "test-utils")]
                live_entries: 0,
                resident_bytes,
                peak_transition_bytes: peak_transition_bytes.max(resident_bytes),
                retired_source_bytes: 0,
            });
        }
        let output_state = if let Some(output) = anticipated_output {
            if output.blocks() != output_blocks {
                return Err(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write anticipated state has the wrong block count",
                ));
            }
            output
        } else {
            let geometry = geometry.ok_or(MetalError::InvalidRegistersReadWriteState(
                "registers read-write sequence is missing output geometry",
            ))?;
            if new_rounds_bound < DIRECT_TRANSITION_BOUND_ROUNDS {
                CycleState::Indexed(IndexedStateBuffers::new_reusing(
                    &self.context,
                    geometry,
                    MTLResourceOptions::StorageModePrivate,
                    self.spare_indexed.take(),
                    self.private_buffer_pool_epoch,
                )?)
            } else if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN
                && new_rounds_bound == WIDE_INDEXED_BOUND_ROUND
            {
                let reusable_common = self.spare_indexed.take().map(|indexed| indexed.common);
                self.spare_common = None;
                CycleState::WideIndexed(WideIndexedStateBuffers::new_reusing_common(
                    &self.context,
                    geometry,
                    MTLResourceOptions::StorageModePrivate,
                    reusable_common,
                    self.private_buffer_pool_epoch,
                )?)
            } else {
                let reusable = self.spare_direct.take();
                let reusable_common =
                    if reusable.is_none() && self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN {
                        self.spare_common
                            .take()
                            .or_else(|| self.spare_indexed.take().map(|indexed| indexed.common))
                    } else {
                        None
                    };
                self.spare_indexed = None;
                self.spare_common = None;
                CycleState::Direct(DirectStateBuffers::new_reusing(
                    &self.context,
                    geometry,
                    MTLResourceOptions::StorageModePrivate,
                    reusable,
                    reusable_common,
                    operand_carry_element_bytes(self.log_t, self.stage1_source, new_rounds_bound),
                    self.private_buffer_pool_epoch,
                )?)
            }
        };
        let future_wide =
            if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN && new_rounds_bound == 3 {
                let future_geometry = self.anticipated_wide_geometry.take().ok_or(
                    MetalError::InvalidRegistersReadWriteState(
                        "registers read-write sequence is missing future wide geometry",
                    ),
                )?;
                if future_geometry.blocks != work_items {
                    return Err(MetalError::InvalidRegistersReadWriteState(
                        "registers read-write future wide geometry has the wrong block count",
                    ));
                }
                Some(WideIndexedStateBuffers::new_reusing_common(
                    &self.context,
                    future_geometry,
                    MTLResourceOptions::StorageModePrivate,
                    None,
                    self.private_buffer_pool_epoch,
                )?)
            } else {
                None
            };
        let future_direct = if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN {
            let future_geometry = match new_rounds_bound {
                4 => self.anticipated_direct_geometry.take(),
                5 => self.anticipated_direct_next_geometry.take(),
                _ => None,
            };
            match future_geometry {
                Some(future_geometry) => {
                    if future_geometry.blocks != work_items {
                        return Err(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write future direct geometry has the wrong block count",
                        ));
                    }
                    Some(DirectPlaneBuffers::new(
                        &self.context,
                        future_geometry,
                        MTLResourceOptions::StorageModePrivate,
                        operand_carry_element_bytes(
                            self.log_t,
                            self.stage1_source,
                            new_rounds_bound + 1,
                        ),
                        self.private_buffer_pool_epoch,
                    )?)
                }
                None => None,
            }
        } else {
            None
        };
        let next_offsets = new_buffer::<u16>(
            &self.context,
            work_items,
            MTLResourceOptions::StorageModePrivate,
        )?;
        let next_masks = new_buffer::<u64>(
            &self.context,
            work_items,
            MTLResourceOptions::StorageModePrivate,
        )?;
        let allocation = allocation_started.elapsed();
        let input_bytes = self.state.as_ref().map_or(0, CycleState::bytes);
        let peak_transition_bytes = self
            .base_resident_bytes()
            .checked_add(input_bytes)
            .and_then(|bytes| bytes.checked_add(output_state.bytes()))
            .and_then(|bytes| bytes.checked_add(next_offsets.length() as usize))
            .and_then(|bytes| bytes.checked_add(next_masks.length() as usize))
            .and_then(|bytes| bytes.checked_add(self.spare_bytes()))
            .and_then(|bytes| {
                bytes.checked_add(future_wide.as_ref().map_or(0, |state| state.bytes))
            })
            .and_then(|bytes| {
                bytes.checked_add(future_direct.as_ref().map_or(0, |state| state.bytes))
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    self.anticipated_direct_geometry
                        .as_ref()
                        .map_or(0, StateGeometry::bytes)
                        + self
                            .anticipated_direct_next_geometry
                            .as_ref()
                            .map_or(0, StateGeometry::bytes),
                )
            })
            .ok_or(MetalError::InputTooLong(output_state.bytes()))?;
        self.context.validate_additional_working_set(
            u64::try_from(peak_transition_bytes)
                .map_err(|_| MetalError::InputTooLong(peak_transition_bytes))?,
        )?;
        let (dispatch_output_blocks, dispatch_work_items, dispatch_groups) = if self.log_t
            >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN
            && (3..=14).contains(&new_rounds_bound)
        {
            let rows_per_output = 1usize << new_rounds_bound;
            let output_blocks = self.physical_rows.div_ceil(rows_per_output);
            let work_items = self.physical_rows.div_ceil(2 * rows_per_output);
            (output_blocks, work_items, work_items.div_ceil(self.threads))
        } else {
            (output_blocks, work_items, groups)
        };
        let params = SequenceParams {
            row_count: checked_u32(self.physical_rows)?,
            input_blocks: checked_u32(input_blocks)?,
            output_blocks: checked_u32(dispatch_output_blocks)?,
            input_capacity: 0,
            output_capacity: 0,
            work_items: checked_u32(dispatch_work_items)?,
            output_stride: checked_u32(match (&self.state, &output_state) {
                (Some(CycleState::Indexed(_) | CycleState::WideIndexed(_)), _)
                    if dispatch_work_items >= SPARSE_COOPERATIVE_WORK_ITEMS_MIN =>
                {
                    dispatch_work_items.div_ceil(SPARSE_COOPERATIVE_WORK_ITEMS_PER_GROUP)
                }
                (Some(CycleState::Direct(_)), CycleState::Direct(_))
                    if new_rounds_bound > DIRECT_TRANSITION_BOUND_ROUNDS
                        && dispatch_work_items >= direct_cooperative_work_items_min(self.log_t) =>
                {
                    dispatch_work_items.div_ceil(DIRECT_COOPERATIVE_WORK_ITEMS_PER_GROUP)
                }
                _ => dispatch_groups,
            })?,
            e_in_length: checked_u32(e_in.len())?,
            ra_lut_bits,
            wa_lut_bits,
            source_stride: checked_u32(self.cycles)?,
            emit_message: 1,
            reserved: operand_carry_kind(self.log_t, self.stage1_source, new_rounds_bound),
        };
        let deferred_challenge = challenge;
        let prefill = future_wide
            .as_ref()
            .map(PrefillTarget::Wide)
            .or_else(|| future_direct.as_ref().map(PrefillTarget::Direct));
        if self.log_t >= ASYNC_SOURCE_RETIREMENT_LOG_T_MIN && new_rounds_bound == 4 {
            self.start_source_retirement()?;
        }
        let (quadratic, wall, gpu_active, prefill_gpu_active) = self.submit_bind(
            challenge,
            params,
            &output_state,
            dispatch_groups,
            work_items,
            (5..=14).contains(&new_rounds_bound),
            true,
            Some(&next_offsets),
            Some(&next_masks),
            prefill,
        )?;
        self.pending_geometry =
            Some(self.read_geometry(work_items, groups, next_offsets, next_masks)?);
        if let Some(future_wide) = future_wide {
            self.anticipated_wide_indexed = Some(future_wide);
        }
        if let Some(future_direct) = future_direct {
            match new_rounds_bound {
                4 => self.anticipated_direct = Some(future_direct),
                5 => self.anticipated_direct_next = Some(future_direct),
                _ => {
                    return Err(MetalError::InvalidRegistersReadWriteState(
                        "registers read-write direct prefill was scheduled at the wrong round",
                    ));
                }
            }
        }
        if matches!(&output_state, CycleState::WideIndexed(_)) {
            self.deferred_wide_challenge = Some(deferred_challenge);
        } else if matches!(self.state, Some(CycleState::WideIndexed(_))) {
            self.deferred_wide_challenge = None;
        }
        if new_rounds_bound == 3 {
            self.deferred_bootstrap_challenge = None;
            self.deferred_replay_challenge = None;
        }
        self.install_state(output_state);
        self.rounds_bound = new_rounds_bound;
        let retired_source_bytes = if new_rounds_bound == 3 {
            self.retire_source()?
        } else {
            0
        };
        let resident_bytes = self.resident_bytes();
        let peak_transition_bytes = peak_transition_bytes.max(resident_bytes);
        Ok(RegistersReadWriteCycleObservation {
            quadratic,
            wall,
            gpu_active,
            prefill_gpu_active,
            allocation,
            #[cfg(feature = "test-utils")]
            live_entries: 0,
            resident_bytes,
            peak_transition_bytes,
            retired_source_bytes,
        })
    }

    pub(crate) fn finish(
        &mut self,
        challenge: AkitaField,
    ) -> Result<RegistersReadWriteCycleFinish, MetalError> {
        if let Some(retirement) = self.source_retirement.take() {
            let join = retirement.join()?;
            tracing::info!(
                target: "jolt::metal",
                join_ns = u64::try_from(join.as_nanos()).unwrap_or(u64::MAX),
                "joined registers read-write source retirement"
            );
        }
        if !self.initial_message_emitted || self.rounds_bound + 1 != self.log_t {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write finish needs the final cycle challenge",
            ));
        }
        let input_blocks = self.cycles >> self.rounds_bound;
        let output_blocks = input_blocks / 2;
        if output_blocks != 1 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write cycle finish did not reach one block",
            ));
        }
        let allocation_started = Instant::now();
        let geometry =
            self.pending_geometry
                .take()
                .ok_or(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write finish is missing anticipated geometry",
                ))?;
        if geometry.blocks != output_blocks {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write final geometry has the wrong block count",
            ));
        }
        self.spare_indexed = None;
        self.spare_common = None;
        self.spare_direct = None;
        let output_state = CycleState::Direct(DirectStateBuffers::new(
            &self.context,
            geometry,
            MTLResourceOptions::StorageModeShared,
            operand_carry_element_bytes(self.log_t, self.stage1_source, self.log_t),
            self.private_buffer_pool_epoch,
        )?);
        let allocation = allocation_started.elapsed();
        let input_bytes = self.state.as_ref().map_or(0, CycleState::bytes);
        let peak_transition_bytes = self
            .base_resident_bytes()
            .checked_add(input_bytes)
            .and_then(|bytes| bytes.checked_add(output_state.bytes()))
            .ok_or(MetalError::InputTooLong(output_state.bytes()))?;
        let params = SequenceParams {
            row_count: checked_u32(self.physical_rows)?,
            input_blocks: checked_u32(input_blocks)?,
            output_blocks: 1,
            input_capacity: 0,
            output_capacity: 0,
            work_items: 1,
            output_stride: 1,
            e_in_length: 0,
            ra_lut_bits: 0,
            wa_lut_bits: 0,
            emit_message: 0,
            reserved: operand_carry_kind(self.log_t, self.stage1_source, self.log_t),
            source_stride: checked_u32(self.cycles)?,
        };
        let (_, wall, gpu_active, _) = self.submit_bind(
            challenge,
            params,
            &output_state,
            1,
            1,
            false,
            false,
            None,
            None,
            None,
        )?;
        self.state = Some(output_state);
        self.rounds_bound += 1;
        let (roots, increment) = self.read_final_state()?;
        let resident_bytes = self.resident_bytes();
        let peak_transition_bytes = peak_transition_bytes.max(resident_bytes);
        Ok(RegistersReadWriteCycleFinish {
            roots,
            increment,
            wall,
            gpu_active,
            allocation,
            resident_bytes,
            peak_transition_bytes,
        })
    }

    pub(crate) fn operand_claims(
        &self,
        cycle_point: &[AkitaField],
        address_point: &[AkitaField],
    ) -> Result<RegistersReadWriteOperandClaimsObservation, MetalError> {
        if self.rounds_bound != self.log_t
            || cycle_point.len() != self.log_t
            || address_point.is_empty()
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write operand claims need the fully bound cycle state",
            ));
        }
        let prepare_started = Instant::now();
        let (combined_claim, carried_rs1_claim) = self.final_operand_claims(address_point)?;
        let gamma_sq_inv = (self.gamma * self.gamma).inverse();
        if let (Some(rs1_claim), Some(gamma_sq_inv)) = (carried_rs1_claim, gamma_sq_inv) {
            let rs2_claim = (combined_claim - self.gamma * rs1_claim) * gamma_sq_inv;
            return Ok(RegistersReadWriteOperandClaimsObservation {
                claims: [rs1_claim, rs2_claim],
                prepare: prepare_started.elapsed(),
                wall: Duration::ZERO,
                gpu_active: Duration::ZERO,
            });
        }
        let compact_rs1_source = gamma_sq_inv.and(self.compact_rs1_source.as_ref());
        let joint_point = cycle_point
            .iter()
            .chain(address_point)
            .copied()
            .collect::<Vec<_>>();
        let high_bits = core::cmp::min(self.log_t, joint_point.len().div_ceil(2));
        let low_cycle_bits = joint_point
            .len()
            .checked_sub(high_bits + address_point.len())
            .ok_or(MetalError::InvalidRegistersReadWriteState(
                "registers read-write operand-claim equality split is invalid",
            ))?;
        let cycles_per_high_block = 1usize
            .checked_shl(
                u32::try_from(low_cycle_bits)
                    .map_err(|_| MetalError::InputTooLong(low_cycle_bits))?,
            )
            .ok_or(MetalError::InputTooLong(low_cycle_bits))?;
        let high_evals = EqPolynomial::<AkitaField>::evals(&joint_point[..high_bits], None);
        let low_evals = EqPolynomial::<AkitaField>::evals(&joint_point[high_bits..], None);
        let high_blocks = self.physical_rows.div_ceil(cycles_per_high_block);
        let high_evals =
            high_evals
                .get(..high_blocks)
                .ok_or(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write operand-claim high table is too short",
                ))?;
        let claim_columns = if compact_rs1_source.is_some() { 1 } else { 2 };
        let partial_fields = high_blocks
            .checked_mul(claim_columns)
            .ok_or(MetalError::InputTooLong(high_blocks))?;
        if partial_fields
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(partial_fields))?
            > self.scratch.partial_a.length() as usize
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write operand claims exceed sequence scratch",
            ));
        }
        write_fields(&self.scratch.e_in, high_evals)?;
        write_fields(&self.scratch.e_out, &low_evals)?;
        let params = OperandClaimsParams {
            row_count: checked_u32(self.physical_rows)?,
            cycles_per_high_block: checked_u32(cycles_per_high_block)?,
            address_bits: checked_u32(address_point.len())?,
            output_stride: checked_u32(high_blocks)?,
            remap_indices: u32::from(self.stage1_source && self.remap_registers),
        };
        let prepare = prepare_started.elapsed();
        let (claims, wall, gpu_active) = autoreleasepool(|| {
            let command = self.context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            let (pipeline, source) = if let Some(source) = compact_rs1_source {
                (&self.pipelines.compact_rs1_claim, source)
            } else {
                (&self.pipelines.operand_claims, self.source()?)
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(source), 0);
            encoder.set_buffer(1, Some(&self.scratch.e_in), 0);
            encoder.set_buffer(2, Some(&self.scratch.e_out), 0);
            encoder.set_buffer(3, Some(&self.scratch.partial_a), 0);
            set_inline_bytes(encoder, 4, &params);
            if compact_rs1_source.is_some() {
                encoder.set_buffer(5, Some(&self.register_map), 0);
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: high_blocks as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.scratch.partial_a,
                &self.scratch.partial_b,
                high_blocks,
                claim_columns,
                REGISTERS_READ_WRITE_SIMD_WIDTH,
            )?;
            encoder.end_encoding();
            let started = Instant::now();
            command.commit();
            command.wait_until_completed();
            let wall = started.elapsed();
            let gpu_active = completed_command_gpu_time(command)?;
            let output = if final_in_a {
                &self.scratch.partial_a
            } else {
                &self.scratch.partial_b
            };
            let claims = if let (Some(gamma_sq_inv), Some(_)) = (gamma_sq_inv, compact_rs1_source) {
                let rs1 = buffer_slice::<Fp128>(output, 1);
                self.context
                    .validate_inputs("registers read-write compact rs1", rs1)?;
                let rs1_claim = rs1
                    .first()
                    .copied()
                    .ok_or(MetalError::InvalidRegistersReadWriteState(
                        "registers read-write compact rs1 output is empty",
                    ))?
                    .into_jolt_field();
                let rs2_claim = (combined_claim - self.gamma * rs1_claim) * gamma_sq_inv;
                [rs1_claim, rs2_claim]
            } else {
                read_quadratic(&self.context, output)?
            };
            Ok::<_, MetalError>((claims, wall, gpu_active))
        })?;
        Ok(RegistersReadWriteOperandClaimsObservation {
            claims,
            prepare,
            wall,
            gpu_active,
        })
    }

    fn final_operand_claims(
        &self,
        address_point: &[AkitaField],
    ) -> Result<(AkitaField, Option<AkitaField>), MetalError> {
        let Some(CycleState::Direct(state)) = self.state.as_ref() else {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write final cycle state is not direct",
            ));
        };
        if state.common.blocks != 1 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write final cycle state has more than one block",
            ));
        }
        let length = buffer_slice::<u8>(&state.common.lengths, 1)[0] as usize;
        if length > MAX_REGISTER_BLOCK_CAPACITY {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write final cycle state exceeds the register domain",
            ));
        }
        let columns = buffer_slice::<u8>(&state.common.columns, length);
        let combined_values = buffer_slice::<Fp128>(&state.ra, length);
        self.context
            .validate_inputs("registers read-write combined ra", combined_values)?;
        let carried_rs1_values = state
            .operand
            .as_ref()
            .map(|operand| buffer_slice::<Fp128>(operand, length));
        if let Some(values) = carried_rs1_values {
            self.context
                .validate_inputs("registers read-write carried rs1", values)?;
        }
        let address_evals = EqPolynomial::<AkitaField>::evals(address_point, None);
        let mut combined_claim = AkitaField::zero();
        let mut carried_rs1_claim = carried_rs1_values.map(|_| AkitaField::zero());
        for index in 0..length {
            let column = usize::from(self.original_register_column(columns[index])?);
            let weight =
                address_evals
                    .get(column)
                    .ok_or(MetalError::InvalidRegistersReadWriteState(
                        "registers read-write final column is outside the address domain",
                    ))?;
            let combined_value: AkitaField = combined_values[index].into_jolt_field();
            combined_claim += *weight * combined_value;
            if let (Some(values), Some(claim)) = (carried_rs1_values, carried_rs1_claim.as_mut()) {
                let rs1_value: AkitaField = values[index].into_jolt_field();
                *claim += *weight * rs1_value;
            }
        }
        Ok((combined_claim, carried_rs1_claim))
    }

    #[cfg(feature = "test-utils")]
    pub(crate) const fn limits(&self) -> PipelineLimits {
        self.limits
    }

    #[cfg(feature = "test-utils")]
    pub(crate) const fn threads(&self) -> usize {
        self.threads
    }

    pub(crate) const fn source_bytes(&self) -> usize {
        self.source_bytes
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.base_resident_bytes()
            + self.state.as_ref().map_or(0, CycleState::bytes)
            + self.spare_bytes()
            + self
                .pending_geometry
                .as_ref()
                .map_or(0, StateGeometry::bytes)
            + self
                .anticipated_geometry
                .as_ref()
                .map_or(0, StateGeometry::bytes)
            + self
                .anticipated_indexed
                .as_ref()
                .map_or(0, |state| state.bytes)
            + self
                .anticipated_wide_geometry
                .as_ref()
                .map_or(0, StateGeometry::bytes)
            + self
                .anticipated_wide_indexed
                .as_ref()
                .map_or(0, |state| state.bytes)
            + self
                .anticipated_direct_geometry
                .as_ref()
                .map_or(0, StateGeometry::bytes)
            + self
                .anticipated_direct_next_geometry
                .as_ref()
                .map_or(0, StateGeometry::bytes)
            + self
                .anticipated_direct
                .as_ref()
                .map_or(0, |state| state.bytes)
            + self
                .anticipated_direct_next
                .as_ref()
                .map_or(0, |state| state.bytes)
    }

    fn base_resident_bytes(&self) -> usize {
        self.source.as_ref().map_or(0, |_| self.source_bytes)
            + self.compact_rs1_indices_bytes
            + self.scratch.bytes
            + self.register_map.length() as usize
    }

    fn original_register_column(&self, column: u8) -> Result<u8, MetalError> {
        if !self.remap_registers {
            return Ok(column);
        }
        self.register_unmap.get(usize::from(column)).copied().ok_or(
            MetalError::InvalidRegistersReadWriteState(
                "registers read-write dense column is outside the active register map",
            ),
        )
    }

    fn spare_bytes(&self) -> usize {
        self.spare_indexed.as_ref().map_or(0, |state| state.bytes)
            + self.spare_common.as_ref().map_or(0, |state| state.bytes)
            + self.spare_direct.as_ref().map_or(0, |state| state.bytes)
    }

    fn install_state(&mut self, output: CycleState) {
        let output_is_indexed = matches!(&output, CycleState::Indexed(_));
        let output_is_wide = matches!(&output, CycleState::WideIndexed(_));
        let output_is_direct = matches!(&output, CycleState::Direct(_));
        let previous = self.state.replace(output);
        match previous {
            Some(CycleState::Indexed(state)) if output_is_indexed => {
                self.spare_indexed = Some(state);
            }
            Some(CycleState::Indexed(state))
                if self.log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN =>
            {
                self.spare_common = Some(state.common);
            }
            Some(CycleState::WideIndexed(state)) if output_is_direct => {
                self.spare_common = Some(state.common);
            }
            Some(CycleState::Direct(state)) if output_is_direct => {
                self.spare_direct = Some(state);
            }
            _ => {}
        }
        if output_is_indexed {
            self.spare_common = None;
            self.spare_direct = None;
        } else if output_is_wide {
            self.spare_indexed = None;
            self.spare_direct = None;
        } else {
            self.spare_indexed = None;
        }
    }

    fn read_geometry(
        &self,
        blocks: usize,
        groups: usize,
        offsets: Buffer,
        masks: Buffer,
    ) -> Result<StateGeometry, MetalError> {
        self.read_geometry_with_counts(
            blocks,
            groups,
            offsets,
            masks,
            &self.scratch.geometry_counts,
        )
    }

    fn read_geometry_with_counts(
        &self,
        blocks: usize,
        groups: usize,
        offsets: Buffer,
        masks: Buffer,
        counts_buffer: &Buffer,
    ) -> Result<StateGeometry, MetalError> {
        if groups != blocks.div_ceil(self.threads) {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write geometry has the wrong group count",
            ));
        }
        let offset_bytes = blocks
            .checked_mul(size_of::<u16>())
            .ok_or(MetalError::InputTooLong(blocks))?;
        if offsets.length() < offset_bytes as u64 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write geometry offset buffer is too short",
            ));
        }
        let mask_bytes = blocks
            .checked_mul(size_of::<u64>())
            .ok_or(MetalError::InputTooLong(blocks))?;
        if masks.length() < mask_bytes as u64 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write geometry mask buffer is too short",
            ));
        }
        let count_bytes = groups
            .checked_mul(size_of::<u32>())
            .ok_or(MetalError::InputTooLong(groups))?;
        if counts_buffer.length() < count_bytes as u64 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write geometry count buffer is too short",
            ));
        }
        let counts = buffer_slice::<u32>(counts_buffer, groups);
        let mut tile_bases = Vec::with_capacity(groups);
        let mut slots = 0u64;
        for &count in counts {
            tile_bases.push(u32::try_from(slots).map_err(|_| {
                MetalError::InvalidRegistersReadWriteState(
                    "registers read-write live state exceeds u32 indexing",
                )
            })?);
            if count > u32::from(u16::MAX) {
                return Err(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write tile offset exceeds u16",
                ));
            }
            slots = slots
                .checked_add(u64::from(count))
                .ok_or(MetalError::InputTooLong(blocks))?;
        }
        let slots = usize::try_from(slots).map_err(|_| MetalError::InputTooLong(blocks))?;
        Ok(StateGeometry {
            blocks,
            slots,
            tile_bases,
            offsets,
            masks,
        })
    }

    fn write_weights(
        &self,
        rounds_bound: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(), MetalError> {
        if e_in.is_empty()
            || e_out.is_empty()
            || e_in.len() * e_out.len() != 1usize << (self.log_t - rounds_bound - 1)
        {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write equality weights have the wrong domain",
            ));
        }
        write_fields(&self.scratch.e_in, e_in)?;
        write_fields(&self.scratch.e_out, e_out)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the arguments mirror four stateless prefill levels with distinct buffers"
    )]
    fn submit_stateless_bootstrap(
        &self,
        challenge: AkitaField,
        params: SequenceParams,
        groups: usize,
        geometry_offsets: &Buffer,
        geometry_masks: &Buffer,
        future_params: SequenceParams,
        future_groups: usize,
        future_counts: &Buffer,
        future_offsets: &Buffer,
        future_masks: &Buffer,
        wide_future_params: SequenceParams,
        wide_future_groups: usize,
        wide_future_counts: &Buffer,
        wide_future_offsets: &Buffer,
        wide_future_masks: &Buffer,
        direct_future_params: SequenceParams,
        direct_future_groups: usize,
        direct_future_counts: &Buffer,
        direct_future_offsets: &Buffer,
        direct_future_masks: &Buffer,
        direct_next_params: SequenceParams,
        direct_next_groups: usize,
        direct_next_counts: &Buffer,
        direct_next_offsets: &Buffer,
        direct_next_masks: &Buffer,
    ) -> Result<([AkitaField; 2], Duration, Duration), MetalError> {
        let geometry_blocks = params.output_blocks as usize / 2;
        let geometry_groups = geometry_blocks.div_ceil(self.threads);
        let active_blocks = params.work_items as usize;
        let zero_suffix = if active_blocks < geometry_blocks {
            let active_offset_bytes = active_blocks
                .checked_mul(size_of::<u16>())
                .ok_or(MetalError::InputTooLong(active_blocks))?;
            let offset_suffix_bytes = geometry_blocks
                .checked_sub(active_blocks)
                .and_then(|blocks| blocks.checked_mul(size_of::<u16>()))
                .ok_or(MetalError::InputTooLong(geometry_blocks))?;
            let active_mask_bytes = active_blocks
                .checked_mul(size_of::<u64>())
                .ok_or(MetalError::InputTooLong(active_blocks))?;
            let mask_suffix_bytes = geometry_blocks
                .checked_sub(active_blocks)
                .and_then(|blocks| blocks.checked_mul(size_of::<u64>()))
                .ok_or(MetalError::InputTooLong(geometry_blocks))?;
            let active_count_bytes = groups
                .checked_mul(size_of::<u32>())
                .ok_or(MetalError::InputTooLong(groups))?;
            let count_suffix_bytes = geometry_groups
                .checked_sub(groups)
                .and_then(|groups| groups.checked_mul(size_of::<u32>()))
                .ok_or(MetalError::InputTooLong(geometry_groups))?;
            Some((
                NSRange::new(
                    checked_u64(active_offset_bytes)?,
                    checked_u64(offset_suffix_bytes)?,
                ),
                NSRange::new(
                    checked_u64(active_mask_bytes)?,
                    checked_u64(mask_suffix_bytes)?,
                ),
                NSRange::new(
                    checked_u64(active_count_bytes)?,
                    checked_u64(count_suffix_bytes)?,
                ),
            ))
        } else {
            None
        };
        let challenge = Fp128::from_jolt_field(&challenge);
        autoreleasepool(|| {
            let command = self.context.queue.new_command_buffer();
            if let Some((offsets, masks, counts)) = zero_suffix {
                let encoder = command.new_blit_command_encoder();
                encoder.fill_buffer(geometry_offsets, offsets, 0);
                encoder.fill_buffer(geometry_masks, masks, 0);
                encoder.fill_buffer(&self.scratch.geometry_counts, counts, 0);
                encoder.end_encoding();
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.stateless_bootstrap);
            self.bind_source(encoder)?;
            encoder.set_buffer(1, Some(&self.scratch.ra_lut), 0);
            encoder.set_buffer(2, Some(&self.scratch.wa_lut), 0);
            encoder.set_buffer(3, Some(&self.scratch.e_in), 0);
            encoder.set_buffer(4, Some(&self.scratch.e_out), 0);
            encoder.set_buffer(5, Some(&self.scratch.partial_a), 0);
            set_inline_bytes(encoder, 6, &challenge);
            set_inline_bytes(encoder, 7, &params);
            encoder.set_buffer(8, Some(&self.scratch.geometry_counts), 0);
            encoder.set_buffer(9, Some(geometry_offsets), 0);
            encoder.set_buffer(10, Some(geometry_masks), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_compute_pipeline_state(&self.pipelines.indexed_state_geometry);
            encoder.set_buffer(0, Some(geometry_masks), 0);
            encoder.set_buffer(1, Some(future_counts), 0);
            encoder.set_buffer(2, Some(future_offsets), 0);
            encoder.set_buffer(3, Some(future_masks), 0);
            set_inline_bytes(encoder, 4, &future_params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: future_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_buffer(0, Some(future_masks), 0);
            encoder.set_buffer(1, Some(wide_future_counts), 0);
            encoder.set_buffer(2, Some(wide_future_offsets), 0);
            encoder.set_buffer(3, Some(wide_future_masks), 0);
            set_inline_bytes(encoder, 4, &wide_future_params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: wide_future_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_buffer(0, Some(wide_future_masks), 0);
            encoder.set_buffer(1, Some(direct_future_counts), 0);
            encoder.set_buffer(2, Some(direct_future_offsets), 0);
            encoder.set_buffer(3, Some(direct_future_masks), 0);
            set_inline_bytes(encoder, 4, &direct_future_params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: direct_future_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_buffer(0, Some(direct_future_masks), 0);
            encoder.set_buffer(1, Some(direct_next_counts), 0);
            encoder.set_buffer(2, Some(direct_next_offsets), 0);
            encoder.set_buffer(3, Some(direct_next_masks), 0);
            set_inline_bytes(encoder, 4, &direct_next_params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: direct_next_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.scratch.partial_a,
                &self.scratch.partial_b,
                groups,
                2,
                REGISTERS_READ_WRITE_SIMD_WIDTH,
            )?;
            encoder.end_encoding();
            let started = Instant::now();
            command.commit();
            command.wait_until_completed();
            let wall = started.elapsed();
            let gpu_active = completed_command_gpu_time(command)?;
            let output = if final_in_a {
                &self.scratch.partial_a
            } else {
                &self.scratch.partial_b
            };
            Ok((read_quadratic(&self.context, output)?, wall, gpu_active))
        })
    }

    fn submit_stateless_replay_bootstrap(
        &self,
        challenge: AkitaField,
        params: SequenceParams,
        groups: usize,
        future_state: &IndexedStateBuffers,
    ) -> Result<([AkitaField; 2], Duration, Duration, Duration), MetalError> {
        let deferred_challenge = self.deferred_bootstrap_challenge.as_ref().ok_or(
            MetalError::InvalidRegistersReadWriteState(
                "registers read-write stateless replay is missing its first challenge",
            ),
        )?;
        let deferred_challenge = Fp128::from_jolt_field(deferred_challenge);
        let challenge = Fp128::from_jolt_field(&challenge);
        let prefill_queue =
            self.prefill_queue
                .as_ref()
                .ok_or(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write stateless replay is missing its prefill queue",
                ))?;
        autoreleasepool(|| {
            let prefill_buffers = [
                &future_state.common.lengths,
                &future_state.common.columns,
                &future_state.common.previous,
                &future_state.common.next,
                &future_state.common.values,
                &future_state.common.increments,
                &future_state.ra,
                &future_state.wa,
            ];
            let prefill_command = prefill_buffers
                .iter()
                .any(|buffer| !buffer.was_reused())
                .then(|| {
                    let command = prefill_queue.new_command_buffer();
                    let encoder = command.new_blit_command_encoder();
                    for buffer in prefill_buffers {
                        if !buffer.was_reused() {
                            encoder.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
                        }
                    }
                    encoder.end_encoding();
                    command
                });

            let command = self.context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.stateless_replay_bootstrap);
            self.bind_source(encoder)?;
            encoder.set_buffer(1, Some(&self.scratch.ra_lut), 0);
            encoder.set_buffer(2, Some(&self.scratch.wa_lut), 0);
            encoder.set_buffer(3, Some(&self.scratch.e_in), 0);
            encoder.set_buffer(4, Some(&self.scratch.e_out), 0);
            encoder.set_buffer(5, Some(&self.scratch.partial_a), 0);
            set_inline_bytes(encoder, 6, &deferred_challenge);
            set_inline_bytes(encoder, 7, &challenge);
            set_inline_bytes(encoder, 8, &params);
            encoder.set_buffer(9, Some(&self.scratch.geometry_counts), 0);
            encoder.set_buffer(10, Some(&future_state.common.offsets), 0);
            encoder.set_buffer(11, Some(&future_state.common.masks), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.scratch.partial_a,
                &self.scratch.partial_b,
                groups,
                2,
                REGISTERS_READ_WRITE_SIMD_WIDTH,
            )?;
            encoder.end_encoding();
            let started = Instant::now();
            if let Some(prefill_command) = prefill_command.as_ref() {
                prefill_command.commit();
            }
            command.commit();
            command.wait_until_completed();
            if let Some(prefill_command) = prefill_command.as_ref() {
                prefill_command.wait_until_completed();
            }
            let wall = started.elapsed();
            let main_gpu_active = completed_command_gpu_time(command)?;
            let prefill_gpu_active = prefill_command
                .as_ref()
                .map(|command| completed_command_gpu_time(command))
                .transpose()?
                .unwrap_or_default();
            let gpu_active = main_gpu_active + prefill_gpu_active;
            let output = if final_in_a {
                &self.scratch.partial_a
            } else {
                &self.scratch.partial_b
            };
            Ok((
                read_quadratic(&self.context, output)?,
                wall,
                gpu_active,
                prefill_gpu_active,
            ))
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the bind submission keeps state, geometry, and optional prefill dependencies explicit"
    )]
    fn submit_bind(
        &self,
        challenge: AkitaField,
        params: SequenceParams,
        output_state: &CycleState,
        groups: usize,
        logical_geometry_blocks: usize,
        clear_output_suffix: bool,
        emit_message: bool,
        geometry_offsets: Option<&Buffer>,
        geometry_masks: Option<&Buffer>,
        prefill: Option<PrefillTarget<'_>>,
    ) -> Result<([AkitaField; 2], Duration, Duration, Duration), MetalError> {
        let message_groups = params.output_stride as usize;
        let partial_capacity = self.scratch.partial_a.length() as usize / (2 * size_of::<Fp128>());
        if emit_message && message_groups > partial_capacity {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write partial scratch is too short",
            ));
        }
        let challenge = Fp128::from_jolt_field(&challenge);
        let output_common = match output_state {
            CycleState::Indexed(state) => &state.common,
            CycleState::WideIndexed(state) => &state.common,
            CycleState::Direct(state) => &state.common,
        };
        let output_offsets = &output_common.offsets;
        let geometry_offsets = geometry_offsets.unwrap_or(output_offsets);
        let output_masks = &output_common.masks;
        let geometry_masks = geometry_masks.unwrap_or(output_masks);
        let active_geometry_blocks = params.work_items as usize;
        let logical_geometry_groups = logical_geometry_blocks.div_ceil(self.threads);
        let zero_suffix = if active_geometry_blocks < logical_geometry_blocks {
            let active_offset_bytes = active_geometry_blocks
                .checked_mul(size_of::<u16>())
                .ok_or(MetalError::InputTooLong(active_geometry_blocks))?;
            let offset_suffix_bytes = logical_geometry_blocks
                .checked_sub(active_geometry_blocks)
                .and_then(|blocks| blocks.checked_mul(size_of::<u16>()))
                .ok_or(MetalError::InputTooLong(logical_geometry_blocks))?;
            let active_mask_bytes = active_geometry_blocks
                .checked_mul(size_of::<u64>())
                .ok_or(MetalError::InputTooLong(active_geometry_blocks))?;
            let mask_suffix_bytes = logical_geometry_blocks
                .checked_sub(active_geometry_blocks)
                .and_then(|blocks| blocks.checked_mul(size_of::<u64>()))
                .ok_or(MetalError::InputTooLong(logical_geometry_blocks))?;
            let active_count_bytes = groups
                .checked_mul(size_of::<u32>())
                .ok_or(MetalError::InputTooLong(groups))?;
            let count_suffix_bytes = logical_geometry_groups
                .checked_sub(groups)
                .and_then(|groups| groups.checked_mul(size_of::<u32>()))
                .ok_or(MetalError::InputTooLong(logical_geometry_groups))?;
            Some((
                NSRange::new(
                    checked_u64(active_offset_bytes)?,
                    checked_u64(offset_suffix_bytes)?,
                ),
                NSRange::new(
                    checked_u64(active_mask_bytes)?,
                    checked_u64(mask_suffix_bytes)?,
                ),
                NSRange::new(
                    checked_u64(active_count_bytes)?,
                    checked_u64(count_suffix_bytes)?,
                ),
            ))
        } else {
            None
        };
        let active_output_blocks = params.output_blocks as usize;
        let output_zero_suffix = if clear_output_suffix
            && active_output_blocks < output_common.blocks
        {
            let active_length_bytes = active_output_blocks
                .checked_mul(size_of::<u8>())
                .ok_or(MetalError::InputTooLong(active_output_blocks))?;
            let length_suffix_bytes = output_common
                .blocks
                .checked_sub(active_output_blocks)
                .and_then(|blocks| blocks.checked_mul(size_of::<u8>()))
                .ok_or(MetalError::InputTooLong(output_common.blocks))?;
            let active_increment_bytes = active_output_blocks
                .checked_mul(size_of::<Fp128>())
                .ok_or(MetalError::InputTooLong(active_output_blocks))?;
            let increment_suffix_bytes = output_common
                .blocks
                .checked_sub(active_output_blocks)
                .and_then(|blocks| blocks.checked_mul(size_of::<Fp128>()))
                .ok_or(MetalError::InputTooLong(output_common.blocks))?;
            Some((
                NSRange::new(
                    checked_u64(active_length_bytes)?,
                    checked_u64(length_suffix_bytes)?,
                ),
                NSRange::new(
                    checked_u64(active_increment_bytes)?,
                    checked_u64(increment_suffix_bytes)?,
                ),
            ))
        } else {
            None
        };
        let prefill = prefill.filter(|target| target.needs_prefill());
        let prefill_queue = match prefill {
            Some(_) => Some(self.prefill_queue.as_ref().ok_or(
                MetalError::InvalidRegistersReadWriteState(
                    "registers read-write bind is missing its prefill queue",
                ),
            )?),
            None => None,
        };
        autoreleasepool(|| {
            let prefill_command = match (prefill, prefill_queue) {
                (Some(target), Some(queue)) => {
                    let command = queue.new_command_buffer();
                    let encoder = command.new_blit_command_encoder();
                    target.encode(encoder);
                    encoder.end_encoding();
                    Some(command)
                }
                _ => None,
            };
            let command = self.context.queue.new_command_buffer();
            if zero_suffix.is_some() || output_zero_suffix.is_some() {
                let encoder = command.new_blit_command_encoder();
                if let Some((offsets, masks, counts)) = zero_suffix {
                    encoder.fill_buffer(geometry_offsets, offsets, 0);
                    encoder.fill_buffer(geometry_masks, masks, 0);
                    encoder.fill_buffer(&self.scratch.geometry_counts, counts, 0);
                }
                if let Some((lengths, increments)) = output_zero_suffix {
                    encoder.fill_buffer(&output_common.lengths, lengths, 0);
                    encoder.fill_buffer(&output_common.increments, increments, 0);
                }
                encoder.end_encoding();
            }
            let encoder = command.new_compute_command_encoder();
            match (self.state.as_ref(), output_state) {
                (None, CycleState::Indexed(output)) => {
                    if let (Some(first_challenge), Some(second_challenge)) = (
                        self.deferred_bootstrap_challenge.as_ref(),
                        self.deferred_replay_challenge.as_ref(),
                    ) {
                        let first_challenge = Fp128::from_jolt_field(first_challenge);
                        let second_challenge = Fp128::from_jolt_field(second_challenge);
                        encoder
                            .set_compute_pipeline_state(&self.pipelines.replay_three_materialize);
                        self.bind_source(encoder)?;
                        bind_common_output(encoder, 1, &output.common);
                        encoder.set_buffer(6, Some(&output.ra), 0);
                        encoder.set_buffer(7, Some(&output.wa), 0);
                        encoder.set_buffer(8, Some(&output.common.increments), 0);
                        set_inline_bytes(encoder, 9, &first_challenge);
                        set_inline_bytes(encoder, 10, &second_challenge);
                        set_inline_bytes(encoder, 11, &challenge);
                        set_inline_bytes(encoder, 12, &params);
                        encoder.set_buffer(13, Some(&output.common.offsets), 0);
                        encoder.set_buffer(14, Some(&output.common.tile_bases), 0);
                        let materialize_groups =
                            (params.output_blocks as usize).div_ceil(self.threads);
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: materialize_groups as u64,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: self.threads as u64,
                                height: 1,
                                depth: 1,
                            },
                        );

                        encoder.set_compute_pipeline_state(&self.pipelines.indexed_state_message);
                        encoder.set_buffer(0, Some(&output.common.masks), 0);
                        encoder.set_buffer(1, Some(&output.common.previous), 0);
                        encoder.set_buffer(2, Some(&output.common.next), 0);
                        encoder.set_buffer(3, Some(&output.common.values), 0);
                        encoder.set_buffer(4, Some(&output.ra), 0);
                        encoder.set_buffer(5, Some(&output.wa), 0);
                        encoder.set_buffer(6, Some(&output.common.increments), 0);
                        encoder.set_buffer(7, Some(&self.scratch.ra_lut), 0);
                        encoder.set_buffer(8, Some(&self.scratch.wa_lut), 0);
                        encoder.set_buffer(9, Some(&self.scratch.e_in), 0);
                        encoder.set_buffer(10, Some(&self.scratch.e_out), 0);
                        encoder.set_buffer(11, Some(&self.scratch.partial_a), 0);
                        set_inline_bytes(encoder, 12, &params);
                        encoder.set_buffer(13, Some(&output.common.offsets), 0);
                        encoder.set_buffer(14, Some(&self.scratch.geometry_counts), 0);
                        encoder.set_buffer(15, Some(&output.common.tile_bases), 0);
                        encoder.set_buffer(16, Some(geometry_offsets), 0);
                        encoder.set_buffer(17, Some(geometry_masks), 0);
                        encoder.set_buffer(18, Some(&output.common.lengths), 0);
                    } else if let Some(deferred_challenge) =
                        self.deferred_bootstrap_challenge.as_ref()
                    {
                        let deferred_challenge = Fp128::from_jolt_field(deferred_challenge);
                        encoder.set_compute_pipeline_state(&self.pipelines.replay_bootstrap);
                        self.bind_source(encoder)?;
                        bind_common_output(encoder, 1, &output.common);
                        encoder.set_buffer(6, Some(&output.ra), 0);
                        encoder.set_buffer(7, Some(&output.wa), 0);
                        encoder.set_buffer(8, Some(&output.common.increments), 0);
                        encoder.set_buffer(9, Some(&self.scratch.ra_lut), 0);
                        encoder.set_buffer(10, Some(&self.scratch.wa_lut), 0);
                        encoder.set_buffer(11, Some(&self.scratch.e_in), 0);
                        encoder.set_buffer(12, Some(&self.scratch.e_out), 0);
                        encoder.set_buffer(13, Some(&self.scratch.partial_a), 0);
                        set_inline_bytes(encoder, 14, &deferred_challenge);
                        set_inline_bytes(encoder, 15, &challenge);
                        set_inline_bytes(encoder, 16, &params);
                        encoder.set_buffer(17, Some(&output.common.offsets), 0);
                        encoder.set_buffer(18, Some(&self.scratch.geometry_counts), 0);
                        encoder.set_buffer(19, Some(&output.common.tile_bases), 0);
                        encoder.set_buffer(20, Some(geometry_offsets), 0);
                        encoder.set_buffer(21, Some(geometry_masks), 0);
                    } else {
                        encoder.set_compute_pipeline_state(&self.pipelines.bootstrap);
                        self.bind_source(encoder)?;
                        bind_common_output(encoder, 1, &output.common);
                        encoder.set_buffer(6, Some(&output.ra), 0);
                        encoder.set_buffer(7, Some(&output.wa), 0);
                        encoder.set_buffer(8, Some(&output.common.increments), 0);
                        encoder.set_buffer(9, Some(&self.scratch.ra_lut), 0);
                        encoder.set_buffer(10, Some(&self.scratch.wa_lut), 0);
                        encoder.set_buffer(11, Some(&self.scratch.e_in), 0);
                        encoder.set_buffer(12, Some(&self.scratch.e_out), 0);
                        encoder.set_buffer(13, Some(&self.scratch.partial_a), 0);
                        set_inline_bytes(encoder, 14, &challenge);
                        set_inline_bytes(encoder, 15, &params);
                        encoder.set_buffer(16, Some(&output.common.offsets), 0);
                        encoder.set_buffer(17, Some(&self.scratch.geometry_counts), 0);
                        encoder.set_buffer(18, Some(&output.common.tile_bases), 0);
                        encoder.set_buffer(19, Some(geometry_offsets), 0);
                        encoder.set_buffer(20, Some(geometry_masks), 0);
                    }
                }
                (Some(CycleState::Indexed(input)), CycleState::Indexed(output)) => {
                    let cooperative = emit_message
                        && params.work_items as usize >= SPARSE_COOPERATIVE_WORK_ITEMS_MIN;
                    if cooperative {
                        encoder.set_compute_pipeline_state(&self.pipelines.direct_geometry);
                        encoder.set_buffer(0, Some(&input.common.masks), 0);
                        encoder.set_buffer(1, Some(&self.scratch.geometry_counts), 0);
                        encoder.set_buffer(2, Some(geometry_offsets), 0);
                        encoder.set_buffer(3, Some(geometry_masks), 0);
                        set_inline_bytes(encoder, 4, &params);
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: groups as u64,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: self.threads as u64,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    let pipeline = if cooperative {
                        &self.pipelines.indexed_cooperative
                    } else {
                        &self.pipelines.indexed
                    };
                    encoder.set_compute_pipeline_state(pipeline);
                    bind_indexed_input(encoder, input);
                    bind_common_output(encoder, 8, &output.common);
                    encoder.set_buffer(13, Some(&output.ra), 0);
                    encoder.set_buffer(14, Some(&output.wa), 0);
                    encoder.set_buffer(15, Some(&output.common.increments), 0);
                    encoder.set_buffer(16, Some(&self.scratch.ra_lut), 0);
                    encoder.set_buffer(17, Some(&self.scratch.wa_lut), 0);
                    encoder.set_buffer(18, Some(&self.scratch.e_in), 0);
                    encoder.set_buffer(19, Some(&self.scratch.e_out), 0);
                    encoder.set_buffer(20, Some(&self.scratch.partial_a), 0);
                    set_inline_bytes(encoder, 21, &challenge);
                    set_inline_bytes(encoder, 22, &params);
                    encoder.set_buffer(23, Some(&input.common.offsets), 0);
                    encoder.set_buffer(24, Some(&output.common.offsets), 0);
                    encoder.set_buffer(25, Some(&self.scratch.geometry_counts), 0);
                    encoder.set_buffer(26, Some(&input.common.tile_bases), 0);
                    encoder.set_buffer(27, Some(&output.common.tile_bases), 0);
                    encoder.set_buffer(28, Some(geometry_offsets), 0);
                    encoder.set_buffer(29, Some(&input.common.masks), 0);
                    encoder.set_buffer(30, Some(geometry_masks), 0);
                }
                (Some(CycleState::Indexed(input)), CycleState::WideIndexed(output)) => {
                    if !emit_message
                        || (params.work_items as usize) < SPARSE_COOPERATIVE_WORK_ITEMS_MIN
                    {
                        return Err(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write wide indexed transition is too small",
                        ));
                    }
                    encoder.set_compute_pipeline_state(&self.pipelines.direct_geometry);
                    encoder.set_buffer(0, Some(&input.common.masks), 0);
                    encoder.set_buffer(1, Some(&self.scratch.geometry_counts), 0);
                    encoder.set_buffer(2, Some(geometry_offsets), 0);
                    encoder.set_buffer(3, Some(geometry_masks), 0);
                    set_inline_bytes(encoder, 4, &params);
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: groups as u64,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: self.threads as u64,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoder.set_compute_pipeline_state(&self.pipelines.wide_indexed_cooperative);
                    bind_indexed_input(encoder, input);
                    bind_common_output(encoder, 8, &output.common);
                    encoder.set_buffer(13, Some(&output.ra), 0);
                    encoder.set_buffer(14, Some(&output.wa), 0);
                    encoder.set_buffer(15, Some(&output.common.increments), 0);
                    encoder.set_buffer(16, Some(&self.scratch.ra_lut), 0);
                    encoder.set_buffer(17, Some(&self.scratch.wa_lut), 0);
                    encoder.set_buffer(18, Some(&self.scratch.e_in), 0);
                    encoder.set_buffer(19, Some(&self.scratch.e_out), 0);
                    encoder.set_buffer(20, Some(&self.scratch.partial_a), 0);
                    set_inline_bytes(encoder, 21, &challenge);
                    set_inline_bytes(encoder, 22, &params);
                    encoder.set_buffer(23, Some(&input.common.offsets), 0);
                    encoder.set_buffer(24, Some(&output.common.offsets), 0);
                    encoder.set_buffer(25, Some(&self.scratch.geometry_counts), 0);
                    encoder.set_buffer(26, Some(&input.common.tile_bases), 0);
                    encoder.set_buffer(27, Some(&output.common.tile_bases), 0);
                    encoder.set_buffer(28, Some(geometry_offsets), 0);
                    encoder.set_buffer(29, Some(&input.common.masks), 0);
                    encoder.set_buffer(30, Some(geometry_masks), 0);
                }
                (Some(CycleState::Indexed(input)), CycleState::Direct(output)) => {
                    let cooperative = emit_message
                        && params.work_items as usize >= SPARSE_COOPERATIVE_WORK_ITEMS_MIN;
                    if cooperative {
                        encoder.set_compute_pipeline_state(&self.pipelines.direct_geometry);
                        encoder.set_buffer(0, Some(&input.common.masks), 0);
                        encoder.set_buffer(1, Some(&self.scratch.geometry_counts), 0);
                        encoder.set_buffer(2, Some(geometry_offsets), 0);
                        encoder.set_buffer(3, Some(geometry_masks), 0);
                        set_inline_bytes(encoder, 4, &params);
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: groups as u64,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: self.threads as u64,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    let pipeline = if cooperative {
                        &self.pipelines.transition_cooperative
                    } else {
                        &self.pipelines.transition
                    };
                    encoder.set_compute_pipeline_state(pipeline);
                    bind_indexed_input(encoder, input);
                    bind_common_output(encoder, 8, &output.common);
                    encoder.set_buffer(13, Some(&output.ra), 0);
                    encoder.set_buffer(14, Some(&output.wa), 0);
                    encoder.set_buffer(15, Some(&output.common.increments), 0);
                    encoder.set_buffer(16, Some(&self.scratch.ra_lut), 0);
                    encoder.set_buffer(17, Some(&self.scratch.wa_lut), 0);
                    encoder.set_buffer(18, Some(&self.scratch.e_in), 0);
                    encoder.set_buffer(19, Some(&self.scratch.e_out), 0);
                    encoder.set_buffer(20, Some(&self.scratch.partial_a), 0);
                    set_inline_bytes(encoder, 21, &challenge);
                    set_inline_bytes(encoder, 22, &params);
                    encoder.set_buffer(23, Some(&input.common.offsets), 0);
                    encoder.set_buffer(24, Some(&output.common.offsets), 0);
                    encoder.set_buffer(25, Some(&self.scratch.geometry_counts), 0);
                    encoder.set_buffer(26, Some(&input.common.tile_bases), 0);
                    encoder.set_buffer(27, Some(&output.common.tile_bases), 0);
                    encoder.set_buffer(28, Some(geometry_offsets), 0);
                    encoder.set_buffer(29, Some(&input.common.masks), 0);
                    encoder.set_buffer(30, Some(geometry_masks), 0);
                }
                (Some(CycleState::WideIndexed(input)), CycleState::Direct(output)) => {
                    if !emit_message
                        || (params.work_items as usize) < SPARSE_COOPERATIVE_WORK_ITEMS_MIN
                    {
                        return Err(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write wide direct transition is too small",
                        ));
                    }
                    let deferred_challenge = self.deferred_wide_challenge.as_ref().ok_or(
                        MetalError::InvalidRegistersReadWriteState(
                            "registers read-write wide state is missing its challenge",
                        ),
                    )?;
                    let deferred_challenge = Fp128::from_jolt_field(deferred_challenge);
                    let output_operand = output.operand.as_deref().unwrap_or(&self.scratch.ra_lut);
                    encoder.set_compute_pipeline_state(&self.pipelines.direct_geometry);
                    encoder.set_buffer(0, Some(&input.common.masks), 0);
                    encoder.set_buffer(1, Some(&self.scratch.geometry_counts), 0);
                    encoder.set_buffer(2, Some(geometry_offsets), 0);
                    encoder.set_buffer(3, Some(geometry_masks), 0);
                    set_inline_bytes(encoder, 4, &params);
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: groups as u64,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: self.threads as u64,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoder.set_compute_pipeline_state(&self.pipelines.wide_transition_cooperative);
                    set_inline_bytes(encoder, 0, &deferred_challenge);
                    bind_wide_indexed_input(encoder, input);
                    bind_common_output(encoder, 8, &output.common);
                    encoder.set_buffer(13, Some(&output.ra), 0);
                    encoder.set_buffer(14, Some(&output.wa), 0);
                    encoder.set_buffer(15, Some(&output.common.increments), 0);
                    encoder.set_buffer(16, Some(&self.scratch.ra_lut), 0);
                    encoder.set_buffer(17, Some(&self.scratch.wa_lut), 0);
                    encoder.set_buffer(18, Some(&self.scratch.e_in), 0);
                    encoder.set_buffer(19, Some(&self.scratch.e_out), 0);
                    encoder.set_buffer(20, Some(&self.scratch.partial_a), 0);
                    set_inline_bytes(encoder, 21, &challenge);
                    set_inline_bytes(encoder, 22, &params);
                    encoder.set_buffer(23, Some(&input.common.offsets), 0);
                    encoder.set_buffer(24, Some(&output.common.offsets), 0);
                    encoder.set_buffer(25, Some(output_operand), 0);
                    encoder.set_buffer(26, Some(&input.common.tile_bases), 0);
                    encoder.set_buffer(27, Some(&output.common.tile_bases), 0);
                    encoder.set_buffer(28, Some(geometry_offsets), 0);
                    encoder.set_buffer(29, Some(&input.common.masks), 0);
                    encoder.set_buffer(30, Some(geometry_masks), 0);
                }
                (Some(CycleState::Direct(input)), CycleState::Direct(output)) => {
                    let cooperative = emit_message
                        && params.work_items as usize
                            >= direct_cooperative_work_items_min(self.log_t);
                    if cooperative {
                        encoder.set_compute_pipeline_state(&self.pipelines.direct_geometry);
                        encoder.set_buffer(0, Some(&input.common.masks), 0);
                        encoder.set_buffer(1, Some(&self.scratch.geometry_counts), 0);
                        encoder.set_buffer(2, Some(geometry_offsets), 0);
                        encoder.set_buffer(3, Some(geometry_masks), 0);
                        set_inline_bytes(encoder, 4, &params);
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: groups as u64,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: self.threads as u64,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    let pipelines = if cooperative {
                        &self.pipelines.direct_cooperative
                    } else {
                        &self.pipelines.direct
                    };
                    let pipeline = pipelines
                        .get(params.reserved as usize)
                        .and_then(Option::as_ref)
                        .ok_or(MetalError::InvalidRegistersReadWriteState(
                            "registers read-write operand carry kind is unsupported",
                        ))?;
                    encoder.set_compute_pipeline_state(pipeline);
                    bind_direct_input(encoder, input);
                    bind_common_output(encoder, 8, &output.common);
                    encoder.set_buffer(13, Some(&output.ra), 0);
                    encoder.set_buffer(14, Some(&output.wa), 0);
                    encoder.set_buffer(15, Some(&output.common.increments), 0);
                    encoder.set_buffer(16, Some(&self.scratch.e_in), 0);
                    encoder.set_buffer(17, Some(&self.scratch.e_out), 0);
                    encoder.set_buffer(18, Some(&self.scratch.partial_a), 0);
                    set_inline_bytes(encoder, 19, &challenge);
                    set_inline_bytes(encoder, 20, &params);
                    encoder.set_buffer(21, Some(&input.common.offsets), 0);
                    encoder.set_buffer(22, Some(&output.common.offsets), 0);
                    encoder.set_buffer(23, Some(&self.scratch.geometry_counts), 0);
                    encoder.set_buffer(24, Some(&input.common.tile_bases), 0);
                    encoder.set_buffer(25, Some(&output.common.tile_bases), 0);
                    encoder.set_buffer(26, Some(geometry_offsets), 0);
                    encoder.set_buffer(27, Some(&input.common.masks), 0);
                    encoder.set_buffer(28, Some(geometry_masks), 0);
                    let (input_operand, output_operand): (&Buffer, &Buffer) =
                        match (input.operand.as_ref(), output.operand.as_ref()) {
                            (Some(input), Some(output)) => (input, output),
                            (None, None) => (&self.scratch.ra_lut, &self.scratch.ra_lut),
                            _ => {
                                return Err(MetalError::InvalidRegistersReadWriteState(
                                "registers read-write operand carry changed across direct states",
                            ));
                            }
                        };
                    if cooperative {
                        encoder.set_buffer(23, Some(&self.scratch.ra_lut), 0);
                    } else {
                        encoder.set_buffer(27, Some(&self.scratch.ra_lut), 0);
                    }
                    encoder.set_buffer(29, Some(input_operand), 0);
                    encoder.set_buffer(30, Some(output_operand), 0);
                }
                _ => {
                    return Err(MetalError::InvalidRegistersReadWriteState(
                        "registers read-write sequence state transition is invalid",
                    ));
                }
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: message_groups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = if emit_message {
                Some(encode_column_reductions(
                    encoder,
                    &self.pipelines.reduction,
                    &self.scratch.partial_a,
                    &self.scratch.partial_b,
                    message_groups,
                    2,
                    REGISTERS_READ_WRITE_SIMD_WIDTH,
                )?)
            } else {
                None
            };
            encoder.end_encoding();
            let started = Instant::now();
            if let Some(prefill_command) = prefill_command.as_ref() {
                prefill_command.commit();
            }
            command.commit();
            command.wait_until_completed();
            if let Some(prefill_command) = prefill_command.as_ref() {
                prefill_command.wait_until_completed();
            }
            let wall = started.elapsed();
            let main_gpu_active = completed_command_gpu_time(command)?;
            let prefill_gpu_active = prefill_command
                .as_ref()
                .map(|command| completed_command_gpu_time(command))
                .transpose()?
                .unwrap_or_default();
            let gpu_active = main_gpu_active + prefill_gpu_active;
            let quadratic = match final_in_a {
                Some(true) => read_quadratic(&self.context, &self.scratch.partial_a)?,
                Some(false) => read_quadratic(&self.context, &self.scratch.partial_b)?,
                None => [AkitaField::zero(); 2],
            };
            Ok((quadratic, wall, gpu_active, prefill_gpu_active))
        })
    }

    fn read_final_state(
        &self,
    ) -> Result<(Vec<BoundRegisterCycleRoot<AkitaField>>, AkitaField), MetalError> {
        let Some(CycleState::Direct(state)) = &self.state else {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write final state is not direct",
            ));
        };
        if state.common.blocks != 1 {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write final state has more than one block",
            ));
        }
        let length = buffer_slice::<u8>(&state.common.lengths, 1)[0] as usize;
        if length > MAX_REGISTER_BLOCK_CAPACITY {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write final block exceeds the register domain",
            ));
        }
        let columns = buffer_slice::<u8>(&state.common.columns, length);
        let previous = buffer_slice::<u64>(&state.common.previous, length);
        let next = buffer_slice::<u64>(&state.common.next, length);
        let values = buffer_slice::<Fp128>(&state.common.values, length);
        let ra = buffer_slice::<Fp128>(&state.ra, length);
        let wa = buffer_slice::<Fp128>(&state.wa, length);
        self.context
            .validate_inputs("registers read-write final values", values)?;
        self.context
            .validate_inputs("registers read-write final ra", ra)?;
        self.context
            .validate_inputs("registers read-write final wa", wa)?;
        let mut roots = Vec::with_capacity(length);
        for index in 0..length {
            roots.push(BoundRegisterCycleRoot {
                column: self.original_register_column(columns[index])?,
                previous: previous[index],
                next: next[index],
                value: values[index].into_jolt_field(),
                ra: ra[index].into_jolt_field(),
                wa: wa[index].into_jolt_field(),
            });
        }
        let increments = buffer_slice::<Fp128>(&state.common.increments, 1);
        self.context
            .validate_inputs("registers read-write final increment", increments)?;
        Ok((roots, increments[0].into_jolt_field()))
    }
}

impl IndexedStateBuffers {
    fn new_reusing(
        context: &SolinasMetal,
        geometry: StateGeometry,
        options: MTLResourceOptions,
        reusable: Option<Self>,
        pool_epoch: u64,
    ) -> Result<Self, MetalError> {
        let Some(reusable) = reusable else {
            return Self::new(context, geometry, options, pool_epoch);
        };
        let slots = geometry.slots;
        require_buffer_capacity::<u16>(&reusable.ra, slots)?;
        require_buffer_capacity::<u8>(&reusable.wa, slots)?;
        let common = CommonStateBuffers::reuse_payload(context, geometry, reusable.common)?;
        let bytes = common
            .bytes
            .checked_add(reusable.ra.length() as usize)
            .and_then(|bytes| bytes.checked_add(reusable.wa.length() as usize))
            .ok_or(MetalError::InputTooLong(slots))?;
        Ok(Self {
            common,
            ra: reusable.ra,
            wa: reusable.wa,
            bytes,
        })
    }

    fn new(
        context: &SolinasMetal,
        geometry: StateGeometry,
        options: MTLResourceOptions,
        pool_epoch: u64,
    ) -> Result<Self, MetalError> {
        let common = CommonStateBuffers::new(context, geometry, options, pool_epoch)?;
        let slots = common.slots;
        let ra = new_private_payload_buffer::<u16>(context, slots, options, pool_epoch)?;
        let wa = new_private_payload_buffer::<u8>(context, slots, options, pool_epoch)?;
        let bytes = common
            .bytes
            .checked_add(ra.length() as usize)
            .and_then(|bytes| bytes.checked_add(wa.length() as usize))
            .ok_or(MetalError::InputTooLong(slots))?;
        Ok(Self {
            common,
            ra,
            wa,
            bytes,
        })
    }
}

impl WideIndexedStateBuffers {
    fn new_reusing_common(
        context: &SolinasMetal,
        geometry: StateGeometry,
        options: MTLResourceOptions,
        reusable_common: Option<CommonStateBuffers>,
        pool_epoch: u64,
    ) -> Result<Self, MetalError> {
        let slots = geometry.slots;
        let common = if let Some(reusable_common) = reusable_common {
            CommonStateBuffers::reuse_payload(context, geometry, reusable_common)?
        } else {
            CommonStateBuffers::new(context, geometry, options, pool_epoch)?
        };
        let ra = new_private_payload_buffer::<u32>(context, slots, options, pool_epoch)?;
        let wa = new_private_payload_buffer::<u16>(context, slots, options, pool_epoch)?;
        let bytes = common
            .bytes
            .checked_add(ra.length() as usize)
            .and_then(|bytes| bytes.checked_add(wa.length() as usize))
            .ok_or(MetalError::InputTooLong(slots))?;
        Ok(Self {
            common,
            ra,
            wa,
            bytes,
        })
    }
}

impl DirectPlaneBuffers {
    fn new(
        context: &SolinasMetal,
        geometry: StateGeometry,
        options: MTLResourceOptions,
        operand_element_bytes: Option<usize>,
        pool_epoch: u64,
    ) -> Result<Self, MetalError> {
        let slots = geometry.slots;
        let geometry_bytes = geometry.bytes();
        let ra = new_private_payload_buffer::<Fp128>(context, slots, options, pool_epoch)?;
        let wa = new_private_payload_buffer::<Fp128>(context, slots, options, pool_epoch)?;
        let operand = operand_element_bytes
            .map(|bytes| {
                new_private_payload_byte_buffer(context, slots, bytes, options, pool_epoch)
            })
            .transpose()?;
        let bytes = geometry_bytes
            .checked_add(ra.length() as usize)
            .and_then(|bytes| bytes.checked_add(wa.length() as usize))
            .and_then(|bytes| {
                bytes.checked_add(
                    operand
                        .as_ref()
                        .map_or(0, |buffer| buffer.length() as usize),
                )
            })
            .ok_or(MetalError::InputTooLong(slots))?;
        Ok(Self {
            geometry,
            ra,
            wa,
            operand,
            bytes,
        })
    }

    fn into_state(
        self,
        context: &SolinasMetal,
        reusable_common: CommonStateBuffers,
    ) -> Result<DirectStateBuffers, MetalError> {
        let common = CommonStateBuffers::reuse_payload(context, self.geometry, reusable_common)?;
        let bytes = common
            .bytes
            .checked_add(self.ra.length() as usize)
            .and_then(|bytes| bytes.checked_add(self.wa.length() as usize))
            .and_then(|bytes| {
                bytes.checked_add(
                    self.operand
                        .as_ref()
                        .map_or(0, |buffer| buffer.length() as usize),
                )
            })
            .ok_or(MetalError::InputTooLong(common.slots))?;
        Ok(DirectStateBuffers {
            common,
            ra: self.ra,
            wa: self.wa,
            operand: self.operand,
            bytes,
        })
    }
}

impl DirectStateBuffers {
    fn new_reusing(
        context: &SolinasMetal,
        geometry: StateGeometry,
        options: MTLResourceOptions,
        reusable: Option<Self>,
        reusable_common: Option<CommonStateBuffers>,
        operand_element_bytes: Option<usize>,
        pool_epoch: u64,
    ) -> Result<Self, MetalError> {
        if let Some(reusable) = reusable {
            if reusable_common.is_some() {
                return Err(MetalError::InvalidRegistersReadWriteState(
                    "registers read-write direct reuse has two payload donors",
                ));
            }
            let slots = geometry.slots;
            let Self {
                common: reusable_common,
                ra,
                wa,
                operand: reusable_operand,
                ..
            } = reusable;
            require_buffer_capacity::<Fp128>(&ra, slots)?;
            require_buffer_capacity::<Fp128>(&wa, slots)?;
            let operand = match operand_element_bytes {
                Some(element_bytes) => {
                    let required = slots
                        .checked_mul(element_bytes)
                        .ok_or(MetalError::InputTooLong(slots))?;
                    match reusable_operand {
                        Some(buffer) if buffer.length() as usize >= required => Some(buffer),
                        _ => Some(new_private_payload_byte_buffer(
                            context,
                            slots,
                            element_bytes,
                            options,
                            pool_epoch,
                        )?),
                    }
                }
                None => None,
            };
            let common = CommonStateBuffers::reuse_payload(context, geometry, reusable_common)?;
            let bytes = common
                .bytes
                .checked_add(ra.length() as usize)
                .and_then(|bytes| bytes.checked_add(wa.length() as usize))
                .and_then(|bytes| {
                    bytes.checked_add(
                        operand
                            .as_ref()
                            .map_or(0, |buffer| buffer.length() as usize),
                    )
                })
                .ok_or(MetalError::InputTooLong(slots))?;
            return Ok(Self {
                common,
                ra,
                wa,
                operand,
                bytes,
            });
        }
        let Some(reusable_common) = reusable_common else {
            return Self::new(
                context,
                geometry,
                options,
                operand_element_bytes,
                pool_epoch,
            );
        };
        let slots = geometry.slots;
        let common = CommonStateBuffers::reuse_payload(context, geometry, reusable_common)?;
        let ra = new_private_payload_buffer::<Fp128>(context, slots, options, pool_epoch)?;
        let wa = new_private_payload_buffer::<Fp128>(context, slots, options, pool_epoch)?;
        let operand = operand_element_bytes
            .map(|bytes| {
                new_private_payload_byte_buffer(context, slots, bytes, options, pool_epoch)
            })
            .transpose()?;
        let bytes = common
            .bytes
            .checked_add(ra.length() as usize)
            .and_then(|bytes| bytes.checked_add(wa.length() as usize))
            .and_then(|bytes| {
                bytes.checked_add(
                    operand
                        .as_ref()
                        .map_or(0, |buffer| buffer.length() as usize),
                )
            })
            .ok_or(MetalError::InputTooLong(slots))?;
        Ok(Self {
            common,
            ra,
            wa,
            operand,
            bytes,
        })
    }

    fn new(
        context: &SolinasMetal,
        geometry: StateGeometry,
        options: MTLResourceOptions,
        operand_element_bytes: Option<usize>,
        pool_epoch: u64,
    ) -> Result<Self, MetalError> {
        let common = CommonStateBuffers::new(context, geometry, options, pool_epoch)?;
        let slots = common.slots;
        let ra = new_private_payload_buffer::<Fp128>(context, slots, options, pool_epoch)?;
        let wa = new_private_payload_buffer::<Fp128>(context, slots, options, pool_epoch)?;
        let operand = operand_element_bytes
            .map(|bytes| {
                new_private_payload_byte_buffer(context, slots, bytes, options, pool_epoch)
            })
            .transpose()?;
        let bytes = common
            .bytes
            .checked_add(ra.length() as usize)
            .and_then(|bytes| bytes.checked_add(wa.length() as usize))
            .and_then(|bytes| {
                bytes.checked_add(
                    operand
                        .as_ref()
                        .map_or(0, |buffer| buffer.length() as usize),
                )
            })
            .ok_or(MetalError::InputTooLong(slots))?;
        Ok(Self {
            common,
            ra,
            wa,
            operand,
            bytes,
        })
    }
}

impl CommonStateBuffers {
    fn reuse_payload(
        context: &SolinasMetal,
        geometry: StateGeometry,
        reusable: Self,
    ) -> Result<Self, MetalError> {
        let StateGeometry {
            blocks,
            slots,
            tile_bases,
            offsets,
            masks,
        } = geometry;
        if tile_bases.len() != blocks.div_ceil(STATE_TILE_BLOCKS) {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write geometry has the wrong tile count",
            ));
        }
        require_buffer_capacity::<u8>(&reusable.lengths, blocks)?;
        require_buffer_capacity::<u8>(&reusable.columns, slots)?;
        require_buffer_capacity::<u64>(&reusable.previous, slots)?;
        require_buffer_capacity::<u64>(&reusable.next, slots)?;
        require_buffer_capacity::<Fp128>(&reusable.values, slots)?;
        require_buffer_capacity::<Fp128>(&reusable.increments, blocks)?;

        let tile_bases_buffer = new_buffer::<u32>(
            context,
            tile_bases.len(),
            MTLResourceOptions::StorageModeShared,
        )?;
        buffer_slice_mut::<u32>(&tile_bases_buffer, tile_bases.len()).copy_from_slice(&tile_bases);
        let bytes = [
            reusable.lengths.length(),
            offsets.length(),
            tile_bases_buffer.length(),
            reusable.columns.length(),
            reusable.previous.length(),
            reusable.next.length(),
            reusable.values.length(),
            reusable.increments.length(),
            masks.length(),
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes as usize)
                .ok_or(MetalError::InputTooLong(total))
        })?;
        Ok(Self {
            lengths: reusable.lengths,
            offsets,
            tile_bases: tile_bases_buffer,
            columns: reusable.columns,
            previous: reusable.previous,
            next: reusable.next,
            values: reusable.values,
            increments: reusable.increments,
            masks,
            blocks,
            slots,
            bytes,
        })
    }

    fn new(
        context: &SolinasMetal,
        geometry: StateGeometry,
        options: MTLResourceOptions,
        pool_epoch: u64,
    ) -> Result<Self, MetalError> {
        let StateGeometry {
            blocks,
            slots,
            tile_bases,
            offsets,
            masks,
        } = geometry;
        if tile_bases.len() != blocks.div_ceil(STATE_TILE_BLOCKS) {
            return Err(MetalError::InvalidRegistersReadWriteState(
                "registers read-write geometry has the wrong tile count",
            ));
        }
        let lengths = new_private_payload_buffer::<u8>(context, blocks, options, pool_epoch)?;
        let tile_bases_buffer = new_buffer::<u32>(
            context,
            tile_bases.len(),
            MTLResourceOptions::StorageModeShared,
        )?;
        buffer_slice_mut::<u32>(&tile_bases_buffer, tile_bases.len()).copy_from_slice(&tile_bases);
        let columns = new_private_payload_buffer::<u8>(context, slots, options, pool_epoch)?;
        let previous = new_private_payload_buffer::<u64>(context, slots, options, pool_epoch)?;
        let next = new_private_payload_buffer::<u64>(context, slots, options, pool_epoch)?;
        let values = new_private_payload_buffer::<Fp128>(context, slots, options, pool_epoch)?;
        let increments = new_private_payload_buffer::<Fp128>(context, blocks, options, pool_epoch)?;
        let bytes = [
            lengths.length(),
            offsets.length(),
            tile_bases_buffer.length(),
            columns.length(),
            previous.length(),
            next.length(),
            values.length(),
            increments.length(),
            masks.length(),
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes as usize)
                .ok_or(MetalError::InputTooLong(total))
        })?;
        Ok(Self {
            lengths,
            offsets,
            tile_bases: tile_bases_buffer,
            columns,
            previous,
            next,
            values,
            increments,
            masks,
            blocks,
            slots,
            bytes,
        })
    }
}

fn require_buffer_capacity<T>(buffer: &Buffer, elements: usize) -> Result<(), MetalError> {
    let required = elements
        .checked_mul(size_of::<T>())
        .ok_or(MetalError::InputTooLong(elements))?;
    if required > buffer.length() as usize {
        return Err(MetalError::InvalidRegistersReadWriteState(
            "registers read-write reused payload capacity is too short",
        ));
    }
    Ok(())
}

fn direct_transition_bound(log_t: usize) -> usize {
    if log_t >= CROSS_REPRESENTATION_REUSE_LOG_T_MIN {
        WIDE_INDEXED_BOUND_ROUND + 1
    } else {
        DIRECT_TRANSITION_BOUND_ROUNDS
    }
}

fn operand_carry_element_bytes(
    log_t: usize,
    stage1_source: bool,
    rounds_bound: usize,
) -> Option<usize> {
    if !uses_operand_carry(log_t, stage1_source) || rounds_bound < 5 {
        return None;
    }
    Some(match rounds_bound {
        5 => size_of::<u32>(),
        6 => size_of::<u64>(),
        _ => size_of::<Fp128>(),
    })
}

fn operand_carry_kind(log_t: usize, stage1_source: bool, rounds_bound: usize) -> u32 {
    if !uses_operand_carry(log_t, stage1_source) {
        return 0;
    }
    match rounds_bound {
        5 => 5,
        6 => 1,
        7 => 2,
        8 => 3,
        9.. => 4,
        _ => 0,
    }
}

fn bind_common_output(
    encoder: &metal::ComputeCommandEncoderRef,
    first: u64,
    state: &CommonStateBuffers,
) {
    encoder.set_buffer(first, Some(&state.lengths), 0);
    encoder.set_buffer(first + 1, Some(&state.columns), 0);
    encoder.set_buffer(first + 2, Some(&state.previous), 0);
    encoder.set_buffer(first + 3, Some(&state.next), 0);
    encoder.set_buffer(first + 4, Some(&state.values), 0);
}

fn bind_indexed_input(encoder: &metal::ComputeCommandEncoderRef, state: &IndexedStateBuffers) {
    encoder.set_buffer(0, Some(&state.common.lengths), 0);
    encoder.set_buffer(1, Some(&state.common.columns), 0);
    encoder.set_buffer(2, Some(&state.common.previous), 0);
    encoder.set_buffer(3, Some(&state.common.next), 0);
    encoder.set_buffer(4, Some(&state.common.values), 0);
    encoder.set_buffer(5, Some(&state.ra), 0);
    encoder.set_buffer(6, Some(&state.wa), 0);
    encoder.set_buffer(7, Some(&state.common.increments), 0);
}

fn bind_wide_indexed_input(
    encoder: &metal::ComputeCommandEncoderRef,
    state: &WideIndexedStateBuffers,
) {
    encoder.set_buffer(1, Some(&state.common.columns), 0);
    encoder.set_buffer(2, Some(&state.common.previous), 0);
    encoder.set_buffer(3, Some(&state.common.next), 0);
    encoder.set_buffer(4, Some(&state.common.values), 0);
    encoder.set_buffer(5, Some(&state.ra), 0);
    encoder.set_buffer(6, Some(&state.wa), 0);
    encoder.set_buffer(7, Some(&state.common.increments), 0);
}

fn bind_direct_input(encoder: &metal::ComputeCommandEncoderRef, state: &DirectStateBuffers) {
    encoder.set_buffer(0, Some(&state.common.lengths), 0);
    encoder.set_buffer(1, Some(&state.common.columns), 0);
    encoder.set_buffer(2, Some(&state.common.previous), 0);
    encoder.set_buffer(3, Some(&state.common.next), 0);
    encoder.set_buffer(4, Some(&state.common.values), 0);
    encoder.set_buffer(5, Some(&state.ra), 0);
    encoder.set_buffer(6, Some(&state.wa), 0);
    encoder.set_buffer(7, Some(&state.common.increments), 0);
}

fn bind_lut(values: &mut Vec<AkitaField>, challenge: AkitaField) -> Result<(), MetalError> {
    let length = values.len();
    let output_length = length
        .checked_mul(length)
        .ok_or(MetalError::InputTooLong(length))?;
    if output_length > 1 << 16 {
        return Err(MetalError::InvalidRegistersReadWriteState(
            "registers read-write coefficient LUT overflowed",
        ));
    }
    let old = values.as_slice();
    let mut output = Vec::with_capacity(output_length);
    for index in 0..output_length {
        let high = old[index / length];
        let low = old[index % length];
        output.push(low + challenge * (high - low));
    }
    *values = output;
    Ok(())
}

fn new_buffer<T>(
    context: &SolinasMetal,
    elements: usize,
    options: MTLResourceOptions,
) -> Result<Buffer, MetalError> {
    let bytes = elements
        .max(1)
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))?;
    context.validate_buffer_length(bytes)?;
    Ok(context.device.new_buffer(bytes, options))
}

fn new_private_payload_buffer<T>(
    context: &SolinasMetal,
    elements: usize,
    options: MTLResourceOptions,
    pool_epoch: u64,
) -> Result<PooledPrivateBuffer, MetalError> {
    new_private_payload_byte_buffer(context, elements, size_of::<T>(), options, pool_epoch)
}

fn new_private_payload_byte_buffer(
    context: &SolinasMetal,
    elements: usize,
    element_bytes: usize,
    options: MTLResourceOptions,
    pool_epoch: u64,
) -> Result<PooledPrivateBuffer, MetalError> {
    let bytes = elements
        .max(1)
        .checked_mul(element_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))?;
    context.new_pooled_private_buffer(
        bytes,
        options,
        pool_epoch,
        PRIVATE_PAYLOAD_POOL_THRESHOLD_BYTES,
    )
}

fn write_fields(buffer: &Buffer, values: &[AkitaField]) -> Result<(), MetalError> {
    let required = values
        .len()
        .checked_mul(size_of::<Fp128>())
        .ok_or(MetalError::InputTooLong(values.len()))?;
    if required > buffer.length() as usize {
        return Err(MetalError::InputTooLong(values.len()));
    }
    let output = buffer_slice_mut::<Fp128>(buffer, values.len());
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn read_quadratic(context: &SolinasMetal, buffer: &Buffer) -> Result<[AkitaField; 2], MetalError> {
    let values = buffer_slice::<Fp128>(buffer, 2);
    context.validate_inputs("registers read-write cycle message", values)?;
    Ok([values[0].into_jolt_field(), values[1].into_jolt_field()])
}

fn checked_u32(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}

fn checked_u64(value: usize) -> Result<u64, MetalError> {
    u64::try_from(value).map_err(|_| MetalError::InputTooLong(value))
}

#[cfg(test)]
#[expect(
    clippy::items_after_test_module,
    clippy::unwrap_used,
    reason = "Metal source-primer oracle setup"
)]
mod tests {
    use super::*;
    use crate::metal::solinas::buffer_from_slice;

    #[test]
    fn stage1_retains_operand_carry_at_compact_source_threshold() {
        assert!(uses_operand_carry(COMPACT_RS1_SOURCE_LOG_T_MIN, true));
        assert!(!uses_operand_carry(COMPACT_RS1_SOURCE_LOG_T_MIN, false));
        assert!(uses_operand_carry(COMPACT_RS1_SOURCE_LOG_T_MIN - 1, false));
        assert!(!uses_operand_carry(
            CROSS_REPRESENTATION_REUSE_LOG_T_MIN - 1,
            true
        ));
    }

    #[test]
    fn source_primer_touches_read_raf_pages_without_mutating_sources() {
        let context = SolinasMetal::for_akita().unwrap();
        let page_words = SOURCE_PRIMER_PAGE_BYTES / size_of::<u64>();
        let word_counts = [2 * page_words + 17, page_words + 3, 3 * page_words - 1];
        let values: [Vec<u64>; 3] = std::array::from_fn(|source| {
            (0..word_counts[source])
                .map(|word| {
                    ((source as u64 + 1) << 60) ^ (word as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                })
                .collect()
        });
        let sources = std::array::from_fn(|source| {
            buffer_from_slice(&context.device, values[source].as_slice())
        });

        let pending = context
            .submit_registers_read_write_source_primer_buffers(
                sources.clone(),
                [false, true, false],
            )
            .unwrap();
        let (observation, checksums) = pending.join_with_checksums().unwrap();
        let page_counts = [0, word_counts[1].div_ceil(page_words), 0];
        let total_pages = page_counts.into_iter().sum::<usize>();
        assert_eq!(observation.pages, total_pages);
        assert_eq!(observation.read_bytes, total_pages * size_of::<u64>());

        let expected = (0..SOURCE_PRIMER_THREADS)
            .map(|gid| {
                let mut checksum = 0x9e37_79b9_u32 ^ gid as u32;
                for page in (gid..total_pages).step_by(SOURCE_PRIMER_THREADS) {
                    let (source, local_page) = if page < page_counts[0] {
                        (0, page)
                    } else if page < page_counts[0] + page_counts[1] {
                        (1, page - page_counts[0])
                    } else {
                        (2, page - page_counts[0] - page_counts[1])
                    };
                    let value = values[source][local_page * page_words];
                    checksum ^= value as u32 ^ (value >> 32) as u32 ^ page as u32;
                    checksum = checksum.rotate_left(5).wrapping_mul(0x85eb_ca6b);
                }
                checksum
            })
            .collect::<Vec<_>>();
        assert_eq!(checksums, expected);

        for (source, expected) in sources.iter().zip(&values) {
            // SAFETY: the shared test buffer remains alive and has exactly this many words.
            let actual =
                unsafe { slice::from_raw_parts(source.contents().cast::<u64>(), expected.len()) };
            assert_eq!(actual, expected.as_slice());
        }
    }
}

fn buffer_slice<T>(buffer: &Buffer, length: usize) -> &[T] {
    debug_assert!(length * size_of::<T>() <= buffer.length() as usize);
    // SAFETY: callers use the allocation's element type after GPU completion.
    unsafe { slice::from_raw_parts(buffer.contents().cast::<T>(), length) }
}

#[expect(
    clippy::mut_from_ref,
    reason = "Metal shared buffers provide interior-mutable mapped storage"
)]
fn buffer_slice_mut<T>(buffer: &Buffer, length: usize) -> &mut [T] {
    debug_assert!(length * size_of::<T>() <= buffer.length() as usize);
    // SAFETY: callers write shared allocations before overlapping GPU access.
    unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<T>(), length) }
}
