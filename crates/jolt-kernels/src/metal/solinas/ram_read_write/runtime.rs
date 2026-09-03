use std::{
    ffi::c_void,
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::{One as _, Zero as _};
use jolt_field::{Prime128OffsetA7F7 as AkitaField, Ring};
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePassDescriptor, ComputePipelineState, CounterSampleBuffer, MTLResourceOptions, MTLSize,
    NSRange,
};
#[cfg(feature = "test-utils")]
use metal::{CounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLStorageMode};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    CycleProductRoot, HotChunk, HotSegment, PhaseParams, PrefixPhaseParams, Segment,
    RAM_READ_WRITE_ADDRESS_BOUNDED_PIPELINE, RAM_READ_WRITE_ADDRESS_HOT_COUNT_PIPELINE,
    RAM_READ_WRITE_ADDRESS_HOT_MESSAGE_PIPELINE, RAM_READ_WRITE_ADDRESS_HOT_PREFIX_PIPELINE,
    RAM_READ_WRITE_ADDRESS_HOT_SCATTER_PIPELINE, RAM_READ_WRITE_ADDRESS_PIPELINE,
    RAM_READ_WRITE_BOUNDED_SEGMENT_MAX, RAM_READ_WRITE_BOUNDED_THREADGROUP_BYTES_MAX,
    RAM_READ_WRITE_CYCLE_PIPELINE, RAM_READ_WRITE_CYCLE_THREADGROUP_BYTES_MAX,
    RAM_READ_WRITE_CYCLE_TILE_LOG2, RAM_READ_WRITE_HOT_COMPACTION_THREADS,
    RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE, RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD,
    RAM_READ_WRITE_HOT_THREADGROUP_BYTES_MAX, RAM_READ_WRITE_INITIAL_SCATTER_PIPELINE,
    RAM_READ_WRITE_PREFIX_ADDRESS_PIPELINE, RAM_READ_WRITE_PREFIX_ADDRESS_TRANSITION_PIPELINE,
    RAM_READ_WRITE_PREFIX_CYCLE_PIPELINE, RAM_READ_WRITE_PREFIX_CYCLE_TRANSITION_PIPELINE,
    RAM_READ_WRITE_PREFIX_HOT_TRANSITION_PIPELINE, RAM_READ_WRITE_RECORD_PREFIX_LOG_T_MIN,
    RAM_READ_WRITE_RECORD_PREFIX_ROUNDS, RAM_READ_WRITE_REDUCTION_PIPELINE,
    RAM_READ_WRITE_REDUCTION_WIDTH, RAM_READ_WRITE_SIMD_WIDTH, RAM_READ_WRITE_THREADS,
};
use crate::metal::ram_records::{
    AlignedRamReadWriteRecordArena, RamReadWriteRecordChunks, NO_ACCESS,
};
use crate::metal::solinas::{
    completed_command_gpu_time, encode_column_reductions, ram_raf_split_equality, set_inline_bytes,
    Fp128, MetalError, SolinasMetal,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RamReadWriteBucketStats {
    pub accesses: usize,
    pub active_addresses: usize,
    pub maximum_segment: usize,
    pub p50_segment: usize,
    pub p95_segment: usize,
    pub p99_segment: usize,
    pub hot_addresses: usize,
    pub hot_message_chunks: usize,
    pub hot_state_entries: usize,
    pub hot_compaction_threads: usize,
    pub hot_compaction_threadgroup_bytes: u64,
    pub hot_auxiliary_bytes: usize,
    pub address_bytes: usize,
    pub cycle_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RamReadWritePreparationTiming {
    pub bucket_plan: Duration,
    pub allocation: Duration,
    pub initialization_and_scatter: Duration,
    pub pipeline_setup: Duration,
    pub gpu_scatter_wall: Duration,
    pub gpu_scatter_active: Duration,
    pub total: Duration,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RamReadWriteDispatchTiming {
    pub address: Duration,
    pub hot_count: Duration,
    pub hot_prefix: Duration,
    pub hot_scatter: Duration,
    pub hot_message: Duration,
    pub cycle: Duration,
    pub reductions: Duration,
}

impl std::ops::AddAssign for RamReadWriteDispatchTiming {
    fn add_assign(&mut self, rhs: Self) {
        self.address += rhs.address;
        self.hot_count += rhs.hot_count;
        self.hot_prefix += rhs.hot_prefix;
        self.hot_scatter += rhs.hot_scatter;
        self.hot_message += rhs.hot_message;
        self.cycle += rhs.cycle;
        self.reductions += rhs.reductions;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AddressCycleRoot {
    pub address: usize,
    pub previous: u64,
    pub next: u64,
    pub value: AkitaField,
    pub ra: AkitaField,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RamReadWriteSequenceObservation {
    pub address_quadratic: [AkitaField; 2],
    pub cycle_quadratic: Option<[AkitaField; 2]>,
    pub cycle_roots: Option<Vec<CycleProductRoot>>,
    pub address_live_entries: usize,
    pub cycle_live_entries: usize,
    pub wall: Duration,
    pub gpu_active: Duration,
    pub dispatch_timing: Option<RamReadWriteDispatchTiming>,
}

pub(crate) struct RamReadWriteFinish {
    pub address_roots: Vec<AddressCycleRoot>,
    pub cycle_roots: Option<Vec<CycleProductRoot>>,
    pub wall: Duration,
    pub gpu_active: Duration,
    pub dispatch_timing: Option<RamReadWriteDispatchTiming>,
}

struct AddressBuffers {
    bounded_segments: Buffer,
    hot_segments: Buffer,
    hot_message_chunks: Buffer,
    hot_chunk_counts: Buffer,
    hot_chunk_offsets: Buffer,
    hot_source_lengths: Buffer,
    segments: Buffer,
    blocks: Buffer,
    previous: Buffer,
    next: Buffer,
    values: Buffer,
    ra: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
    hot_partial_a: Buffer,
    hot_partial_b: Buffer,
    aux_blocks: Buffer,
    aux_previous: Buffer,
    aux_next: Buffer,
    aux_values: Buffer,
    aux_ra: Buffer,
}

impl AddressBuffers {
    fn resident_bytes(&self) -> usize {
        self.bounded_segments.length() as usize
            + self.hot_segments.length() as usize
            + self.hot_message_chunks.length() as usize
            + self.hot_chunk_counts.length() as usize
            + self.hot_chunk_offsets.length() as usize
            + self.hot_source_lengths.length() as usize
            + self.segments.length() as usize
            + self.blocks.length() as usize
            + self.previous.length() as usize
            + self.next.length() as usize
            + self.values.length() as usize
            + self.ra.length() as usize
            + self.partial_a.length() as usize
            + self.partial_b.length() as usize
            + self.hot_partial_a.length() as usize
            + self.hot_partial_b.length() as usize
            + self.aux_blocks.length() as usize
            + self.aux_previous.length() as usize
            + self.aux_next.length() as usize
            + self.aux_values.length() as usize
            + self.aux_ra.length() as usize
    }
}

struct CycleBuffers {
    segments: Buffer,
    blocks: Buffer,
    address_indices: Option<Buffer>,
    hamming: Buffer,
    increments: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

impl CycleBuffers {
    fn resident_bytes(&self) -> usize {
        self.segments.length() as usize
            + self.blocks.length() as usize
            + self
                .address_indices
                .as_ref()
                .map_or(0, |buffer| buffer.length() as usize)
            + self.hamming.length() as usize
            + self.increments.length() as usize
            + self.partial_a.length() as usize
            + self.partial_b.length() as usize
    }
}

struct RamReadWriteBuffers {
    address: AddressBuffers,
    cycle: CycleBuffers,
    e_in: Buffer,
    e_out: Buffer,
}

#[derive(Clone)]
pub(crate) struct RamRafSegmentedAddressPlane {
    segments: Buffer,
    blocks: Buffer,
    bounded_segments: Buffer,
    hot_segments: Buffer,
    hot_message_chunks: Buffer,
    rows: usize,
    addresses: usize,
    accesses: usize,
    bounded_address_count: usize,
    hot_address_count: usize,
    hot_message_chunk_count: usize,
    cold_segment_threshold: usize,
    hot_message_chunk_size: usize,
    device_registry_id: u64,
}

impl RamRafSegmentedAddressPlane {
    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) const fn addresses(&self) -> usize {
        self.addresses
    }

    pub(crate) const fn accesses(&self) -> usize {
        self.accesses
    }

    pub(crate) const fn bounded_address_count(&self) -> usize {
        self.bounded_address_count
    }

    pub(crate) const fn hot_address_count(&self) -> usize {
        self.hot_address_count
    }

    pub(crate) const fn hot_message_chunk_count(&self) -> usize {
        self.hot_message_chunk_count
    }

    pub(crate) const fn cold_segment_threshold(&self) -> usize {
        self.cold_segment_threshold
    }

    pub(crate) const fn hot_message_chunk_size(&self) -> usize {
        self.hot_message_chunk_size
    }

    pub(crate) const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    pub(crate) fn storage_id(&self) -> usize {
        self.segments.as_ptr() as usize
    }

    pub(crate) fn resource_identities(&self) -> [usize; 5] {
        [
            self.segments.as_ptr() as usize,
            self.blocks.as_ptr() as usize,
            self.bounded_segments.as_ptr() as usize,
            self.hot_segments.as_ptr() as usize,
            self.hot_message_chunks.as_ptr() as usize,
        ]
    }

    pub(crate) fn borrowed_bytes(&self) -> usize {
        self.segments.length() as usize
            + self.blocks.length() as usize
            + self.bounded_segments.length() as usize
            + self.hot_segments.length() as usize
            + self.hot_message_chunks.length() as usize
    }

    pub(crate) fn cpu_pushforward(
        &self,
        cycle_point: &[AkitaField],
    ) -> Result<Vec<AkitaField>, MetalError> {
        let (e_lo, e_hi) = ram_raf_split_equality(cycle_point)?;
        let inner_length = e_lo.len();
        if inner_length == 0
            || !inner_length.is_power_of_two()
            || self.rows / inner_length != e_hi.len()
        {
            return Err(MetalError::InvalidRamRafState(
                "segmented CPU pushforward has inconsistent equality geometry",
            ));
        }
        let segments = buffer_slice::<Segment>(&self.segments, self.addresses);
        let blocks = buffer_slice::<u32>(&self.blocks, self.accesses);
        if segments.iter().any(|segment| {
            (segment.offset as usize)
                .checked_add(segment.length as usize)
                .is_none_or(|end| end > blocks.len())
        }) {
            return Err(MetalError::InvalidRamRafState(
                "segmented CPU pushforward has an out-of-range segment",
            ));
        }
        let inner_mask = inner_length - 1;
        let fold_segment = |segment: &Segment| {
            let start = segment.offset as usize;
            let end = start + segment.length as usize;
            blocks[start..end]
                .iter()
                .fold(AkitaField::zero(), |sum, &cycle| {
                    let cycle = (cycle & RAM_READ_WRITE_PREFIX_BLOCK_MASK) as usize;
                    debug_assert!(cycle < self.rows);
                    sum + e_lo[cycle & inner_mask] * e_hi[cycle / inner_length]
                })
        };
        #[cfg(feature = "parallel")]
        let masses = segments.par_iter().map(fold_segment).collect();
        #[cfg(not(feature = "parallel"))]
        let masses = segments.iter().map(fold_segment).collect();
        Ok(masses)
    }

    ref_field_getters! { pub(crate), {
        segments: Buffer,
        blocks: Buffer,
        bounded_segments: Buffer,
        hot_segments: Buffer,
        hot_message_chunks: Buffer,
    }}
}

struct RamReadWritePipelines {
    address: ComputePipelineState,
    address_bounded: ComputePipelineState,
    address_hot_count: ComputePipelineState,
    address_hot_prefix: ComputePipelineState,
    address_hot_scatter: ComputePipelineState,
    address_hot_message: ComputePipelineState,
    cycle: ComputePipelineState,
    prefix_address: ComputePipelineState,
    prefix_address_transition: ComputePipelineState,
    prefix_hot_transition: ComputePipelineState,
    prefix_cycle: ComputePipelineState,
    prefix_cycle_transition: ComputePipelineState,
    reduction: ComputePipelineState,
}

struct RecordPrefixState {
    rounds: usize,
    table_entries: usize,
    weight_table: Buffer,
    suffix_table: Buffer,
    address_segments: Buffer,
    bounded_segments: Buffer,
    hot_segments: Buffer,
    hot_message_chunks: Buffer,
    cycle_segments: Buffer,
    cycle_output_segments: Buffer,
    cycle_output_blocks: Buffer,
    hot_destinations: Buffer,
    bounded_address_count: usize,
    hot_address_count: usize,
    hot_message_chunk_count: usize,
    hot_state_entries: usize,
    address_live_entries: Vec<usize>,
    cycle_live_entries: Vec<usize>,
}

impl RecordPrefixState {
    fn resident_bytes(&self) -> usize {
        self.weight_table.length() as usize
            + self.suffix_table.length() as usize
            + self.address_segments.length() as usize
            + self.bounded_segments.length() as usize
            + self.hot_segments.length() as usize
            + self.hot_message_chunks.length() as usize
            + self.cycle_segments.length() as usize
            + self.cycle_output_segments.length() as usize
            + self.cycle_output_blocks.length() as usize
            + self.hot_destinations.length() as usize
    }
}

struct RamReadWriteDispatchProfiler {
    samples: CounterSampleBuffer,
    resolved: Buffer,
}

pub(crate) struct RamReadWriteSequence {
    context: SolinasMetal,
    pipelines: RamReadWritePipelines,
    buffers: RamReadWriteBuffers,
    log_t: usize,
    address_count: usize,
    hot_segment_threshold: usize,
    bounded_address_count: usize,
    hot_address_count: usize,
    hot_message_chunk_count: usize,
    hot_state_entries: usize,
    hot_address_threads: usize,
    tile_count: usize,
    tile_log: usize,
    rounds_bound: usize,
    initial_message_emitted: bool,
    cycle_resident: bool,
    finished: bool,
    stats: RamReadWriteBucketStats,
    preparation_timing: RamReadWritePreparationTiming,
    timing_wall: Duration,
    timing_gpu_active: Duration,
    dispatch_profiler: Option<RamReadWriteDispatchProfiler>,
    record_prefix: Option<RecordPrefixState>,
    hot_source_aux: bool,
}

struct SequenceCommand {
    command_buffer: CommandBuffer,
    address_output: Option<Buffer>,
    hot_address_output: Option<Buffer>,
    cycle_output: Option<Buffer>,
    submitted_at: Instant,
    timestamp_calibration_start: Option<(u64, u64)>,
    dispatch_activity: [bool; RAM_READ_WRITE_DISPATCH_STAGES as usize],
    live_entries: Option<(usize, usize)>,
}

type PhaseOutputs = (Option<Buffer>, Option<Buffer>, Option<Buffer>);
type RamReadWriteIncrementChunk = (Vec<u64>, Vec<i128>);
type RamReadWriteIncrementChunks = Vec<RamReadWriteIncrementChunk>;
type PreparedRamReadWriteSequence = (RamReadWriteSequence, Option<RamReadWriteIncrementChunks>);

const RAM_READ_WRITE_DISPATCH_STAGES: u64 = 7;
const RAM_READ_WRITE_DISPATCH_SAMPLES: u64 = 2 * RAM_READ_WRITE_DISPATCH_STAGES;
const RAM_READ_WRITE_PREFIX_SEGMENT_START: u32 = 1 << 31;
const RAM_READ_WRITE_PREFIX_BLOCK_MASK: u32 = !RAM_READ_WRITE_PREFIX_SEGMENT_START;

#[derive(Clone, Copy)]
struct BucketWriters {
    address_blocks: *mut u32,
    address_previous: *mut u64,
    address_next: *mut u64,
    cycle_blocks: *mut u32,
    cycle_address_indices: *mut u32,
    cycle_hamming: *mut Fp128,
    cycle_increments: *mut Fp128,
}

// SAFETY: each scatter worker owns disjoint destination ranges assigned by
// `bucket_plan`; the buffers outlive every worker.
unsafe impl Send for BucketWriters {}
// SAFETY: sharing only copies buffer base pointers. Writes remain confined to
// the disjoint ranges assigned to each worker.
unsafe impl Sync for BucketWriters {}

impl BucketWriters {
    fn new(address: &AddressBuffers, cycle: &CycleBuffers) -> Self {
        Self {
            address_blocks: address.blocks.contents().cast::<u32>(),
            address_previous: address.previous.contents().cast::<u64>(),
            address_next: address.next.contents().cast::<u64>(),
            cycle_blocks: cycle.blocks.contents().cast::<u32>(),
            cycle_address_indices: cycle
                .address_indices
                .as_ref()
                .map_or(std::ptr::null_mut(), |buffer| {
                    buffer.contents().cast::<u32>()
                }),
            cycle_hamming: cycle.hamming.contents().cast::<Fp128>(),
            cycle_increments: cycle.increments.contents().cast::<Fp128>(),
        }
    }

    unsafe fn write_address(self, index: usize, cycle: u32, previous: u64, next: u64) {
        // SAFETY: `bucket_plan` assigns this record a unique initialized slot
        // inside every equally sized address plane.
        unsafe {
            self.address_blocks.add(index).write(cycle);
            self.address_previous.add(index).write(previous);
            self.address_next.add(index).write(next);
        }
    }

    unsafe fn write_cycle(
        self,
        index: usize,
        address_index: usize,
        cycle: u32,
        previous: u64,
        next: u64,
    ) {
        // SAFETY: chronological worker prefixes assign every access record one
        // unique slot inside both host-written cycle planes.
        unsafe {
            self.cycle_blocks.add(index).write(cycle);
            if self.cycle_address_indices.is_null() {
                let increment = if previous == next {
                    Fp128::ZERO
                } else {
                    Fp128::from_jolt_field(&AkitaField::from_i128(
                        i128::from(next) - i128::from(previous),
                    ))
                };
                self.cycle_hamming
                    .add(index)
                    .write(Fp128::from_u128(address_index as u128));
                self.cycle_increments.add(index).write(increment);
            } else {
                self.cycle_address_indices
                    .add(index)
                    .write(address_index as u32);
            }
        }
    }
}

struct WorkerCounts {
    addresses: Vec<u32>,
    tiles: Vec<u32>,
    accesses: u32,
    first_cycle: Option<u32>,
    last_cycle: Option<u32>,
}

struct WorkerPlan {
    address_cursors: Vec<u32>,
    cycle_cursor: u32,
    accesses: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct InitialScatterParams {
    records: u32,
    address_count: u32,
    cycle_offset: u32,
    reserved: u32,
}

const _: [(); 16] = [(); size_of::<InitialScatterParams>()];

struct BucketPlan {
    workers: Vec<WorkerPlan>,
    chunk_length: usize,
    address_lengths: Vec<u32>,
    tile_lengths: Vec<u32>,
    bounded_segments: Vec<u32>,
    hot_segments: Vec<HotSegment>,
    hot_message_chunks: Vec<HotChunk>,
    hot_state_entries: usize,
    stats: RamReadWriteBucketStats,
}

#[derive(Clone, Copy)]
struct RamReadWritePreparationOptions {
    hot_segment_threshold: usize,
    gpu_record_scatter: bool,
}

#[derive(Clone, Copy)]
enum RamReadWritePreparationSource<'a> {
    Dense {
        addresses: &'a [u32],
        previous: &'a [u64],
        next: &'a [u64],
    },
    Direct {
        addresses: &'a [u32],
        value_at: &'a (dyn Fn(usize) -> Result<(u64, u64), MetalError> + Sync),
    },
    Records {
        source: &'a RamReadWriteRecordChunks,
    },
}

impl SolinasMetal {
    pub(crate) fn prepare_ram_read_write_sequence(
        &self,
        addresses: &[u32],
        previous: &[u64],
        next: &[u64],
        log_t: usize,
        address_count: usize,
    ) -> Result<RamReadWriteSequence, MetalError> {
        self.prepare_ram_read_write_sequence_with_hot_threshold(
            addresses,
            previous,
            next,
            log_t,
            address_count,
            RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD,
        )
    }

    pub(crate) fn prepare_ram_read_write_sequence_with_hot_threshold(
        &self,
        addresses: &[u32],
        previous: &[u64],
        next: &[u64],
        log_t: usize,
        address_count: usize,
        hot_segment_threshold: usize,
    ) -> Result<RamReadWriteSequence, MetalError> {
        self.prepare_ram_read_write_sequence_with_options(
            RamReadWritePreparationSource::Dense {
                addresses,
                previous,
                next,
            },
            log_t,
            address_count,
            RamReadWritePreparationOptions {
                hot_segment_threshold,
                gpu_record_scatter: false,
            },
        )
        .map(|(sequence, _)| sequence)
    }

    pub(crate) fn prepare_ram_read_write_direct_sequence(
        &self,
        addresses: &[u32],
        log_t: usize,
        address_count: usize,
        value_at: &(dyn Fn(usize) -> Result<(u64, u64), MetalError> + Sync),
    ) -> Result<(RamReadWriteSequence, RamReadWriteIncrementChunks), MetalError> {
        let (sequence, activity) = self.prepare_ram_read_write_sequence_with_options(
            RamReadWritePreparationSource::Direct {
                addresses,
                value_at,
            },
            log_t,
            address_count,
            RamReadWritePreparationOptions {
                hot_segment_threshold: RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD,
                gpu_record_scatter: false,
            },
        )?;
        let activity = activity.ok_or(MetalError::InvalidRamReadWriteState(
            "direct RAM read-write preparation lost its increment activity",
        ))?;
        Ok((sequence, activity))
    }

    pub(crate) fn prepare_ram_read_write_record_sequence(
        &self,
        source: &RamReadWriteRecordChunks,
        log_t: usize,
        address_count: usize,
        gpu_record_scatter: bool,
    ) -> Result<RamReadWriteSequence, MetalError> {
        self.prepare_ram_read_write_sequence_with_options(
            RamReadWritePreparationSource::Records { source },
            log_t,
            address_count,
            RamReadWritePreparationOptions {
                hot_segment_threshold: RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD,
                gpu_record_scatter,
            },
        )
        .map(|(sequence, _)| sequence)
    }

    fn prepare_ram_read_write_sequence_with_options(
        &self,
        source: RamReadWritePreparationSource<'_>,
        log_t: usize,
        address_count: usize,
        options: RamReadWritePreparationOptions,
    ) -> Result<PreparedRamReadWriteSequence, MetalError> {
        let prepare_started = Instant::now();
        let RamReadWritePreparationOptions {
            hot_segment_threshold,
            gpu_record_scatter,
        } = options;
        let use_record_prefix = matches!(source, RamReadWritePreparationSource::Records { .. })
            && !gpu_record_scatter
            && (cfg!(test) || log_t >= RAM_READ_WRITE_RECORD_PREFIX_LOG_T_MIN)
            && log_t > RAM_READ_WRITE_RECORD_PREFIX_ROUNDS;
        if log_t == 0
            || log_t > u32::BITS as usize
            || address_count == 0
            || hot_segment_threshold == 0
            || hot_segment_threshold > u32::MAX as usize
        {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write source has inconsistent geometry",
            ));
        }
        let rows = 1usize << log_t;
        if let RamReadWritePreparationSource::Dense {
            addresses,
            previous,
            next,
        } = source
        {
            if addresses.len() != previous.len() || addresses.len() != next.len() {
                return Err(MetalError::LengthMismatch {
                    lhs: addresses.len(),
                    rhs: previous.len().min(next.len()),
                });
            }
            if addresses.len() != rows {
                return Err(MetalError::InvalidRamReadWriteState(
                    "RAM read-write source has inconsistent geometry",
                ));
            }
        }
        if let RamReadWritePreparationSource::Direct { addresses, .. } = source {
            if addresses.len() != rows {
                return Err(MetalError::InvalidRamReadWriteState(
                    "RAM read-write direct source has inconsistent geometry",
                ));
            }
        }
        if let RamReadWritePreparationSource::Records { source } = source {
            if source.access_count() == 0 {
                return Err(MetalError::InvalidRamReadWriteState(
                    "RAM read-write record source is empty",
                ));
            }
        }
        let tile_log = log_t.min(RAM_READ_WRITE_CYCLE_TILE_LOG2);
        let tile_count = 1usize << (log_t - tile_log);
        let bucket_plan_started = Instant::now();
        let mut plan = match source {
            RamReadWritePreparationSource::Dense { addresses, .. } => bucket_plan(
                addresses,
                address_count,
                tile_log,
                tile_count,
                hot_segment_threshold,
            )?,
            RamReadWritePreparationSource::Direct { addresses, .. } => bucket_plan(
                addresses,
                address_count,
                tile_log,
                tile_count,
                hot_segment_threshold,
            )?,
            RamReadWritePreparationSource::Records { source } => bucket_plan_records(
                source,
                address_count,
                tile_log,
                tile_count,
                hot_segment_threshold,
            )?,
        };
        let bucket_plan_wall = bucket_plan_started.elapsed();
        let record_capacity = plan.stats.accesses.max(1);
        let cycle_state_capacity = if use_record_prefix {
            record_capacity.min(1usize << (log_t - RAM_READ_WRITE_RECORD_PREFIX_ROUNDS))
        } else {
            record_capacity
        };
        let maximum_eq_fields = 1usize << log_t.div_ceil(2);
        let allocation_started = Instant::now();
        let base_buffer_bytes = ram_read_write_buffer_bytes(
            record_capacity,
            address_count,
            tile_count,
            maximum_eq_fields,
            use_record_prefix,
            cycle_state_capacity,
            &plan,
        )?;
        let prefix_buffer_bytes = if use_record_prefix {
            ram_read_write_prefix_buffer_bytes(
                record_capacity,
                address_count,
                tile_count,
                cycle_state_capacity,
                &plan,
            )?
        } else {
            0
        };
        self.validate_additional_working_set(
            base_buffer_bytes
                .checked_add(prefix_buffer_bytes)
                .ok_or(MetalError::InputTooLong(record_capacity))?,
        )?;
        let bounded_address_capacity = plan.bounded_segments.len().max(1);
        let hot_address_capacity = plan.hot_segments.len().max(1);
        let hot_message_chunk_capacity = plan.hot_message_chunks.len().max(1);
        let hot_state_capacity = if use_record_prefix {
            1
        } else {
            plan.hot_state_entries.max(1)
        };
        let address_partial_capacity =
            address_count.max(record_capacity.div_ceil(RAM_READ_WRITE_THREADS));
        let mut address = AddressBuffers {
            bounded_segments: self.new_ram_read_write_buffer::<u32>(bounded_address_capacity)?,
            hot_segments: self.new_ram_read_write_buffer::<HotSegment>(hot_address_capacity)?,
            hot_message_chunks: self
                .new_ram_read_write_buffer::<HotChunk>(hot_message_chunk_capacity)?,
            hot_chunk_counts: self.new_ram_read_write_buffer::<u32>(hot_message_chunk_capacity)?,
            hot_chunk_offsets: self.new_ram_read_write_buffer::<u32>(hot_message_chunk_capacity)?,
            hot_source_lengths: self.new_ram_read_write_buffer::<u32>(hot_address_capacity)?,
            segments: self.new_ram_read_write_buffer::<Segment>(address_count)?,
            blocks: self.new_ram_read_write_buffer::<u32>(record_capacity)?,
            previous: self.new_ram_read_write_buffer::<u64>(record_capacity)?,
            next: self.new_ram_read_write_buffer::<u64>(record_capacity)?,
            values: self.new_ram_read_write_buffer::<Fp128>(record_capacity)?,
            ra: self.new_ram_read_write_buffer::<Fp128>(record_capacity)?,
            partial_a: self.new_ram_read_write_buffer::<Fp128>(2 * address_partial_capacity)?,
            partial_b: self.new_ram_read_write_buffer::<Fp128>(2 * address_partial_capacity)?,
            hot_partial_a: self
                .new_ram_read_write_buffer::<Fp128>(2 * hot_message_chunk_capacity)?,
            hot_partial_b: self
                .new_ram_read_write_buffer::<Fp128>(2 * hot_message_chunk_capacity)?,
            aux_blocks: self.new_ram_read_write_buffer::<u32>(hot_state_capacity)?,
            aux_previous: self.new_ram_read_write_buffer::<u64>(hot_state_capacity)?,
            aux_next: self.new_ram_read_write_buffer::<u64>(hot_state_capacity)?,
            aux_values: self.new_ram_read_write_buffer::<Fp128>(hot_state_capacity)?,
            aux_ra: self.new_ram_read_write_buffer::<Fp128>(hot_state_capacity)?,
        };
        let cycle = CycleBuffers {
            segments: self.new_ram_read_write_buffer::<Segment>(tile_count)?,
            blocks: self.new_ram_read_write_buffer::<u32>(record_capacity)?,
            address_indices: use_record_prefix
                .then(|| self.new_ram_read_write_buffer::<u32>(record_capacity))
                .transpose()?,
            hamming: self.new_ram_read_write_buffer::<Fp128>(cycle_state_capacity)?,
            increments: self.new_ram_read_write_buffer::<Fp128>(cycle_state_capacity)?,
            partial_a: self.new_ram_read_write_buffer::<Fp128>(2 * tile_count)?,
            partial_b: self.new_ram_read_write_buffer::<Fp128>(2 * tile_count)?,
        };
        let e_in = self.new_ram_read_write_buffer::<Fp128>(maximum_eq_fields)?;
        let e_out = self.new_ram_read_write_buffer::<Fp128>(maximum_eq_fields)?;
        let allocation_wall = allocation_started.elapsed();
        let initialization_started = Instant::now();
        buffer_slice_mut::<u32>(&address.bounded_segments, plan.bounded_segments.len())
            .copy_from_slice(&plan.bounded_segments);
        buffer_slice_mut::<HotSegment>(&address.hot_segments, plan.hot_segments.len())
            .copy_from_slice(&plan.hot_segments);
        buffer_slice_mut::<HotChunk>(&address.hot_message_chunks, plan.hot_message_chunks.len())
            .copy_from_slice(&plan.hot_message_chunks);
        initialize_segments(
            &address.segments,
            &cycle.segments,
            &plan,
            address_count,
            tile_count,
        );
        let address_partials = buffer_slice_mut::<Fp128>(&address.partial_a, 2 * address_count);
        for hot in &plan.hot_segments {
            let index = hot.segment_index as usize;
            address_partials[index] = Fp128::ZERO;
            address_partials[address_count + index] = Fp128::ZERO;
        }
        let workers = std::mem::take(&mut plan.workers);
        let mut gpu_scatter_wall = Duration::ZERO;
        let mut gpu_scatter_active = Duration::ZERO;
        let activity = match source {
            RamReadWritePreparationSource::Dense {
                addresses,
                previous,
                next,
            } => {
                let writers = BucketWriters::new(&address, &cycle);
                scatter_buckets(
                    addresses,
                    previous,
                    next,
                    plan.chunk_length,
                    workers,
                    writers,
                );
                None
            }
            RamReadWritePreparationSource::Direct {
                addresses,
                value_at,
            } => {
                let writers = BucketWriters::new(&address, &cycle);
                Some(scatter_direct_buckets(
                    addresses,
                    plan.chunk_length,
                    workers,
                    writers,
                    value_at,
                )?)
            }
            RamReadWritePreparationSource::Records { source } if gpu_record_scatter => {
                let timing = self.scatter_record_buckets_gpu(
                    source,
                    &workers,
                    address_count,
                    &address,
                    &cycle,
                )?;
                gpu_scatter_wall = timing.0;
                gpu_scatter_active = timing.1;
                None
            }
            RamReadWritePreparationSource::Records { source } => {
                let writers = BucketWriters::new(&address, &cycle);
                scatter_record_buckets(source.chunks(), plan.chunk_length, workers, writers)?;
                None
            }
        };
        let record_prefix = use_record_prefix
            .then(|| {
                self.prepare_record_prefix_state(
                    &address,
                    &cycle,
                    &plan,
                    address_count,
                    tile_count,
                    RAM_READ_WRITE_RECORD_PREFIX_ROUNDS,
                )
            })
            .transpose()?;
        if let Some(prefix) = &record_prefix {
            let capacity = prefix.hot_state_entries.max(1);
            address.aux_blocks = self.new_ram_read_write_buffer::<u32>(capacity)?;
            address.aux_previous = self.new_ram_read_write_buffer::<u64>(capacity)?;
            address.aux_next = self.new_ram_read_write_buffer::<u64>(capacity)?;
            address.aux_values = self.new_ram_read_write_buffer::<Fp128>(capacity)?;
            address.aux_ra = self.new_ram_read_write_buffer::<Fp128>(capacity)?;
        }
        let initialization_and_scatter_wall = initialization_started.elapsed();

        let pipeline_setup_started = Instant::now();
        let address_pipeline = self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_PIPELINE)?;
        let address_bounded_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_BOUNDED_PIPELINE)?;
        let address_hot_count_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_COUNT_PIPELINE)?;
        let address_hot_prefix_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_PREFIX_PIPELINE)?;
        let address_hot_scatter_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_SCATTER_PIPELINE)?;
        let address_hot_message_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_MESSAGE_PIPELINE)?;
        let cycle_pipeline = self.compile_named_pipeline(RAM_READ_WRITE_CYCLE_PIPELINE)?;
        let prefix_address = self.compile_named_pipeline(RAM_READ_WRITE_PREFIX_ADDRESS_PIPELINE)?;
        let prefix_address_transition =
            self.compile_named_pipeline(RAM_READ_WRITE_PREFIX_ADDRESS_TRANSITION_PIPELINE)?;
        let prefix_hot_transition =
            self.compile_named_pipeline(RAM_READ_WRITE_PREFIX_HOT_TRANSITION_PIPELINE)?;
        let prefix_cycle = self.compile_named_pipeline(RAM_READ_WRITE_PREFIX_CYCLE_PIPELINE)?;
        let prefix_cycle_transition =
            self.compile_named_pipeline(RAM_READ_WRITE_PREFIX_CYCLE_TRANSITION_PIPELINE)?;
        let reduction = self.compile_named_pipeline(RAM_READ_WRITE_REDUCTION_PIPELINE)?;
        let address_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_THREADS),
            Self::limits(&address_pipeline),
        )?;
        let address_bounded_limits = Self::limits(&address_bounded_pipeline);
        let address_bounded_threads =
            Self::resolve_threadgroup_width(Some(RAM_READ_WRITE_THREADS), address_bounded_limits)?;
        let cycle_limits = Self::limits(&cycle_pipeline);
        let cycle_threads =
            Self::resolve_threadgroup_width(Some(RAM_READ_WRITE_THREADS), cycle_limits)?;
        let address_hot_count_limits = Self::limits(&address_hot_count_pipeline);
        let address_hot_prefix_limits = Self::limits(&address_hot_prefix_pipeline);
        let address_hot_scatter_limits = Self::limits(&address_hot_scatter_pipeline);
        let address_hot_count_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_HOT_COMPACTION_THREADS),
            address_hot_count_limits,
        )?;
        let address_hot_prefix_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_HOT_COMPACTION_THREADS),
            address_hot_prefix_limits,
        )?;
        let address_hot_scatter_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_HOT_COMPACTION_THREADS),
            address_hot_scatter_limits,
        )?;
        let address_hot_message_limits = Self::limits(&address_hot_message_pipeline);
        let address_hot_message_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_THREADS),
            address_hot_message_limits,
        )?;
        let reduction_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_REDUCTION_WIDTH),
            Self::limits(&reduction),
        )?;
        let prefix_schedule_valid = [
            (&prefix_address, RAM_READ_WRITE_HOT_THREADGROUP_BYTES_MAX),
            (
                &prefix_address_transition,
                RAM_READ_WRITE_BOUNDED_THREADGROUP_BYTES_MAX,
            ),
            (
                &prefix_hot_transition,
                RAM_READ_WRITE_HOT_THREADGROUP_BYTES_MAX,
            ),
            (&prefix_cycle, RAM_READ_WRITE_CYCLE_THREADGROUP_BYTES_MAX),
            (
                &prefix_cycle_transition,
                RAM_READ_WRITE_CYCLE_THREADGROUP_BYTES_MAX,
            ),
        ]
        .into_iter()
        .try_fold(true, |valid, (pipeline, memory_limit)| {
            let limits = Self::limits(pipeline);
            Ok::<_, MetalError>(
                valid
                    && limits.thread_execution_width == RAM_READ_WRITE_SIMD_WIDTH
                    && Self::resolve_threadgroup_width(Some(RAM_READ_WRITE_THREADS), limits)?
                        == RAM_READ_WRITE_THREADS
                    && limits.static_threadgroup_memory_length <= memory_limit,
            )
        })?;
        let hot_threadgroup_bytes = address_hot_count_limits
            .static_threadgroup_memory_length
            .max(address_hot_prefix_limits.static_threadgroup_memory_length)
            .max(address_hot_scatter_limits.static_threadgroup_memory_length);
        if address_hot_count_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_hot_prefix_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_hot_scatter_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_hot_message_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_bounded_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || cycle_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_threads != RAM_READ_WRITE_THREADS
            || address_bounded_threads != RAM_READ_WRITE_THREADS
            || address_hot_count_threads != RAM_READ_WRITE_HOT_COMPACTION_THREADS
            || address_hot_prefix_threads != RAM_READ_WRITE_HOT_COMPACTION_THREADS
            || address_hot_scatter_threads != RAM_READ_WRITE_HOT_COMPACTION_THREADS
            || address_hot_message_threads != RAM_READ_WRITE_THREADS
            || cycle_threads != RAM_READ_WRITE_THREADS
            || reduction_threads != RAM_READ_WRITE_REDUCTION_WIDTH
            || hot_threadgroup_bytes > RAM_READ_WRITE_HOT_THREADGROUP_BYTES_MAX
            || address_bounded_limits.static_threadgroup_memory_length
                > RAM_READ_WRITE_BOUNDED_THREADGROUP_BYTES_MAX
            || cycle_limits.static_threadgroup_memory_length
                > RAM_READ_WRITE_CYCLE_THREADGROUP_BYTES_MAX
            || !prefix_schedule_valid
        {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write pipeline width differs from its checked schedule",
            ));
        }
        let pipeline_setup_wall = pipeline_setup_started.elapsed();

        plan.stats.hot_compaction_threads = address_hot_scatter_threads;
        plan.stats.hot_compaction_threadgroup_bytes = hot_threadgroup_bytes;
        if let Some(prefix) = &record_prefix {
            plan.stats.hot_message_chunks = prefix.hot_message_chunk_count;
            plan.stats.hot_state_entries = prefix.hot_state_entries;
            plan.stats.hot_auxiliary_bytes = prefix
                .hot_state_entries
                .saturating_mul(size_of::<u32>() + 2 * size_of::<u64>() + 2 * size_of::<Fp128>());
        }
        plan.stats.address_bytes = address.resident_bytes();
        plan.stats.cycle_bytes = cycle.resident_bytes();
        Ok((
            RamReadWriteSequence {
                context: self.clone(),
                pipelines: RamReadWritePipelines {
                    address: address_pipeline,
                    address_bounded: address_bounded_pipeline,
                    address_hot_count: address_hot_count_pipeline,
                    address_hot_prefix: address_hot_prefix_pipeline,
                    address_hot_scatter: address_hot_scatter_pipeline,
                    address_hot_message: address_hot_message_pipeline,
                    cycle: cycle_pipeline,
                    prefix_address,
                    prefix_address_transition,
                    prefix_hot_transition,
                    prefix_cycle,
                    prefix_cycle_transition,
                    reduction,
                },
                buffers: RamReadWriteBuffers {
                    address,
                    cycle,
                    e_in,
                    e_out,
                },
                log_t,
                address_count,
                hot_segment_threshold,
                bounded_address_count: plan.bounded_segments.len(),
                hot_address_count: plan.hot_segments.len(),
                hot_message_chunk_count: plan.hot_message_chunks.len(),
                hot_state_entries: record_prefix
                    .as_ref()
                    .map_or(hot_state_capacity, |prefix| prefix.hot_state_entries),
                hot_address_threads: address_hot_scatter_threads,
                tile_count,
                tile_log,
                rounds_bound: 0,
                initial_message_emitted: false,
                cycle_resident: true,
                finished: false,
                stats: plan.stats,
                preparation_timing: RamReadWritePreparationTiming {
                    bucket_plan: bucket_plan_wall,
                    allocation: allocation_wall,
                    initialization_and_scatter: initialization_and_scatter_wall,
                    pipeline_setup: pipeline_setup_wall,
                    gpu_scatter_wall,
                    gpu_scatter_active,
                    total: prepare_started.elapsed(),
                },
                timing_wall: Duration::ZERO,
                timing_gpu_active: Duration::ZERO,
                dispatch_profiler: None,
                record_prefix,
                hot_source_aux: false,
            },
            activity,
        ))
    }

    pub(crate) fn prepare_ram_raf_segmented_accesses(
        &self,
        rows: usize,
        address_count: usize,
        cycles: &[u32],
        addresses: &[u32],
    ) -> Result<RamRafSegmentedAddressPlane, MetalError> {
        if cycles.len() != addresses.len() {
            return Err(MetalError::LengthMismatch {
                lhs: cycles.len(),
                rhs: addresses.len(),
            });
        }
        if cycles.is_empty()
            || rows < 1 << 15
            || !rows.is_power_of_two()
            || address_count == 0
            || !address_count.is_power_of_two()
        {
            return Err(MetalError::InvalidRamRafState(
                "retained RAM RAF source has inconsistent geometry",
            ));
        }

        let mut lengths = vec![0u32; address_count];
        let mut previous_cycle = None;
        for (&cycle, &address) in cycles.iter().zip(addresses) {
            if cycle as usize >= rows
                || address as usize >= address_count
                || previous_cycle.is_some_and(|previous| previous >= cycle)
            {
                return Err(MetalError::InvalidRamRafState(
                    "retained RAM RAF source has an invalid access",
                ));
            }
            lengths[address as usize] = lengths[address as usize]
                .checked_add(1)
                .ok_or(MetalError::InputTooLong(cycles.len()))?;
            previous_cycle = Some(cycle);
        }

        let mut segments = Vec::with_capacity(address_count);
        let mut offset = 0u32;
        for &length in &lengths {
            segments.push(Segment {
                offset,
                length,
                capacity: length,
                aux_offset: 0,
            });
            offset = offset
                .checked_add(length)
                .ok_or(MetalError::InputTooLong(cycles.len()))?;
        }
        if offset as usize != cycles.len() {
            return Err(MetalError::InvalidRamRafState(
                "retained RAM RAF segment lengths disagree",
            ));
        }

        let mut cursors = segments
            .iter()
            .map(|segment| segment.offset)
            .collect::<Vec<_>>();
        let mut blocks = vec![0u32; cycles.len()];
        for (&cycle, &address) in cycles.iter().zip(addresses) {
            let cursor = &mut cursors[address as usize];
            blocks[*cursor as usize] = cycle;
            *cursor += 1;
        }

        let bounded_segments = lengths
            .iter()
            .enumerate()
            .filter_map(|(address, &length)| {
                ((length as usize) > RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD
                    && (length as usize) <= RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE)
                    .then_some(address as u32)
            })
            .collect::<Vec<_>>();
        let mut hot_segments = Vec::new();
        let mut hot_message_chunks = Vec::new();
        for (address, &length) in lengths.iter().enumerate() {
            if length as usize <= RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE {
                continue;
            }
            let first_chunk = hot_message_chunks.len();
            let hot_index = hot_segments.len();
            for local_offset in (0..length).step_by(RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE) {
                hot_message_chunks.push(HotChunk {
                    hot_index: hot_index as u32,
                    local_offset,
                });
            }
            hot_segments.push(HotSegment {
                segment_index: address as u32,
                first_chunk: first_chunk as u32,
                chunk_count: (hot_message_chunks.len() - first_chunk) as u32,
                aux_offset: 0,
            });
        }

        let storage_bytes = segments
            .len()
            .checked_mul(size_of::<Segment>())
            .and_then(|bytes| bytes.checked_add(blocks.len() * size_of::<u32>()))
            .and_then(|bytes| bytes.checked_add(bounded_segments.len() * size_of::<u32>()))
            .and_then(|bytes| bytes.checked_add(hot_segments.len() * size_of::<HotSegment>()))
            .and_then(|bytes| bytes.checked_add(hot_message_chunks.len() * size_of::<HotChunk>()))
            .ok_or(MetalError::InputTooLong(cycles.len()))?;
        self.validate_additional_working_set(
            u64::try_from(storage_bytes).map_err(|_| MetalError::InputTooLong(storage_bytes))?,
        )?;

        let segment_buffer = self.new_ram_read_write_buffer::<Segment>(segments.len())?;
        buffer_slice_mut::<Segment>(&segment_buffer, segments.len()).copy_from_slice(&segments);
        let block_buffer = self.new_ram_read_write_buffer::<u32>(blocks.len())?;
        buffer_slice_mut::<u32>(&block_buffer, blocks.len()).copy_from_slice(&blocks);
        let bounded_buffer =
            self.new_ram_read_write_buffer::<u32>(bounded_segments.len().max(1))?;
        buffer_slice_mut::<u32>(&bounded_buffer, bounded_segments.len())
            .copy_from_slice(&bounded_segments);
        let hot_segment_buffer =
            self.new_ram_read_write_buffer::<HotSegment>(hot_segments.len().max(1))?;
        buffer_slice_mut::<HotSegment>(&hot_segment_buffer, hot_segments.len())
            .copy_from_slice(&hot_segments);
        let hot_chunk_buffer =
            self.new_ram_read_write_buffer::<HotChunk>(hot_message_chunks.len().max(1))?;
        buffer_slice_mut::<HotChunk>(&hot_chunk_buffer, hot_message_chunks.len())
            .copy_from_slice(&hot_message_chunks);

        Ok(RamRafSegmentedAddressPlane {
            segments: segment_buffer,
            blocks: block_buffer,
            bounded_segments: bounded_buffer,
            hot_segments: hot_segment_buffer,
            hot_message_chunks: hot_chunk_buffer,
            rows,
            addresses: address_count,
            accesses: cycles.len(),
            bounded_address_count: bounded_segments.len(),
            hot_address_count: hot_segments.len(),
            hot_message_chunk_count: hot_message_chunks.len(),
            cold_segment_threshold: RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD,
            hot_message_chunk_size: RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE,
            device_registry_id: self.device_registry_id(),
        })
    }

    fn scatter_record_buckets_gpu(
        &self,
        source: &RamReadWriteRecordChunks,
        workers: &[WorkerPlan],
        address_count: usize,
        address: &AddressBuffers,
        cycle: &CycleBuffers,
    ) -> Result<(Duration, Duration), MetalError> {
        if source.chunks().len() != workers.len()
            || source.chunks().iter().zip(workers).any(|(arena, worker)| {
                arena.records().len() != worker.accesses
                    || arena.ranks().len() != worker.accesses
                    || worker.address_cursors.len() != address_count
            })
        {
            return Err(MetalError::InvalidRamReadWriteState(
                "GPU RAM scatter source disagrees with its stable worker plan",
            ));
        }
        let cursor_elements = workers
            .len()
            .checked_mul(address_count)
            .ok_or(MetalError::InputTooLong(address_count))?;
        let cursor_buffer = self.new_ram_read_write_buffer::<u32>(cursor_elements.max(1))?;
        let cursors = buffer_slice_mut::<u32>(&cursor_buffer, cursor_elements);
        for (destination, worker) in cursors.chunks_mut(address_count).zip(workers) {
            destination.copy_from_slice(&worker.address_cursors);
        }
        let mut source_buffers = Vec::with_capacity(source.chunks().len());
        for arena in source.chunks() {
            let record_bytes = u64::try_from(arena.record_allocation_bytes())
                .map_err(|_| MetalError::InputTooLong(arena.records().len()))?;
            let rank_bytes = u64::try_from(arena.rank_allocation_bytes())
                .map_err(|_| MetalError::InputTooLong(arena.ranks().len()))?;
            self.validate_buffer_length(record_bytes)?;
            self.validate_buffer_length(rank_bytes)?;
            let records = self.device.new_buffer_with_bytes_no_copy(
                arena.record_pointer().cast_mut().cast::<c_void>(),
                record_bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            let ranks = self.device.new_buffer_with_bytes_no_copy(
                arena.rank_pointer().cast_mut().cast::<c_void>(),
                rank_bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            source_buffers.push((records, ranks));
        }
        let pipeline = self.compile_named_pipeline(RAM_READ_WRITE_INITIAL_SCATTER_PIPELINE)?;
        let limits = Self::limits(&pipeline);
        if limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH {
            return Err(MetalError::InvalidRamReadWriteState(
                "GPU RAM scatter SIMD width differs from its checked schedule",
            ));
        }
        let threads = Self::resolve_threadgroup_width(Some(RAM_READ_WRITE_THREADS), limits)?;
        let address_count_u32 =
            u32::try_from(address_count).map_err(|_| MetalError::InputTooLong(address_count))?;
        let span = tracing::info_span!(
            "MetalRamReadWrite::gpu_initial_scatter",
            records = source.access_count(),
            workers = workers.len(),
            wall_ns = tracing::field::Empty,
            gpu_active_ns = tracing::field::Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let command_buffer = autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer().to_owned();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(2, Some(&cursor_buffer), 0);
            encoder.set_buffer(3, Some(&address.blocks), 0);
            encoder.set_buffer(4, Some(&address.previous), 0);
            encoder.set_buffer(5, Some(&address.next), 0);
            encoder.set_buffer(6, Some(&cycle.blocks), 0);
            encoder.set_buffer(7, Some(&cycle.increments), 0);
            for (worker_index, ((arena, worker), (records, ranks))) in source
                .chunks()
                .iter()
                .zip(workers)
                .zip(&source_buffers)
                .enumerate()
            {
                if worker.accesses == 0 {
                    continue;
                }
                let params = InitialScatterParams {
                    records: u32::try_from(worker.accesses)
                        .map_err(|_| MetalError::InputTooLong(worker.accesses))?,
                    address_count: address_count_u32,
                    cycle_offset: worker.cycle_cursor,
                    reserved: 0,
                };
                let cursor_offset = worker_index
                    .checked_mul(address_count)
                    .and_then(|elements| elements.checked_mul(size_of::<u32>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or(MetalError::InputTooLong(address_count))?;
                encoder.set_buffer(0, Some(records), 0);
                encoder.set_buffer(1, Some(ranks), 0);
                encoder.set_buffer(2, Some(&cursor_buffer), cursor_offset);
                set_inline_bytes(encoder, 8, &params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: arena.records().len().div_ceil(threads) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.end_encoding();
            command_buffer.commit();
            Ok::<_, MetalError>(command_buffer)
        })?;
        command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command_buffer)?;
        let wall = started.elapsed();
        let _ = span.record(
            "wall_ns",
            u64::try_from(wall.as_nanos()).unwrap_or(u64::MAX),
        );
        let _ = span.record(
            "gpu_active_ns",
            u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX),
        );
        Ok((wall, gpu_active))
    }

    fn prepare_record_prefix_state(
        &self,
        address: &AddressBuffers,
        cycle: &CycleBuffers,
        plan: &BucketPlan,
        address_count: usize,
        tile_count: usize,
        rounds: usize,
    ) -> Result<RecordPrefixState, MetalError> {
        let accesses = plan.stats.accesses;
        let address_blocks = buffer_slice_mut::<u32>(&address.blocks, accesses.max(1));
        let cycle_blocks = buffer_slice::<u32>(&cycle.blocks, accesses.max(1));
        let mut address_live_entries = vec![0usize; rounds + 1];
        let mut cycle_live_entries = vec![0usize; rounds + 1];
        let mut transition_segments = Vec::with_capacity(address_count);
        let mut hot_destinations = vec![u32::MAX; accesses.max(1)];
        let mut hot_segments = Vec::with_capacity(plan.hot_segments.len());
        let mut hot_message_chunks = Vec::new();
        let mut hot_state_entries = 0usize;
        let mut address_offset = 0usize;
        let mut source_hot = plan.hot_segments.iter().peekable();
        for (segment_index, &length) in plan.address_lengths.iter().enumerate() {
            let length = length as usize;
            let begin = address_offset;
            let end = begin + length;
            let blocks = &address_blocks[begin..end];
            let counts = (0..=rounds)
                .map(|round| count_shifted_blocks(blocks, round))
                .collect::<Vec<_>>();
            for (total, count) in address_live_entries.iter_mut().zip(&counts) {
                *total += count;
            }
            let hot = source_hot
                .next_if(|hot| hot.segment_index as usize == segment_index)
                .copied();
            let aux_offset =
                u32::try_from(hot_state_entries).map_err(|_| MetalError::InputTooLong(accesses))?;
            transition_segments.push(Segment {
                offset: u32::try_from(begin).map_err(|_| MetalError::InputTooLong(accesses))?,
                length: u32::try_from(counts[rounds])
                    .map_err(|_| MetalError::InputTooLong(accesses))?,
                capacity: u32::try_from(length).map_err(|_| MetalError::InputTooLong(accesses))?,
                aux_offset: if hot.is_some() { aux_offset } else { 0 },
            });
            if hot.is_some() {
                let first_chunk = hot_message_chunks.len();
                for local_offset in
                    (0..counts[rounds]).step_by(RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE)
                {
                    hot_message_chunks.push(HotChunk {
                        hot_index: hot_segments.len() as u32,
                        local_offset: local_offset as u32,
                    });
                }
                hot_segments.push(HotSegment {
                    segment_index: segment_index as u32,
                    first_chunk: first_chunk as u32,
                    chunk_count: (hot_message_chunks.len() - first_chunk) as u32,
                    aux_offset,
                });
                let mut destination = aux_offset;
                let mut previous = None;
                for index in begin..end {
                    let block = address_blocks[index] >> rounds;
                    if previous != Some(block) {
                        hot_destinations[index] = destination;
                        destination = destination
                            .checked_add(1)
                            .ok_or(MetalError::InputTooLong(accesses))?;
                        previous = Some(block);
                    }
                }
                if destination - aux_offset != counts[rounds] as u32 {
                    return Err(MetalError::InvalidRamReadWriteState(
                        "RAM record prefix hot destinations disagree with the frontier",
                    ));
                }
                hot_state_entries = hot_state_entries
                    .checked_add(counts[rounds])
                    .ok_or(MetalError::InputTooLong(accesses))?;
            }
            if length != 0 {
                address_blocks[begin] |= RAM_READ_WRITE_PREFIX_SEGMENT_START;
            }
            address_offset = end;
        }
        if address_offset != accesses {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM record prefix address segments disagree with access count",
            ));
        }
        let mut transition_cycle_segments = Vec::with_capacity(tile_count);
        let mut output_cycle_segments = Vec::with_capacity(tile_count);
        let mut cycle_offset = 0usize;
        let mut cycle_output_offset = 0usize;
        for &length in &plan.tile_lengths {
            let length = length as usize;
            let begin = cycle_offset;
            let end = begin + length;
            let blocks = &cycle_blocks[begin..end];
            let counts = (0..=rounds)
                .map(|round| count_shifted_blocks(blocks, round))
                .collect::<Vec<_>>();
            for (total, count) in cycle_live_entries.iter_mut().zip(&counts) {
                *total += count;
            }
            transition_cycle_segments.push(Segment {
                offset: u32::try_from(begin).map_err(|_| MetalError::InputTooLong(accesses))?,
                length: u32::try_from(counts[rounds])
                    .map_err(|_| MetalError::InputTooLong(accesses))?,
                capacity: u32::try_from(length).map_err(|_| MetalError::InputTooLong(accesses))?,
                aux_offset: u32::try_from(cycle_output_offset)
                    .map_err(|_| MetalError::InputTooLong(accesses))?,
            });
            output_cycle_segments.push(Segment {
                offset: u32::try_from(cycle_output_offset)
                    .map_err(|_| MetalError::InputTooLong(accesses))?,
                length: u32::try_from(counts[rounds])
                    .map_err(|_| MetalError::InputTooLong(accesses))?,
                capacity: u32::try_from(counts[rounds])
                    .map_err(|_| MetalError::InputTooLong(accesses))?,
                aux_offset: 0,
            });
            cycle_output_offset = cycle_output_offset
                .checked_add(counts[rounds])
                .ok_or(MetalError::InputTooLong(accesses))?;
            cycle_offset = end;
        }
        if cycle_offset != accesses {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM record prefix cycle segments disagree with access count",
            ));
        }
        if cycle_output_offset * size_of::<Fp128>() > cycle.hamming.length() as usize {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM record prefix cycle frontier exceeds its output allocation",
            ));
        }

        let address_segment_buffer = self.new_ram_read_write_buffer::<Segment>(address_count)?;
        buffer_slice_mut::<Segment>(&address_segment_buffer, address_count)
            .copy_from_slice(&transition_segments);
        let bounded_segment_buffer =
            self.new_ram_read_write_buffer::<u32>(plan.bounded_segments.len().max(1))?;
        buffer_slice_mut::<u32>(&bounded_segment_buffer, plan.bounded_segments.len())
            .copy_from_slice(&plan.bounded_segments);
        let hot_segment_buffer =
            self.new_ram_read_write_buffer::<HotSegment>(hot_segments.len().max(1))?;
        buffer_slice_mut::<HotSegment>(&hot_segment_buffer, hot_segments.len())
            .copy_from_slice(&hot_segments);
        let hot_chunk_buffer =
            self.new_ram_read_write_buffer::<HotChunk>(hot_message_chunks.len().max(1))?;
        buffer_slice_mut::<HotChunk>(&hot_chunk_buffer, hot_message_chunks.len())
            .copy_from_slice(&hot_message_chunks);
        let cycle_segment_buffer = self.new_ram_read_write_buffer::<Segment>(tile_count)?;
        buffer_slice_mut::<Segment>(&cycle_segment_buffer, tile_count)
            .copy_from_slice(&transition_cycle_segments);
        let cycle_output_segment_buffer = self.new_ram_read_write_buffer::<Segment>(tile_count)?;
        buffer_slice_mut::<Segment>(&cycle_output_segment_buffer, tile_count)
            .copy_from_slice(&output_cycle_segments);
        let cycle_output_block_buffer =
            self.new_ram_read_write_buffer::<u32>(cycle_output_offset.max(1))?;
        let hot_destination_buffer = self.new_ram_read_write_buffer::<u32>(accesses.max(1))?;
        buffer_slice_mut::<u32>(&hot_destination_buffer, accesses.max(1))
            .copy_from_slice(&hot_destinations);
        let table_capacity = 1usize << rounds;
        let weight_table = self.new_ram_read_write_buffer::<Fp128>(table_capacity)?;
        let suffix_table = self.new_ram_read_write_buffer::<Fp128>(table_capacity)?;
        buffer_slice_mut::<Fp128>(&weight_table, table_capacity).fill(Fp128::ZERO);
        buffer_slice_mut::<Fp128>(&suffix_table, table_capacity).fill(Fp128::ZERO);
        buffer_slice_mut::<Fp128>(&weight_table, table_capacity)[0] = Fp128::ONE;

        Ok(RecordPrefixState {
            rounds,
            table_entries: 1,
            weight_table,
            suffix_table,
            address_segments: address_segment_buffer,
            bounded_segments: bounded_segment_buffer,
            hot_segments: hot_segment_buffer,
            hot_message_chunks: hot_chunk_buffer,
            cycle_segments: cycle_segment_buffer,
            cycle_output_segments: cycle_output_segment_buffer,
            cycle_output_blocks: cycle_output_block_buffer,
            hot_destinations: hot_destination_buffer,
            bounded_address_count: plan.bounded_segments.len(),
            hot_address_count: hot_segments.len(),
            hot_message_chunk_count: hot_message_chunks.len(),
            hot_state_entries,
            address_live_entries,
            cycle_live_entries,
        })
    }

    fn new_ram_read_write_buffer<T>(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = elements
            .checked_mul(size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(elements))?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl RamReadWriteSequence {
    pub(crate) const fn bucket_stats(&self) -> RamReadWriteBucketStats {
        self.stats
    }

    pub(crate) const fn preparation_timing(&self) -> RamReadWritePreparationTiming {
        self.preparation_timing
    }

    pub(crate) fn ram_raf_segmented_address_plane(&self) -> RamRafSegmentedAddressPlane {
        RamRafSegmentedAddressPlane {
            segments: self.buffers.address.segments.clone(),
            blocks: self.buffers.address.blocks.clone(),
            bounded_segments: self.buffers.address.bounded_segments.clone(),
            hot_segments: self.buffers.address.hot_segments.clone(),
            hot_message_chunks: self.buffers.address.hot_message_chunks.clone(),
            rows: 1usize << self.log_t,
            addresses: self.address_count,
            accesses: self.stats.accesses,
            bounded_address_count: self.bounded_address_count,
            hot_address_count: self.hot_address_count,
            hot_message_chunk_count: self.hot_message_chunk_count,
            cold_segment_threshold: self.hot_segment_threshold,
            hot_message_chunk_size: RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE,
            device_registry_id: self.context.device_registry_id(),
        }
    }

    #[cfg(feature = "test-utils")]
    pub(crate) fn enable_dispatch_timing(&mut self) -> Result<(), MetalError> {
        if self.dispatch_profiler.is_some() {
            return Ok(());
        }
        if !self
            .context
            .device
            .supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Err(MetalError::InvalidRamReadWriteState(
                "Metal device does not support compute-stage counter sampling",
            ));
        }
        let counter_set = self
            .context
            .device
            .counter_sets()
            .into_iter()
            .find(|counter_set| counter_set.name() == "timestamp")
            .ok_or(MetalError::InvalidRamReadWriteState(
                "Metal device has no timestamp counter set",
            ))?;
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count(RAM_READ_WRITE_DISPATCH_SAMPLES);
        descriptor.set_counter_set(&counter_set);
        let samples = self
            .context
            .device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(|_| {
                MetalError::InvalidRamReadWriteState(
                    "Metal timestamp counter sample buffer creation failed",
                )
            })?;
        let resolved = self.context.device.new_buffer(
            RAM_READ_WRITE_DISPATCH_SAMPLES * size_of::<u64>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_profiler = Some(RamReadWriteDispatchProfiler { samples, resolved });
        Ok(())
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.stats.address_bytes
            + self.stats.cycle_bytes
            + self.buffers.e_in.length() as usize
            + self.buffers.e_out.length() as usize
            + self
                .record_prefix
                .as_ref()
                .map_or(0, RecordPrefixState::resident_bytes)
    }

    pub(crate) fn apply_initial_memory(&self, memory: &mut [u64]) -> Result<(), MetalError> {
        if memory.len() != self.address_count {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write initial memory has the wrong address domain",
            ));
        }
        let segments = buffer_slice::<Segment>(&self.buffers.address.segments, self.address_count);
        let previous =
            buffer_slice::<u64>(&self.buffers.address.previous, self.stats.accesses.max(1));
        for (address, segment) in segments.iter().enumerate() {
            if segment.length != 0 {
                memory[address] = previous[segment.offset as usize];
            }
        }
        Ok(())
    }

    pub(crate) fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<RamReadWriteSequenceObservation, MetalError> {
        if self.initial_message_emitted || self.finished {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write initial message was emitted twice",
            ));
        }
        self.write_weights(0, e_in, e_out)?;
        let command = if self.record_prefix.is_some() {
            self.submit_prefix_phase(0, e_in.len())?
        } else {
            self.submit_phase(None, true, true, e_in.len())?
        };
        let observation = self.complete_phase(command, false)?;
        self.initial_message_emitted = true;
        Ok(observation)
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<RamReadWriteSequenceObservation, MetalError> {
        if !self.initial_message_emitted || self.finished || self.rounds_bound + 1 >= self.log_t {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write transition is outside the cycle-message phase",
            ));
        }
        let new_rounds_bound = self.rounds_bound + 1;
        self.write_weights(new_rounds_bound, e_in, e_out)?;
        if let Some(prefix_rounds) = self.record_prefix.as_ref().map(|prefix| prefix.rounds) {
            self.expand_record_prefix_table(challenge, new_rounds_bound)?;
            if new_rounds_bound < prefix_rounds {
                let command = self.submit_prefix_phase(new_rounds_bound, e_in.len())?;
                let observation = self.complete_phase(command, false)?;
                self.rounds_bound = new_rounds_bound;
                return Ok(observation);
            }
            if new_rounds_bound == prefix_rounds {
                return self.transition_record_prefix(e_in.len());
            }
        }
        let cycle_message = self.cycle_resident && new_rounds_bound < self.tile_log;
        let handoff_cycle = self.cycle_resident && new_rounds_bound == self.tile_log;
        let command = self.submit_phase(Some(challenge), true, cycle_message, e_in.len())?;
        let mut observation = self.complete_phase(command, handoff_cycle)?;
        self.rounds_bound = new_rounds_bound;
        if self.hot_address_count != 0 {
            self.hot_source_aux = !self.hot_source_aux;
        }
        if handoff_cycle {
            self.cycle_resident = false;
            observation.cycle_quadratic = None;
        }
        Ok(observation)
    }

    pub(crate) fn finish(
        &mut self,
        challenge: AkitaField,
    ) -> Result<RamReadWriteFinish, MetalError> {
        if !self.initial_message_emitted || self.finished || self.rounds_bound + 1 != self.log_t {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write finish needs the final cycle challenge",
            ));
        }
        let handoff_cycle = self.cycle_resident;
        let command = self.submit_phase(Some(challenge), false, false, 0)?;
        let observation = self.complete_phase(command, handoff_cycle)?;
        if self.hot_address_count != 0 {
            self.hot_source_aux = !self.hot_source_aux;
        }
        self.rounds_bound += 1;
        self.cycle_resident = false;
        self.finished = true;
        let address_roots = self.read_address_roots()?;
        Ok(RamReadWriteFinish {
            address_roots,
            cycle_roots: observation.cycle_roots,
            wall: observation.wall,
            gpu_active: observation.gpu_active,
            dispatch_timing: observation.dispatch_timing,
        })
    }

    fn expand_record_prefix_table(
        &mut self,
        challenge: AkitaField,
        rounds_bound: usize,
    ) -> Result<(), MetalError> {
        let prefix = self
            .record_prefix
            .as_mut()
            .ok_or(MetalError::InvalidRamReadWriteState(
                "RAM record prefix state disappeared",
            ))?;
        if rounds_bound == 0 || rounds_bound > prefix.rounds {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM record prefix table expansion is outside its retained rounds",
            ));
        }
        let previous_entries = 1usize << (rounds_bound - 1);
        if prefix.table_entries != previous_entries {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM record prefix table is out of sequence",
            ));
        }
        let entries = 2 * previous_entries;
        let one_minus_challenge = AkitaField::one() - challenge;
        let weights = buffer_slice_mut::<Fp128>(&prefix.weight_table, entries);
        let suffixes = buffer_slice_mut::<Fp128>(&prefix.suffix_table, entries);
        for index in 0..previous_entries {
            let weight = weights[index].into_jolt_field::<AkitaField>();
            let suffix = suffixes[index].into_jolt_field::<AkitaField>();
            weights[index] = Fp128::from_jolt_field(&(one_minus_challenge * weight));
            suffixes[index] = Fp128::from_jolt_field(&(challenge + one_minus_challenge * suffix));
            weights[previous_entries + index] = Fp128::from_jolt_field(&(challenge * weight));
            suffixes[previous_entries + index] = Fp128::from_jolt_field(&(challenge * suffix));
        }
        prefix.table_entries = entries;
        Ok(())
    }

    fn submit_prefix_phase(
        &self,
        rounds_bound: usize,
        e_in_length: usize,
    ) -> Result<SequenceCommand, MetalError> {
        let prefix = self
            .record_prefix
            .as_ref()
            .ok_or(MetalError::InvalidRamReadWriteState(
                "RAM record prefix state disappeared",
            ))?;
        if rounds_bound >= prefix.rounds {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM record prefix message exceeded its retained rounds",
            ));
        }
        let address_groups = self.stats.accesses.div_ceil(RAM_READ_WRITE_THREADS);
        if 2 * address_groups * size_of::<Fp128>()
            > self.buffers.address.partial_a.length() as usize
        {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM record prefix address partials exceed their allocation",
            ));
        }
        let address_params = PrefixPhaseParams {
            records: u32::try_from(self.stats.accesses)
                .map_err(|_| MetalError::InputTooLong(self.stats.accesses))?,
            output_stride: u32::try_from(address_groups)
                .map_err(|_| MetalError::InputTooLong(address_groups))?,
            e_in_length: u32::try_from(e_in_length)
                .map_err(|_| MetalError::InputTooLong(e_in_length))?,
            rounds_bound: u32::try_from(rounds_bound)
                .map_err(|_| MetalError::InputTooLong(rounds_bound))?,
            bind: u32::from(rounds_bound != 0),
            reserved: [0; 3],
        };
        let mut cycle_params = address_params;
        cycle_params.output_stride = u32::try_from(self.tile_count)
            .map_err(|_| MetalError::InputTooLong(self.tile_count))?;
        let address_live_entries = *prefix.address_live_entries.get(rounds_bound).ok_or(
            MetalError::InvalidRamReadWriteState("RAM record prefix address frontier is missing"),
        )?;
        let cycle_live_entries = *prefix.cycle_live_entries.get(rounds_bound).ok_or(
            MetalError::InvalidRamReadWriteState("RAM record prefix cycle frontier is missing"),
        )?;
        let submitted_at = Instant::now();
        let timestamp_calibration_start = self.dispatch_profiler.as_ref().map(|_| {
            let mut cpu = 0;
            let mut gpu = 0;
            self.context.device.sample_timestamps(&mut cpu, &mut gpu);
            (cpu, gpu)
        });
        let dispatch_activity = [true, false, false, false, false, true, true];
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer().to_owned();
            let outputs = if self.dispatch_profiler.is_some() {
                let mut outputs = (None, None, None);
                for stage in 0..RAM_READ_WRITE_DISPATCH_STAGES {
                    let encoder = self.profiled_encoder(&command_buffer, stage)?;
                    match stage {
                        0 => self.encode_prefix_address(encoder, prefix, address_params),
                        5 => self.encode_prefix_cycle(encoder, prefix, cycle_params)?,
                        6 => {
                            outputs = self.encode_prefix_reductions(
                                encoder,
                                address_groups,
                                self.tile_count,
                            )?;
                        }
                        _ => {}
                    }
                    encoder.end_encoding();
                }
                outputs
            } else {
                let encoder = command_buffer.new_compute_command_encoder();
                self.encode_prefix_address(encoder, prefix, address_params);
                self.encode_prefix_cycle(encoder, prefix, cycle_params)?;
                let outputs =
                    self.encode_prefix_reductions(encoder, address_groups, self.tile_count)?;
                encoder.end_encoding();
                outputs
            };
            self.resolve_dispatch_counters(&command_buffer);
            command_buffer.commit();
            Ok(SequenceCommand {
                command_buffer,
                address_output: outputs.0,
                hot_address_output: None,
                cycle_output: outputs.2,
                submitted_at,
                timestamp_calibration_start,
                dispatch_activity,
                live_entries: Some((address_live_entries, cycle_live_entries)),
            })
        })
    }

    fn transition_record_prefix(
        &mut self,
        e_in_length: usize,
    ) -> Result<RamReadWriteSequenceObservation, MetalError> {
        let rounds = self
            .record_prefix
            .as_ref()
            .map(|prefix| prefix.rounds)
            .ok_or(MetalError::InvalidRamReadWriteState(
                "RAM record prefix state disappeared",
            ))?;
        let command = self.submit_prefix_transition()?;
        let transition = self.complete_phase(command, false)?;
        self.activate_record_prefix()?;
        self.rounds_bound = rounds;
        self.hot_source_aux = self.hot_address_count != 0;

        let cycle_message = self.cycle_resident && rounds < self.tile_log;
        let handoff_cycle = self.cycle_resident && rounds == self.tile_log;
        let command = self.submit_phase(None, true, cycle_message, e_in_length)?;
        let mut observation = self.complete_phase(command, handoff_cycle)?;
        observation.wall += transition.wall;
        observation.gpu_active += transition.gpu_active;
        match (&mut observation.dispatch_timing, transition.dispatch_timing) {
            (Some(total), Some(transition)) => *total += transition,
            (None, Some(transition)) => observation.dispatch_timing = Some(transition),
            _ => {}
        }
        if handoff_cycle {
            self.cycle_resident = false;
            observation.cycle_quadratic = None;
        }
        Ok(observation)
    }

    fn submit_prefix_transition(&self) -> Result<SequenceCommand, MetalError> {
        let prefix = self
            .record_prefix
            .as_ref()
            .ok_or(MetalError::InvalidRamReadWriteState(
                "RAM record prefix state disappeared",
            ))?;
        let rounds = prefix.rounds;
        let params = PrefixPhaseParams {
            records: u32::try_from(self.stats.accesses)
                .map_err(|_| MetalError::InputTooLong(self.stats.accesses))?,
            output_stride: 0,
            e_in_length: 0,
            rounds_bound: u32::try_from(rounds).map_err(|_| MetalError::InputTooLong(rounds))?,
            bind: 1,
            reserved: [
                u32::try_from(self.address_count)
                    .map_err(|_| MetalError::InputTooLong(self.address_count))?,
                0,
                0,
            ],
        };
        let address_live_entries = prefix.address_live_entries[rounds];
        let cycle_live_entries = prefix.cycle_live_entries[rounds];
        let submitted_at = Instant::now();
        let timestamp_calibration_start = self.dispatch_profiler.as_ref().map(|_| {
            let mut cpu = 0;
            let mut gpu = 0;
            self.context.device.sample_timestamps(&mut cpu, &mut gpu);
            (cpu, gpu)
        });
        let dispatch_activity = [
            true,
            false,
            false,
            prefix.hot_address_count != 0,
            false,
            true,
            false,
        ];
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer().to_owned();
            if self.dispatch_profiler.is_some() {
                for stage in [5, 0, 1, 2, 3, 4, 6] {
                    let encoder = self.profiled_encoder(&command_buffer, stage)?;
                    match stage {
                        5 => {
                            self.encode_prefix_cycle_transition(encoder, prefix, params)?;
                        }
                        0 => self.encode_prefix_address_transition(encoder, prefix, params),
                        3 => self.encode_prefix_hot_transition(encoder, prefix, params),
                        _ => {}
                    }
                    encoder.end_encoding();
                }
            } else {
                let encoder = command_buffer.new_compute_command_encoder();
                self.encode_prefix_cycle_transition(encoder, prefix, params)?;
                self.encode_prefix_address_transition(encoder, prefix, params);
                self.encode_prefix_hot_transition(encoder, prefix, params);
                encoder.end_encoding();
            }
            self.resolve_dispatch_counters(&command_buffer);
            command_buffer.commit();
            Ok(SequenceCommand {
                command_buffer,
                address_output: None,
                hot_address_output: None,
                cycle_output: None,
                submitted_at,
                timestamp_calibration_start,
                dispatch_activity,
                live_entries: Some((address_live_entries, cycle_live_entries)),
            })
        })
    }

    fn activate_record_prefix(&mut self) -> Result<(), MetalError> {
        let prefix = self
            .record_prefix
            .take()
            .ok_or(MetalError::InvalidRamReadWriteState(
                "RAM record prefix state disappeared",
            ))?;
        self.buffers.address.segments = prefix.address_segments;
        self.buffers.address.bounded_segments = prefix.bounded_segments;
        self.buffers.address.hot_segments = prefix.hot_segments;
        self.buffers.address.hot_message_chunks = prefix.hot_message_chunks;
        self.buffers.cycle.segments = prefix.cycle_output_segments;
        self.buffers.cycle.blocks = prefix.cycle_output_blocks;
        self.buffers.cycle.address_indices = None;
        self.bounded_address_count = prefix.bounded_address_count;
        self.hot_address_count = prefix.hot_address_count;
        self.hot_message_chunk_count = prefix.hot_message_chunk_count;
        self.stats.address_bytes = self.buffers.address.resident_bytes();
        self.stats.cycle_bytes = self.buffers.cycle.resident_bytes();
        Ok(())
    }

    fn submit_phase(
        &self,
        challenge: Option<AkitaField>,
        emit_address_message: bool,
        emit_cycle_message: bool,
        e_in_length: usize,
    ) -> Result<SequenceCommand, MetalError> {
        let bind = u32::from(challenge.is_some());
        let challenge = challenge
            .as_ref()
            .map_or(Fp128::ZERO, Fp128::from_jolt_field);
        self.context
            .validate_inputs("RAM read-write challenge", &[challenge])?;
        let address_params = PhaseParams {
            work_items: u32::try_from(self.address_count)
                .map_err(|_| MetalError::InputTooLong(self.address_count))?,
            output_stride: u32::try_from(self.address_count)
                .map_err(|_| MetalError::InputTooLong(self.address_count))?,
            e_in_length: u32::try_from(e_in_length)
                .map_err(|_| MetalError::InputTooLong(e_in_length))?,
            bind,
            emit_message: u32::from(emit_address_message),
            hot_source_aux: u32::from(self.hot_source_aux),
            hot_threshold: self.hot_segment_threshold as u32,
            source_initial: u32::from(self.rounds_bound == 0),
        };
        let cycle_params = PhaseParams {
            work_items: u32::try_from(self.tile_count)
                .map_err(|_| MetalError::InputTooLong(self.tile_count))?,
            output_stride: u32::try_from(self.tile_count)
                .map_err(|_| MetalError::InputTooLong(self.tile_count))?,
            e_in_length: address_params.e_in_length,
            bind,
            emit_message: u32::from(emit_cycle_message),
            hot_source_aux: 0,
            hot_threshold: self.hot_segment_threshold as u32,
            source_initial: u32::from(self.rounds_bound == 0),
        };
        let submitted_at = Instant::now();
        let timestamp_calibration_start = self.dispatch_profiler.as_ref().map(|_| {
            let mut cpu = 0;
            let mut gpu = 0;
            self.context.device.sample_timestamps(&mut cpu, &mut gpu);
            (cpu, gpu)
        });
        let dispatch_activity = [
            true,
            bind != 0 && self.hot_message_chunk_count != 0,
            bind != 0 && self.hot_address_count != 0,
            bind != 0 && self.hot_message_chunk_count != 0,
            emit_address_message && self.hot_message_chunk_count != 0,
            self.cycle_resident,
            emit_address_message || emit_cycle_message,
        ];
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer().to_owned();
            let (address_output, hot_address_output, cycle_output) = if self
                .dispatch_profiler
                .is_some()
            {
                {
                    let encoder = self.profiled_encoder(&command_buffer, 0)?;
                    self.encode_address(encoder, challenge, address_params);
                    self.encode_bounded_addresses(encoder, challenge, address_params);
                    encoder.end_encoding();
                }
                {
                    let encoder = self.profiled_encoder(&command_buffer, 1)?;
                    if bind != 0 {
                        self.encode_hot_address_counts(encoder, address_params);
                    }
                    encoder.end_encoding();
                }
                {
                    let encoder = self.profiled_encoder(&command_buffer, 2)?;
                    if bind != 0 {
                        self.encode_hot_address_prefixes(encoder, address_params);
                    }
                    encoder.end_encoding();
                }
                {
                    let encoder = self.profiled_encoder(&command_buffer, 3)?;
                    if bind != 0 {
                        self.encode_hot_address_scatter(encoder, challenge, address_params);
                    }
                    encoder.end_encoding();
                }
                {
                    let encoder = self.profiled_encoder(&command_buffer, 4)?;
                    if emit_address_message {
                        let mut message_params = address_params;
                        message_params.hot_source_aux ^= bind;
                        message_params.source_initial &= 1 - bind;
                        self.encode_hot_address_messages(encoder, message_params);
                    }
                    encoder.end_encoding();
                }
                {
                    let encoder = self.profiled_encoder(&command_buffer, 5)?;
                    if self.cycle_resident {
                        self.encode_cycle(encoder, challenge, cycle_params);
                    }
                    encoder.end_encoding();
                }
                let outputs = {
                    let encoder = self.profiled_encoder(&command_buffer, 6)?;
                    let outputs =
                        self.encode_reductions(encoder, emit_address_message, emit_cycle_message)?;
                    encoder.end_encoding();
                    outputs
                };
                outputs
            } else {
                let encoder = command_buffer.new_compute_command_encoder();
                self.encode_address(encoder, challenge, address_params);
                self.encode_bounded_addresses(encoder, challenge, address_params);
                if bind != 0 {
                    self.encode_hot_address_counts(encoder, address_params);
                    self.encode_hot_address_prefixes(encoder, address_params);
                    self.encode_hot_address_scatter(encoder, challenge, address_params);
                }
                if emit_address_message {
                    let mut message_params = address_params;
                    message_params.hot_source_aux ^= bind;
                    message_params.source_initial &= 1 - bind;
                    self.encode_hot_address_messages(encoder, message_params);
                }
                if self.cycle_resident {
                    self.encode_cycle(encoder, challenge, cycle_params);
                }
                let outputs =
                    self.encode_reductions(encoder, emit_address_message, emit_cycle_message)?;
                encoder.end_encoding();
                outputs
            };
            if let Some(profiler) = &self.dispatch_profiler {
                let blit = command_buffer.new_blit_command_encoder();
                blit.resolve_counters(
                    &profiler.samples,
                    NSRange::new(0, RAM_READ_WRITE_DISPATCH_SAMPLES),
                    &profiler.resolved,
                    0,
                );
                blit.end_encoding();
            }
            command_buffer.commit();
            Ok(SequenceCommand {
                command_buffer,
                address_output,
                hot_address_output,
                cycle_output,
                submitted_at,
                timestamp_calibration_start,
                dispatch_activity,
                live_entries: None,
            })
        })
    }

    fn profiled_encoder<'command>(
        &self,
        command_buffer: &'command metal::CommandBufferRef,
        stage: u64,
    ) -> Result<&'command metal::ComputeCommandEncoderRef, MetalError> {
        let Some(profiler) = &self.dispatch_profiler else {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write dispatch profiler disappeared during encoding",
            ));
        };
        if stage >= RAM_READ_WRITE_DISPATCH_STAGES {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write dispatch profile stage is out of range",
            ));
        }
        let descriptor = ComputePassDescriptor::new();
        let attachment = descriptor.sample_buffer_attachments().object_at(0).ok_or(
            MetalError::InvalidRamReadWriteState(
                "Metal compute pass has no counter attachment slot",
            ),
        )?;
        attachment.set_sample_buffer(&profiler.samples);
        attachment.set_start_of_encoder_sample_index(2 * stage);
        attachment.set_end_of_encoder_sample_index(2 * stage + 1);
        Ok(command_buffer.compute_command_encoder_with_descriptor(descriptor))
    }

    fn resolve_dispatch_counters(&self, command_buffer: &metal::CommandBufferRef) {
        if let Some(profiler) = &self.dispatch_profiler {
            let blit = command_buffer.new_blit_command_encoder();
            blit.resolve_counters(
                &profiler.samples,
                NSRange::new(0, RAM_READ_WRITE_DISPATCH_SAMPLES),
                &profiler.resolved,
                0,
            );
            blit.end_encoding();
        }
    }

    fn encode_prefix_address(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        prefix: &RecordPrefixState,
        params: PrefixPhaseParams,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.prefix_address);
        encoder.set_buffer(0, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(3, Some(&prefix.weight_table), 0);
        encoder.set_buffer(4, Some(&prefix.suffix_table), 0);
        encoder.set_buffer(5, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(6, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(7, Some(&self.buffers.address.partial_a), 0);
        set_inline_bytes(encoder, 8, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: params.output_stride as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: RAM_READ_WRITE_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_prefix_cycle(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        prefix: &RecordPrefixState,
        params: PrefixPhaseParams,
    ) -> Result<(), MetalError> {
        let address_indices = self.buffers.cycle.address_indices.as_ref().ok_or(
            MetalError::InvalidRamReadWriteState(
                "RAM record prefix lost its cycle address indices",
            ),
        )?;
        encoder.set_compute_pipeline_state(&self.pipelines.prefix_cycle);
        encoder.set_buffer(0, Some(&prefix.cycle_segments), 0);
        encoder.set_buffer(1, Some(&self.buffers.cycle.blocks), 0);
        encoder.set_buffer(2, Some(address_indices), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(5, Some(&prefix.weight_table), 0);
        encoder.set_buffer(6, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(7, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(8, Some(&self.buffers.cycle.partial_a), 0);
        set_inline_bytes(encoder, 9, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.tile_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: RAM_READ_WRITE_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_prefix_reductions(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        address_groups: usize,
        cycle_groups: usize,
    ) -> Result<PhaseOutputs, MetalError> {
        let address_in_a = encode_column_reductions(
            encoder,
            &self.pipelines.reduction,
            &self.buffers.address.partial_a,
            &self.buffers.address.partial_b,
            address_groups,
            2,
            RAM_READ_WRITE_REDUCTION_WIDTH,
        )?;
        let cycle_in_a = encode_column_reductions(
            encoder,
            &self.pipelines.reduction,
            &self.buffers.cycle.partial_a,
            &self.buffers.cycle.partial_b,
            cycle_groups,
            2,
            RAM_READ_WRITE_REDUCTION_WIDTH,
        )?;
        Ok((
            Some(if address_in_a {
                self.buffers.address.partial_a.clone()
            } else {
                self.buffers.address.partial_b.clone()
            }),
            None,
            Some(if cycle_in_a {
                self.buffers.cycle.partial_a.clone()
            } else {
                self.buffers.cycle.partial_b.clone()
            }),
        ))
    }

    fn encode_prefix_cycle_transition(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        prefix: &RecordPrefixState,
        params: PrefixPhaseParams,
    ) -> Result<(), MetalError> {
        let address_indices = self.buffers.cycle.address_indices.as_ref().ok_or(
            MetalError::InvalidRamReadWriteState(
                "RAM record prefix lost its cycle address indices",
            ),
        )?;
        encoder.set_compute_pipeline_state(&self.pipelines.prefix_cycle_transition);
        encoder.set_buffer(0, Some(&prefix.cycle_segments), 0);
        encoder.set_buffer(1, Some(&self.buffers.cycle.blocks), 0);
        encoder.set_buffer(2, Some(address_indices), 0);
        encoder.set_buffer(3, Some(&prefix.cycle_output_blocks), 0);
        encoder.set_buffer(4, Some(&self.buffers.cycle.hamming), 0);
        encoder.set_buffer(5, Some(&self.buffers.cycle.increments), 0);
        encoder.set_buffer(6, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(7, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(8, Some(&prefix.weight_table), 0);
        set_inline_bytes(encoder, 9, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.tile_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: RAM_READ_WRITE_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_prefix_address_transition(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        prefix: &RecordPrefixState,
        params: PrefixPhaseParams,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.prefix_address_transition);
        encoder.set_buffer(0, Some(&prefix.address_segments), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.values), 0);
        encoder.set_buffer(5, Some(&self.buffers.address.ra), 0);
        encoder.set_buffer(6, Some(&self.buffers.address.partial_a), 0);
        encoder.set_buffer(7, Some(&prefix.weight_table), 0);
        encoder.set_buffer(8, Some(&prefix.suffix_table), 0);
        set_inline_bytes(encoder, 9, &params);
        dispatch_segments(encoder, self.address_count);
    }

    fn encode_prefix_hot_transition(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        prefix: &RecordPrefixState,
        params: PrefixPhaseParams,
    ) {
        if prefix.hot_address_count == 0 {
            return;
        }
        encoder.set_compute_pipeline_state(&self.pipelines.prefix_hot_transition);
        encoder.set_buffer(0, Some(&prefix.hot_destinations), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(4, Some(&prefix.weight_table), 0);
        encoder.set_buffer(5, Some(&prefix.suffix_table), 0);
        encoder.set_buffer(6, Some(&self.buffers.address.aux_blocks), 0);
        encoder.set_buffer(7, Some(&self.buffers.address.aux_previous), 0);
        encoder.set_buffer(8, Some(&self.buffers.address.aux_next), 0);
        encoder.set_buffer(9, Some(&self.buffers.address.aux_values), 0);
        encoder.set_buffer(10, Some(&self.buffers.address.aux_ra), 0);
        set_inline_bytes(encoder, 11, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.stats.accesses.div_ceil(RAM_READ_WRITE_THREADS) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: RAM_READ_WRITE_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_reductions(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        emit_address_message: bool,
        emit_cycle_message: bool,
    ) -> Result<PhaseOutputs, MetalError> {
        let address_output = if emit_address_message {
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.buffers.address.partial_a,
                &self.buffers.address.partial_b,
                self.address_count,
                2,
                RAM_READ_WRITE_REDUCTION_WIDTH,
            )?;
            Some(if final_in_a {
                self.buffers.address.partial_a.clone()
            } else {
                self.buffers.address.partial_b.clone()
            })
        } else {
            None
        };
        let hot_address_output = if emit_address_message && self.hot_message_chunk_count != 0 {
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.buffers.address.hot_partial_a,
                &self.buffers.address.hot_partial_b,
                self.hot_message_chunk_count,
                2,
                RAM_READ_WRITE_REDUCTION_WIDTH,
            )?;
            Some(if final_in_a {
                self.buffers.address.hot_partial_a.clone()
            } else {
                self.buffers.address.hot_partial_b.clone()
            })
        } else {
            None
        };
        let cycle_output = if emit_cycle_message {
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.buffers.cycle.partial_a,
                &self.buffers.cycle.partial_b,
                self.tile_count,
                2,
                RAM_READ_WRITE_REDUCTION_WIDTH,
            )?;
            Some(if final_in_a {
                self.buffers.cycle.partial_a.clone()
            } else {
                self.buffers.cycle.partial_b.clone()
            })
        } else {
            None
        };
        Ok((address_output, hot_address_output, cycle_output))
    }

    fn encode_address(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        challenge: Fp128,
        params: PhaseParams,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.address);
        encoder.set_buffer(0, Some(&self.buffers.address.segments), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.values), 0);
        encoder.set_buffer(5, Some(&self.buffers.address.ra), 0);
        encoder.set_buffer(6, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(7, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(8, Some(&self.buffers.address.partial_a), 0);
        set_inline_bytes(encoder, 9, &challenge);
        set_inline_bytes(encoder, 10, &params);
        dispatch_segments(encoder, self.address_count);
    }

    fn encode_bounded_addresses(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        challenge: Fp128,
        mut params: PhaseParams,
    ) {
        if self.bounded_address_count == 0 {
            return;
        }
        params.work_items = self.bounded_address_count as u32;
        encoder.set_compute_pipeline_state(&self.pipelines.address_bounded);
        encoder.set_buffer(0, Some(&self.buffers.address.bounded_segments), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.segments), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(5, Some(&self.buffers.address.values), 0);
        encoder.set_buffer(6, Some(&self.buffers.address.ra), 0);
        encoder.set_buffer(7, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(8, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(9, Some(&self.buffers.address.partial_a), 0);
        set_inline_bytes(encoder, 10, &challenge);
        set_inline_bytes(encoder, 11, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.bounded_address_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: RAM_READ_WRITE_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_cycle(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        challenge: Fp128,
        params: PhaseParams,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.cycle);
        encoder.set_buffer(0, Some(&self.buffers.cycle.segments), 0);
        encoder.set_buffer(1, Some(&self.buffers.cycle.blocks), 0);
        encoder.set_buffer(2, Some(&self.buffers.cycle.hamming), 0);
        encoder.set_buffer(3, Some(&self.buffers.cycle.increments), 0);
        encoder.set_buffer(4, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(5, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(6, Some(&self.buffers.cycle.partial_a), 0);
        set_inline_bytes(encoder, 7, &challenge);
        set_inline_bytes(encoder, 8, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.tile_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: RAM_READ_WRITE_THREADS as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_hot_address_counts(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        mut params: PhaseParams,
    ) {
        if self.hot_message_chunk_count == 0 {
            return;
        }
        params.work_items = self.hot_message_chunk_count as u32;
        encoder.set_compute_pipeline_state(&self.pipelines.address_hot_count);
        encoder.set_buffer(0, Some(&self.buffers.address.hot_message_chunks), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.hot_segments), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.segments), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.aux_blocks), 0);
        encoder.set_buffer(5, Some(&self.buffers.address.hot_chunk_counts), 0);
        set_inline_bytes(encoder, 6, &params);
        self.dispatch_hot_chunks(encoder);
    }

    fn encode_hot_address_prefixes(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        mut params: PhaseParams,
    ) {
        if self.hot_address_count == 0 {
            return;
        }
        params.work_items = self.hot_address_count as u32;
        encoder.set_compute_pipeline_state(&self.pipelines.address_hot_prefix);
        encoder.set_buffer(0, Some(&self.buffers.address.hot_segments), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.segments), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.hot_chunk_counts), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.hot_chunk_offsets), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.hot_source_lengths), 0);
        encoder.set_buffer(5, Some(&self.buffers.address.partial_a), 0);
        set_inline_bytes(encoder, 6, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.hot_address_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.hot_address_threads as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_hot_address_scatter(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        challenge: Fp128,
        mut params: PhaseParams,
    ) {
        if self.hot_message_chunk_count == 0 {
            return;
        }
        params.work_items = self.hot_message_chunk_count as u32;
        encoder.set_compute_pipeline_state(&self.pipelines.address_hot_scatter);
        encoder.set_buffer(0, Some(&self.buffers.address.hot_message_chunks), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.hot_segments), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.segments), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.hot_chunk_offsets), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.hot_source_lengths), 0);
        encoder.set_buffer(5, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(6, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(7, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(8, Some(&self.buffers.address.values), 0);
        encoder.set_buffer(9, Some(&self.buffers.address.ra), 0);
        encoder.set_buffer(10, Some(&self.buffers.address.aux_blocks), 0);
        encoder.set_buffer(11, Some(&self.buffers.address.aux_previous), 0);
        encoder.set_buffer(12, Some(&self.buffers.address.aux_next), 0);
        encoder.set_buffer(13, Some(&self.buffers.address.aux_values), 0);
        encoder.set_buffer(14, Some(&self.buffers.address.aux_ra), 0);
        set_inline_bytes(encoder, 15, &challenge);
        set_inline_bytes(encoder, 16, &params);
        self.dispatch_hot_chunks(encoder);
    }

    fn encode_hot_address_messages(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        mut params: PhaseParams,
    ) {
        if self.hot_message_chunk_count == 0 {
            return;
        }
        params.work_items = self.hot_message_chunk_count as u32;
        params.output_stride = params.work_items;
        params.bind = 0;
        encoder.set_compute_pipeline_state(&self.pipelines.address_hot_message);
        encoder.set_buffer(0, Some(&self.buffers.address.hot_message_chunks), 0);
        encoder.set_buffer(1, Some(&self.buffers.address.hot_segments), 0);
        encoder.set_buffer(2, Some(&self.buffers.address.segments), 0);
        encoder.set_buffer(3, Some(&self.buffers.address.blocks), 0);
        encoder.set_buffer(4, Some(&self.buffers.address.previous), 0);
        encoder.set_buffer(5, Some(&self.buffers.address.next), 0);
        encoder.set_buffer(6, Some(&self.buffers.address.values), 0);
        encoder.set_buffer(7, Some(&self.buffers.address.ra), 0);
        encoder.set_buffer(8, Some(&self.buffers.address.aux_blocks), 0);
        encoder.set_buffer(9, Some(&self.buffers.address.aux_previous), 0);
        encoder.set_buffer(10, Some(&self.buffers.address.aux_next), 0);
        encoder.set_buffer(11, Some(&self.buffers.address.aux_values), 0);
        encoder.set_buffer(12, Some(&self.buffers.address.aux_ra), 0);
        encoder.set_buffer(13, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(14, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(15, Some(&self.buffers.address.hot_partial_a), 0);
        set_inline_bytes(encoder, 16, &params);
        self.dispatch_hot_chunks(encoder);
    }

    fn dispatch_hot_chunks(&self, encoder: &metal::ComputeCommandEncoderRef) {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.hot_message_chunk_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.hot_address_threads as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn complete_phase(
        &mut self,
        command: SequenceCommand,
        read_cycle_roots: bool,
    ) -> Result<RamReadWriteSequenceObservation, MetalError> {
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        let wall = command.submitted_at.elapsed();
        let dispatch_timing = command
            .timestamp_calibration_start
            .map(|(cpu_start, gpu_start)| {
                self.read_dispatch_timing(cpu_start, gpu_start, command.dispatch_activity)
            })
            .transpose()?;
        let mut address_quadratic = command
            .address_output
            .as_ref()
            .map_or(Ok([AkitaField::zero(); 2]), |buffer| {
                read_quadratic(&self.context, buffer, "RAM read-write address message")
            })?;
        if let Some(buffer) = command.hot_address_output.as_ref() {
            let hot = read_quadratic(&self.context, buffer, "RAM read-write hot-address message")?;
            address_quadratic[0] += hot[0];
            address_quadratic[1] += hot[1];
        }
        let cycle_quadratic = command
            .cycle_output
            .as_ref()
            .map(|buffer| read_quadratic(&self.context, buffer, "RAM read-write cycle message"))
            .transpose()?;
        let cycle_roots = read_cycle_roots
            .then(|| self.read_cycle_roots())
            .transpose()?;
        let (address_live_entries, cycle_live_entries) =
            command.live_entries.unwrap_or_else(|| {
                (
                    read_segment_total(&self.buffers.address.segments, self.address_count),
                    if self.cycle_resident {
                        read_segment_total(&self.buffers.cycle.segments, self.tile_count)
                    } else {
                        0
                    },
                )
            });
        self.timing_wall += wall;
        self.timing_gpu_active += gpu_active;
        Ok(RamReadWriteSequenceObservation {
            address_quadratic,
            cycle_quadratic,
            cycle_roots,
            address_live_entries,
            cycle_live_entries,
            wall,
            gpu_active,
            dispatch_timing,
        })
    }

    fn read_dispatch_timing(
        &self,
        cpu_start: u64,
        gpu_start: u64,
        activity: [bool; RAM_READ_WRITE_DISPATCH_STAGES as usize],
    ) -> Result<RamReadWriteDispatchTiming, MetalError> {
        let Some(profiler) = &self.dispatch_profiler else {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write dispatch profiler disappeared during submission",
            ));
        };
        let mut cpu_end = 0;
        let mut gpu_end = 0;
        self.context
            .device
            .sample_timestamps(&mut cpu_end, &mut gpu_end);
        if cpu_end <= cpu_start || gpu_end <= gpu_start {
            return Err(MetalError::InvalidRamReadWriteState(
                "Metal timestamp calibration is not increasing",
            ));
        }
        // SAFETY: `resolved` is a shared buffer allocated for exactly fourteen
        // u64 timestamp samples and the resolving command buffer has completed.
        let samples = unsafe {
            slice::from_raw_parts(
                profiler.resolved.contents().cast::<u64>(),
                RAM_READ_WRITE_DISPATCH_SAMPLES as usize,
            )
        };
        let duration = |stage: usize| {
            if !activity[stage] {
                return Ok(Duration::ZERO);
            }
            dispatch_duration(
                samples[2 * stage],
                samples[2 * stage + 1],
                cpu_end - cpu_start,
                gpu_end - gpu_start,
            )
        };
        Ok(RamReadWriteDispatchTiming {
            address: duration(0)?,
            hot_count: duration(1)?,
            hot_prefix: duration(2)?,
            hot_scatter: duration(3)?,
            hot_message: duration(4)?,
            cycle: duration(5)?,
            reductions: duration(6)?,
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
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write equality weights do not cover the current cycle pairs",
            ));
        }
        write_fields(&self.buffers.e_in, e_in)?;
        write_fields(&self.buffers.e_out, e_out)
    }

    fn read_cycle_roots(&self) -> Result<Vec<CycleProductRoot>, MetalError> {
        let segments = buffer_slice::<Segment>(&self.buffers.cycle.segments, self.tile_count);
        let blocks = buffer_slice::<u32>(
            &self.buffers.cycle.blocks,
            self.buffers.cycle.blocks.length() as usize / size_of::<u32>(),
        );
        let hamming = buffer_slice::<Fp128>(
            &self.buffers.cycle.hamming,
            self.buffers.cycle.hamming.length() as usize / size_of::<Fp128>(),
        );
        let increments = buffer_slice::<Fp128>(
            &self.buffers.cycle.increments,
            self.buffers.cycle.increments.length() as usize / size_of::<Fp128>(),
        );
        let mut roots = Vec::with_capacity(self.tile_count);
        for segment in segments {
            if segment.length == 0 {
                continue;
            }
            if segment.length != 1 {
                return Err(MetalError::InvalidRamReadWriteState(
                    "RAM read-write cycle tile did not compact to one root",
                ));
            }
            let index = segment.offset as usize;
            validate_field(&self.context, hamming[index], index)?;
            validate_field(&self.context, increments[index], index)?;
            roots.push(CycleProductRoot {
                block: blocks[index] as usize,
                hamming: hamming[index].into_jolt_field(),
                increment: increments[index].into_jolt_field(),
            });
        }
        Ok(roots)
    }

    fn read_address_roots(&self) -> Result<Vec<AddressCycleRoot>, MetalError> {
        let segments = buffer_slice::<Segment>(&self.buffers.address.segments, self.address_count);
        let capacity = self.stats.accesses.max(1);
        let primary_previous = buffer_slice::<u64>(&self.buffers.address.previous, capacity);
        let primary_next = buffer_slice::<u64>(&self.buffers.address.next, capacity);
        let primary_values = buffer_slice::<Fp128>(&self.buffers.address.values, capacity);
        let primary_ra = buffer_slice::<Fp128>(&self.buffers.address.ra, capacity);
        let hot_capacity = self.hot_state_entries.max(1);
        let aux_previous = buffer_slice::<u64>(&self.buffers.address.aux_previous, hot_capacity);
        let aux_next = buffer_slice::<u64>(&self.buffers.address.aux_next, hot_capacity);
        let aux_values = buffer_slice::<Fp128>(&self.buffers.address.aux_values, hot_capacity);
        let aux_ra = buffer_slice::<Fp128>(&self.buffers.address.aux_ra, hot_capacity);
        let hot_roots_in_aux = self.hot_source_aux;
        let mut roots = Vec::with_capacity(self.stats.active_addresses);
        for (address, segment) in segments.iter().enumerate() {
            if segment.length == 0 {
                continue;
            }
            if segment.length != 1 {
                return Err(MetalError::InvalidRamReadWriteState(
                    "RAM read-write address segment did not compact to one root",
                ));
            }
            let in_aux = hot_roots_in_aux
                && segment.capacity as usize
                    > self
                        .hot_segment_threshold
                        .max(RAM_READ_WRITE_BOUNDED_SEGMENT_MAX);
            let index = if in_aux {
                segment.aux_offset as usize
            } else {
                segment.offset as usize
            };
            let (previous, next, value, ra) = if in_aux {
                (
                    aux_previous[index],
                    aux_next[index],
                    aux_values[index],
                    aux_ra[index],
                )
            } else {
                (
                    primary_previous[index],
                    primary_next[index],
                    primary_values[index],
                    primary_ra[index],
                )
            };
            validate_field(&self.context, value, index)?;
            validate_field(&self.context, ra, index)?;
            roots.push(AddressCycleRoot {
                address,
                previous,
                next,
                value: value.into_jolt_field(),
                ra: ra.into_jolt_field(),
            });
        }
        Ok(roots)
    }
}

fn bucket_plan(
    addresses: &[u32],
    address_count: usize,
    tile_log: usize,
    tile_count: usize,
    hot_segment_threshold: usize,
) -> Result<BucketPlan, MetalError> {
    let worker_count = worker_count(addresses.len());
    let chunk_length = addresses.len().div_ceil(worker_count);
    #[cfg(feature = "parallel")]
    let mut counts = addresses
        .par_chunks(chunk_length)
        .enumerate()
        .map(|(worker, chunk)| {
            count_worker(
                chunk,
                worker * chunk_length,
                address_count,
                tile_log,
                tile_count,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let mut counts = vec![count_worker(
        addresses,
        0,
        address_count,
        tile_log,
        tile_count,
    )?];

    finish_bucket_plan(
        &mut counts,
        addresses.len(),
        chunk_length,
        address_count,
        tile_count,
        hot_segment_threshold,
    )
}

fn count_shifted_blocks(blocks: &[u32], shift: usize) -> usize {
    let mut previous = None;
    blocks
        .iter()
        .filter(|&&block| {
            let block = (block & RAM_READ_WRITE_PREFIX_BLOCK_MASK) >> shift;
            let changed = previous != Some(block);
            previous = Some(block);
            changed
        })
        .count()
}

fn bucket_plan_records(
    source: &RamReadWriteRecordChunks,
    address_count: usize,
    tile_log: usize,
    tile_count: usize,
    hot_segment_threshold: usize,
) -> Result<BucketPlan, MetalError> {
    let chunks = source.chunks();
    if source.address_count() == address_count
        && source.tile_log() == tile_log
        && source.worker_census().len() == chunks.len()
        && source.worker_census().iter().all(|census| {
            census.address_counts().len() == address_count
                && census.tile_counts().len() == tile_count
        })
    {
        let mut counts = source
            .worker_census()
            .iter()
            .map(|census| {
                Ok(WorkerCounts {
                    addresses: census.address_counts().to_vec(),
                    tiles: census.tile_counts().to_vec(),
                    accesses: u32::try_from(census.accesses())
                        .map_err(|_| MetalError::InputTooLong(census.accesses()))?,
                    first_cycle: census.first_cycle(),
                    last_cycle: census.last_cycle(),
                })
            })
            .collect::<Result<Vec<_>, MetalError>>()?;
        return finish_bucket_plan(
            &mut counts,
            source.access_count(),
            1,
            address_count,
            tile_count,
            hot_segment_threshold,
        );
    }
    let worker_count = worker_count(chunks.len());
    let chunks_per_worker = chunks.len().div_ceil(worker_count);
    #[cfg(feature = "parallel")]
    let mut counts = chunks
        .par_chunks(chunks_per_worker)
        .map(|chunks| count_record_worker(chunks, address_count, tile_log, tile_count))
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let mut counts = vec![count_record_worker(
        chunks,
        address_count,
        tile_log,
        tile_count,
    )?];
    let source_len = counts.iter().map(|worker| worker.accesses as usize).sum();
    finish_bucket_plan(
        &mut counts,
        source_len,
        chunks_per_worker,
        address_count,
        tile_count,
        hot_segment_threshold,
    )
}

fn finish_bucket_plan(
    counts: &mut [WorkerCounts],
    source_len: usize,
    chunk_length: usize,
    address_count: usize,
    tile_count: usize,
    hot_segment_threshold: usize,
) -> Result<BucketPlan, MetalError> {
    let mut previous_cycle = None;
    for worker in counts.iter() {
        if let Some(first_cycle) = worker.first_cycle {
            if previous_cycle.is_some_and(|previous| previous >= first_cycle) {
                return Err(MetalError::InvalidRamReadWriteState(
                    "RAM read-write records are not chronologically ordered",
                ));
            }
            previous_cycle = worker.last_cycle;
        }
    }
    let mut address_lengths = vec![0u32; address_count];
    let mut address_offset = 0u32;
    for (address, address_length) in address_lengths.iter_mut().enumerate() {
        let length = counts.iter().try_fold(0u32, |sum, worker| {
            sum.checked_add(worker.addresses[address])
        });
        let Some(length) = length else {
            return Err(MetalError::InputTooLong(source_len));
        };
        *address_length = length;
        let mut cursor = address_offset;
        for worker in counts.iter_mut() {
            let count = worker.addresses[address];
            worker.addresses[address] = cursor;
            cursor = cursor
                .checked_add(count)
                .ok_or(MetalError::InputTooLong(source_len))?;
        }
        address_offset = cursor;
    }

    let mut tile_lengths = vec![0u32; tile_count];
    for (tile, length) in tile_lengths.iter_mut().enumerate() {
        *length = counts
            .iter()
            .try_fold(0u32, |sum, worker| sum.checked_add(worker.tiles[tile]))
            .ok_or(MetalError::InputTooLong(source_len))?;
    }
    let accesses = address_offset as usize;
    if counts
        .iter()
        .map(|worker| worker.accesses as usize)
        .sum::<usize>()
        != accesses
        || tile_lengths
            .iter()
            .map(|&length| length as usize)
            .sum::<usize>()
            != accesses
    {
        return Err(MetalError::InvalidRamReadWriteState(
            "RAM read-write bucket counts disagree",
        ));
    }
    let mut cycle_cursor = 0u32;
    let workers = counts
        .iter_mut()
        .map(|worker| {
            let start = cycle_cursor;
            cycle_cursor += worker.accesses;
            WorkerPlan {
                address_cursors: std::mem::take(&mut worker.addresses),
                cycle_cursor: start,
                accesses: worker.accesses as usize,
            }
        })
        .collect();
    let bounded_segments = address_lengths
        .iter()
        .enumerate()
        .filter_map(|(segment_index, &length)| {
            ((length as usize) > hot_segment_threshold
                && (length as usize) <= RAM_READ_WRITE_BOUNDED_SEGMENT_MAX)
                .then_some(segment_index as u32)
        })
        .collect::<Vec<_>>();
    let auxiliary_threshold = hot_segment_threshold.max(RAM_READ_WRITE_BOUNDED_SEGMENT_MAX);
    let hot_address_count = address_lengths
        .iter()
        .filter(|&&length| length as usize > auxiliary_threshold)
        .count();
    let mut hot_segments = Vec::with_capacity(hot_address_count);
    let mut hot_message_chunks = Vec::with_capacity(
        accesses.div_ceil(RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE) + hot_address_count,
    );
    let mut hot_state_entries = 0u32;
    for (segment_index, &length) in address_lengths.iter().enumerate() {
        if length as usize <= auxiliary_threshold {
            continue;
        }
        let hot_index = hot_segments.len();
        let first_chunk = hot_message_chunks.len();
        for local_offset in (0..length).step_by(RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE) {
            hot_message_chunks.push(HotChunk {
                hot_index: hot_index as u32,
                local_offset,
            });
        }
        let chunk_count = hot_message_chunks.len() - first_chunk;
        hot_segments.push(HotSegment {
            segment_index: segment_index as u32,
            first_chunk: first_chunk as u32,
            chunk_count: chunk_count as u32,
            aux_offset: hot_state_entries,
        });
        hot_state_entries = hot_state_entries
            .checked_add(length)
            .ok_or(MetalError::InputTooLong(source_len))?;
    }
    let stats = bucket_stats(
        &address_lengths,
        accesses,
        &tile_lengths,
        bounded_segments.len(),
        hot_segments.len(),
        hot_message_chunks.len(),
        hot_state_entries as usize,
    );
    Ok(BucketPlan {
        workers,
        chunk_length,
        address_lengths,
        tile_lengths,
        bounded_segments,
        hot_segments,
        hot_message_chunks,
        hot_state_entries: hot_state_entries as usize,
        stats,
    })
}

fn count_worker(
    addresses: &[u32],
    base: usize,
    address_count: usize,
    tile_log: usize,
    tile_count: usize,
) -> Result<WorkerCounts, MetalError> {
    let mut address_counts = vec![0u32; address_count];
    let mut tile_counts = vec![0u32; tile_count];
    let mut accesses = 0u32;
    let mut first_cycle = None;
    let mut last_cycle = None;
    for (offset, &address) in addresses.iter().enumerate() {
        if address == NO_ACCESS {
            continue;
        }
        let address = address as usize;
        if address >= address_count {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write access exceeds the address domain",
            ));
        }
        address_counts[address] = address_counts[address]
            .checked_add(1)
            .ok_or(MetalError::InputTooLong(addresses.len()))?;
        let cycle = base + offset;
        let cycle_u32 =
            u32::try_from(cycle).map_err(|_| MetalError::InputTooLong(addresses.len()))?;
        let _ = first_cycle.get_or_insert(cycle_u32);
        last_cycle = Some(cycle_u32);
        let tile = cycle >> tile_log;
        tile_counts[tile] = tile_counts[tile]
            .checked_add(1)
            .ok_or(MetalError::InputTooLong(addresses.len()))?;
        accesses = accesses
            .checked_add(1)
            .ok_or(MetalError::InputTooLong(addresses.len()))?;
    }
    Ok(WorkerCounts {
        addresses: address_counts,
        tiles: tile_counts,
        accesses,
        first_cycle,
        last_cycle,
    })
}

fn count_record_worker(
    chunks: &[AlignedRamReadWriteRecordArena],
    address_count: usize,
    tile_log: usize,
    tile_count: usize,
) -> Result<WorkerCounts, MetalError> {
    let mut address_counts = vec![0u32; address_count];
    let mut tile_counts = vec![0u32; tile_count];
    let mut accesses = 0u32;
    let mut first_cycle = None;
    let mut last_cycle = None;
    for record in chunks.iter().flat_map(|chunk| chunk.records()) {
        let address = record.address as usize;
        let cycle = record.cycle;
        let tile = (cycle as usize) >> tile_log;
        if address >= address_count || tile >= tile_count {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write record exceeds its domain",
            ));
        }
        if last_cycle.is_some_and(|previous| previous >= cycle) {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write records are not chronologically ordered",
            ));
        }
        address_counts[address] = address_counts[address]
            .checked_add(1)
            .ok_or(MetalError::InputTooLong(chunks.len()))?;
        tile_counts[tile] = tile_counts[tile]
            .checked_add(1)
            .ok_or(MetalError::InputTooLong(chunks.len()))?;
        accesses = accesses
            .checked_add(1)
            .ok_or(MetalError::InputTooLong(chunks.len()))?;
        let _ = first_cycle.get_or_insert(cycle);
        last_cycle = Some(cycle);
    }
    Ok(WorkerCounts {
        addresses: address_counts,
        tiles: tile_counts,
        accesses,
        first_cycle,
        last_cycle,
    })
}

fn bucket_stats(
    address_lengths: &[u32],
    accesses: usize,
    tile_lengths: &[u32],
    bounded_addresses: usize,
    hot_addresses: usize,
    hot_message_chunks: usize,
    hot_state_entries: usize,
) -> RamReadWriteBucketStats {
    let mut nonzero = address_lengths
        .iter()
        .copied()
        .filter(|&length| length != 0)
        .collect::<Vec<_>>();
    nonzero.sort_unstable();
    let percentile = |percent: usize| {
        nonzero
            .get((nonzero.len().saturating_sub(1) * percent) / 100)
            .copied()
            .unwrap_or(0) as usize
    };
    let address_partials = address_lengths
        .len()
        .max(accesses.max(1).div_ceil(RAM_READ_WRITE_THREADS));
    let address_bytes = accesses
        .saturating_mul(size_of::<u32>() + 2 * size_of::<u64>() + 2 * size_of::<Fp128>())
        .saturating_add(address_lengths.len() * size_of::<Segment>())
        .saturating_add(4 * address_partials * size_of::<Fp128>())
        .saturating_add(bounded_addresses * size_of::<u32>())
        .saturating_add(hot_addresses * size_of::<HotSegment>())
        .saturating_add(hot_addresses * size_of::<u32>())
        .saturating_add(hot_message_chunks * size_of::<HotChunk>())
        .saturating_add(2 * hot_message_chunks * size_of::<u32>())
        .saturating_add(
            hot_state_entries
                .saturating_mul(size_of::<u32>() + 2 * size_of::<u64>() + 2 * size_of::<Fp128>()),
        )
        .saturating_add(4 * hot_message_chunks * size_of::<Fp128>());
    let cycle_bytes = accesses
        .saturating_mul(size_of::<u32>() + 2 * size_of::<Fp128>())
        .saturating_add(tile_lengths.len() * size_of::<Segment>())
        .saturating_add(4 * tile_lengths.len() * size_of::<Fp128>());
    RamReadWriteBucketStats {
        accesses,
        active_addresses: nonzero.len(),
        maximum_segment: nonzero.last().copied().unwrap_or(0) as usize,
        p50_segment: percentile(50),
        p95_segment: percentile(95),
        p99_segment: percentile(99),
        hot_addresses,
        hot_message_chunks,
        hot_state_entries,
        hot_compaction_threads: 0,
        hot_compaction_threadgroup_bytes: 0,
        hot_auxiliary_bytes: hot_state_entries
            .saturating_mul(size_of::<u32>() + 2 * size_of::<u64>() + 2 * size_of::<Fp128>()),
        address_bytes,
        cycle_bytes,
    }
}

fn initialize_segments(
    address_buffer: &Buffer,
    cycle_buffer: &Buffer,
    plan: &BucketPlan,
    address_count: usize,
    tile_count: usize,
) {
    let address_segments = buffer_slice_mut::<Segment>(address_buffer, address_count);
    let mut offset = 0u32;
    let mut hot_segments = plan.hot_segments.iter().peekable();
    for (segment_index, (segment, &length)) in address_segments
        .iter_mut()
        .zip(&plan.address_lengths)
        .enumerate()
    {
        let aux_offset = hot_segments
            .next_if(|hot| hot.segment_index as usize == segment_index)
            .map_or(0, |hot| hot.aux_offset);
        *segment = Segment {
            offset,
            length,
            capacity: length,
            aux_offset,
        };
        offset += length;
    }
    let cycle_segments = buffer_slice_mut::<Segment>(cycle_buffer, tile_count);
    offset = 0;
    for (segment, &length) in cycle_segments.iter_mut().zip(&plan.tile_lengths) {
        *segment = Segment {
            offset,
            length,
            capacity: length,
            aux_offset: 0,
        };
        offset += length;
    }
}

fn scatter_buckets(
    addresses: &[u32],
    previous: &[u64],
    next: &[u64],
    chunk_length: usize,
    workers: Vec<WorkerPlan>,
    writers: BucketWriters,
) {
    #[cfg(feature = "parallel")]
    addresses
        .par_chunks(chunk_length)
        .zip(previous.par_chunks(chunk_length))
        .zip(next.par_chunks(chunk_length))
        .zip(workers.into_par_iter())
        .enumerate()
        .for_each(|(worker, (((addresses, previous), next), mut plan))| {
            scatter_worker(
                worker * chunk_length,
                addresses,
                previous,
                next,
                &mut plan,
                writers,
            );
        });
    #[cfg(not(feature = "parallel"))]
    {
        if let Some(mut plan) = workers.into_iter().next() {
            scatter_worker(0, addresses, previous, next, &mut plan, writers);
        }
    }
}

fn scatter_direct_buckets(
    addresses: &[u32],
    chunk_length: usize,
    workers: Vec<WorkerPlan>,
    writers: BucketWriters,
    value_at: &(dyn Fn(usize) -> Result<(u64, u64), MetalError> + Sync),
) -> Result<RamReadWriteIncrementChunks, MetalError> {
    #[cfg(feature = "parallel")]
    {
        addresses
            .par_chunks(chunk_length)
            .zip(workers.into_par_iter())
            .enumerate()
            .map(|(worker, (addresses, mut plan))| {
                scatter_direct_worker(
                    worker * chunk_length,
                    addresses,
                    &mut plan,
                    writers,
                    value_at,
                )
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut workers = workers.into_iter();
        let Some(mut plan) = workers.next() else {
            return Ok(Vec::new());
        };
        Ok(vec![scatter_direct_worker(
            0, addresses, &mut plan, writers, value_at,
        )?])
    }
}

fn scatter_record_buckets(
    chunks: &[AlignedRamReadWriteRecordArena],
    chunks_per_worker: usize,
    workers: Vec<WorkerPlan>,
    writers: BucketWriters,
) -> Result<(), MetalError> {
    #[cfg(feature = "parallel")]
    {
        chunks
            .par_chunks(chunks_per_worker)
            .zip(workers.into_par_iter())
            .try_for_each(|(chunks, mut plan)| scatter_record_worker(chunks, &mut plan, writers))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut workers = workers.into_iter();
        let Some(mut plan) = workers.next() else {
            return Ok(());
        };
        scatter_record_worker(chunks, &mut plan, writers)
    }
}

fn scatter_record_worker(
    chunks: &[AlignedRamReadWriteRecordArena],
    plan: &mut WorkerPlan,
    writers: BucketWriters,
) -> Result<(), MetalError> {
    for (local_access, record) in chunks.iter().flat_map(|chunk| chunk.records()).enumerate() {
        let address_cursor = &mut plan.address_cursors[record.address as usize];
        let address_index = *address_cursor as usize;
        *address_cursor += 1;
        // SAFETY: the prefix plan gives this worker disjoint, in-range slots.
        unsafe {
            writers.write_address(
                address_index,
                record.cycle,
                record.pre_value,
                record.post_value,
            );
            writers.write_cycle(
                plan.cycle_cursor as usize + local_access,
                address_index,
                record.cycle,
                record.pre_value,
                record.post_value,
            );
        }
    }
    if plan.accesses
        != chunks
            .iter()
            .map(|chunk| chunk.records().len())
            .sum::<usize>()
    {
        return Err(MetalError::InvalidRamReadWriteState(
            "RAM read-write record worker count disagrees with its plan",
        ));
    }
    Ok(())
}

fn scatter_direct_worker(
    base: usize,
    addresses: &[u32],
    plan: &mut WorkerPlan,
    writers: BucketWriters,
    value_at: &(dyn Fn(usize) -> Result<(u64, u64), MetalError> + Sync),
) -> Result<(Vec<u64>, Vec<i128>), MetalError> {
    let mut increment_cycles = Vec::with_capacity(plan.accesses / 2);
    let mut increments = Vec::with_capacity(plan.accesses / 2);
    for (local_access, (offset, &address)) in addresses
        .iter()
        .enumerate()
        .filter(|(_, address)| **address != NO_ACCESS)
        .enumerate()
    {
        let cycle = base + offset;
        let cycle_u32 =
            u32::try_from(cycle).map_err(|_| MetalError::InputTooLong(addresses.len()))?;
        let (previous, next) = value_at(cycle)?;
        let address_cursor = &mut plan.address_cursors[address as usize];
        let address_index = *address_cursor as usize;
        *address_cursor += 1;
        // SAFETY: the prefix plan gives this worker disjoint, in-range slots.
        unsafe {
            writers.write_address(address_index, cycle_u32, previous, next);
            writers.write_cycle(
                plan.cycle_cursor as usize + local_access,
                address_index,
                cycle_u32,
                previous,
                next,
            );
        }
        let increment = i128::from(next) - i128::from(previous);
        if increment != 0 {
            increment_cycles.push(cycle as u64);
            increments.push(increment);
        }
    }
    Ok((increment_cycles, increments))
}

fn scatter_worker(
    base: usize,
    addresses: &[u32],
    previous: &[u64],
    next: &[u64],
    plan: &mut WorkerPlan,
    writers: BucketWriters,
) {
    let mut cycle_cursor = plan.cycle_cursor as usize;
    for (offset, ((&address, &previous), &next)) in
        addresses.iter().zip(previous).zip(next).enumerate()
    {
        if address == NO_ACCESS {
            continue;
        }
        let address_cursor = &mut plan.address_cursors[address as usize];
        let address_index = *address_cursor as usize;
        *address_cursor += 1;
        let cycle = (base + offset) as u32;
        // SAFETY: the prefix plan gives this worker disjoint, in-range slots.
        unsafe {
            writers.write_address(address_index, cycle, previous, next);
            writers.write_cycle(cycle_cursor, address_index, cycle, previous, next);
        }
        cycle_cursor += 1;
    }
}

fn worker_count(elements: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().min(elements.max(1))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = elements;
        1
    }
}

fn dispatch_segments(encoder: &metal::ComputeCommandEncoderRef, segments: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: segments.div_ceil(RAM_READ_WRITE_THREADS) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: RAM_READ_WRITE_THREADS as u64,
            height: 1,
            depth: 1,
        },
    );
}

fn write_fields(buffer: &Buffer, values: &[AkitaField]) -> Result<(), MetalError> {
    let required = values
        .len()
        .checked_mul(size_of::<Fp128>())
        .ok_or(MetalError::InputTooLong(values.len()))?;
    if required > buffer.length() as usize {
        return Err(MetalError::InputTooLong(values.len()));
    }
    let destination = buffer_slice_mut::<Fp128>(buffer, values.len());
    for (destination, value) in destination.iter_mut().zip(values) {
        *destination = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn read_quadratic(
    context: &SolinasMetal,
    buffer: &Buffer,
    name: &'static str,
) -> Result<[AkitaField; 2], MetalError> {
    let values = buffer_slice::<Fp128>(buffer, 2);
    context.validate_inputs(name, values)?;
    Ok([values[0].into_jolt_field(), values[1].into_jolt_field()])
}

fn validate_field(context: &SolinasMetal, value: Fp128, index: usize) -> Result<(), MetalError> {
    if value.is_canonical(context.offset) {
        Ok(())
    } else {
        Err(MetalError::NonCanonicalOutput {
            index,
            offset: context.offset,
        })
    }
}

fn read_segment_total(buffer: &Buffer, count: usize) -> usize {
    buffer_slice::<Segment>(buffer, count)
        .iter()
        .map(|segment| segment.length as usize)
        .sum()
}

fn buffer_slice<T>(buffer: &Buffer, length: usize) -> &[T] {
    debug_assert!(length * size_of::<T>() <= buffer.length() as usize);
    // SAFETY: callers use the allocation's element type and a checked length.
    unsafe { slice::from_raw_parts(buffer.contents().cast::<T>(), length) }
}

fn dispatch_duration(
    begin: u64,
    end: u64,
    cpu_time_span: u64,
    gpu_time_span: u64,
) -> Result<Duration, MetalError> {
    let gpu_ticks = end
        .checked_sub(begin)
        .ok_or(MetalError::InvalidRamReadWriteState(
            "Metal dispatch timestamps are not increasing",
        ))?;
    let seconds = (gpu_ticks as f64) / (gpu_time_span as f64) * (cpu_time_span as f64) / 1e9;
    Duration::try_from_secs_f64(seconds).map_err(|_| {
        MetalError::InvalidRamReadWriteState("Metal dispatch timestamp duration is invalid")
    })
}

#[expect(
    clippy::mut_from_ref,
    reason = "Metal shared buffers provide interior-mutable mapped storage"
)]
fn buffer_slice_mut<T>(buffer: &Buffer, length: usize) -> &mut [T] {
    debug_assert!(length * size_of::<T>() <= buffer.length() as usize);
    // SAFETY: callers exclusively initialize or update the shared allocation
    // before any overlapping CPU or GPU access.
    unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<T>(), length) }
}

fn ram_read_write_buffer_bytes(
    records: usize,
    addresses: usize,
    tiles: usize,
    eq_fields: usize,
    record_prefix: bool,
    cycle_state_capacity: usize,
    plan: &BucketPlan,
) -> Result<u64, MetalError> {
    let typed = |elements: usize, element_bytes: usize| {
        elements
            .checked_mul(element_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
    };
    let bounded_addresses = plan.bounded_segments.len();
    let hot_addresses = plan.hot_segments.len();
    let hot_message_chunks = plan.hot_message_chunks.len();
    let hot_state_entries = plan.hot_state_entries;
    let hot_state_capacity = if record_prefix {
        1
    } else {
        hot_state_entries.max(1)
    };
    let address_partials = addresses.max(records.div_ceil(RAM_READ_WRITE_THREADS));
    [
        typed(bounded_addresses.max(1), size_of::<u32>()),
        typed(hot_addresses.max(1), size_of::<HotSegment>()),
        typed(hot_message_chunks.max(1), size_of::<HotChunk>()),
        typed(hot_message_chunks.max(1), size_of::<u32>()),
        typed(hot_message_chunks.max(1), size_of::<u32>()),
        typed(hot_addresses.max(1), size_of::<u32>()),
        typed(addresses, size_of::<Segment>()),
        typed(records, size_of::<u32>()),
        typed(records, size_of::<u64>()),
        typed(records, size_of::<u64>()),
        typed(records, size_of::<Fp128>()),
        typed(records, size_of::<Fp128>()),
        typed(2 * address_partials, size_of::<Fp128>()),
        typed(2 * address_partials, size_of::<Fp128>()),
        typed(2 * hot_message_chunks.max(1), size_of::<Fp128>()),
        typed(2 * hot_message_chunks.max(1), size_of::<Fp128>()),
        typed(hot_state_capacity, size_of::<u32>()),
        typed(hot_state_capacity, size_of::<u64>()),
        typed(hot_state_capacity, size_of::<u64>()),
        typed(hot_state_capacity, size_of::<Fp128>()),
        typed(hot_state_capacity, size_of::<Fp128>()),
        typed(tiles, size_of::<Segment>()),
        typed(records, size_of::<u32>()),
        typed(usize::from(record_prefix) * records, size_of::<u32>()),
        typed(cycle_state_capacity, size_of::<Fp128>()),
        typed(cycle_state_capacity, size_of::<Fp128>()),
        typed(2 * tiles, size_of::<Fp128>()),
        typed(2 * tiles, size_of::<Fp128>()),
        typed(eq_fields, size_of::<Fp128>()),
        typed(eq_fields, size_of::<Fp128>()),
    ]
    .into_iter()
    .try_fold(0u64, |sum, bytes| sum.checked_add(bytes?))
    .ok_or(MetalError::InputTooLong(records))
}

fn ram_read_write_prefix_buffer_bytes(
    records: usize,
    addresses: usize,
    tiles: usize,
    cycle_state_capacity: usize,
    plan: &BucketPlan,
) -> Result<u64, MetalError> {
    let typed = |elements: usize, element_bytes: usize| {
        elements
            .checked_mul(element_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
    };
    [
        typed(
            2 * (1usize << RAM_READ_WRITE_RECORD_PREFIX_ROUNDS),
            size_of::<Fp128>(),
        ),
        typed(addresses, size_of::<Segment>()),
        typed(plan.bounded_segments.len().max(1), size_of::<u32>()),
        typed(plan.hot_segments.len().max(1), size_of::<HotSegment>()),
        typed(plan.hot_message_chunks.len().max(1), size_of::<HotChunk>()),
        typed(2 * tiles, size_of::<Segment>()),
        typed(cycle_state_capacity, size_of::<u32>()),
        typed(records.max(1), size_of::<u32>()),
        typed(plan.hot_state_entries.max(1), size_of::<u32>()),
        typed(plan.hot_state_entries.max(1), size_of::<u64>()),
        typed(plan.hot_state_entries.max(1), size_of::<u64>()),
        typed(plan.hot_state_entries.max(1), size_of::<Fp128>()),
        typed(plan.hot_state_entries.max(1), size_of::<Fp128>()),
    ]
    .into_iter()
    .try_fold(0u64, |sum, bytes| sum.checked_add(bytes?))
    .ok_or(MetalError::InputTooLong(records))
}
