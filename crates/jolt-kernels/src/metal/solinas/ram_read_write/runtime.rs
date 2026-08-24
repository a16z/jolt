use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::{AkitaField, FromPrimitiveInt};
use metal::{
    objc::rc::autoreleasepool, Buffer, CommandBuffer, ComputePipelineState, MTLResourceOptions,
    MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    CycleProductRoot, HotChunk, HotSegment, PhaseParams, Segment,
    RAM_READ_WRITE_ADDRESS_HOT_COUNT_PIPELINE, RAM_READ_WRITE_ADDRESS_HOT_MESSAGE_PIPELINE,
    RAM_READ_WRITE_ADDRESS_HOT_PREFIX_PIPELINE, RAM_READ_WRITE_ADDRESS_HOT_SCATTER_PIPELINE,
    RAM_READ_WRITE_ADDRESS_PIPELINE, RAM_READ_WRITE_CYCLE_PIPELINE, RAM_READ_WRITE_CYCLE_TILE_LOG2,
    RAM_READ_WRITE_HOT_COMPACTION_THREADS, RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE,
    RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD, RAM_READ_WRITE_HOT_THREADGROUP_BYTES_MAX,
    RAM_READ_WRITE_REDUCTION_PIPELINE, RAM_READ_WRITE_REDUCTION_WIDTH, RAM_READ_WRITE_SIMD_WIDTH,
    RAM_READ_WRITE_THREADS,
};
use crate::metal::solinas::{
    completed_command_gpu_time, encode_column_reductions, set_inline_bytes, Fp128, MetalError,
    SolinasMetal,
};
use crate::optimized::ram_trace::NO_ACCESS;

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
}

pub(crate) struct RamReadWriteFinish {
    pub address_roots: Vec<AddressCycleRoot>,
    pub cycle_roots: Option<Vec<CycleProductRoot>>,
    pub gpu_active: Duration,
}

struct AddressBuffers {
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

struct CycleBuffers {
    segments: Buffer,
    blocks: Buffer,
    hamming: Buffer,
    increments: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

struct RamReadWriteBuffers {
    address: AddressBuffers,
    cycle: CycleBuffers,
    e_in: Buffer,
    e_out: Buffer,
}

struct RamReadWritePipelines {
    address: ComputePipelineState,
    address_hot_count: ComputePipelineState,
    address_hot_prefix: ComputePipelineState,
    address_hot_scatter: ComputePipelineState,
    address_hot_message: ComputePipelineState,
    cycle: ComputePipelineState,
    reduction: ComputePipelineState,
}

pub(crate) struct RamReadWriteSequence {
    context: SolinasMetal,
    pipelines: RamReadWritePipelines,
    buffers: RamReadWriteBuffers,
    log_t: usize,
    address_count: usize,
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
    timing_wall: Duration,
    timing_gpu_active: Duration,
}

struct SequenceCommand {
    command_buffer: CommandBuffer,
    address_output: Option<Buffer>,
    hot_address_output: Option<Buffer>,
    cycle_output: Option<Buffer>,
    submitted_at: Instant,
}

#[derive(Clone, Copy)]
struct BucketWriters {
    address_blocks: *mut u32,
    address_previous: *mut u64,
    address_next: *mut u64,
    address_values: *mut Fp128,
    address_ra: *mut Fp128,
    cycle_blocks: *mut u32,
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
    unsafe fn write_address(self, index: usize, cycle: u32, previous: u64, next: u64) {
        // SAFETY: `bucket_plan` assigns this record a unique initialized slot
        // inside every equally sized address plane.
        unsafe {
            self.address_blocks.add(index).write(cycle);
            self.address_previous.add(index).write(previous);
            self.address_next.add(index).write(next);
            self.address_values
                .add(index)
                .write(Fp128::from_u128(u128::from(previous)));
            self.address_ra.add(index).write(Fp128::ONE);
        }
    }

    unsafe fn write_cycle(self, index: usize, cycle: u32, previous: u64, next: u64) {
        let increment = if previous == next {
            Fp128::ZERO
        } else {
            Fp128::from_jolt_field(&AkitaField::from_i128(
                i128::from(next) - i128::from(previous),
            ))
        };
        // SAFETY: chronological worker prefixes assign every access record one
        // unique slot inside all three cycle planes.
        unsafe {
            self.cycle_blocks.add(index).write(cycle);
            self.cycle_hamming.add(index).write(Fp128::ONE);
            self.cycle_increments.add(index).write(increment);
        }
    }
}

struct WorkerCounts {
    addresses: Vec<u32>,
    tiles: Vec<u32>,
    accesses: u32,
}

struct WorkerPlan {
    address_cursors: Vec<u32>,
    cycle_cursor: u32,
}

struct BucketPlan {
    workers: Vec<WorkerPlan>,
    chunk_length: usize,
    address_lengths: Vec<u32>,
    tile_lengths: Vec<u32>,
    hot_segments: Vec<HotSegment>,
    hot_message_chunks: Vec<HotChunk>,
    hot_state_entries: usize,
    stats: RamReadWriteBucketStats,
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
        if addresses.len() != previous.len() || addresses.len() != next.len() {
            return Err(MetalError::LengthMismatch {
                lhs: addresses.len(),
                rhs: previous.len().min(next.len()),
            });
        }
        if log_t == 0
            || log_t > u32::BITS as usize
            || addresses.len() != 1usize.checked_shl(log_t as u32).unwrap_or(0)
            || address_count == 0
        {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write source has inconsistent geometry",
            ));
        }
        let tile_log = log_t.min(RAM_READ_WRITE_CYCLE_TILE_LOG2);
        let tile_count = 1usize << (log_t - tile_log);
        let mut plan = bucket_plan(addresses, address_count, tile_log, tile_count)?;
        let record_capacity = plan.stats.accesses.max(1);
        let maximum_eq_fields = 1usize << log_t.div_ceil(2);
        self.validate_additional_working_set(ram_read_write_buffer_bytes(
            record_capacity,
            address_count,
            tile_count,
            maximum_eq_fields,
            plan.hot_segments.len(),
            plan.hot_message_chunks.len(),
            plan.hot_state_entries,
        )?)?;
        let hot_address_capacity = plan.hot_segments.len().max(1);
        let hot_message_chunk_capacity = plan.hot_message_chunks.len().max(1);
        let hot_state_capacity = plan.hot_state_entries.max(1);
        let address = AddressBuffers {
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
            partial_a: self.new_ram_read_write_buffer::<Fp128>(2 * address_count)?,
            partial_b: self.new_ram_read_write_buffer::<Fp128>(2 * address_count)?,
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
            hamming: self.new_ram_read_write_buffer::<Fp128>(record_capacity)?,
            increments: self.new_ram_read_write_buffer::<Fp128>(record_capacity)?,
            partial_a: self.new_ram_read_write_buffer::<Fp128>(2 * tile_count)?,
            partial_b: self.new_ram_read_write_buffer::<Fp128>(2 * tile_count)?,
        };
        let e_in = self.new_ram_read_write_buffer::<Fp128>(maximum_eq_fields)?;
        let e_out = self.new_ram_read_write_buffer::<Fp128>(maximum_eq_fields)?;
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
        let writers = BucketWriters {
            address_blocks: address.blocks.contents().cast::<u32>(),
            address_previous: address.previous.contents().cast::<u64>(),
            address_next: address.next.contents().cast::<u64>(),
            address_values: address.values.contents().cast::<Fp128>(),
            address_ra: address.ra.contents().cast::<Fp128>(),
            cycle_blocks: cycle.blocks.contents().cast::<u32>(),
            cycle_hamming: cycle.hamming.contents().cast::<Fp128>(),
            cycle_increments: cycle.increments.contents().cast::<Fp128>(),
        };
        scatter_buckets(
            addresses,
            previous,
            next,
            plan.chunk_length,
            std::mem::take(&mut plan.workers),
            writers,
        );

        let address_pipeline = self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_PIPELINE)?;
        let address_hot_count_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_COUNT_PIPELINE)?;
        let address_hot_prefix_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_PREFIX_PIPELINE)?;
        let address_hot_scatter_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_SCATTER_PIPELINE)?;
        let address_hot_message_pipeline =
            self.compile_named_pipeline(RAM_READ_WRITE_ADDRESS_HOT_MESSAGE_PIPELINE)?;
        let cycle_pipeline = self.compile_named_pipeline(RAM_READ_WRITE_CYCLE_PIPELINE)?;
        let reduction = self.compile_named_pipeline(RAM_READ_WRITE_REDUCTION_PIPELINE)?;
        let address_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_THREADS),
            Self::limits(&address_pipeline),
        )?;
        let cycle_threads = Self::resolve_threadgroup_width(
            Some(RAM_READ_WRITE_THREADS),
            Self::limits(&cycle_pipeline),
        )?;
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
        let hot_threadgroup_bytes = address_hot_count_limits
            .static_threadgroup_memory_length
            .max(address_hot_prefix_limits.static_threadgroup_memory_length)
            .max(address_hot_scatter_limits.static_threadgroup_memory_length);
        if address_hot_count_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_hot_prefix_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_hot_scatter_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_hot_message_limits.thread_execution_width != RAM_READ_WRITE_SIMD_WIDTH
            || address_threads != RAM_READ_WRITE_THREADS
            || address_hot_count_threads != RAM_READ_WRITE_HOT_COMPACTION_THREADS
            || address_hot_prefix_threads != RAM_READ_WRITE_HOT_COMPACTION_THREADS
            || address_hot_scatter_threads != RAM_READ_WRITE_HOT_COMPACTION_THREADS
            || address_hot_message_threads != RAM_READ_WRITE_THREADS
            || cycle_threads != RAM_READ_WRITE_THREADS
            || reduction_threads != RAM_READ_WRITE_REDUCTION_WIDTH
            || hot_threadgroup_bytes > RAM_READ_WRITE_HOT_THREADGROUP_BYTES_MAX
        {
            return Err(MetalError::InvalidRamReadWriteState(
                "RAM read-write pipeline width differs from its checked schedule",
            ));
        }

        plan.stats.hot_compaction_threads = address_hot_scatter_threads;
        plan.stats.hot_compaction_threadgroup_bytes = hot_threadgroup_bytes;
        Ok(RamReadWriteSequence {
            context: self.clone(),
            pipelines: RamReadWritePipelines {
                address: address_pipeline,
                address_hot_count: address_hot_count_pipeline,
                address_hot_prefix: address_hot_prefix_pipeline,
                address_hot_scatter: address_hot_scatter_pipeline,
                address_hot_message: address_hot_message_pipeline,
                cycle: cycle_pipeline,
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
            hot_address_count: plan.hot_segments.len(),
            hot_message_chunk_count: plan.hot_message_chunks.len(),
            hot_state_entries: plan.hot_state_entries,
            hot_address_threads: address_hot_scatter_threads,
            tile_count,
            tile_log,
            rounds_bound: 0,
            initial_message_emitted: false,
            cycle_resident: true,
            finished: false,
            stats: plan.stats,
            timing_wall: Duration::ZERO,
            timing_gpu_active: Duration::ZERO,
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

    pub(crate) fn resident_bytes(&self) -> usize {
        self.stats.address_bytes
            + self.stats.cycle_bytes
            + self.buffers.e_in.length() as usize
            + self.buffers.e_out.length() as usize
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
        let command = self.submit_phase(None, true, true, e_in.len())?;
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
        let cycle_message = self.cycle_resident && new_rounds_bound < self.tile_log;
        let handoff_cycle = self.cycle_resident && new_rounds_bound == self.tile_log;
        let command = self.submit_phase(Some(challenge), true, cycle_message, e_in.len())?;
        let mut observation = self.complete_phase(command, handoff_cycle)?;
        self.rounds_bound = new_rounds_bound;
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
        self.rounds_bound += 1;
        self.cycle_resident = false;
        self.finished = true;
        let address_roots = self.read_address_roots()?;
        Ok(RamReadWriteFinish {
            address_roots,
            cycle_roots: observation.cycle_roots,
            gpu_active: observation.gpu_active,
        })
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
            hot_source_aux: (self.rounds_bound % 2) as u32,
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
        };
        let submitted_at = Instant::now();
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer().to_owned();
            let encoder = command_buffer.new_compute_command_encoder();
            self.encode_address(encoder, challenge, address_params);
            if bind != 0 {
                self.encode_hot_address_counts(encoder, address_params);
                self.encode_hot_address_prefixes(encoder, address_params);
                self.encode_hot_address_scatter(encoder, challenge, address_params);
            }
            if emit_address_message {
                let mut message_params = address_params;
                message_params.hot_source_aux ^= bind;
                self.encode_hot_address_messages(encoder, message_params);
            }
            if self.cycle_resident {
                self.encode_cycle(encoder, challenge, cycle_params);
            }
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
            encoder.end_encoding();
            command_buffer.commit();
            Ok(SequenceCommand {
                command_buffer,
                address_output,
                hot_address_output,
                cycle_output,
                submitted_at,
            })
        })
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
        dispatch_segments(encoder, self.tile_count);
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
        let address_live_entries =
            read_segment_total(&self.buffers.address.segments, self.address_count);
        let cycle_live_entries = if self.cycle_resident {
            read_segment_total(&self.buffers.cycle.segments, self.tile_count)
        } else {
            0
        };
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
        let blocks = buffer_slice::<u32>(&self.buffers.cycle.blocks, self.stats.accesses.max(1));
        let hamming =
            buffer_slice::<Fp128>(&self.buffers.cycle.hamming, self.stats.accesses.max(1));
        let increments =
            buffer_slice::<Fp128>(&self.buffers.cycle.increments, self.stats.accesses.max(1));
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
        let hot_roots_in_aux = !self.rounds_bound.is_multiple_of(2);
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
                && segment.capacity as usize > RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD;
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

    let mut address_lengths = vec![0u32; address_count];
    let mut address_offset = 0u32;
    for (address, address_length) in address_lengths.iter_mut().enumerate() {
        let length = counts.iter().try_fold(0u32, |sum, worker| {
            sum.checked_add(worker.addresses[address])
        });
        let Some(length) = length else {
            return Err(MetalError::InputTooLong(addresses.len()));
        };
        *address_length = length;
        let mut cursor = address_offset;
        for worker in &mut counts {
            let count = worker.addresses[address];
            worker.addresses[address] = cursor;
            cursor = cursor
                .checked_add(count)
                .ok_or(MetalError::InputTooLong(addresses.len()))?;
        }
        address_offset = cursor;
    }

    let mut tile_lengths = vec![0u32; tile_count];
    for (tile, length) in tile_lengths.iter_mut().enumerate() {
        *length = counts
            .iter()
            .try_fold(0u32, |sum, worker| sum.checked_add(worker.tiles[tile]))
            .ok_or(MetalError::InputTooLong(addresses.len()))?;
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
        .into_iter()
        .map(|worker| {
            let start = cycle_cursor;
            cycle_cursor += worker.accesses;
            WorkerPlan {
                address_cursors: worker.addresses,
                cycle_cursor: start,
            }
        })
        .collect();
    let hot_address_count = address_lengths
        .iter()
        .filter(|&&length| length as usize > RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD)
        .count();
    let mut hot_segments = Vec::with_capacity(hot_address_count);
    let mut hot_message_chunks = Vec::with_capacity(
        accesses.div_ceil(RAM_READ_WRITE_HOT_MESSAGE_CHUNK_SIZE) + hot_address_count,
    );
    let mut hot_state_entries = 0u32;
    for (segment_index, &length) in address_lengths.iter().enumerate() {
        if length as usize <= RAM_READ_WRITE_HOT_SEGMENT_THRESHOLD {
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
            .ok_or(MetalError::InputTooLong(addresses.len()))?;
    }
    let stats = bucket_stats(
        &address_lengths,
        accesses,
        &tile_lengths,
        hot_segments.len(),
        hot_message_chunks.len(),
        hot_state_entries as usize,
    );
    Ok(BucketPlan {
        workers,
        chunk_length,
        address_lengths,
        tile_lengths,
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
        let tile = (base + offset) >> tile_log;
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
    })
}

fn bucket_stats(
    address_lengths: &[u32],
    accesses: usize,
    tile_lengths: &[u32],
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
    let address_bytes = accesses
        .saturating_mul(size_of::<u32>() + 2 * size_of::<u64>() + 2 * size_of::<Fp128>())
        .saturating_add(address_lengths.len() * size_of::<Segment>())
        .saturating_add(4 * address_lengths.len() * size_of::<Fp128>())
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
            writers.write_cycle(cycle_cursor, cycle, previous, next);
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
    hot_addresses: usize,
    hot_message_chunks: usize,
    hot_state_entries: usize,
) -> Result<u64, MetalError> {
    let typed = |elements: usize, element_bytes: usize| {
        elements
            .checked_mul(element_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
    };
    [
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
        typed(2 * addresses, size_of::<Fp128>()),
        typed(2 * addresses, size_of::<Fp128>()),
        typed(2 * hot_message_chunks.max(1), size_of::<Fp128>()),
        typed(2 * hot_message_chunks.max(1), size_of::<Fp128>()),
        typed(hot_state_entries.max(1), size_of::<u32>()),
        typed(hot_state_entries.max(1), size_of::<u64>()),
        typed(hot_state_entries.max(1), size_of::<u64>()),
        typed(hot_state_entries.max(1), size_of::<Fp128>()),
        typed(hot_state_entries.max(1), size_of::<Fp128>()),
        typed(tiles, size_of::<Segment>()),
        typed(records, size_of::<u32>()),
        typed(records, size_of::<Fp128>()),
        typed(records, size_of::<Fp128>()),
        typed(2 * tiles, size_of::<Fp128>()),
        typed(2 * tiles, size_of::<Fp128>()),
        typed(eq_fields, size_of::<Fp128>()),
        typed(eq_fields, size_of::<Fp128>()),
    ]
    .into_iter()
    .try_fold(0u64, |sum, bytes| sum.checked_add(bytes?))
    .ok_or(MetalError::InputTooLong(records))
}
