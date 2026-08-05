use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use super::{
    command_buffer_timestamp, Fp128, InstructionInputRows, MetalError, PipelineLimits,
    SolinasMetal, SpartanOuterUniskipRows,
};

pub const OUTER_REMAINDER_OPENINGS: usize = 35;
const OUTER_REMAINDER_STREAM_ROWS: usize = 10;
const OUTER_REMAINDER_TILE_ROWS: usize = 64;
const OUTER_REMAINDER_ROW_WORDS: usize = 20;
const SIMD_WIDTH: usize = 32;
const DEVICE_BUFFERS: usize = 9;

const MATERIALIZE_PIPELINE: &str = "solinas_outer_remainder_materialize_b_and_message";
const STREAM_BIND_PIPELINE: &str = "solinas_outer_remainder_stream_bind_and_message";
const TRANSITION_PIPELINE: &str = "solinas_outer_remainder_bind_and_message";
const OPENING_PIPELINE: &str = "solinas_outer_remainder_opening_tiles";
const REDUCTION_PIPELINE: &str = "solinas_outer_remainder_reduce_columns";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterRemainderSequenceConfig {
    pub materialize_threads_per_threadgroup: Option<usize>,
    pub stream_bind_threads_per_threadgroup: Option<usize>,
    pub transition_threads_per_threadgroup: Option<usize>,
    pub opening_threads_per_threadgroup: Option<usize>,
    pub max_threadgroups: usize,
    pub cpu_tail_elements: usize,
}

impl Default for OuterRemainderSequenceConfig {
    fn default() -> Self {
        Self {
            materialize_threads_per_threadgroup: Some(256),
            stream_bind_threads_per_threadgroup: Some(128),
            transition_threads_per_threadgroup: Some(128),
            opening_threads_per_threadgroup: Some(256),
            max_threadgroups: 8192,
            cpu_tail_elements: 1 << 18,
        }
    }
}

#[derive(Clone, Copy)]
struct StorageGeometry {
    current_elements: usize,
    weight_capacity: usize,
    max_threadgroups: usize,
    element_counts: [usize; DEVICE_BUFFERS],
    owned_bytes: u64,
}

pub(crate) fn outer_remainder_sequence_storage_bytes_with_config(
    rows: usize,
    config: OuterRemainderSequenceConfig,
) -> Result<u64, MetalError> {
    Ok(storage_geometry(rows, config)?.owned_bytes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OuterRemainderPhase {
    BeforeMaterialize,
    BOnly,
    Interleaved,
    Exported,
    OpeningsComplete,
    Poisoned,
}

impl OuterRemainderPhase {
    const fn name(self) -> &'static str {
        match self {
            Self::BeforeMaterialize => "before materialization",
            Self::BOnly => "B-only",
            Self::Interleaved => "interleaved",
            Self::Exported => "CPU tail exported",
            Self::OpeningsComplete => "openings complete",
            Self::Poisoned => "poisoned",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OuterRemainderDispatchCounts {
    pub materializations: usize,
    pub stream_transitions: usize,
    pub dense_transitions: usize,
    pub cpu_tail_exports: usize,
    pub opening_scans: usize,
    pub command_buffers: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OuterRemainderStorageStats {
    pub owned_bytes: u64,
    pub buffer_identities: [usize; DEVICE_BUFFERS],
    pub compact_row_identity: usize,
    pub residual_row_identity: usize,
    pub row_device_registry_id: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PhaseParams {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OpeningParams {
    rows: u32,
    e_in_length: u32,
    e_out_length: u32,
    blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReduceParams {
    input_count: u32,
    columns: u32,
    reserved: [u32; 2],
}

struct Pipelines {
    materialize: ComputePipelineState,
    stream_bind: ComputePipelineState,
    transition: ComputePipelineState,
    opening: ComputePipelineState,
    reduction: ComputePipelineState,
}

struct PipelineSetLimits {
    materialize: PipelineLimits,
    stream_bind: PipelineLimits,
    transition: PipelineLimits,
    opening: PipelineLimits,
    reduction: PipelineLimits,
}

struct Threads {
    materialize: usize,
    stream_bind: usize,
    transition: usize,
    opening: usize,
    reduction: usize,
}

struct Buffers {
    state_a: Buffer,
    state_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    lagrange: Buffer,
    message_partials: Buffer,
    message_output: Buffer,
    opening_partials: Buffer,
    opening_output: Buffer,
}

impl Buffers {
    fn all(&self) -> [&Buffer; DEVICE_BUFFERS] {
        [
            &self.state_a,
            &self.state_b,
            &self.e_in,
            &self.e_out,
            &self.lagrange,
            &self.message_partials,
            &self.message_output,
            &self.opening_partials,
            &self.opening_output,
        ]
    }

    fn identities(&self) -> [usize; DEVICE_BUFFERS] {
        self.all().map(|buffer| buffer.as_ptr() as usize)
    }
}

struct Storage {
    context: SolinasMetal,
    pipelines: Pipelines,
    limits: PipelineSetLimits,
    threads: Threads,
    buffers: Buffers,
    weight_capacity: usize,
    max_threadgroups: usize,
    owned_bytes: u64,
}

pub struct OuterRemainderSequence {
    storage: Storage,
    rows: Option<SpartanOuterUniskipRows>,
    config: OuterRemainderSequenceConfig,
    phase: OuterRemainderPhase,
    current_elements: usize,
    dense_in_a: bool,
    gpu_active: Duration,
    dispatch_counts: OuterRemainderDispatchCounts,
}

impl SolinasMetal {
    pub fn prepare_outer_remainder_sequence(
        &self,
        rows: SpartanOuterUniskipRows,
        config: OuterRemainderSequenceConfig,
    ) -> Result<OuterRemainderSequence, MetalError> {
        let cycles = rows.len();
        if cycles < 4 || !cycles.is_power_of_two() {
            return Err(MetalError::InvalidOuterRemainderRows(cycles));
        }
        if rows.device_registry_id() != self.device_registry_id() {
            return Err(MetalError::OuterRemainderRowDevice {
                expected: self.device_registry_id(),
                got: rows.device_registry_id(),
            });
        }
        let geometry = storage_geometry(cycles, config)?;
        let current_elements = geometry.current_elements;
        let weight_capacity = geometry.weight_capacity;
        let max_threadgroups = geometry.max_threadgroups;

        let pipelines = Pipelines {
            materialize: self.compile_named_pipeline(MATERIALIZE_PIPELINE)?,
            stream_bind: self.compile_named_pipeline(STREAM_BIND_PIPELINE)?,
            transition: self.compile_named_pipeline(TRANSITION_PIPELINE)?,
            opening: self.compile_named_pipeline(OPENING_PIPELINE)?,
            reduction: self.compile_named_pipeline(REDUCTION_PIPELINE)?,
        };
        let limits = PipelineSetLimits {
            materialize: Self::limits(&pipelines.materialize),
            stream_bind: Self::limits(&pipelines.stream_bind),
            transition: Self::limits(&pipelines.transition),
            opening: Self::limits(&pipelines.opening),
            reduction: Self::limits(&pipelines.reduction),
        };
        for (pipeline, pipeline_limits) in [
            (MATERIALIZE_PIPELINE, limits.materialize),
            (STREAM_BIND_PIPELINE, limits.stream_bind),
            (TRANSITION_PIPELINE, limits.transition),
            (OPENING_PIPELINE, limits.opening),
            (REDUCTION_PIPELINE, limits.reduction),
        ] {
            if pipeline_limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedOuterRemainderExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: pipeline_limits.thread_execution_width,
                });
            }
        }

        let threads = Threads {
            materialize: Self::resolve_threadgroup_width(
                config.materialize_threads_per_threadgroup,
                limits.materialize,
            )?,
            stream_bind: Self::resolve_threadgroup_width(
                config.stream_bind_threads_per_threadgroup,
                limits.stream_bind,
            )?,
            transition: Self::resolve_threadgroup_width(
                config.transition_threads_per_threadgroup,
                limits.transition,
            )?,
            opening: Self::resolve_threadgroup_width(
                config.opening_threads_per_threadgroup,
                limits.opening,
            )?,
            reduction: Self::resolve_threadgroup_width(None, limits.reduction)?,
        };
        if threads.opening < OUTER_REMAINDER_OPENINGS {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "opening threadgroup needs at least 35 threads",
            ));
        }
        validate_opening_threadgroup_memory(self, limits.opening, threads.opening)?;

        for elements in geometry.element_counts {
            let bytes = field_bytes(elements)?;
            self.validate_buffer_length(bytes)?;
        }
        self.validate_additional_working_set(outer_remainder_sequence_storage_bytes_with_config(
            cycles, config,
        )?)?;
        let buffers = Buffers {
            state_a: new_field_buffer(self, current_elements)?,
            state_b: new_field_buffer(self, current_elements)?,
            e_in: new_field_buffer(self, weight_capacity)?,
            e_out: new_field_buffer(self, weight_capacity)?,
            lagrange: new_field_buffer(self, OUTER_REMAINDER_STREAM_ROWS)?,
            message_partials: new_field_buffer(self, 2 * max_threadgroups)?,
            message_output: new_field_buffer(self, 2)?,
            opening_partials: new_field_buffer(self, OUTER_REMAINDER_OPENINGS * max_threadgroups)?,
            opening_output: new_field_buffer(self, OUTER_REMAINDER_OPENINGS)?,
        };

        Ok(OuterRemainderSequence {
            storage: Storage {
                context: self.clone(),
                pipelines,
                limits,
                threads,
                buffers,
                weight_capacity,
                max_threadgroups,
                owned_bytes: geometry.owned_bytes,
            },
            rows: Some(rows),
            config,
            phase: OuterRemainderPhase::BeforeMaterialize,
            current_elements,
            dense_in_a: true,
            gpu_active: Duration::ZERO,
            dispatch_counts: OuterRemainderDispatchCounts::default(),
        })
    }
}

impl OuterRemainderSequence {
    pub fn materialize_and_first_message(
        &mut self,
        stream_lagrange: &[AkitaField; OUTER_REMAINDER_STREAM_ROWS],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; 2], MetalError> {
        self.require_phase(OuterRemainderPhase::BeforeMaterialize)?;
        self.validate_weights("materialization", self.current_elements / 2, e_in, e_out)?;
        write_fields(
            &self.storage.context,
            &self.storage.buffers.lagrange,
            OUTER_REMAINDER_STREAM_ROWS,
            stream_lagrange,
        )?;
        self.write_weights(e_in, e_out)?;
        let blocks = e_out.len().min(self.storage.max_threadgroups);
        let params = self.phase_params(blocks, e_in.len(), e_out.len())?;
        let rows = self.rows()?;
        let queue = self.storage.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.storage.pipelines.materialize);
            encoder.set_buffer(0, Some(rows.instruction_input_buffer()), 0);
            encoder.set_buffer(1, Some(rows.residual_buffer()), 0);
            encoder.set_buffer(2, Some(&self.storage.buffers.lagrange), 0);
            encoder.set_buffer(3, Some(&self.storage.buffers.e_in), 0);
            encoder.set_buffer(4, Some(&self.storage.buffers.e_out), 0);
            encoder.set_buffer(5, Some(&self.storage.buffers.state_a), 0);
            encoder.set_buffer(6, Some(&self.storage.buffers.message_partials), 0);
            set_inline_bytes(encoder, 7, &params);
            encoder.set_threadgroup_memory_length(
                0,
                message_threadgroup_bytes(self.storage.threads.materialize),
            );
            dispatch(encoder, blocks, self.storage.threads.materialize);
            self.encode_reduction(
                encoder,
                &self.storage.buffers.message_partials,
                &self.storage.buffers.message_output,
                blocks,
                2,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        self.dispatch_counts.command_buffers += 1;
        self.finish_command(command_buffer)?;
        let output = self.storage.buffers.message_output.clone();
        let endpoints = self.read_array::<2>(&output, "outer endpoints")?;
        self.phase = OuterRemainderPhase::BOnly;
        self.dense_in_a = true;
        self.dispatch_counts.materializations += 1;
        Ok(endpoints)
    }

    pub fn bind_stream_and_message(
        &mut self,
        challenge: AkitaField,
        stream_lagrange: &[AkitaField; OUTER_REMAINDER_STREAM_ROWS],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; 2], MetalError> {
        self.require_phase(OuterRemainderPhase::BOnly)?;
        if self.current_elements < 4 {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "at least four source cells for a fused message",
                got: self.phase.name(),
            });
        }
        self.validate_weights("stream transition", self.current_elements / 4, e_in, e_out)?;
        write_fields(
            &self.storage.context,
            &self.storage.buffers.lagrange,
            OUTER_REMAINDER_STREAM_ROWS,
            stream_lagrange,
        )?;
        self.write_weights(e_in, e_out)?;
        let blocks = e_out.len().min(self.storage.max_threadgroups);
        let params = self.phase_params(blocks, e_in.len(), e_out.len())?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.storage
            .context
            .validate_inputs("outer challenge", slice::from_ref(&challenge))?;
        let rows = self.rows()?;
        let queue = self.storage.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.storage.pipelines.stream_bind);
            encoder.set_buffer(0, Some(rows.instruction_input_buffer()), 0);
            encoder.set_buffer(1, Some(&self.storage.buffers.state_a), 0);
            encoder.set_buffer(2, Some(&self.storage.buffers.state_b), 0);
            encoder.set_buffer(3, Some(&self.storage.buffers.lagrange), 0);
            encoder.set_buffer(4, Some(&self.storage.buffers.e_in), 0);
            encoder.set_buffer(5, Some(&self.storage.buffers.e_out), 0);
            encoder.set_buffer(6, Some(&self.storage.buffers.message_partials), 0);
            set_inline_bytes(encoder, 7, &challenge);
            set_inline_bytes(encoder, 8, &params);
            encoder.set_threadgroup_memory_length(
                0,
                message_threadgroup_bytes(self.storage.threads.stream_bind),
            );
            dispatch(encoder, blocks, self.storage.threads.stream_bind);
            self.encode_reduction(
                encoder,
                &self.storage.buffers.message_partials,
                &self.storage.buffers.message_output,
                blocks,
                2,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        self.dispatch_counts.command_buffers += 1;
        self.finish_command(command_buffer)?;
        let output = self.storage.buffers.message_output.clone();
        let endpoints = self.read_array::<2>(&output, "outer endpoints")?;
        self.current_elements /= 2;
        self.dense_in_a = false;
        self.phase = OuterRemainderPhase::Interleaved;
        self.dispatch_counts.stream_transitions += 1;
        Ok(endpoints)
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; 2], MetalError> {
        self.require_phase(OuterRemainderPhase::Interleaved)?;
        if self.current_elements <= self.config.cpu_tail_elements {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "GPU prefix above the configured CPU-tail cutoff",
                got: self.phase.name(),
            });
        }
        if self.current_elements < 4 {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "at least four source cells for a fused message",
                got: self.phase.name(),
            });
        }
        self.validate_weights("dense transition", self.current_elements / 4, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let blocks = e_out.len().min(self.storage.max_threadgroups);
        let params = self.phase_params(blocks, e_in.len(), e_out.len())?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.storage
            .context
            .validate_inputs("outer challenge", slice::from_ref(&challenge))?;
        let (source, destination) = self.dense_buffers();
        let queue = self.storage.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.storage.pipelines.transition);
            encoder.set_buffer(0, Some(source), 0);
            encoder.set_buffer(1, Some(destination), 0);
            encoder.set_buffer(2, Some(&self.storage.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.storage.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.storage.buffers.message_partials), 0);
            set_inline_bytes(encoder, 5, &challenge);
            set_inline_bytes(encoder, 6, &params);
            encoder.set_threadgroup_memory_length(
                0,
                message_threadgroup_bytes(self.storage.threads.transition),
            );
            dispatch(encoder, blocks, self.storage.threads.transition);
            self.encode_reduction(
                encoder,
                &self.storage.buffers.message_partials,
                &self.storage.buffers.message_output,
                blocks,
                2,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        self.dispatch_counts.command_buffers += 1;
        self.finish_command(command_buffer)?;
        let output = self.storage.buffers.message_output.clone();
        let endpoints = self.read_array::<2>(&output, "outer endpoints")?;
        self.current_elements /= 2;
        self.dense_in_a = !self.dense_in_a;
        self.dispatch_counts.dense_transitions += 1;
        Ok(endpoints)
    }

    pub fn export_cpu_tail(
        &mut self,
        az: &mut [AkitaField],
        bz: &mut [AkitaField],
    ) -> Result<(), MetalError> {
        self.require_phase(OuterRemainderPhase::Interleaved)?;
        if self.current_elements > self.config.cpu_tail_elements {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "current table at or below the configured CPU-tail cutoff",
                got: self.phase.name(),
            });
        }
        if az.len() != self.current_elements || bz.len() != self.current_elements {
            return Err(MetalError::OuterRemainderTailLength {
                expected: self.current_elements,
                az: az.len(),
                bz: bz.len(),
            });
        }
        let source = self.dense_source();
        // SAFETY: all commands are completed synchronously, the active buffer has
        // exactly two initialized fields per current cell, and shared storage is
        // CPU-visible for the lifetime of `self`.
        let fields = unsafe {
            slice::from_raw_parts(source.contents().cast::<Fp128>(), 2 * self.current_elements)
        };
        self.storage
            .context
            .validate_inputs("outer CPU tail", fields)?;
        for (index, pair) in fields.chunks_exact(2).enumerate() {
            az[index] = pair[0].into_jolt_field();
            bz[index] = pair[1].into_jolt_field();
        }
        self.phase = OuterRemainderPhase::Exported;
        self.dispatch_counts.cpu_tail_exports += 1;
        Ok(())
    }

    pub fn evaluate_openings(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; OUTER_REMAINDER_OPENINGS], MetalError> {
        self.require_phase(OuterRemainderPhase::Exported)?;
        let cycles = self.rows()?.len();
        self.validate_weights("opening scan", cycles, e_in, e_out)?;
        self.write_weights(e_in, e_out)?;
        let blocks = e_out.len().min(self.storage.max_threadgroups);
        let params = OpeningParams {
            rows: to_u32(cycles)?,
            e_in_length: to_u32(e_in.len())?,
            e_out_length: to_u32(e_out.len())?,
            blocks: to_u32(blocks)?,
        };
        let rows = self.rows()?;
        let threads = self.storage.threads.opening;
        let shards = threads / OUTER_REMAINDER_OPENINGS;
        let queue = self.storage.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.storage.pipelines.opening);
            encoder.set_buffer(0, Some(rows.instruction_input_buffer()), 0);
            encoder.set_buffer(1, Some(rows.residual_buffer()), 0);
            encoder.set_buffer(2, Some(&self.storage.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.storage.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.storage.buffers.opening_partials), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.set_threadgroup_memory_length(
                0,
                (OUTER_REMAINDER_TILE_ROWS * OUTER_REMAINDER_ROW_WORDS * size_of::<u64>()) as u64,
            );
            encoder.set_threadgroup_memory_length(
                1,
                (OUTER_REMAINDER_TILE_ROWS * size_of::<Fp128>()) as u64,
            );
            encoder.set_threadgroup_memory_length(
                2,
                (OUTER_REMAINDER_OPENINGS * shards * size_of::<Fp128>()) as u64,
            );
            dispatch(encoder, blocks, threads);
            self.encode_reduction(
                encoder,
                &self.storage.buffers.opening_partials,
                &self.storage.buffers.opening_output,
                blocks,
                OUTER_REMAINDER_OPENINGS,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        self.dispatch_counts.command_buffers += 1;
        self.finish_command(command_buffer)?;
        let output = self.storage.buffers.opening_output.clone();
        let openings = self.read_array::<OUTER_REMAINDER_OPENINGS>(&output, "outer openings")?;
        self.phase = OuterRemainderPhase::OpeningsComplete;
        self.dispatch_counts.opening_scans += 1;
        Ok(openings)
    }

    pub fn into_instruction_input_rows(mut self) -> Result<InstructionInputRows, MetalError> {
        self.require_phase(OuterRemainderPhase::OpeningsComplete)?;
        let mut rows = self
            .rows
            .take()
            .ok_or(MetalError::InvalidOuterRemainderState {
                expected: "resident rows owned by the completed sequence",
                got: self.phase.name(),
            })?;
        Ok(rows.share_instruction_input_rows())
    }

    pub const fn phase(&self) -> OuterRemainderPhase {
        self.phase
    }

    pub const fn current_elements(&self) -> usize {
        self.current_elements
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active
    }

    pub const fn dispatch_counts(&self) -> OuterRemainderDispatchCounts {
        self.dispatch_counts
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn storage_stats(&self) -> Result<OuterRemainderStorageStats, MetalError> {
        let rows = self.rows()?;
        Ok(OuterRemainderStorageStats {
            owned_bytes: self.storage.owned_bytes,
            buffer_identities: self.storage.buffers.identities(),
            compact_row_identity: rows.instruction_input_allocation_identity(),
            residual_row_identity: rows.allocation_identity(),
            row_device_registry_id: rows.device_registry_id(),
        })
    }

    pub const fn materialize_pipeline_limits(&self) -> PipelineLimits {
        self.storage.limits.materialize
    }

    pub const fn stream_bind_pipeline_limits(&self) -> PipelineLimits {
        self.storage.limits.stream_bind
    }

    pub const fn transition_pipeline_limits(&self) -> PipelineLimits {
        self.storage.limits.transition
    }

    pub const fn opening_pipeline_limits(&self) -> PipelineLimits {
        self.storage.limits.opening
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.storage.limits.reduction
    }

    fn rows(&self) -> Result<&SpartanOuterUniskipRows, MetalError> {
        self.rows
            .as_ref()
            .ok_or(MetalError::InvalidOuterRemainderState {
                expected: "resident split rows",
                got: self.phase.name(),
            })
    }

    fn require_phase(&self, expected: OuterRemainderPhase) -> Result<(), MetalError> {
        if self.phase != expected {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: expected.name(),
                got: self.phase.name(),
            });
        }
        Ok(())
    }

    fn validate_weights(
        &self,
        phase: &'static str,
        expected: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(), MetalError> {
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(expected))?;
        if e_in.is_empty()
            || e_out.is_empty()
            || e_in.len() > self.storage.weight_capacity
            || e_out.len() > self.storage.weight_capacity
            || covered != expected
        {
            return Err(MetalError::OuterRemainderWeightShape {
                phase,
                expected,
                e_in: e_in.len(),
                e_out: e_out.len(),
            });
        }
        Ok(())
    }

    fn write_weights(&self, e_in: &[AkitaField], e_out: &[AkitaField]) -> Result<(), MetalError> {
        write_fields(
            &self.storage.context,
            &self.storage.buffers.e_in,
            self.storage.weight_capacity,
            e_in,
        )?;
        write_fields(
            &self.storage.context,
            &self.storage.buffers.e_out,
            self.storage.weight_capacity,
            e_out,
        )
    }

    fn phase_params(
        &self,
        blocks: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<PhaseParams, MetalError> {
        Ok(PhaseParams {
            source_elements: to_u32(self.current_elements)?,
            e_in_length: to_u32(e_in_length)?,
            e_out_length: to_u32(e_out_length)?,
            blocks: to_u32(blocks)?,
        })
    }

    fn dense_source(&self) -> &Buffer {
        if self.dense_in_a {
            &self.storage.buffers.state_a
        } else {
            &self.storage.buffers.state_b
        }
    }

    fn dense_buffers(&self) -> (&Buffer, &Buffer) {
        if self.dense_in_a {
            (&self.storage.buffers.state_a, &self.storage.buffers.state_b)
        } else {
            (&self.storage.buffers.state_b, &self.storage.buffers.state_a)
        }
    }

    fn encode_reduction(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        input_count: usize,
        columns: usize,
    ) {
        let params = ReduceParams {
            input_count: input_count as u32,
            columns: columns as u32,
            reserved: [0; 2],
        };
        encoder.set_compute_pipeline_state(&self.storage.pipelines.reduction);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        set_inline_bytes(encoder, 2, &params);
        encoder.set_threadgroup_memory_length(
            0,
            ((self.storage.threads.reduction / SIMD_WIDTH) * size_of::<Fp128>()) as u64,
        );
        dispatch(encoder, columns, self.storage.threads.reduction);
    }

    fn finish_command(
        &mut self,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<(), MetalError> {
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            self.phase = OuterRemainderPhase::Poisoned;
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime");
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime");
        let (start, end) = match (start, end) {
            (Ok(start), Ok(end)) => (start, end),
            (Err(error), _) | (_, Err(error)) => {
                self.phase = OuterRemainderPhase::Poisoned;
                return Err(error);
            }
        };
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            self.phase = OuterRemainderPhase::Poisoned;
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        self.gpu_active += Duration::from_secs_f64(end - start);
        Ok(())
    }

    fn read_array<const N: usize>(
        &mut self,
        buffer: &Buffer,
        side: &'static str,
    ) -> Result<[AkitaField; N], MetalError> {
        // SAFETY: every call follows synchronous command completion and the
        // selected output allocation contains at least N fields.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), N) };
        if let Err(error) = self.storage.context.validate_inputs(side, values) {
            self.phase = OuterRemainderPhase::Poisoned;
            return Err(error);
        }
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }
}

fn validate_opening_threadgroup_memory(
    context: &SolinasMetal,
    limits: PipelineLimits,
    threads: usize,
) -> Result<(), MetalError> {
    let shards = threads / OUTER_REMAINDER_OPENINGS;
    let dynamic = OUTER_REMAINDER_TILE_ROWS
        .checked_mul(OUTER_REMAINDER_ROW_WORDS)
        .and_then(|words| words.checked_mul(size_of::<u64>()))
        .and_then(|bytes| bytes.checked_add(OUTER_REMAINDER_TILE_ROWS * size_of::<Fp128>()))
        .and_then(|bytes| bytes.checked_add(OUTER_REMAINDER_OPENINGS * shards * size_of::<Fp128>()))
        .ok_or(MetalError::InvalidOuterRemainderConfig(
            "opening threadgroup byte count overflowed",
        ))? as u64;
    let requested = limits
        .static_threadgroup_memory_length
        .checked_add(dynamic)
        .ok_or(MetalError::InvalidOuterRemainderConfig(
            "opening threadgroup byte count overflowed",
        ))?;
    let maximum = context.device.max_threadgroup_memory_length();
    if requested > maximum {
        return Err(MetalError::OuterRemainderThreadgroupMemory { requested, maximum });
    }
    Ok(())
}

fn storage_geometry(
    cycles: usize,
    config: OuterRemainderSequenceConfig,
) -> Result<StorageGeometry, MetalError> {
    if cycles < 4 || !cycles.is_power_of_two() {
        return Err(MetalError::InvalidOuterRemainderRows(cycles));
    }
    if config.max_threadgroups == 0 {
        return Err(MetalError::InvalidOuterRemainderConfig(
            "max_threadgroups must be nonzero",
        ));
    }
    if config.cpu_tail_elements < 2 || !config.cpu_tail_elements.is_power_of_two() {
        return Err(MetalError::InvalidOuterRemainderConfig(
            "cpu_tail_elements must be a power of two of at least two",
        ));
    }
    let current_elements = cycles
        .checked_mul(2)
        .ok_or(MetalError::InputTooLong(cycles))?;
    validate_u32(current_elements)?;
    let weight_bits = (cycles.ilog2() as usize).div_ceil(2);
    let weight_capacity = 1usize
        .checked_shl(weight_bits as u32)
        .ok_or(MetalError::InputTooLong(cycles))?;
    let max_threadgroups = config.max_threadgroups.min(weight_capacity);
    let message_partials = 2usize
        .checked_mul(max_threadgroups)
        .ok_or(MetalError::InputTooLong(max_threadgroups))?;
    let opening_partials = OUTER_REMAINDER_OPENINGS
        .checked_mul(max_threadgroups)
        .ok_or(MetalError::InputTooLong(max_threadgroups))?;
    let element_counts = [
        current_elements,
        current_elements,
        weight_capacity,
        weight_capacity,
        OUTER_REMAINDER_STREAM_ROWS,
        message_partials,
        2,
        opening_partials,
        OUTER_REMAINDER_OPENINGS,
    ];
    let owned_bytes = element_counts.iter().try_fold(0u64, |total, &elements| {
        total
            .checked_add(field_bytes(elements)?)
            .ok_or(MetalError::InputTooLong(cycles))
    })?;
    Ok(StorageGeometry {
        current_elements,
        weight_capacity,
        max_threadgroups,
        element_counts,
        owned_bytes,
    })
}

fn write_fields(
    context: &SolinasMetal,
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::OuterRemainderStorageLength {
            name: "field input",
            capacity,
            got: values.len(),
        });
    }
    // SAFETY: no command is active while the host overwrites this prefix and
    // the shared allocation has `capacity` fields.
    let destination =
        unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (destination, value) in destination.iter_mut().zip(values) {
        *destination = Fp128::from_jolt_field(value);
    }
    context.validate_inputs("outer host input", &destination[..values.len()])?;
    Ok(())
}

fn new_field_buffer(context: &SolinasMetal, elements: usize) -> Result<Buffer, MetalError> {
    let bytes = field_bytes(elements)?;
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}

fn field_bytes(elements: usize) -> Result<u64, MetalError> {
    let bytes = elements
        .checked_mul(size_of::<Fp128>())
        .ok_or(MetalError::InputTooLong(elements))?;
    u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(elements))
}

fn validate_u32(elements: usize) -> Result<(), MetalError> {
    let _ = to_u32(elements)?;
    Ok(())
}

fn to_u32(elements: usize) -> Result<u32, MetalError> {
    u32::try_from(elements).map_err(|_| MetalError::InputTooLong(elements))
}

fn message_threadgroup_bytes(threads: usize) -> u64 {
    (2 * (threads / SIMD_WIDTH) * size_of::<Fp128>()) as u64
}

fn dispatch(encoder: &metal::ComputeCommandEncoderRef, groups: usize, threads: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups as u64,
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

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

const _: () = assert!(size_of::<PhaseParams>() == 16);
const _: () = assert!(size_of::<OpeningParams>() == 16);
const _: () = assert!(size_of::<ReduceParams>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::{
        field_bytes, message_threadgroup_bytes, outer_remainder_sequence_storage_bytes_with_config,
        OpeningParams, OuterRemainderPhase, OuterRemainderSequenceConfig, PhaseParams,
        ReduceParams, OUTER_REMAINDER_OPENINGS,
    };

    #[test]
    fn default_schedule_reaches_the_log_26_tail_in_nine_transitions() {
        let config = OuterRemainderSequenceConfig::default();
        let mut elements = 1usize << 27;
        let mut transitions = 0;
        while elements > config.cpu_tail_elements {
            elements /= 2;
            transitions += 1;
        }
        assert_eq!(elements, 1 << 18);
        assert_eq!(transitions, 9);
    }

    #[test]
    fn initial_log_26_gruen_shape_excludes_the_active_variable() {
        let current_elements = 1usize << 27;
        let e_in = 1usize << 13;
        let e_out = 1usize << 13;
        assert_eq!(e_in * e_out * 2, current_elements);
    }

    #[test]
    fn opening_tile_memory_is_below_16_kib_at_256_threads() {
        let shards = 256 / OUTER_REMAINDER_OPENINGS;
        let bytes = 64 * 20 * size_of::<u64>()
            + 64 * size_of::<super::Fp128>()
            + OUTER_REMAINDER_OPENINGS * shards * size_of::<super::Fp128>();
        assert_eq!(shards, 7);
        assert!(bytes < 16 * 1024);
    }

    #[test]
    fn abi_params_are_four_words() {
        assert_eq!(size_of::<PhaseParams>(), 16);
        assert_eq!(size_of::<OpeningParams>(), 16);
        assert_eq!(size_of::<ReduceParams>(), 16);
    }

    #[test]
    fn log_26_storage_has_two_two_gib_state_buffers() {
        assert_eq!(field_bytes(1 << 27).unwrap(), 1 << 31);
        assert_eq!(
            outer_remainder_sequence_storage_bytes_with_config(
                1 << 26,
                OuterRemainderSequenceConfig::default(),
            )
            .unwrap(),
            4_300_079_856,
        );
        assert_eq!(message_threadgroup_bytes(256), 256);
    }

    #[test]
    fn phase_names_are_stable_diagnostics() {
        assert_eq!(OuterRemainderPhase::BOnly.name(), "B-only");
        assert_eq!(OuterRemainderPhase::Poisoned.name(), "poisoned");
    }
}
