use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{objc::rc::autoreleasepool, Buffer, MTLCommandBufferStatus, MTLSize};

use super::super::{
    command_buffer_timestamp, Fp128, InstructionInputRows, MetalError, PipelineLimits,
    SolinasMetal, SpartanOuterUniskipRows,
};
use super::{
    api::{
        OuterRemainderDispatchCounts, OuterRemainderPhase, OuterRemainderSequenceConfig,
        OuterRemainderStorageInitializationStats, OuterRemainderStorageStats,
        OUTER_REMAINDER_OPENINGS, OUTER_REMAINDER_STREAM_ROWS,
    },
    plan::{message_threadgroup_bytes, opening_threadgroup_memory_lengths, to_u32, SIMD_WIDTH},
    storage::{write_fields, DenseBuffers, OuterRemainderSequenceStorage, Storage},
};

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct PhaseParams {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct OpeningParams {
    rows: u32,
    e_in_length: u32,
    e_out_length: u32,
    blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ReduceParams {
    input_count: u32,
    columns: u32,
    reserved: [u32; 2],
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

#[cfg(feature = "allocative")]
impl allocative::Allocative for OuterRemainderSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            self.storage.owned_bytes as usize,
        );
        if let Some(rows) = &self.rows {
            visitor.visit_field(allocative::Key::new("resident_rows"), rows);
        }
        visitor.exit();
    }
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
        self.prepare_outer_remainder_sequence_storage(rows.len(), config)?
            .attach(rows)
    }
}

impl OuterRemainderSequenceStorage {
    pub(crate) fn attach(
        self,
        rows: SpartanOuterUniskipRows,
    ) -> Result<OuterRemainderSequence, MetalError> {
        if rows.len() != self.cycles {
            return Err(MetalError::InvalidOuterRemainderRows(rows.len()));
        }
        if rows.device_registry_id() != self.storage.context.device_registry_id() {
            return Err(MetalError::OuterRemainderRowDevice {
                expected: self.storage.context.device_registry_id(),
                got: rows.device_registry_id(),
            });
        }
        Ok(OuterRemainderSequence {
            storage: self.storage,
            rows: Some(rows),
            config: self.config,
            phase: OuterRemainderPhase::BeforeMaterialize,
            current_elements: self.current_elements,
            dense_in_a: true,
            gpu_active: Duration::ZERO,
            dispatch_counts: OuterRemainderDispatchCounts::default(),
        })
    }
}

impl OuterRemainderSequence {
    pub const fn storage_initialization(&self) -> OuterRemainderStorageInitializationStats {
        self.storage.initialization
    }

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
        let dense = self.dense_storage()?;
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
            encoder.set_buffer(5, Some(&dense.state_a), 0);
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
        let dense = self.dense_storage()?;
        let queue = self.storage.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.storage.pipelines.stream_bind);
            encoder.set_buffer(0, Some(rows.instruction_input_buffer()), 0);
            encoder.set_buffer(1, Some(&dense.state_a), 0);
            encoder.set_buffer(2, Some(&dense.state_b), 0);
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
        let (source, destination) = self.dense_buffers()?;
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
        let source = self.dense_source()?.clone();
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
        drop(source);
        let dense =
            self.storage
                .buffers
                .dense
                .take()
                .ok_or(MetalError::InvalidOuterRemainderState {
                    expected: "resident dense buffers before CPU-tail release",
                    got: self.phase.name(),
                })?;
        self.storage.owned_bytes = self
            .storage
            .owned_bytes
            .checked_sub(self.storage.dense_bytes)
            .ok_or(MetalError::InvalidOuterRemainderState {
                expected: "owned storage at least as large as dense storage",
                got: self.phase.name(),
            })?;
        drop(dense);
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
        let threadgroup_memory =
            opening_threadgroup_memory_lengths(self.config.binding_plan, threads)?;
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
            for (index, bytes) in threadgroup_memory.into_iter().enumerate() {
                if bytes != 0 {
                    encoder.set_threadgroup_memory_length(index as u64, bytes);
                }
            }
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

    fn dense_storage(&self) -> Result<&DenseBuffers, MetalError> {
        self.storage
            .buffers
            .dense
            .as_ref()
            .ok_or(MetalError::InvalidOuterRemainderState {
                expected: "resident dense buffers",
                got: self.phase.name(),
            })
    }

    fn dense_source(&self) -> Result<&Buffer, MetalError> {
        let dense = self.dense_storage()?;
        if self.dense_in_a {
            Ok(&dense.state_a)
        } else {
            Ok(&dense.state_b)
        }
    }

    fn dense_buffers(&self) -> Result<(&Buffer, &Buffer), MetalError> {
        let dense = self.dense_storage()?;
        if self.dense_in_a {
            Ok((&dense.state_a, &dense.state_b))
        } else {
            Ok((&dense.state_b, &dense.state_a))
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
