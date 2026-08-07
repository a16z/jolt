use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::MATERIALIZE_PIPELINE;
use crate::metal::solinas::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
    SpartanOuterUniskipRow, SpartanOuterUniskipRows,
};

const PARENT_MATERIALIZE_PIPELINE: &str = "solinas_outer_remainder_materialize_b_and_message";
const REDUCTION_PIPELINE: &str = "solinas_outer_remainder_reduce_columns";
const SIMD_WIDTH: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanOuterDeferredBProbeConfig {
    pub threads_per_threadgroup: Option<usize>,
    pub max_threadgroups: usize,
}

impl Default for SpartanOuterDeferredBProbeConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(256),
            max_threadgroups: 8192,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanOuterDeferredBProbeStats {
    pub wall: Duration,
    pub gpu_active: Duration,
    pub message: [AkitaField; 2],
    pub pipeline_limits: PipelineLimits,
}

struct Pipelines {
    parent: ComputePipelineState,
    candidate: ComputePipelineState,
    reduction: ComputePipelineState,
}

struct Buffers {
    lagrange: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    b_state: Buffer,
    partials: Buffer,
    output: Buffer,
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
struct ReduceParams {
    input_count: u32,
    columns: u32,
    reserved: [u32; 2],
}

pub struct SpartanOuterDeferredBProbe {
    context: SolinasMetal,
    rows: SpartanOuterUniskipRows,
    pipelines: Pipelines,
    parent_limits: PipelineLimits,
    candidate_limits: PipelineLimits,
    buffers: Buffers,
    params: PhaseParams,
    threads: usize,
    reduction_threads: usize,
    completed: bool,
}

impl SolinasMetal {
    pub fn prepare_spartan_outer_deferred_b_synthetic_rows(
        &self,
        rows: usize,
        seed: u64,
    ) -> Result<SpartanOuterUniskipRows, MetalError> {
        self.prepare_spartan_outer_uniskip_rows_with_fill(rows, |compact, residual| {
            #[cfg(feature = "parallel")]
            compact
                .par_iter_mut()
                .zip(residual.par_iter_mut())
                .enumerate()
                .for_each(|(index, (compact, residual))| {
                    (*compact, *residual) = synthetic_row(index, seed).split();
                });
            #[cfg(not(feature = "parallel"))]
            for (index, (compact, residual)) in compact.iter_mut().zip(residual).enumerate() {
                (*compact, *residual) = synthetic_row(index, seed).split();
            }
            Ok(())
        })
    }

    pub fn prepare_spartan_outer_deferred_b_probe(
        &self,
        rows: SpartanOuterUniskipRows,
        lagrange: &[AkitaField; 10],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: SpartanOuterDeferredBProbeConfig,
    ) -> Result<SpartanOuterDeferredBProbe, MetalError> {
        if rows.len() < 4 || !rows.len().is_power_of_two() {
            return Err(MetalError::InvalidOuterRemainderRows(rows.len()));
        }
        if rows.device_registry_id() != self.device_registry_id() {
            return Err(MetalError::OuterRemainderRowDevice {
                expected: self.device_registry_id(),
                got: rows.device_registry_id(),
            });
        }
        if config.max_threadgroups == 0 {
            return Err(MetalError::InvalidOuterRemainderConfig(
                "the deferred-B probe needs at least one threadgroup",
            ));
        }
        let weight_elements = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(rows.len()))?;
        if weight_elements != rows.len() {
            return Err(MetalError::OuterRemainderWeightShape {
                phase: "deferred-B probe",
                expected: rows.len(),
                e_in: e_in.len(),
                e_out: e_out.len(),
            });
        }

        let lagrange = fields(lagrange);
        let e_in = fields(e_in);
        let e_out = fields(e_out);
        self.validate_inputs("deferred-B Lagrange weights", &lagrange)?;
        self.validate_inputs("deferred-B inner weights", &e_in)?;
        self.validate_inputs("deferred-B outer weights", &e_out)?;

        let parent = self.compile_named_pipeline(PARENT_MATERIALIZE_PIPELINE)?;
        let candidate = self.compile_named_pipeline(MATERIALIZE_PIPELINE)?;
        let reduction = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let parent_limits = Self::limits(&parent);
        let candidate_limits = Self::limits(&candidate);
        let reduction_limits = Self::limits(&reduction);
        for (pipeline, limits) in [
            (PARENT_MATERIALIZE_PIPELINE, parent_limits),
            (MATERIALIZE_PIPELINE, candidate_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedOuterRemainderExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads = Self::resolve_threadgroup_width(
            config.threads_per_threadgroup,
            tighter_limits(parent_limits, candidate_limits),
        )?;
        let reduction_threads = Self::resolve_threadgroup_width(None, reduction_limits)?;
        let blocks = e_out.len().min(config.max_threadgroups);

        let state_elements = rows
            .len()
            .checked_mul(2)
            .ok_or(MetalError::InputTooLong(rows.len()))?;
        let partial_elements = blocks
            .checked_mul(2)
            .ok_or(MetalError::InputTooLong(blocks))?;
        let element_counts = [
            lagrange.len(),
            e_in.len(),
            e_out.len(),
            state_elements,
            partial_elements,
            2,
        ];
        let mut additional_bytes = 0_u64;
        for elements in element_counts {
            let bytes = field_bytes(elements)?;
            self.validate_buffer_length(bytes)?;
            additional_bytes = additional_bytes
                .checked_add(bytes)
                .ok_or(MetalError::InputTooLong(elements))?;
        }
        self.validate_additional_working_set(additional_bytes)?;

        let buffers = Buffers {
            lagrange: buffer_from_slice(&self.device, &lagrange),
            e_in: buffer_from_slice(&self.device, &e_in),
            e_out: buffer_from_slice(&self.device, &e_out),
            b_state: new_field_buffer(self, state_elements)?,
            partials: new_field_buffer(self, partial_elements)?,
            output: new_field_buffer(self, 2)?,
        };
        let params = PhaseParams {
            source_elements: to_u32(state_elements)?,
            e_in_length: to_u32(e_in.len())?,
            e_out_length: to_u32(e_out.len())?,
            blocks: to_u32(blocks)?,
        };

        Ok(SpartanOuterDeferredBProbe {
            context: self.clone(),
            rows,
            pipelines: Pipelines {
                parent,
                candidate,
                reduction,
            },
            parent_limits,
            candidate_limits,
            buffers,
            params,
            threads,
            reduction_threads,
            completed: false,
        })
    }
}

impl SpartanOuterDeferredBProbe {
    pub fn run_parent(&mut self) -> Result<SpartanOuterDeferredBProbeStats, MetalError> {
        self.run(false)
    }

    pub fn run_candidate(&mut self) -> Result<SpartanOuterDeferredBProbeStats, MetalError> {
        self.run(true)
    }

    pub fn read_b_state(&self) -> Result<Vec<AkitaField>, MetalError> {
        if !self.completed {
            return Err(MetalError::NotExecuted);
        }
        let elements = self.rows.len() * 2;
        // SAFETY: the probe owns a shared buffer of exactly `elements` fields,
        // and a completed materializer initializes every field before this read.
        let values = unsafe {
            slice::from_raw_parts(self.buffers.b_state.contents().cast::<Fp128>(), elements)
        };
        self.context.validate_inputs("deferred-B state", values)?;
        Ok(values
            .iter()
            .copied()
            .map(Fp128::into_jolt_field::<AkitaField>)
            .collect())
    }

    fn run(&mut self, candidate: bool) -> Result<SpartanOuterDeferredBProbeStats, MetalError> {
        self.completed = false;
        let pipeline = if candidate {
            &self.pipelines.candidate
        } else {
            &self.pipelines.parent
        };
        let pipeline_limits = if candidate {
            self.candidate_limits
        } else {
            self.parent_limits
        };
        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let wall_started = Instant::now();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.rows.instruction_input_buffer()), 0);
            encoder.set_buffer(1, Some(self.rows.residual_buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.lagrange), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(5, Some(&self.buffers.b_state), 0);
            encoder.set_buffer(6, Some(&self.buffers.partials), 0);
            set_inline_bytes(encoder, 7, &self.params);
            encoder.set_threadgroup_memory_length(
                0,
                (2 * (self.threads / SIMD_WIDTH) * size_of::<Fp128>()) as u64,
            );
            dispatch(encoder, self.params.blocks as usize, self.threads);

            let reduce = ReduceParams {
                input_count: self.params.blocks,
                columns: 2,
                reserved: [0; 2],
            };
            encoder.set_compute_pipeline_state(&self.pipelines.reduction);
            encoder.set_buffer(0, Some(&self.buffers.partials), 0);
            encoder.set_buffer(1, Some(&self.buffers.output), 0);
            set_inline_bytes(encoder, 2, &reduce);
            encoder.set_threadgroup_memory_length(
                0,
                ((self.reduction_threads / SIMD_WIDTH) * size_of::<Fp128>()) as u64,
            );
            dispatch(encoder, 2, self.reduction_threads);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        let wall = wall_started.elapsed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(command_buffer.status()));
        }
        let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        self.completed = true;
        let message = self.read_message()?;
        Ok(SpartanOuterDeferredBProbeStats {
            wall,
            gpu_active: Duration::from_secs_f64(end - start),
            message,
            pipeline_limits,
        })
    }

    fn read_message(&self) -> Result<[AkitaField; 2], MetalError> {
        // SAFETY: the completed reduction initializes both fields in the
        // two-element shared output buffer.
        let values =
            unsafe { slice::from_raw_parts(self.buffers.output.contents().cast::<Fp128>(), 2) };
        self.context.validate_inputs("deferred-B message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }
}

fn tighter_limits(left: PipelineLimits, right: PipelineLimits) -> PipelineLimits {
    PipelineLimits {
        thread_execution_width: left.thread_execution_width,
        max_total_threads_per_threadgroup: left
            .max_total_threads_per_threadgroup
            .min(right.max_total_threads_per_threadgroup),
        static_threadgroup_memory_length: left
            .static_threadgroup_memory_length
            .max(right.static_threadgroup_memory_length),
    }
}

fn fields(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn synthetic_row(index: usize, seed: u64) -> SpartanOuterUniskipRow {
    let mut words = [0_u64; 20];
    for (word, value) in words[..19].iter_mut().enumerate() {
        *value = splitmix(seed ^ index as u64 ^ (word as u64).wrapping_mul(0x1000_0001));
    }
    words[2] &= (1 << 24) - 1;
    words[4] &= (1 << 24) - 1;
    words[8] = 0;
    words[15] &= (1 << 24) - 1;
    let selector = splitmix(seed ^ index as u64 ^ 0xa5a5_5a5a);
    let mut flags = 0_u64;
    match selector % 3 {
        1 => flags |= 1 << 0,
        2 => flags |= 1 << 1,
        _ => {}
    }
    match (selector >> 2) % 4 {
        1 => flags |= 1 << 2,
        2 => flags |= 1 << 3,
        3 => flags |= 1 << 4,
        _ => {}
    }
    for bit in 5..=16 {
        flags |= ((selector >> (bit + 7)) & 1) << bit;
    }
    flags |= ((selector >> 40) & 1) << 17;
    flags |= ((selector >> 41) & 1) << 18;
    flags |= ((selector >> 42) & 1) << 19;
    words[19] = flags;
    SpartanOuterUniskipRow::from_words(words)
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn field_bytes(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<Fp128>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

fn new_field_buffer(context: &SolinasMetal, elements: usize) -> Result<Buffer, MetalError> {
    Ok(context.device.new_buffer(
        field_bytes(elements)?,
        MTLResourceOptions::StorageModeShared,
    ))
}

fn to_u32(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::InputTooLong(value))
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
const _: () = assert!(size_of::<ReduceParams>() == 16);
