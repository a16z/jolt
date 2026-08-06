use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits,
    ResidentLookupIndexPlane, SolinasMetal,
};

const GROUPS: usize = 4;
const FACTORS_PER_GROUP: usize = 4;
const FACTORS: usize = GROUPS * FACTORS_PER_GROUP;
const BINS: usize = 256;
const SAMPLES: usize = 4;
const SIMD_WIDTH: usize = 32;
const DEFAULT_THREADS_PER_THREADGROUP: usize = 128;
const MESSAGE_PIPELINE: &str = "solinas_instruction_ra_first_message";
const REDUCE_PIPELINE: &str = "solinas_instruction_ra_reduce";

#[repr(C)]
#[derive(Clone, Copy)]
struct LookupAbi {
    limbs: [u64; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaFirstMessageConfig {
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for InstructionRaFirstMessageConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(DEFAULT_THREADS_PER_THREADGROUP),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FirstMessageParams {
    e_in_length: u32,
    e_out_length: u32,
    _reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReductionParams {
    input_count: u32,
    output_count: u32,
    _reserved: [u32; 2],
}

struct ReductionStep {
    input_a: bool,
    output_count: usize,
    params: Buffer,
}

struct FirstMessageBuffers {
    chunk_tables: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct InstructionRaFirstMessageInvocation {
    context: SolinasMetal,
    lookup_plane: ResidentLookupIndexPlane,
    message_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    message_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: FirstMessageBuffers,
    params: FirstMessageParams,
    reduction_steps: Vec<ReductionStep>,
    final_in_a: bool,
    threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    /// Prepares the production-G4 first message over an explicit resident layout.
    ///
    /// `table_major_lookups` stores the lookup indices in device order and
    /// `cycle_to_table_major` maps each cycle to that order. Buffer population is
    /// preparation work; [`InstructionRaFirstMessageInvocation::execute_timed`]
    /// measures only the resident dispatch and reduction.
    pub fn prepare_instruction_ra_first_message(
        &self,
        table_major_lookups: &[u128],
        cycle_to_table_major: &[u32],
        chunk_tables: &[AkitaField],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: InstructionRaFirstMessageConfig,
    ) -> Result<InstructionRaFirstMessageInvocation, MetalError> {
        if table_major_lookups.len() != cycle_to_table_major.len() {
            return Err(MetalError::InstructionRaPlaneLength {
                name: "cycle-to-table-major",
                expected: table_major_lookups.len() as u64 * size_of::<u32>() as u64,
                got: cycle_to_table_major.len() as u64 * size_of::<u32>() as u64,
            });
        }
        if let Some(&row) = cycle_to_table_major
            .iter()
            .find(|&&row| row as usize >= table_major_lookups.len())
        {
            return Err(MetalError::InputTooLong(row as usize));
        }
        let lookups: Vec<_> = table_major_lookups
            .iter()
            .map(|&lookup| LookupAbi {
                limbs: [lookup as u64, (lookup >> 64) as u64],
            })
            .collect();
        let plane = ResidentLookupIndexPlane::from_buffers(
            buffer_from_slice(&self.device, &lookups),
            buffer_from_slice(&self.device, cycle_to_table_major),
            lookups.len(),
            self.device.registry_id(),
        );
        self.prepare_instruction_ra_first_message_with_plane(
            &plane,
            chunk_tables,
            e_in,
            e_out,
            config,
        )
    }

    pub(crate) fn prepare_instruction_ra_first_message_with_plane(
        &self,
        lookup_plane: &ResidentLookupIndexPlane,
        chunk_tables: &[AkitaField],
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: InstructionRaFirstMessageConfig,
    ) -> Result<InstructionRaFirstMessageInvocation, MetalError> {
        let rows = lookup_plane.len();
        if rows < 2 || !rows.is_power_of_two() {
            return Err(MetalError::InvalidInstructionRaRows(rows));
        }
        let expected_table_values = FACTORS * BINS;
        if chunk_tables.len() != expected_table_values {
            return Err(MetalError::InstructionRaStorageLength {
                expected: expected_table_values,
                got: chunk_tables.len(),
            });
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(rows))?;
        if e_in.is_empty() || e_out.is_empty() || covered != rows / 2 {
            return Err(MetalError::InstructionRaWeightShape {
                expected: rows / 2,
                covered,
            });
        }
        if u32::try_from(covered).is_err() {
            return Err(MetalError::InputTooLong(covered));
        }
        self.validate_instruction_ra_lookup_plane(lookup_plane)?;

        let chunk_tables: Vec<Fp128> = chunk_tables.iter().map(Fp128::from_jolt_field).collect();
        let e_in: Vec<Fp128> = e_in.iter().map(Fp128::from_jolt_field).collect();
        let e_out: Vec<Fp128> = e_out.iter().map(Fp128::from_jolt_field).collect();
        self.validate_inputs("instruction RA chunk tables", &chunk_tables)?;
        self.validate_inputs("instruction RA e_in", &e_in)?;
        self.validate_inputs("instruction RA e_out", &e_out)?;

        let message_pipeline = self.compile_named_pipeline(MESSAGE_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let message_limits = Self::limits(&message_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (MESSAGE_PIPELINE, message_limits),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedInstructionRaExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, message_limits)?;

        let partial_elements = SAMPLES
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(e_out.len()))?;
        let partial_a = self.new_instruction_ra_buffer(partial_elements)?;
        let partial_b = self.new_instruction_ra_buffer(partial_elements)?;

        let mut reduction_steps = Vec::new();
        let mut input_count = e_out.len();
        let mut input_a = true;
        while input_count > 1 {
            let output_count = input_count.div_ceil(SIMD_WIDTH);
            let params = ReductionParams {
                input_count: u32::try_from(input_count)
                    .map_err(|_| MetalError::InputTooLong(input_count))?,
                output_count: u32::try_from(output_count)
                    .map_err(|_| MetalError::InputTooLong(output_count))?,
                _reserved: [0; 2],
            };
            reduction_steps.push(ReductionStep {
                input_a,
                output_count,
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
            });
            input_count = output_count;
            input_a = !input_a;
        }

        let params = FirstMessageParams {
            e_in_length: u32::try_from(e_in.len())
                .map_err(|_| MetalError::InputTooLong(e_in.len()))?,
            e_out_length: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            _reserved: [0; 2],
        };
        Ok(InstructionRaFirstMessageInvocation {
            context: self.clone(),
            lookup_plane: lookup_plane.clone(),
            message_pipeline,
            reduction_pipeline,
            message_limits,
            reduction_limits,
            buffers: FirstMessageBuffers {
                chunk_tables: buffer_from_slice(&self.device, &chunk_tables),
                e_in: buffer_from_slice(&self.device, &e_in),
                e_out: buffer_from_slice(&self.device, &e_out),
                partial_a,
                partial_b,
            },
            params,
            reduction_steps,
            final_in_a: input_a,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }

    fn validate_instruction_ra_lookup_plane(
        &self,
        plane: &ResidentLookupIndexPlane,
    ) -> Result<(), MetalError> {
        let expected_lookups = byte_length::<LookupAbi>(plane.len())?;
        let expected_inverse = byte_length::<u32>(plane.len())?;
        if plane.lookups().length() != expected_lookups {
            return Err(MetalError::InstructionRaPlaneLength {
                name: "lookup-index",
                expected: expected_lookups,
                got: plane.lookups().length(),
            });
        }
        if plane.cycle_to_table_major().length() != expected_inverse {
            return Err(MetalError::InstructionRaPlaneLength {
                name: "cycle-to-table-major",
                expected: expected_inverse,
                got: plane.cycle_to_table_major().length(),
            });
        }
        let expected = self.device.registry_id();
        let got = plane.device_registry_id();
        if got != expected {
            return Err(MetalError::InstructionRaPlaneDevice { expected, got });
        }
        Ok(())
    }

    fn new_instruction_ra_buffer(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = byte_length::<Fp128>(elements)?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl InstructionRaFirstMessageInvocation {
    pub const fn name(&self) -> &'static str {
        MESSAGE_PIPELINE
    }

    pub const fn rows(&self) -> usize {
        self.lookup_plane.len()
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.message_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduction_limits
    }

    pub const fn dynamic_threadgroup_memory_bytes(&self) -> usize {
        SAMPLES * (self.threads_per_threadgroup / SIMD_WIDTH) * size_of::<Fp128>()
    }

    pub const fn useful_multiplications(&self) -> u64 {
        44 * (self.rows() / 2) as u64 + SAMPLES as u64 * self.params.e_out_length as u64
    }

    pub const fn logical_lookup_plane_bytes(&self) -> u64 {
        20 * self.rows() as u64
    }

    pub const fn logical_branch_bytes(&self) -> u64 {
        256 * self.rows() as u64
    }

    pub const fn logical_weight_bytes(&self) -> u64 {
        8 * self.rows() as u64 + 16 * self.params.e_out_length as u64
    }

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        self.completed.set(false);
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.message_pipeline);
            encoder.set_buffer(0, Some(self.lookup_plane.lookups()), 0);
            encoder.set_buffer(1, Some(self.lookup_plane.cycle_to_table_major()), 0);
            encoder.set_buffer(2, Some(&self.buffers.chunk_tables), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 6, &self.params);
            encoder
                .set_threadgroup_memory_length(0, self.dynamic_threadgroup_memory_bytes() as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.params.e_out_length as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );

            for step in &self.reduction_steps {
                encoder.set_compute_pipeline_state(&self.reduction_pipeline);
                let (input, output) = if step.input_a {
                    (&self.buffers.partial_a, &self.buffers.partial_b)
                } else {
                    (&self.buffers.partial_b, &self.buffers.partial_a)
                };
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_buffer(2, Some(&step.params), 0);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: step.output_count as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: SIMD_WIDTH as u64,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
    }

    pub fn read_message(&self) -> Result<[AkitaField; SAMPLES], MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let buffer = if self.final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the completed reduction leaves four fields at the front of
        // the selected shared buffer.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), SAMPLES) };
        self.context
            .validate_inputs("instruction RA first message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

const _: () = assert!(size_of::<LookupAbi>() == 16);
const _: () = assert!(size_of::<FirstMessageParams>() == 16);
const _: () = assert!(size_of::<ReductionParams>() == 16);
