use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

const SIMD_WIDTH: usize = 32;
const SAMPLES: usize = 2;
const DEFAULT_THREADS_PER_THREADGROUP: usize = 128;
const MESSAGE_PIPELINE: &str = "solinas_registers_rw_first_message";
const SECOND_MESSAGE_PIPELINE: &str = "solinas_registers_rw_second_message";
const REDUCTION_PIPELINE: &str = "solinas_registers_rw_reduce";
const ABSENT_REGISTER: u8 = u8::MAX;

/// The resident per-cycle input needed by registers read/write checking.
///
/// Register indices occupy the low three bytes of `metadata`, in rs1/rs2/rd
/// order. `0xff` denotes an absent access. The layout is shared verbatim with
/// Metal and intentionally stays at 40 bytes per cycle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterAccessRow {
    rs1_value: u64,
    rs2_value: u64,
    rd_pre_value: u64,
    rd_post_value: u64,
    metadata: u64,
}

impl RegisterAccessRow {
    pub const fn new(
        rs1: Option<(u8, u64)>,
        rs2: Option<(u8, u64)>,
        rd: Option<(u8, u64, u64)>,
    ) -> Self {
        let (rs1_index, rs1_value) = match rs1 {
            Some((index, value)) => (index, value),
            None => (ABSENT_REGISTER, 0),
        };
        let (rs2_index, rs2_value) = match rs2 {
            Some((index, value)) => (index, value),
            None => (ABSENT_REGISTER, 0),
        };
        let (rd_index, rd_pre_value, rd_post_value) = match rd {
            Some((index, pre, post)) => (index, pre, post),
            None => (ABSENT_REGISTER, 0, 0),
        };
        Self {
            rs1_value,
            rs2_value,
            rd_pre_value,
            rd_post_value,
            metadata: (rs1_index as u64) | ((rs2_index as u64) << 8) | ((rd_index as u64) << 16),
        }
    }

    pub const fn rs1(self) -> Option<(u8, u64)> {
        let index = self.index(0);
        if index == ABSENT_REGISTER {
            None
        } else {
            Some((index, self.rs1_value))
        }
    }

    pub const fn rs2(self) -> Option<(u8, u64)> {
        let index = self.index(1);
        if index == ABSENT_REGISTER {
            None
        } else {
            Some((index, self.rs2_value))
        }
    }

    pub const fn rd(self) -> Option<(u8, u64, u64)> {
        let index = self.index(2);
        if index == ABSENT_REGISTER {
            None
        } else {
            Some((index, self.rd_pre_value, self.rd_post_value))
        }
    }

    const fn index(self, slot: u32) -> u8 {
        ((self.metadata >> (8 * slot)) & 0xff) as u8
    }

    fn validate(self) -> Result<(), MetalError> {
        for slot in 0..3 {
            let index = self.index(slot);
            if index != ABSENT_REGISTER && index >= 128 {
                return Err(MetalError::InvalidRegistersReadWriteIndex(index));
            }
        }
        Ok(())
    }
}

impl Default for RegisterAccessRow {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersReadWriteMessageConfig {
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for RegistersReadWriteMessageConfig {
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

#[derive(Clone)]
struct FirstMessageBuffers {
    rows: Buffer,
    inc: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    gamma: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct RegistersReadWriteFirstMessageInvocation {
    context: SolinasMetal,
    message_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    message_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: FirstMessageBuffers,
    params: FirstMessageParams,
    reduction_steps: Vec<ReductionStep>,
    rows: usize,
    final_in_a: bool,
    threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

struct SecondMessageBuffers {
    rows: Buffer,
    inc: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    gamma: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct RegistersReadWriteSecondMessageInvocation {
    context: SolinasMetal,
    message_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    message_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: SecondMessageBuffers,
    params: FirstMessageParams,
    reduction_steps: Vec<ReductionStep>,
    rows: usize,
    final_in_a: bool,
    threads_per_threadgroup: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    /// Prepares the exact first cycle-round inner endpoints `[q(0), q(infinity)]`.
    ///
    /// Upload and allocation happen here. `execute_timed` measures only the
    /// resident message dispatch and its two-field reduction.
    pub fn prepare_registers_read_write_first_message(
        &self,
        rows: &[RegisterAccessRow],
        inc: &[AkitaField],
        gamma: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: RegistersReadWriteMessageConfig,
    ) -> Result<RegistersReadWriteFirstMessageInvocation, MetalError> {
        if rows.len() < 2 || !rows.len().is_power_of_two() {
            return Err(MetalError::InvalidRegistersReadWriteRows(rows.len()));
        }
        if inc.len() != rows.len() {
            return Err(MetalError::RegistersReadWriteIncLength {
                expected: rows.len(),
                got: inc.len(),
            });
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(rows.len()))?;
        if e_in.is_empty() || e_out.is_empty() || covered != rows.len() / 2 {
            return Err(MetalError::RegistersReadWriteWeightShape {
                expected: rows.len() / 2,
                covered,
            });
        }
        for row in rows {
            row.validate()?;
        }

        let inc = inc.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let e_in = e_in.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let e_out = e_out.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let gamma_sq = gamma * gamma;
        let gamma = [
            Fp128::from_jolt_field(&gamma),
            Fp128::from_jolt_field(&gamma_sq),
        ];
        self.validate_inputs("registers read/write inc", &inc)?;
        self.validate_inputs("registers read/write e_in", &e_in)?;
        self.validate_inputs("registers read/write e_out", &e_out)?;
        self.validate_inputs("registers read/write gamma", &gamma)?;

        let message_pipeline = self.compile_named_pipeline(MESSAGE_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let message_limits = Self::limits(&message_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (MESSAGE_PIPELINE, message_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedRegistersReadWriteExecutionWidth {
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
        let partial_a = self.new_registers_read_write_buffer(partial_elements)?;
        let partial_b = self.new_registers_read_write_buffer(partial_elements)?;
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
        Ok(RegistersReadWriteFirstMessageInvocation {
            context: self.clone(),
            message_pipeline,
            reduction_pipeline,
            message_limits,
            reduction_limits,
            buffers: FirstMessageBuffers {
                rows: buffer_from_slice(&self.device, rows),
                inc: buffer_from_slice(&self.device, &inc),
                e_in: buffer_from_slice(&self.device, &e_in),
                e_out: buffer_from_slice(&self.device, &e_out),
                gamma: buffer_from_slice(&self.device, &gamma),
                partial_a,
                partial_b,
            },
            params,
            reduction_steps,
            rows: rows.len(),
            final_in_a: input_a,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }

    fn new_registers_read_write_buffer(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = byte_length::<Fp128>(elements)?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl RegistersReadWriteFirstMessageInvocation {
    /// Prepares round two while retaining the original resident row, inc, and
    /// gamma buffers. Only the new split-eq weights and reduction scratch are
    /// allocated.
    pub fn prepare_second_message(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: RegistersReadWriteMessageConfig,
    ) -> Result<RegistersReadWriteSecondMessageInvocation, MetalError> {
        if self.rows < 4 {
            return Err(MetalError::InvalidRegistersReadWriteRows(self.rows));
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(self.rows))?;
        if e_in.is_empty() || e_out.is_empty() || covered != self.rows / 4 {
            return Err(MetalError::RegistersReadWriteWeightShape {
                expected: self.rows / 4,
                covered,
            });
        }
        let e_in = e_in.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let e_out = e_out.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        self.context
            .validate_inputs("registers read/write second e_in", &e_in)?;
        self.context
            .validate_inputs("registers read/write second e_out", &e_out)?;

        let message_pipeline = self
            .context
            .compile_named_pipeline(SECOND_MESSAGE_PIPELINE)?;
        let reduction_pipeline = self.context.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let message_limits = SolinasMetal::limits(&message_pipeline);
        let reduction_limits = SolinasMetal::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (SECOND_MESSAGE_PIPELINE, message_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedRegistersReadWriteExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup = SolinasMetal::resolve_threadgroup_width(
            config.threads_per_threadgroup,
            message_limits,
        )?;

        let partial_elements = SAMPLES
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(e_out.len()))?;
        let partial_a = self
            .context
            .new_registers_read_write_buffer(partial_elements)?;
        let partial_b = self
            .context
            .new_registers_read_write_buffer(partial_elements)?;
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
                params: buffer_from_slice(&self.context.device, slice::from_ref(&params)),
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
        Ok(RegistersReadWriteSecondMessageInvocation {
            context: self.context.clone(),
            message_pipeline,
            reduction_pipeline,
            message_limits,
            reduction_limits,
            buffers: SecondMessageBuffers {
                rows: self.buffers.rows.clone(),
                inc: self.buffers.inc.clone(),
                e_in: buffer_from_slice(&self.context.device, &e_in),
                e_out: buffer_from_slice(&self.context.device, &e_out),
                gamma: self.buffers.gamma.clone(),
                partial_a,
                partial_b,
            },
            params,
            reduction_steps,
            rows: self.rows,
            final_in_a: input_a,
            threads_per_threadgroup,
            completed: Cell::new(false),
        })
    }

    pub const fn name(&self) -> &'static str {
        MESSAGE_PIPELINE
    }

    pub const fn rows(&self) -> usize {
        self.rows
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

    pub const fn logical_row_bytes(&self) -> u64 {
        self.rows as u64 * size_of::<RegisterAccessRow>() as u64
    }

    pub const fn logical_inc_bytes(&self) -> u64 {
        self.rows as u64 * size_of::<Fp128>() as u64
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
            encoder.set_buffer(0, Some(&self.buffers.rows), 0);
            encoder.set_buffer(1, Some(&self.buffers.inc), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.buffers.gamma), 0);
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
        // SAFETY: the completed reduction leaves two fields at the front of
        // the selected shared buffer.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), SAMPLES) };
        self.context
            .validate_inputs("registers read/write first message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }
}

impl RegistersReadWriteSecondMessageInvocation {
    pub const fn name(&self) -> &'static str {
        SECOND_MESSAGE_PIPELINE
    }

    pub const fn rows(&self) -> usize {
        self.rows
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

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn execute(&self, challenge: AkitaField) -> Result<(), MetalError> {
        self.execute_timed(challenge).map(|_| ())
    }

    pub fn execute_timed(&self, challenge: AkitaField) -> Result<Duration, MetalError> {
        self.completed.set(false);
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("registers read/write second challenge", &[challenge])?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.message_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.rows), 0);
            encoder.set_buffer(1, Some(&self.buffers.inc), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.buffers.gamma), 0);
            encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 6, &challenge);
            set_inline_bytes(encoder, 7, &self.params);
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
        // SAFETY: the completed reduction leaves two fields at the front of
        // the selected shared buffer.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), SAMPLES) };
        self.context
            .validate_inputs("registers read/write second message", values)?;
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

const _: () = assert!(size_of::<RegisterAccessRow>() == 40);
const _: () = assert!(size_of::<FirstMessageParams>() == 16);
const _: () = assert!(size_of::<ReductionParams>() == 16);
