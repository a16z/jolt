use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use jolt_poly::{EqPolynomial, LtPolynomial};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

const SIMD_WIDTH: usize = 32;
const SAMPLES: usize = 3;
const ADDRESS_BITS: usize = 7;
const ABSENT_REGISTER: u8 = u8::MAX;
const MESSAGE_PIPELINE: &str = "solinas_registers_val_first_message_factorized";
const REDUCTION_PIPELINE: &str = "solinas_registers_val_reduce";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValFirstMessageConfig {
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for RegistersValFirstMessageConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(32),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MessageParams {
    cycles: u32,
    high_blocks: u32,
    lt_lo_length: u32,
    _reserved: u32,
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

struct Buffers {
    inc: Buffer,
    rd: Buffer,
    eq_address: Buffer,
    lt_lo: Buffer,
    lt_hi: Buffer,
    eq_hi: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct RegistersValFirstMessageInvocation {
    context: SolinasMetal,
    message_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    message_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    params: MessageParams,
    reduction_steps: Vec<ReductionStep>,
    cycles: usize,
    threadgroups: usize,
    threads_per_threadgroup: usize,
    final_in_a: bool,
    completed: Cell<bool>,
}

impl SolinasMetal {
    /// Prepares the first cycle-round evaluations at `t = 0, 2, 3` for
    /// `LT(cycle, r_cycle) * inc(cycle) * eq(r_address, rd(cycle))`.
    pub fn prepare_registers_val_first_message(
        &self,
        inc: &[AkitaField],
        rd: &[u8],
        r_address: &[AkitaField],
        r_cycle: &[AkitaField],
        config: RegistersValFirstMessageConfig,
    ) -> Result<RegistersValFirstMessageInvocation, MetalError> {
        if inc.len() < 4 || !inc.len().is_power_of_two() {
            return Err(MetalError::InvalidRegistersValCycles(inc.len()));
        }
        if rd.len() != inc.len() {
            return Err(MetalError::RegistersValIndexLength {
                expected: inc.len(),
                got: rd.len(),
            });
        }
        if r_address.len() != ADDRESS_BITS || r_cycle.len() != inc.len().ilog2() as usize {
            return Err(MetalError::RegistersValPointShape {
                address_bits: r_address.len(),
                cycle_bits: r_cycle.len(),
                cycles: inc.len(),
            });
        }
        if let Some(&index) = rd
            .iter()
            .find(|&&index| index != ABSENT_REGISTER && index >= (1 << ADDRESS_BITS))
        {
            return Err(MetalError::InvalidRegistersValIndex(index));
        }

        let message_pipeline = self.compile_named_pipeline(MESSAGE_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let message_limits = Self::limits(&message_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (MESSAGE_PIPELINE, message_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedRegistersValExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, message_limits)?;

        let inc = inc.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        let eq_address = EqPolynomial::<AkitaField>::evals(r_address, None)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        let mid = r_cycle.len() / 2;
        let (r_hi, r_lo) = r_cycle.split_at(r_cycle.len() - mid);
        let lt_lo = LtPolynomial::<AkitaField>::evaluations(r_lo)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        let lt_hi = LtPolynomial::<AkitaField>::evaluations(r_hi)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        let eq_hi = EqPolynomial::<AkitaField>::evals(r_hi, None)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        for (name, values) in [
            ("registers val inc", inc.as_slice()),
            ("registers val eq address", eq_address.as_slice()),
            ("registers val lt lo", lt_lo.as_slice()),
            ("registers val lt hi", lt_hi.as_slice()),
            ("registers val eq hi", eq_hi.as_slice()),
        ] {
            self.validate_inputs(name, values)?;
        }

        let threadgroups = lt_hi.len();
        let partial_count = threadgroups;
        let params = MessageParams {
            cycles: u32::try_from(inc.len()).map_err(|_| MetalError::InputTooLong(inc.len()))?,
            high_blocks: u32::try_from(threadgroups)
                .map_err(|_| MetalError::InputTooLong(threadgroups))?,
            lt_lo_length: u32::try_from(lt_lo.len())
                .map_err(|_| MetalError::InputTooLong(lt_lo.len()))?,
            _reserved: 0,
        };
        let partial_elements = SAMPLES
            .checked_mul(partial_count)
            .ok_or(MetalError::InputTooLong(partial_count))?;
        let partial_a = self.new_registers_val_buffer(partial_elements)?;
        let partial_b = self.new_registers_val_buffer(partial_elements)?;

        let mut reduction_steps = Vec::new();
        let mut input_count = partial_count;
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

        Ok(RegistersValFirstMessageInvocation {
            context: self.clone(),
            message_pipeline,
            reduction_pipeline,
            message_limits,
            reduction_limits,
            buffers: Buffers {
                inc: buffer_from_slice(&self.device, &inc),
                rd: buffer_from_slice(&self.device, rd),
                eq_address: buffer_from_slice(&self.device, &eq_address),
                lt_lo: buffer_from_slice(&self.device, &lt_lo),
                lt_hi: buffer_from_slice(&self.device, &lt_hi),
                eq_hi: buffer_from_slice(&self.device, &eq_hi),
                partial_a,
                partial_b,
            },
            params,
            reduction_steps,
            cycles: inc.len(),
            threadgroups,
            threads_per_threadgroup,
            final_in_a: input_a,
            completed: Cell::new(false),
        })
    }

    fn new_registers_val_buffer(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = elements
            .checked_mul(size_of::<Fp128>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(elements))?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl RegistersValFirstMessageInvocation {
    pub const fn name(&self) -> &'static str {
        MESSAGE_PIPELINE
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub const fn threadgroups(&self) -> usize {
        self.threadgroups
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

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn dynamic_threadgroup_memory_bytes(&self) -> usize {
        2 * SAMPLES * (self.threads_per_threadgroup / SIMD_WIDTH) * size_of::<Fp128>()
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
            encoder.set_buffer(0, Some(&self.buffers.inc), 0);
            encoder.set_buffer(1, Some(&self.buffers.rd), 0);
            encoder.set_buffer(2, Some(&self.buffers.eq_address), 0);
            encoder.set_buffer(3, Some(&self.buffers.lt_lo), 0);
            encoder.set_buffer(4, Some(&self.buffers.lt_hi), 0);
            encoder.set_buffer(5, Some(&self.buffers.eq_hi), 0);
            encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 7, &self.params);
            encoder
                .set_threadgroup_memory_length(0, self.dynamic_threadgroup_memory_bytes() as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.threadgroups as u64,
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
        // SAFETY: the completed reduction leaves three fields at the front of
        // the selected shared buffer.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), SAMPLES) };
        self.context
            .validate_inputs("registers val first message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

const _: () = assert!(size_of::<MessageParams>() == 16);
const _: () = assert!(size_of::<ReductionParams>() == 16);
