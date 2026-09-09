use std::{
    cell::Cell,
    mem::{size_of, size_of_val},
    slice,
    time::Duration,
};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
#[cfg(feature = "test-utils")]
use jolt_field::Zero as _;
use jolt_poly::{EqPolynomial, LtPolynomial};
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLResourceOptions, MTLSize,
};

use super::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, Fp128,
    InstructionReadRafStage1Lease, MetalError, PipelineLimits, SolinasMetal,
};

mod stage1;

pub(crate) use stage1::{
    RegistersValInstructionSourceLease, RegistersValInstructionSourceReceipt,
    RegistersValInstructionSourceRequest,
};

const SIMD_WIDTH: usize = 32;
const SAMPLES: usize = 3;
const ADDRESS_BITS: usize = 7;
const ABSENT_REGISTER: u8 = u8::MAX;
const SOURCE_MATERIALIZED: u32 = 0;
const SOURCE_INSTRUCTION_ROWS: u32 = 1;
const MESSAGE_PIPELINE: &str = "solinas_registers_val_first_message_factorized";
const NATIVE_TRANSITION_PIPELINE: &str = "solinas_registers_val_native_transition";
const DENSE_TRANSITION_PIPELINE: &str = "solinas_registers_val_dense_transition";
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValTransitionConfig {
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for RegistersValTransitionConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(32),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValDenseConfig {
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for RegistersValDenseConfig {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: Some(64),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MessageParams {
    cycles: u32,
    high_blocks: u32,
    lt_lo_length: u32,
    source_layout: u32,
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
    dense_a: Buffer,
    dense_b: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

enum RegistersValFirstMessageSource<'a> {
    Uploaded {
        inc: &'a [Fp128],
        rd: &'a [u8],
    },
    InstructionRows {
        receipt: Box<RegistersValInstructionSourceReceipt>,
        source: InstructionReadRafStage1Lease,
    },
}

impl RegistersValFirstMessageSource<'_> {
    fn cycles(&self) -> usize {
        match self {
            Self::Uploaded { inc, .. } => inc.len(),
            Self::InstructionRows { receipt, .. } => receipt.cycles(),
        }
    }

    fn resident_receipt(&self) -> Option<RegistersValInstructionSourceReceipt> {
        match self {
            Self::Uploaded { .. } => None,
            Self::InstructionRows { receipt, .. } => Some(**receipt),
        }
    }
}

pub struct RegistersValFirstMessageInvocation {
    context: SolinasMetal,
    message_pipeline: ComputePipelineState,
    native_transition_pipeline: ComputePipelineState,
    dense_transition_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    message_limits: PipelineLimits,
    native_transition_limits: PipelineLimits,
    dense_transition_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    params: MessageParams,
    reduction_steps: Vec<ReductionStep>,
    cycles: usize,
    threadgroups: usize,
    threads_per_threadgroup: usize,
    final_in_a: bool,
    resident_source: Option<RegistersValInstructionSourceReceipt>,
    instruction_source: Option<InstructionReadRafStage1Lease>,
    completed: Cell<bool>,
}

struct RegistersValFirstMessageCommand {
    command_buffer: CommandBuffer,
}

#[must_use = "a submitted registers-value message must be joined before use"]
pub(crate) struct PendingRegistersValFirstMessage {
    invocation: Option<RegistersValFirstMessageInvocation>,
    command: Option<RegistersValFirstMessageCommand>,
}

impl Drop for PendingRegistersValFirstMessage {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

impl PendingRegistersValFirstMessage {
    pub(crate) fn cycles(&self) -> Option<usize> {
        self.invocation
            .as_ref()
            .map(RegistersValFirstMessageInvocation::cycles)
    }

    pub(crate) fn join(
        mut self,
    ) -> Result<(RegistersValFirstMessageInvocation, Duration), MetalError> {
        let invocation = self
            .invocation
            .take()
            .ok_or(MetalError::InvalidRegistersValState(
                "submitted first message lost its invocation",
            ))?;
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidRegistersValState(
                "submitted first message lost its command",
            ))?;
        let gpu_active = invocation.complete_first_message(command)?;
        Ok((invocation, gpu_active))
    }
}

struct TransitionBuffers {
    inc: Buffer,
    rd: Buffer,
    eq_address: Buffer,
    lt_lo: Buffer,
    lt_hi: Buffer,
    eq_hi: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct RegistersValFirstTransitionInvocation {
    context: SolinasMetal,
    transition_pipeline: ComputePipelineState,
    dense_transition_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    transition_limits: PipelineLimits,
    dense_transition_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: TransitionBuffers,
    params: MessageParams,
    reduction_steps: Vec<ReductionStep>,
    cycles: usize,
    lt_lo_capacity: usize,
    threadgroups: usize,
    threads_per_threadgroup: usize,
    final_in_a: bool,
    instruction_source: Option<InstructionReadRafStage1Lease>,
    completed: Cell<bool>,
}

struct SequenceBuffers {
    lt_lo: Buffer,
    lt_hi: Buffer,
    eq_hi: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct RegistersValSequence {
    context: SolinasMetal,
    transition_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    transition_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: SequenceBuffers,
    reduction_steps: Vec<ReductionStep>,
    current_elements: usize,
    current_lt_lo_length: usize,
    lt_lo_capacity: usize,
    threadgroups: usize,
    threads_per_threadgroup: usize,
    source_in_a: bool,
    final_in_a: bool,
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

        let inc = inc.iter().map(Fp128::from_jolt_field).collect::<Vec<_>>();
        self.prepare_registers_val_first_message_from_source(
            RegistersValFirstMessageSource::Uploaded { inc: &inc, rd },
            r_address,
            r_cycle,
            config,
        )
    }

    pub(crate) fn prepare_registers_val_first_message_instruction_rows(
        &self,
        lease: RegistersValInstructionSourceLease,
        r_address: &[AkitaField],
        r_cycle: &[AkitaField],
        config: RegistersValFirstMessageConfig,
    ) -> Result<RegistersValFirstMessageInvocation, MetalError> {
        let cycles = lease.receipt().cycles();
        let (receipt, source) = lease.into_parts(self, cycles)?;
        self.prepare_registers_val_first_message_from_source(
            RegistersValFirstMessageSource::InstructionRows {
                receipt: Box::new(receipt),
                source,
            },
            r_address,
            r_cycle,
            config,
        )
    }

    fn prepare_registers_val_first_message_from_source(
        &self,
        source: RegistersValFirstMessageSource<'_>,
        r_address: &[AkitaField],
        r_cycle: &[AkitaField],
        config: RegistersValFirstMessageConfig,
    ) -> Result<RegistersValFirstMessageInvocation, MetalError> {
        let cycles = source.cycles();
        if cycles < 4 || !cycles.is_power_of_two() {
            return Err(MetalError::InvalidRegistersValCycles(cycles));
        }
        if r_address.len() != ADDRESS_BITS || r_cycle.len() != cycles.ilog2() as usize {
            return Err(MetalError::RegistersValPointShape {
                address_bits: r_address.len(),
                cycle_bits: r_cycle.len(),
                cycles,
            });
        }
        let inc_bytes = cycles
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(cycles))?;
        let rd_bytes = cycles;
        match &source {
            RegistersValFirstMessageSource::Uploaded { inc, rd } => {
                if inc.len() != cycles || rd.len() != cycles {
                    return Err(MetalError::RegistersValIndexLength {
                        expected: cycles,
                        got: rd.len(),
                    });
                }
            }
            RegistersValFirstMessageSource::InstructionRows { receipt, source } => {
                if receipt.cycles() != cycles
                    || receipt.device_registry_id() != self.device_registry_id()
                    || receipt.instruction_rows_storage_id()
                        != source.receipt().row_allocation_identity()
                    || receipt.instruction_rows_bytes() != source.receipt().row_bytes()
                    || source.row_buffer().device().registry_id() != self.device_registry_id()
                    || source.row_buffer().as_ptr() as usize
                        != receipt.instruction_rows_storage_id()
                    || source.row_buffer().length() != receipt.instruction_rows_bytes()
                {
                    return Err(MetalError::InvalidRegistersValState(
                        "instruction-row source does not match its sealed receipt",
                    ));
                }
            }
        }

        let message_pipeline = self.compile_named_pipeline(MESSAGE_PIPELINE)?;
        let native_transition_pipeline = self.compile_named_pipeline(NATIVE_TRANSITION_PIPELINE)?;
        let dense_transition_pipeline = self.compile_named_pipeline(DENSE_TRANSITION_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let message_limits = Self::limits(&message_pipeline);
        let native_transition_limits = Self::limits(&native_transition_pipeline);
        let dense_transition_limits = Self::limits(&dense_transition_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (MESSAGE_PIPELINE, message_limits),
            (NATIVE_TRANSITION_PIPELINE, native_transition_limits),
            (DENSE_TRANSITION_PIPELINE, dense_transition_limits),
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
            ("registers val eq address", eq_address.as_slice()),
            ("registers val lt lo", lt_lo.as_slice()),
            ("registers val lt hi", lt_hi.as_slice()),
            ("registers val eq hi", eq_hi.as_slice()),
        ] {
            self.validate_inputs(name, values)?;
        }

        let threadgroups = lt_hi.len();
        let partial_count = threadgroups;
        let partial_elements = SAMPLES
            .checked_mul(partial_count)
            .ok_or(MetalError::InputTooLong(partial_count))?;
        let field_bytes = |elements: usize| {
            elements
                .checked_mul(size_of::<Fp128>())
                .ok_or(MetalError::InputTooLong(elements))
        };
        let new_buffer_bytes = [
            size_of_val(eq_address.as_slice()),
            size_of_val(lt_lo.as_slice()),
            size_of_val(lt_hi.as_slice()),
            size_of_val(eq_hi.as_slice()),
            field_bytes(cycles)?,
            field_bytes(cycles / 2)?,
            field_bytes(partial_elements)?,
            field_bytes(partial_elements)?,
        ];
        for &bytes in [inc_bytes, rd_bytes].iter().chain(new_buffer_bytes.iter()) {
            self.validate_buffer_length(
                u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(cycles))?,
            )?;
        }
        self.validate_buffer_length(size_of::<ReductionParams>() as u64)?;
        let reduction_param_bytes = reduction_step_count(partial_count)
            .checked_mul(size_of::<ReductionParams>())
            .ok_or(MetalError::InputTooLong(partial_count))?;
        let source_allocation_bytes = match &source {
            RegistersValFirstMessageSource::Uploaded { .. } => inc_bytes + rd_bytes,
            RegistersValFirstMessageSource::InstructionRows { .. } => 0,
        };
        let additional_bytes = new_buffer_bytes
            .into_iter()
            .try_fold(reduction_param_bytes, usize::checked_add)
            .and_then(|bytes| bytes.checked_add(source_allocation_bytes))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(cycles))?;
        self.validate_additional_working_set(additional_bytes)?;
        let params = MessageParams {
            cycles: u32::try_from(cycles).map_err(|_| MetalError::InputTooLong(cycles))?,
            high_blocks: u32::try_from(threadgroups)
                .map_err(|_| MetalError::InputTooLong(threadgroups))?,
            lt_lo_length: u32::try_from(lt_lo.len())
                .map_err(|_| MetalError::InputTooLong(lt_lo.len()))?,
            source_layout: match &source {
                RegistersValFirstMessageSource::InstructionRows { .. } => SOURCE_INSTRUCTION_ROWS,
                RegistersValFirstMessageSource::Uploaded { .. } => SOURCE_MATERIALIZED,
            },
        };
        let partial_a = self.new_registers_val_buffer(partial_elements)?;
        let partial_b = self.new_registers_val_buffer(partial_elements)?;
        let dense_a = self.new_registers_val_buffer(cycles)?;
        let dense_b = self.new_registers_val_buffer(cycles / 2)?;

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

        let resident_source = source.resident_receipt();
        let (inc, rd, instruction_source) = match source {
            RegistersValFirstMessageSource::Uploaded { inc, rd } => (
                buffer_from_slice(&self.device, inc),
                buffer_from_slice(&self.device, rd),
                None,
            ),
            RegistersValFirstMessageSource::InstructionRows { source, .. } => {
                let rows = source.row_buffer().to_owned();
                (rows.clone(), rows, Some(source))
            }
        };
        Ok(RegistersValFirstMessageInvocation {
            context: self.clone(),
            message_pipeline,
            native_transition_pipeline,
            dense_transition_pipeline,
            reduction_pipeline,
            message_limits,
            native_transition_limits,
            dense_transition_limits,
            reduction_limits,
            buffers: Buffers {
                inc,
                rd,
                eq_address: buffer_from_slice(&self.device, &eq_address),
                lt_lo: buffer_from_slice(&self.device, &lt_lo),
                lt_hi: buffer_from_slice(&self.device, &lt_hi),
                eq_hi: buffer_from_slice(&self.device, &eq_hi),
                dense_a,
                dense_b,
                partial_a,
                partial_b,
            },
            params,
            reduction_steps,
            cycles,
            threadgroups,
            threads_per_threadgroup,
            final_in_a: input_a,
            resident_source,
            instruction_source,
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
    pub fn into_first_transition(
        self,
        bound_lt_lo: &[AkitaField],
        config: RegistersValTransitionConfig,
    ) -> Result<RegistersValFirstTransitionInvocation, MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let expected = self.params.lt_lo_length as usize / 2;
        if expected < 2 || bound_lt_lo.len() != expected {
            return Err(MetalError::RegistersValLtLength {
                expected,
                got: bound_lt_lo.len(),
            });
        }
        let threads_per_threadgroup = SolinasMetal::resolve_threadgroup_width(
            config.threads_per_threadgroup,
            self.native_transition_limits,
        )?;
        write_registers_val_fields(
            &self.buffers.lt_lo,
            self.params.lt_lo_length as usize,
            bound_lt_lo,
        )?;
        let params = MessageParams {
            cycles: u32::try_from(self.cycles)
                .map_err(|_| MetalError::InputTooLong(self.cycles))?,
            high_blocks: self.params.high_blocks,
            lt_lo_length: u32::try_from(bound_lt_lo.len())
                .map_err(|_| MetalError::InputTooLong(bound_lt_lo.len()))?,
            source_layout: self.params.source_layout,
        };
        let Buffers {
            inc,
            rd,
            eq_address,
            lt_lo,
            lt_hi,
            eq_hi,
            dense_a,
            dense_b,
            partial_a,
            partial_b,
        } = self.buffers;
        Ok(RegistersValFirstTransitionInvocation {
            context: self.context,
            transition_pipeline: self.native_transition_pipeline,
            dense_transition_pipeline: self.dense_transition_pipeline,
            reduction_pipeline: self.reduction_pipeline,
            transition_limits: self.native_transition_limits,
            dense_transition_limits: self.dense_transition_limits,
            reduction_limits: self.reduction_limits,
            buffers: TransitionBuffers {
                inc,
                rd,
                eq_address,
                lt_lo,
                lt_hi,
                eq_hi,
                dense_a,
                dense_b,
                partial_a,
                partial_b,
            },
            params,
            reduction_steps: self.reduction_steps,
            cycles: self.cycles,
            lt_lo_capacity: self.params.lt_lo_length as usize,
            threadgroups: self.threadgroups,
            threads_per_threadgroup,
            final_in_a: self.final_in_a,
            instruction_source: self.instruction_source,
            completed: Cell::new(false),
        })
    }

    pub const fn name(&self) -> &'static str {
        MESSAGE_PIPELINE
    }

    copy_field_getters! { pub, {
        cycles: usize,
        threadgroups: usize,
        threads_per_threadgroup: usize,
        pipeline_limits => message_limits: PipelineLimits,
        reduction_pipeline_limits => reduction_limits: PipelineLimits,
    }}

    copy_field_getters! { pub(crate), {
        resident_source_receipt => resident_source: Option<RegistersValInstructionSourceReceipt>,
    }}

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
        let command = self.submit_first_message();
        self.complete_first_message(command)
    }

    pub(crate) fn submit(self) -> PendingRegistersValFirstMessage {
        let command = self.submit_first_message();
        PendingRegistersValFirstMessage {
            invocation: Some(self),
            command: Some(command),
        }
    }

    fn submit_first_message(&self) -> RegistersValFirstMessageCommand {
        self.completed.set(false);
        let command_buffer = self.context.queue.new_command_buffer().to_owned();
        autoreleasepool(|| {
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

            encode_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.reduction_steps,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
            );
            encoder.end_encoding();
            command_buffer.commit();
        });
        RegistersValFirstMessageCommand { command_buffer }
    }

    fn complete_first_message(
        &self,
        command: RegistersValFirstMessageCommand,
    ) -> Result<Duration, MetalError> {
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        self.completed.set(true);
        Ok(gpu_active)
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

impl RegistersValFirstTransitionInvocation {
    pub fn into_sequence(
        self,
        config: RegistersValDenseConfig,
    ) -> Result<RegistersValSequence, MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let threads_per_threadgroup = SolinasMetal::resolve_threadgroup_width(
            config.threads_per_threadgroup,
            self.dense_transition_limits,
        )?;
        let current_elements = self.current_elements();
        drop(self.instruction_source);
        let TransitionBuffers {
            inc: _,
            rd: _,
            eq_address: _,
            lt_lo,
            lt_hi,
            eq_hi,
            dense_a,
            dense_b,
            partial_a,
            partial_b,
        } = self.buffers;
        let current_lt_lo_length = self.params.lt_lo_length as usize;
        Ok(RegistersValSequence {
            context: self.context,
            transition_pipeline: self.dense_transition_pipeline,
            reduction_pipeline: self.reduction_pipeline,
            transition_limits: self.dense_transition_limits,
            reduction_limits: self.reduction_limits,
            buffers: SequenceBuffers {
                lt_lo,
                lt_hi,
                eq_hi,
                dense_a,
                dense_b,
                partial_a,
                partial_b,
            },
            reduction_steps: self.reduction_steps,
            current_elements,
            current_lt_lo_length,
            lt_lo_capacity: self.lt_lo_capacity,
            threadgroups: self.threadgroups,
            threads_per_threadgroup,
            source_in_a: true,
            final_in_a: self.final_in_a,
        })
    }

    pub const fn name(&self) -> &'static str {
        NATIVE_TRANSITION_PIPELINE
    }

    pub const fn current_elements(&self) -> usize {
        self.cycles / 2
    }

    copy_field_getters! { pub, {
        source_cycles => cycles: usize,
        threads_per_threadgroup: usize,
        pipeline_limits => transition_limits: PipelineLimits,
        reduction_pipeline_limits => reduction_limits: PipelineLimits,
    }}

    pub const fn execute_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn dynamic_threadgroup_memory_bytes(&self) -> usize {
        2 * SAMPLES * (self.threads_per_threadgroup / SIMD_WIDTH) * size_of::<Fp128>()
    }

    pub fn execute(&self, challenge: AkitaField) -> Result<(), MetalError> {
        self.execute_timed(challenge).map(|_| ())
    }

    pub fn execute_timed(&self, challenge: AkitaField) -> Result<Duration, MetalError> {
        self.completed.set(false);
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("registers val first transition challenge", &[challenge])?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.transition_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.inc), 0);
            encoder.set_buffer(1, Some(&self.buffers.rd), 0);
            encoder.set_buffer(2, Some(&self.buffers.eq_address), 0);
            encoder.set_buffer(3, Some(&self.buffers.lt_lo), 0);
            encoder.set_buffer(4, Some(&self.buffers.lt_hi), 0);
            encoder.set_buffer(5, Some(&self.buffers.eq_hi), 0);
            encoder.set_buffer(6, Some(&self.buffers.dense_a), 0);
            encoder.set_buffer(7, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 8, &challenge);
            set_inline_bytes(encoder, 9, &self.params);
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

            encode_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.reduction_steps,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = completed_command_gpu_time(command_buffer)?;
            self.completed.set(true);
            Ok(gpu_active)
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
            .validate_inputs("registers val second message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    /// Copies the dense state after the native first transition.
    pub fn read_dense_state_into(&self, output: &mut [[AkitaField; 2]]) -> Result<(), MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let expected = self.current_elements();
        if output.len() != expected {
            return Err(MetalError::RegistersValStateLength {
                expected,
                got: output.len(),
            });
        }
        // SAFETY: the transition writes two fields for every current element.
        let values = unsafe {
            slice::from_raw_parts(
                self.buffers.dense_a.contents().cast::<Fp128>(),
                2 * expected,
            )
        };
        self.context
            .validate_inputs("registers val dense state", values)?;
        for (output, row) in output.iter_mut().zip(values.chunks_exact(2)) {
            *output = [row[0].into_jolt_field(), row[1].into_jolt_field()];
        }
        Ok(())
    }

    #[cfg(feature = "test-utils")]
    pub fn read_dense_state(&self) -> Result<Vec<[AkitaField; 2]>, MetalError> {
        let mut output = vec![[AkitaField::zero(); 2]; self.current_elements()];
        self.read_dense_state_into(&mut output)?;
        Ok(output)
    }
}

impl RegistersValSequence {
    /// Binds the current dense `inc` and write-address tables, then evaluates
    /// the next round message against the host-bound LT-low table.
    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        bound_lt_lo: &[AkitaField],
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        self.bind_and_message_timed(challenge, bound_lt_lo)
            .map(|(message, _)| message)
    }

    pub fn bind_and_message_timed(
        &mut self,
        challenge: AkitaField,
        bound_lt_lo: &[AkitaField],
    ) -> Result<([AkitaField; SAMPLES], Duration), MetalError> {
        let (message, active_time, next_elements, next_lt_lo_length) =
            self.execute_current_bind_and_message(challenge, bound_lt_lo)?;
        self.current_elements = next_elements;
        self.current_lt_lo_length = next_lt_lo_length;
        self.source_in_a = !self.source_in_a;
        Ok((message, active_time))
    }

    fn execute_current_bind_and_message(
        &self,
        challenge: AkitaField,
        bound_lt_lo: &[AkitaField],
    ) -> Result<([AkitaField; SAMPLES], Duration, usize, usize), MetalError> {
        if self.current_lt_lo_length < 4 {
            return Err(MetalError::RegistersValSplitLtExhausted(
                self.current_lt_lo_length,
            ));
        }
        let expected_lt_lo_length = self.current_lt_lo_length / 2;
        if bound_lt_lo.len() != expected_lt_lo_length {
            return Err(MetalError::RegistersValLtLength {
                expected: expected_lt_lo_length,
                got: bound_lt_lo.len(),
            });
        }
        let next_elements = self.current_elements / 2;
        let covered = self
            .threadgroups
            .checked_mul(expected_lt_lo_length)
            .ok_or(MetalError::InputTooLong(next_elements))?;
        if covered != next_elements {
            return Err(MetalError::RegistersValLtLength {
                expected: next_elements / self.threadgroups,
                got: expected_lt_lo_length,
            });
        }
        write_registers_val_fields(&self.buffers.lt_lo, self.lt_lo_capacity, bound_lt_lo)?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("registers val dense transition challenge", &[challenge])?;
        let params = MessageParams {
            cycles: u32::try_from(next_elements)
                .map_err(|_| MetalError::InputTooLong(next_elements))?,
            high_blocks: u32::try_from(self.threadgroups)
                .map_err(|_| MetalError::InputTooLong(self.threadgroups))?,
            lt_lo_length: u32::try_from(expected_lt_lo_length)
                .map_err(|_| MetalError::InputTooLong(expected_lt_lo_length))?,
            source_layout: SOURCE_MATERIALIZED,
        };

        let (message, active_time) = autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.transition_pipeline);
            encoder.set_buffer(0, Some(self.source_buffer()), 0);
            encoder.set_buffer(1, Some(self.destination_buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.lt_lo), 0);
            encoder.set_buffer(3, Some(&self.buffers.lt_hi), 0);
            encoder.set_buffer(4, Some(&self.buffers.eq_hi), 0);
            encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 6, &challenge);
            set_inline_bytes(encoder, 7, &params);
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

            encode_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.reduction_steps,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let gpu_active = completed_command_gpu_time(command_buffer)?;
            let final_buffer = if self.final_in_a {
                &self.buffers.partial_a
            } else {
                &self.buffers.partial_b
            };
            // SAFETY: the dispatch and reductions completed and leave three
            // fields at the front of the selected shared buffer.
            let values =
                unsafe { slice::from_raw_parts(final_buffer.contents().cast::<Fp128>(), SAMPLES) };
            self.context
                .validate_inputs("registers val dense message", values)?;
            let message = std::array::from_fn(|index| values[index].into_jolt_field());
            Ok::<_, MetalError>((message, gpu_active))
        })?;

        Ok((message, active_time, next_elements, expected_lt_lo_length))
    }

    copy_field_getters! { pub, {
        current_elements: usize,
        current_lt_lo_length: usize,
        threads_per_threadgroup: usize,
        pipeline_limits => transition_limits: PipelineLimits,
        reduction_pipeline_limits => reduction_limits: PipelineLimits,
    }}

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn dynamic_threadgroup_memory_bytes(&self) -> usize {
        2 * SAMPLES * (self.threads_per_threadgroup / SIMD_WIDTH) * size_of::<Fp128>()
    }

    /// Copies the current resident `(inc, wa)` rows into caller-owned storage.
    pub fn read_current_dense_state_into(
        &self,
        output: &mut [[AkitaField; 2]],
    ) -> Result<(), MetalError> {
        if output.len() != self.current_elements {
            return Err(MetalError::RegistersValStateLength {
                expected: self.current_elements,
                got: output.len(),
            });
        }
        let elements = 2 * self.current_elements;
        // SAFETY: each dense buffer has capacity for at least `elements`
        // fields and every transition waits for completion before advancing.
        let values = unsafe {
            slice::from_raw_parts(self.source_buffer().contents().cast::<Fp128>(), elements)
        };
        self.context
            .validate_inputs("registers val resident dense state", values)?;
        for (output, row) in output.iter_mut().zip(values.chunks_exact(2)) {
            *output = [row[0].into_jolt_field(), row[1].into_jolt_field()];
        }
        Ok(())
    }

    #[cfg(feature = "test-utils")]
    pub fn read_current_dense_state(&self) -> Result<Vec<[AkitaField; 2]>, MetalError> {
        let mut output = vec![[AkitaField::zero(); 2]; self.current_elements];
        self.read_current_dense_state_into(&mut output)?;
        Ok(output)
    }

    fn source_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.dense_a
        } else {
            &self.buffers.dense_b
        }
    }

    fn destination_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.dense_b
        } else {
            &self.buffers.dense_a
        }
    }
}

fn reduction_step_count(mut input_count: usize) -> usize {
    let mut steps = 0;
    while input_count > 1 {
        input_count = input_count.div_ceil(SIMD_WIDTH);
        steps += 1;
    }
    steps
}

fn encode_reductions(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    steps: &[ReductionStep],
    partial_a: &Buffer,
    partial_b: &Buffer,
) {
    for step in steps {
        encoder.set_compute_pipeline_state(pipeline);
        let (input, output) = if step.input_a {
            (partial_a, partial_b)
        } else {
            (partial_b, partial_a)
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
}

fn write_registers_val_fields(
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::RegistersValLtLength {
            expected: capacity,
            got: values.len(),
        });
    }
    // SAFETY: the shared buffer holds `capacity` fields and no command buffer
    // is active while the host updates the LT-low prefix.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

const _: () = assert!(size_of::<MessageParams>() == 16);
const _: () = assert!(size_of::<ReductionParams>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity setup")]
mod tests {
    use jolt_field::{Ring as _, Zero as _};
    use jolt_poly::{BindingOrder, Polynomial};

    use super::*;

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len)
            .map(|index| AkitaField::from_u64(seed + 19 * index as u64))
            .collect()
    }

    fn bind(values: Vec<AkitaField>, challenge: AkitaField) -> Vec<AkitaField> {
        let mut values = Polynomial::new(values);
        values.bind_with_order(challenge, BindingOrder::LowToHigh);
        values.evals().to_vec()
    }

    #[test]
    fn production_export_matches_two_native_binds() {
        let Ok(context) = SolinasMetal::for_akita() else {
            return;
        };
        let cycles = 64usize;
        let inc = (0..cycles)
            .map(|index| AkitaField::from_u64(3 + 17 * index as u64))
            .collect::<Vec<_>>();
        let rd = (0..cycles)
            .map(|index| {
                if index.is_multiple_of(5) {
                    ABSENT_REGISTER
                } else {
                    index as u8
                }
            })
            .collect::<Vec<_>>();
        let r_address = point(ADDRESS_BITS, 101);
        let r_cycle = point(cycles.ilog2() as usize, 211);
        let first = context
            .prepare_registers_val_first_message(
                &inc,
                &rd,
                &r_address,
                &r_cycle,
                RegistersValFirstMessageConfig::default(),
            )
            .unwrap();
        first.execute().unwrap();

        let low_bits = r_cycle.len() / 2;
        let r_lo = &r_cycle[r_cycle.len() - low_bits..];
        let challenge_0 = AkitaField::from_u64(307);
        let challenge_1 = AkitaField::from_u64(401);
        let lt_lo = bind(LtPolynomial::<AkitaField>::evaluations(r_lo), challenge_0);
        let transition = first
            .into_first_transition(&lt_lo, RegistersValTransitionConfig::default())
            .unwrap();
        transition.execute(challenge_0).unwrap();
        let mut sequence = transition
            .into_sequence(RegistersValDenseConfig::default())
            .unwrap();
        let lt_lo = bind(lt_lo, challenge_1);
        let _ = sequence.bind_and_message(challenge_1, &lt_lo).unwrap();

        let mut exported = vec![[AkitaField::zero(); 2]; sequence.current_elements()];
        sequence
            .read_current_dense_state_into(&mut exported)
            .unwrap();

        let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
        let wa = rd
            .iter()
            .map(|&index| {
                if index == ABSENT_REGISTER {
                    AkitaField::zero()
                } else {
                    eq_address[index as usize]
                }
            })
            .collect::<Vec<_>>();
        let expected_inc = bind(bind(inc, challenge_0), challenge_1);
        let expected_wa = bind(bind(wa, challenge_0), challenge_1);
        let expected = expected_inc
            .into_iter()
            .zip(expected_wa)
            .map(|(inc, wa)| [inc, wa])
            .collect::<Vec<_>>();
        assert_eq!(exported, expected);

        let mut short = vec![[AkitaField::zero(); 2]; exported.len() - 1];
        assert!(matches!(
            sequence.read_current_dense_state_into(&mut short),
            Err(MetalError::RegistersValStateLength {
                expected: 16,
                got: 15
            })
        ));
    }
}
