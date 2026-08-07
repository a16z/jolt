use std::{cell::Cell, mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

mod tiled_transition;

pub use tiled_transition::{
    DenseTransitionError, DenseTransitionInvocation, DenseTransitionObservation,
    DenseTransitionParams, DenseTransitionTile,
};

pub(super) const TILED_TRANSITION_SOURCE: &str = include_str!("tiled_transition.metal");

pub const PRODUCT5_FACTORS: usize = 5;

const PRODUCT5_SIMD_WIDTH: usize = 32;
const MESSAGE_DEFAULT_SIMDGROUPS: usize = 4;
const TRANSITION_DEFAULT_SIMDGROUPS: usize = 2;
const MESSAGE_PIPELINE: &str = "solinas_product5_message";
const TRANSITION_PIPELINE: &str = "solinas_product5_fused_transition";
const REDUCE_PIPELINE: &str = "solinas_product5_reduce";

/// Dispatch controls for the five-factor sumcheck probes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Product5Config {
    /// Threads assigned to each outer equality-table block.
    ///
    /// The default is four SIMD groups for a message and two for a transition.
    /// Explicit values must be a supported multiple of the execution width.
    pub threads_per_threadgroup: Option<usize>,
}

/// Dispatch controls for a resident five-factor round sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Product5SequenceConfig {
    pub message_threads_per_threadgroup: Option<usize>,
    pub transition_threads_per_threadgroup: Option<usize>,
}

impl Default for Product5SequenceConfig {
    fn default() -> Self {
        Self {
            message_threads_per_threadgroup: Some(PRODUCT5_SIMD_WIDTH * MESSAGE_DEFAULT_SIMDGROUPS),
            transition_threads_per_threadgroup: Some(
                PRODUCT5_SIMD_WIDTH * TRANSITION_DEFAULT_SIMDGROUPS,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Product5Mode {
    Message,
    FusedTransition,
}

impl Product5Mode {
    const fn pipeline(self) -> &'static str {
        match self {
            Self::Message => MESSAGE_PIPELINE,
            Self::FusedTransition => TRANSITION_PIPELINE,
        }
    }

    const fn minimum_elements(self) -> usize {
        match self {
            Self::Message => 2,
            Self::FusedTransition => 4,
        }
    }

    const fn message_pairs(self, source_elements: usize) -> usize {
        match self {
            Self::Message => source_elements / 2,
            Self::FusedTransition => source_elements / 4,
        }
    }

    const fn default_simdgroups(self) -> usize {
        match self {
            Self::Message => MESSAGE_DEFAULT_SIMDGROUPS,
            Self::FusedTransition => TRANSITION_DEFAULT_SIMDGROUPS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Product5Params {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
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

struct Product5Buffers {
    tables: Buffer,
    bound: Option<Buffer>,
    e_in: Buffer,
    e_out: Buffer,
    challenge: Option<Buffer>,
    params: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

/// A prepared five-factor message or fused bind-and-message dispatch.
///
/// Input and output buffers stay allocated across calls to [`Self::execute`].
pub struct Product5Invocation<'a> {
    context: &'a SolinasMetal,
    mode: Product5Mode,
    main_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    main_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: Product5Buffers,
    reduction_steps: Vec<ReductionStep>,
    final_in_a: bool,
    threads_per_threadgroup: usize,
    source_elements: usize,
    e_out_length: usize,
    completed: Cell<bool>,
}

impl SolinasMetal {
    /// Prepares five Gruen-grid evaluations of a five-factor product sum.
    pub fn prepare_product5_message(
        &self,
        tables: &[Fp128],
        elements_per_table: usize,
        e_in: &[Fp128],
        e_out: &[Fp128],
        config: Product5Config,
    ) -> Result<Product5Invocation<'_>, MetalError> {
        self.prepare_product5(
            Product5Mode::Message,
            tables,
            elements_per_table,
            None,
            e_in,
            e_out,
            config,
        )
    }

    /// Prepares a fused bind followed by the next five-factor message.
    ///
    /// Each factor consumes four source values, writes two bound values, and
    /// evaluates the resulting pair without reading it back from device memory.
    pub fn prepare_product5_fused_transition(
        &self,
        tables: &[Fp128],
        elements_per_table: usize,
        challenge: Fp128,
        e_in: &[Fp128],
        e_out: &[Fp128],
        config: Product5Config,
    ) -> Result<Product5Invocation<'_>, MetalError> {
        self.prepare_product5(
            Product5Mode::FusedTransition,
            tables,
            elements_per_table,
            Some(challenge),
            e_in,
            e_out,
            config,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the product relation has three input buffers"
    )]
    fn prepare_product5(
        &self,
        mode: Product5Mode,
        tables: &[Fp128],
        elements_per_table: usize,
        challenge: Option<Fp128>,
        e_in: &[Fp128],
        e_out: &[Fp128],
        config: Product5Config,
    ) -> Result<Product5Invocation<'_>, MetalError> {
        if elements_per_table < mode.minimum_elements() || !elements_per_table.is_power_of_two() {
            return Err(MetalError::InvalidProduct5TableLength {
                minimum: mode.minimum_elements(),
                got: elements_per_table,
            });
        }
        let expected_storage = PRODUCT5_FACTORS
            .checked_mul(elements_per_table)
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        if expected_storage > u32::MAX as usize {
            return Err(MetalError::InputTooLong(expected_storage));
        }
        if tables.len() != expected_storage {
            return Err(MetalError::Product5StorageLength {
                expected: expected_storage,
                got: tables.len(),
            });
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        let expected_pairs = mode.message_pairs(elements_per_table);
        if e_in.is_empty() || e_out.is_empty() || covered != expected_pairs {
            return Err(MetalError::Product5WeightShape {
                expected: expected_pairs,
                covered,
            });
        }
        let source_elements = u32::try_from(elements_per_table)
            .map_err(|_| MetalError::InputTooLong(elements_per_table))?;
        let e_in_length =
            u32::try_from(e_in.len()).map_err(|_| MetalError::InputTooLong(e_in.len()))?;
        let e_out_length =
            u32::try_from(e_out.len()).map_err(|_| MetalError::InputTooLong(e_out.len()))?;
        self.validate_inputs("product5 tables", tables)?;
        self.validate_inputs("product5 e_in", e_in)?;
        self.validate_inputs("product5 e_out", e_out)?;
        if let Some(challenge) = challenge {
            self.validate_inputs("product5 challenge", slice::from_ref(&challenge))?;
        }

        let main_pipeline = self.compile_named_pipeline(mode.pipeline())?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let main_limits = Self::limits(&main_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (mode.pipeline(), main_limits),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != PRODUCT5_SIMD_WIDTH {
                return Err(MetalError::UnsupportedProduct5ExecutionWidth {
                    pipeline,
                    expected: PRODUCT5_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let default_width = (main_limits.thread_execution_width * mode.default_simdgroups())
            .min(main_limits.max_total_threads_per_threadgroup);
        let threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.threads_per_threadgroup.or(Some(default_width)),
            main_limits,
        )?;

        let table_bytes = buffer_bytes(tables.len())?;
        self.validate_buffer_length(table_bytes)?;
        let bound_elements = if mode == Product5Mode::FusedTransition {
            Some(expected_storage / 2)
        } else {
            None
        };
        let bound = bound_elements
            .map(|elements| {
                let bytes = buffer_bytes(elements)?;
                self.validate_buffer_length(bytes)?;
                Ok::<Buffer, MetalError>(
                    self.device
                        .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
                )
            })
            .transpose()?;
        let partial_elements = PRODUCT5_FACTORS
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(e_out.len()))?;
        let partial_bytes = buffer_bytes(partial_elements)?;
        self.validate_buffer_length(partial_bytes)?;

        let mut reduction_steps = Vec::new();
        let mut input_count = e_out.len();
        let mut input_a = true;
        while input_count > 1 {
            let output_count = input_count.div_ceil(reduction_limits.thread_execution_width);
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

        let params = Product5Params {
            source_elements,
            e_in_length,
            e_out_length,
            _reserved: 0,
        };
        let challenge =
            challenge.map(|value| buffer_from_slice(&self.device, slice::from_ref(&value)));
        Ok(Product5Invocation {
            context: self,
            mode,
            main_pipeline,
            reduction_pipeline,
            main_limits,
            reduction_limits,
            buffers: Product5Buffers {
                tables: buffer_from_slice(&self.device, tables),
                bound,
                e_in: buffer_from_slice(&self.device, e_in),
                e_out: buffer_from_slice(&self.device, e_out),
                challenge,
                params: buffer_from_slice(&self.device, slice::from_ref(&params)),
                partial_a: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
                partial_b: self
                    .device
                    .new_buffer(partial_bytes, MTLResourceOptions::StorageModeShared),
            },
            reduction_steps,
            final_in_a: input_a,
            threads_per_threadgroup,
            source_elements: elements_per_table,
            e_out_length: e_out.len(),
            completed: Cell::new(false),
        })
    }
}

impl Product5Invocation<'_> {
    /// Returns the Metal entry point used by this invocation.
    pub const fn name(&self) -> &'static str {
        self.mode.pipeline()
    }

    /// Returns the number of source elements in each of the five tables.
    pub const fn source_elements(&self) -> usize {
        self.source_elements
    }

    /// Returns the selected threadgroup width.
    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    /// Returns resource limits for the main product kernel.
    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.main_limits
    }

    /// Returns resource limits for the recursive message reduction kernel.
    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduction_limits
    }

    /// Returns the dynamically allocated reduction scratch per main threadgroup.
    pub const fn dynamic_threadgroup_memory_bytes(&self) -> usize {
        PRODUCT5_FACTORS * (self.threads_per_threadgroup / PRODUCT5_SIMD_WIDTH) * size_of::<Fp128>()
    }

    /// Counts field multiplications that directly implement the relation.
    ///
    /// This includes equality weights and, for a transition, binding. It does
    /// not count additions, reductions, or address arithmetic.
    pub const fn useful_multiplications(&self) -> u64 {
        let core = match self.mode {
            Product5Mode::Message => 11 * self.source_elements as u64,
            Product5Mode::FusedTransition => 8 * self.source_elements as u64,
        };
        core + PRODUCT5_FACTORS as u64 * self.e_out_length as u64
    }

    /// Counts source-factor reads plus bound-factor writes.
    ///
    /// Equality weights and reduction scratch traffic are intentionally omitted
    /// so this is the optimistic resident-weight traffic model.
    pub const fn logical_factor_bytes(&self) -> u64 {
        match self.mode {
            Product5Mode::Message => 80 * self.source_elements as u64,
            Product5Mode::FusedTransition => 120 * self.source_elements as u64,
        }
    }

    /// Executes the complete dispatch and waits for its result.
    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    /// Executes the complete dispatch and returns Metal's active GPU duration.
    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.main_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.tables), 0);
            match self.mode {
                Product5Mode::Message => {
                    encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
                    encoder.set_buffer(4, Some(&self.buffers.params), 0);
                }
                Product5Mode::FusedTransition => {
                    encoder.set_buffer(1, self.buffers.bound.as_deref(), 0);
                    encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                    encoder.set_buffer(5, self.buffers.challenge.as_deref(), 0);
                    encoder.set_buffer(6, Some(&self.buffers.params), 0);
                }
            }
            encoder
                .set_threadgroup_memory_length(0, self.dynamic_threadgroup_memory_bytes() as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.e_out_length as u64,
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
                        width: self.reduction_limits.thread_execution_width as u64,
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

    /// Reads `q(1)`, `q(2)`, `q(3)`, `q(4)`, and `q(infinity)` after execution.
    pub fn read_message(&self) -> Result<[Fp128; PRODUCT5_FACTORS], MetalError> {
        self.require_completed()?;
        let buffer = if self.final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the final reduction buffer contains five initialized `Fp128`
        // values after the completed command buffer.
        let values =
            unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), PRODUCT5_FACTORS) };
        self.context.validate_inputs("product5 message", values)?;
        Ok(std::array::from_fn(|index| values[index]))
    }

    /// Reads the bound factor tables produced by a fused transition.
    ///
    /// Message-only invocations return `None`.
    pub fn read_bound_tables(&self) -> Result<Option<Vec<Fp128>>, MetalError> {
        self.require_completed()?;
        let Some(buffer) = &self.buffers.bound else {
            return Ok(None);
        };
        let elements = PRODUCT5_FACTORS * self.source_elements / 2;
        // SAFETY: `bound` was allocated for exactly `elements` `Fp128` values
        // and the transition dispatch has completed.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), elements) };
        self.context
            .validate_inputs("product5 bound output", values)?;
        Ok(Some(values.to_vec()))
    }

    fn require_completed(&self) -> Result<(), MetalError> {
        if self.completed.get() {
            Ok(())
        } else {
            Err(MetalError::NotExecuted)
        }
    }
}

fn buffer_bytes(elements: usize) -> Result<u64, MetalError> {
    let bytes = elements
        .checked_mul(size_of::<Fp128>())
        .ok_or(MetalError::InputTooLong(elements))?;
    u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(elements))
}

struct Product5SequenceBuffers {
    tables_a: Buffer,
    tables_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

/// Five factor tables retained in Metal buffers across a sumcheck tail.
pub struct Product5Sequence {
    context: SolinasMetal,
    message_pipeline: ComputePipelineState,
    transition_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    reduction_limits: PipelineLimits,
    buffers: Product5SequenceBuffers,
    message_threads_per_threadgroup: usize,
    transition_threads_per_threadgroup: usize,
    initial_elements: usize,
    current_elements: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    source_in_a: bool,
    gpu_active_time: Duration,
}

impl SolinasMetal {
    /// Allocates the buffers used by a complete five-factor sumcheck tail.
    pub fn prepare_product5_sequence(
        &self,
        tables: &[AkitaField],
        elements_per_table: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: Product5SequenceConfig,
    ) -> Result<Product5Sequence, MetalError> {
        if elements_per_table < 2 || !elements_per_table.is_power_of_two() {
            return Err(MetalError::InvalidProduct5TableLength {
                minimum: 2,
                got: elements_per_table,
            });
        }
        let table_elements = PRODUCT5_FACTORS
            .checked_mul(elements_per_table)
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        if tables.len() != table_elements {
            return Err(MetalError::Product5StorageLength {
                expected: table_elements,
                got: tables.len(),
            });
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        if e_in.is_empty() || e_out.is_empty() || covered != elements_per_table / 2 {
            return Err(MetalError::Product5WeightShape {
                expected: elements_per_table / 2,
                covered,
            });
        }

        let sequence =
            self.prepare_empty_product5_sequence(elements_per_table, e_in, e_out, config)?;
        write_akita_fields(&sequence.buffers.tables_a, table_elements, tables)?;
        Ok(sequence)
    }

    pub(crate) fn prepare_product5_sequence_from_fn(
        &self,
        elements_per_table: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: Product5SequenceConfig,
        value: impl Fn(usize) -> AkitaField + Send + Sync,
    ) -> Result<Product5Sequence, MetalError> {
        let sequence =
            self.prepare_empty_product5_sequence(elements_per_table, e_in, e_out, config)?;
        let table_elements = PRODUCT5_FACTORS
            .checked_mul(elements_per_table)
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        write_akita_fields_from_fn(&sequence.buffers.tables_a, table_elements, value);
        Ok(sequence)
    }

    fn prepare_empty_product5_sequence(
        &self,
        elements_per_table: usize,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        config: Product5SequenceConfig,
    ) -> Result<Product5Sequence, MetalError> {
        if elements_per_table < 2 || !elements_per_table.is_power_of_two() {
            return Err(MetalError::InvalidProduct5TableLength {
                minimum: 2,
                got: elements_per_table,
            });
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        if e_in.is_empty() || e_out.is_empty() || covered != elements_per_table / 2 {
            return Err(MetalError::Product5WeightShape {
                expected: elements_per_table / 2,
                covered,
            });
        }

        let sequence = self.prepare_product5_sequence_storage(
            elements_per_table,
            e_in.len(),
            e_out.len(),
            config,
        )?;
        write_akita_fields(&sequence.buffers.e_in, e_in.len(), e_in)?;
        write_akita_fields(&sequence.buffers.e_out, e_out.len(), e_out)?;
        Ok(sequence)
    }

    pub(super) fn prepare_product5_sequence_storage(
        &self,
        elements_per_table: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: Product5SequenceConfig,
    ) -> Result<Product5Sequence, MetalError> {
        if elements_per_table < 2 || !elements_per_table.is_power_of_two() {
            return Err(MetalError::InvalidProduct5TableLength {
                minimum: 2,
                got: elements_per_table,
            });
        }
        let table_elements = PRODUCT5_FACTORS
            .checked_mul(elements_per_table)
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        let covered = e_in_capacity
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(elements_per_table))?;
        if e_in_capacity == 0 || e_out_capacity == 0 || covered != elements_per_table / 2 {
            return Err(MetalError::Product5WeightShape {
                expected: elements_per_table / 2,
                covered,
            });
        }

        let message_pipeline = self.compile_named_pipeline(MESSAGE_PIPELINE)?;
        let transition_pipeline = self.compile_named_pipeline(TRANSITION_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCE_PIPELINE)?;
        let message_limits = Self::limits(&message_pipeline);
        let transition_limits = Self::limits(&transition_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (MESSAGE_PIPELINE, message_limits),
            (TRANSITION_PIPELINE, transition_limits),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != PRODUCT5_SIMD_WIDTH {
                return Err(MetalError::UnsupportedProduct5ExecutionWidth {
                    pipeline,
                    expected: PRODUCT5_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let message_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.message_threads_per_threadgroup,
            message_limits,
        )?;
        let transition_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.transition_threads_per_threadgroup,
            transition_limits,
        )?;

        let tables_a = self.new_product5_buffer(table_elements)?;
        let tables_b = self.new_product5_buffer(table_elements / 2)?;
        let e_in_buffer = self.new_product5_buffer(e_in_capacity)?;
        let e_out_buffer = self.new_product5_buffer(e_out_capacity)?;
        let partial_elements = PRODUCT5_FACTORS
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(e_out_capacity))?;
        let partial_a = self.new_product5_buffer(partial_elements)?;
        let partial_b = self.new_product5_buffer(partial_elements)?;

        Ok(Product5Sequence {
            context: self.clone(),
            message_pipeline,
            transition_pipeline,
            reduction_pipeline,
            reduction_limits,
            buffers: Product5SequenceBuffers {
                tables_a,
                tables_b,
                e_in: e_in_buffer,
                e_out: e_out_buffer,
                partial_a,
                partial_b,
            },
            message_threads_per_threadgroup,
            transition_threads_per_threadgroup,
            initial_elements: elements_per_table,
            current_elements: elements_per_table,
            e_in_capacity,
            e_out_capacity,
            source_in_a: true,
            gpu_active_time: Duration::ZERO,
        })
    }

    fn new_product5_buffer(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = buffer_bytes(elements)?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl Product5Sequence {
    pub(super) fn initial_table_buffer(&self) -> &Buffer {
        &self.buffers.tables_a
    }

    pub(super) fn record_gpu_active_time(&mut self, duration: Duration) {
        self.gpu_active_time += duration;
    }

    /// Restores the initial tables without reallocating device buffers.
    pub fn reset(&mut self, tables: &[AkitaField]) -> Result<(), MetalError> {
        let expected = PRODUCT5_FACTORS * self.initial_elements;
        if tables.len() != expected {
            return Err(MetalError::Product5StorageLength {
                expected,
                got: tables.len(),
            });
        }
        write_akita_fields(&self.buffers.tables_a, expected, tables)?;
        self.current_elements = self.initial_elements;
        self.source_in_a = true;
        self.gpu_active_time = Duration::ZERO;
        Ok(())
    }

    /// Computes the message for the current resident tables.
    pub fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; PRODUCT5_FACTORS], MetalError> {
        self.execute_round(Product5Mode::Message, None, e_in, e_out)
    }

    /// Binds every resident table and computes the following round message.
    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; PRODUCT5_FACTORS], MetalError> {
        self.execute_round(Product5Mode::FusedTransition, Some(challenge), e_in, e_out)
    }

    /// Copies the current resident factor tables into an existing host slice.
    pub fn read_current_tables(&self, output: &mut [AkitaField]) -> Result<(), MetalError> {
        let elements = PRODUCT5_FACTORS * self.current_elements;
        if output.len() != elements {
            return Err(MetalError::Product5StorageLength {
                expected: elements,
                got: output.len(),
            });
        }
        let source = self.source_buffer();
        // SAFETY: both table buffers were allocated for at least `elements`
        // `Fp128` values and all writes finish before this method returns.
        let values = unsafe { slice::from_raw_parts(source.contents().cast::<Fp128>(), elements) };
        self.context
            .validate_inputs("product5 resident tables", values)?;
        for (output, value) in output.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(())
    }

    pub(crate) fn read_current_factor_tables(
        &self,
        output: &mut [Vec<AkitaField>; PRODUCT5_FACTORS],
    ) -> Result<(), MetalError> {
        if output
            .iter()
            .any(|table| table.len() < self.current_elements)
        {
            return Err(MetalError::Product5StorageLength {
                expected: self.current_elements,
                got: output.iter().map(Vec::len).min().unwrap_or(0),
            });
        }
        let source = self.source_buffer();
        let elements = PRODUCT5_FACTORS * self.current_elements;
        // SAFETY: the source buffer was allocated for at least `elements`
        // values and the preceding command completed before returning.
        let values = unsafe { slice::from_raw_parts(source.contents().cast::<Fp128>(), elements) };
        self.context
            .validate_inputs("product5 resident tables", values)?;
        for (factor, table) in output.iter_mut().enumerate() {
            let source =
                &values[factor * self.current_elements..(factor + 1) * self.current_elements];
            for (output, value) in table.iter_mut().zip(source) {
                *output = value.into_jolt_field();
            }
            table.truncate(self.current_elements);
        }
        Ok(())
    }

    pub const fn current_elements(&self) -> usize {
        self.current_elements
    }

    pub const fn resident_buffer_count(&self) -> usize {
        6
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active_time
    }

    fn execute_round(
        &mut self,
        mode: Product5Mode,
        challenge: Option<AkitaField>,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; PRODUCT5_FACTORS], MetalError> {
        let minimum = mode.minimum_elements();
        if self.current_elements < minimum {
            return Err(MetalError::InvalidProduct5TableLength {
                minimum,
                got: self.current_elements,
            });
        }
        let covered = e_in
            .len()
            .checked_mul(e_out.len())
            .ok_or(MetalError::InputTooLong(self.current_elements))?;
        let expected = mode.message_pairs(self.current_elements);
        if e_in.is_empty() || e_out.is_empty() || covered != expected {
            return Err(MetalError::Product5WeightShape { expected, covered });
        }
        write_akita_fields(&self.buffers.e_in, self.e_in_capacity, e_in)?;
        write_akita_fields(&self.buffers.e_out, self.e_out_capacity, e_out)?;

        let source_elements = u32::try_from(self.current_elements)
            .map_err(|_| MetalError::InputTooLong(self.current_elements))?;
        let params = Product5Params {
            source_elements,
            e_in_length: u32::try_from(e_in.len())
                .map_err(|_| MetalError::InputTooLong(e_in.len()))?,
            e_out_length: u32::try_from(e_out.len())
                .map_err(|_| MetalError::InputTooLong(e_out.len()))?,
            _reserved: 0,
        };
        let (pipeline, threads_per_threadgroup) = match mode {
            Product5Mode::Message => (
                self.message_pipeline.clone(),
                self.message_threads_per_threadgroup,
            ),
            Product5Mode::FusedTransition => (
                self.transition_pipeline.clone(),
                self.transition_threads_per_threadgroup,
            ),
        };

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(self.source_buffer()), 0);
            match mode {
                Product5Mode::Message => {
                    encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 4, &params);
                }
                Product5Mode::FusedTransition => {
                    encoder.set_buffer(1, Some(self.destination_buffer()), 0);
                    encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                    let challenge = Fp128::from_jolt_field(&challenge.ok_or(
                        MetalError::InvalidProduct5TableLength {
                            minimum: 4,
                            got: self.current_elements,
                        },
                    )?);
                    set_inline_bytes(encoder, 5, &challenge);
                    set_inline_bytes(encoder, 6, &params);
                }
            }
            let dynamic_memory = PRODUCT5_FACTORS
                * (threads_per_threadgroup / PRODUCT5_SIMD_WIDTH)
                * size_of::<Fp128>();
            encoder.set_threadgroup_memory_length(0, dynamic_memory as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: e_out.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );

            let mut input_count = e_out.len();
            let mut input_a = true;
            while input_count > 1 {
                let output_count =
                    input_count.div_ceil(self.reduction_limits.thread_execution_width);
                let reduction_params = ReductionParams {
                    input_count: input_count as u32,
                    output_count: output_count as u32,
                    _reserved: [0; 2],
                };
                encoder.set_compute_pipeline_state(&self.reduction_pipeline);
                let (input, output) = if input_a {
                    (&self.buffers.partial_a, &self.buffers.partial_b)
                } else {
                    (&self.buffers.partial_b, &self.buffers.partial_a)
                };
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(output), 0);
                set_inline_bytes(encoder, 2, &reduction_params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: output_count as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.reduction_limits.thread_execution_width as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                input_count = output_count;
                input_a = !input_a;
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(command_buffer.status()));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.gpu_active_time += Duration::from_secs_f64(end - start);

            let final_buffer = if input_a {
                &self.buffers.partial_a
            } else {
                &self.buffers.partial_b
            };
            // SAFETY: the main dispatch and reductions wrote five values and
            // the command buffer completed successfully.
            let values = unsafe {
                slice::from_raw_parts(final_buffer.contents().cast::<Fp128>(), PRODUCT5_FACTORS)
            };
            self.context
                .validate_inputs("product5 sequence message", values)?;
            let message = std::array::from_fn(|index| values[index].into_jolt_field());
            if mode == Product5Mode::FusedTransition {
                self.current_elements /= 2;
                self.source_in_a = !self.source_in_a;
            }
            Ok(message)
        })
    }

    fn source_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.tables_a
        } else {
            &self.buffers.tables_b
        }
    }

    fn destination_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.tables_b
        } else {
            &self.buffers.tables_a
        }
    }
}

fn write_akita_fields(
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::Product5StorageLength {
            expected: capacity,
            got: values.len(),
        });
    }
    // SAFETY: the buffer has `capacity * size_of::<Fp128>()` bytes and shared
    // storage remains CPU-writable for the buffer's lifetime.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn write_akita_fields_from_fn(
    buffer: &Buffer,
    elements: usize,
    value: impl Fn(usize) -> AkitaField + Send + Sync,
) {
    // SAFETY: the buffer has `elements * size_of::<Fp128>()` bytes and is not
    // visible to a command buffer while the initial table is populated.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), elements) };
    #[cfg(feature = "parallel")]
    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| *output = Fp128::from_jolt_field(&value(index)));
    #[cfg(not(feature = "parallel"))]
    for (index, output) in output.iter_mut().enumerate() {
        *output = Fp128::from_jolt_field(&value(index));
    }
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}
