use std::{mem::size_of, slice};

use jolt_field::AkitaField;
use metal::{objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    encode_column_reductions, set_inline_bytes, validate_completed_command, Fp128, MetalError,
    PipelineLimits, SolinasMetal,
};

pub const PRODUCT5_FACTORS: usize = 5;

const PRODUCT5_SIMD_WIDTH: usize = 32;
const MESSAGE_DEFAULT_SIMDGROUPS: usize = 4;
const TRANSITION_DEFAULT_SIMDGROUPS: usize = 2;
const MESSAGE_PIPELINE: &str = "solinas_product5_message";
const TRANSITION_PIPELINE: &str = "solinas_product5_fused_transition";
const REDUCE_PIPELINE: &str = "solinas_product5_reduce";

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
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Product5Params {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    _reserved: u32,
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

    copy_field_getters! { pub, { current_elements: usize }}

    pub const fn resident_buffer_count(&self) -> usize {
        6
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
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

            let final_in_a = encode_column_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                PRODUCT5_FACTORS,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            validate_completed_command(command_buffer)?;
            let final_buffer = if final_in_a {
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
