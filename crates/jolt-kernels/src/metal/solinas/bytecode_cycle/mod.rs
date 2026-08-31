use std::{mem::size_of, slice};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};

use super::{
    encode_column_reductions, set_inline_bytes, validate_completed_command, Fp128, MetalError,
    PipelineLimits, SolinasMetal, AKITA_OFFSET_FFFFA7F7,
};

pub const BYTECODE_CYCLE_TABLES: usize = 5;
pub const BYTECODE_CYCLE_SAMPLES: usize = 4;

const SIMD_WIDTH: usize = 32;
const MESSAGE_PIPELINE: &str = "solinas_bytecode_cycle_q10_message";
const TRANSITION_PIPELINE: &str = "solinas_bytecode_cycle_q10_transition";
const REDUCE_PIPELINE: &str = "solinas_instruction_ra_reduce";

#[derive(Clone, Copy, Debug)]
pub struct BytecodeCycleTables<'a> {
    pub combined: &'a [AkitaField],
    pub fused_combined: &'a [AkitaField],
    pub fused_inc: &'a [AkitaField],
    pub ra0: &'a [AkitaField],
    pub ra1: &'a [AkitaField],
}

#[derive(Debug)]
pub struct BytecodeCycleTablesMut<'a> {
    pub combined: &'a mut [AkitaField],
    pub fused_combined: &'a mut [AkitaField],
    pub fused_inc: &'a mut [AkitaField],
    pub ra0: &'a mut [AkitaField],
    pub ra1: &'a mut [AkitaField],
}

impl<'a> BytecodeCycleTables<'a> {
    fn planes(self) -> [(&'static str, &'a [AkitaField]); BYTECODE_CYCLE_TABLES] {
        [
            ("combined", self.combined),
            ("fused_combined", self.fused_combined),
            ("fused_inc", self.fused_inc),
            ("ra0", self.ra0),
            ("ra1", self.ra1),
        ]
    }

    fn validate(self, expected: usize) -> Result<(), MetalError> {
        for (plane, values) in self.planes() {
            if values.len() != expected {
                return Err(MetalError::BytecodeCyclePlaneLength {
                    plane,
                    expected,
                    got: values.len(),
                });
            }
        }
        Ok(())
    }
}

impl<'a> BytecodeCycleTablesMut<'a> {
    fn validate(&self, expected: usize) -> Result<(), MetalError> {
        for (plane, got) in [
            ("combined", self.combined.len()),
            ("fused_combined", self.fused_combined.len()),
            ("fused_inc", self.fused_inc.len()),
            ("ra0", self.ra0.len()),
            ("ra1", self.ra1.len()),
        ] {
            if got != expected {
                return Err(MetalError::BytecodeCyclePlaneLength {
                    plane,
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }

    fn into_planes(self) -> [&'a mut [AkitaField]; BYTECODE_CYCLE_TABLES] {
        [
            self.combined,
            self.fused_combined,
            self.fused_inc,
            self.ra0,
            self.ra1,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeCycleSequenceConfig {
    pub message_threads_per_threadgroup: Option<usize>,
    pub transition_threads_per_threadgroup: Option<usize>,
    pub max_threadgroups: usize,
}

impl Default for BytecodeCycleSequenceConfig {
    fn default() -> Self {
        Self {
            message_threads_per_threadgroup: Some(256),
            transition_threads_per_threadgroup: Some(128),
            max_threadgroups: 1 << 13,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Params {
    source_elements: u32,
    message_pairs: u32,
    threadgroups: u32,
    reserved: u32,
}

struct Pipelines {
    message: ComputePipelineState,
    transition: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct Buffers {
    tables_a: Vec<Buffer>,
    tables_b: Vec<Buffer>,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct BytecodeCycleSequence {
    context: SolinasMetal,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    message_threads_per_threadgroup: usize,
    transition_threads_per_threadgroup: usize,
    max_threadgroups: usize,
    initial_elements: usize,
    current_elements: usize,
    source_in_a: bool,
}

impl SolinasMetal {
    pub fn prepare_bytecode_cycle_sequence(
        &self,
        tables: BytecodeCycleTables<'_>,
        config: BytecodeCycleSequenceConfig,
    ) -> Result<BytecodeCycleSequence, MetalError> {
        let elements_per_table = tables.combined.len();
        tables.validate(elements_per_table)?;
        let mut sequence =
            self.prepare_empty_bytecode_cycle_sequence(elements_per_table, config)?;
        sequence.reset(tables)?;
        Ok(sequence)
    }

    fn prepare_empty_bytecode_cycle_sequence(
        &self,
        elements_per_table: usize,
        config: BytecodeCycleSequenceConfig,
    ) -> Result<BytecodeCycleSequence, MetalError> {
        self.prepare_empty_bytecode_cycle_sequence_with_partial_capacity(
            elements_per_table,
            config,
            0,
        )
    }

    pub(super) fn prepare_empty_bytecode_cycle_sequence_with_partial_capacity(
        &self,
        elements_per_table: usize,
        config: BytecodeCycleSequenceConfig,
        minimum_partial_capacity: usize,
    ) -> Result<BytecodeCycleSequence, MetalError> {
        if self.offset != AKITA_OFFSET_FFFFA7F7 {
            return Err(MetalError::UnexpectedSolinasOffset {
                expected: AKITA_OFFSET_FFFFA7F7,
                got: self.offset,
            });
        }
        if elements_per_table < 4 || !elements_per_table.is_power_of_two() {
            return Err(MetalError::InvalidBytecodeCycleTableLength {
                minimum: 4,
                got: elements_per_table,
            });
        }
        if config.max_threadgroups == 0 {
            return Err(MetalError::InvalidBytecodeCycleThreadgroups(0));
        }
        if minimum_partial_capacity > config.max_threadgroups {
            return Err(MetalError::InvalidBytecodeCycleThreadgroups(
                minimum_partial_capacity,
            ));
        }
        let _ = u32::try_from(elements_per_table)
            .map_err(|_| MetalError::InputTooLong(elements_per_table))?;

        let pipelines = Pipelines {
            message: self.compile_named_pipeline(MESSAGE_PIPELINE)?,
            transition: self.compile_named_pipeline(TRANSITION_PIPELINE)?,
            reduce: self.compile_named_pipeline(REDUCE_PIPELINE)?,
        };
        let message_limits = Self::limits(&pipelines.message);
        let transition_limits = Self::limits(&pipelines.transition);
        let reduction_limits = Self::limits(&pipelines.reduce);
        for (pipeline, limits) in [
            (MESSAGE_PIPELINE, message_limits),
            (TRANSITION_PIPELINE, transition_limits),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedBytecodeCycleExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
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

        let message_partial_capacity = config
            .max_threadgroups
            .min((elements_per_table / 2).div_ceil(message_threads_per_threadgroup));
        let transition_partial_capacity = config
            .max_threadgroups
            .min((elements_per_table / 4).div_ceil(transition_threads_per_threadgroup));
        let partial_capacity = message_partial_capacity
            .max(transition_partial_capacity)
            .max(minimum_partial_capacity);
        let partial_elements = BYTECODE_CYCLE_SAMPLES
            .checked_mul(partial_capacity)
            .ok_or(MetalError::InputTooLong(partial_capacity))?;
        Ok(BytecodeCycleSequence {
            context: self.clone(),
            pipelines,
            reduction_limits,
            buffers: Buffers {
                tables_a: self.new_bytecode_cycle_buffers(elements_per_table)?,
                tables_b: self.new_bytecode_cycle_buffers(elements_per_table / 2)?,
                partial_a: self.new_bytecode_cycle_buffer(partial_elements)?,
                partial_b: self.new_bytecode_cycle_buffer(partial_elements)?,
            },
            message_threads_per_threadgroup,
            transition_threads_per_threadgroup,
            max_threadgroups: config.max_threadgroups,
            initial_elements: elements_per_table,
            current_elements: elements_per_table,
            source_in_a: true,
        })
    }

    fn new_bytecode_cycle_buffer(&self, elements: usize) -> Result<Buffer, MetalError> {
        let bytes = byte_length(elements)?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }

    fn new_bytecode_cycle_buffers(&self, elements: usize) -> Result<Vec<Buffer>, MetalError> {
        (0..BYTECODE_CYCLE_TABLES)
            .map(|_| self.new_bytecode_cycle_buffer(elements))
            .collect()
    }
}

impl BytecodeCycleSequence {
    pub fn reset(&mut self, tables: BytecodeCycleTables<'_>) -> Result<(), MetalError> {
        tables.validate(self.initial_elements)?;
        for (buffer, table) in self
            .buffers
            .tables_a
            .iter()
            .zip(tables.planes().map(|(_, values)| values))
        {
            write_fields(buffer, self.initial_elements, table);
        }
        self.current_elements = self.initial_elements;
        self.source_in_a = true;
        Ok(())
    }

    pub fn message(&mut self) -> Result<[AkitaField; BYTECODE_CYCLE_SAMPLES], MetalError> {
        self.execute_round(None)
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; BYTECODE_CYCLE_SAMPLES], MetalError> {
        self.execute_round(Some(challenge))
    }

    /// Restores the initial source after exactly one transition without copying it.
    pub fn rewind_initial_state(&mut self) -> Result<(), MetalError> {
        if self.source_in_a || self.current_elements != self.initial_elements / 2 {
            return Err(MetalError::InvalidBytecodeCycleState(
                "rewind requires exactly one transition from the initial state",
            ));
        }
        self.current_elements = self.initial_elements;
        self.source_in_a = true;
        Ok(())
    }

    pub fn read_current_tables(
        &self,
        output: BytecodeCycleTablesMut<'_>,
    ) -> Result<(), MetalError> {
        output.validate(self.current_elements)?;
        for (buffer, output) in self.source_buffers().iter().zip(output.into_planes()) {
            // SAFETY: each resident factor buffer has at least
            // `current_elements` fields and every command has completed.
            let values = unsafe {
                slice::from_raw_parts(buffer.contents().cast::<Fp128>(), self.current_elements)
            };
            self.context
                .validate_inputs("bytecode cycle resident table", values)?;
            for (output, value) in output.iter_mut().zip(values) {
                *output = value.into_jolt_field();
            }
        }
        Ok(())
    }

    copy_field_getters! { pub, { current_elements: usize }}

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    fn execute_round(
        &mut self,
        challenge: Option<AkitaField>,
    ) -> Result<[AkitaField; BYTECODE_CYCLE_SAMPLES], MetalError> {
        let (pipeline, threads_per_threadgroup, divisor, minimum) = match challenge {
            Some(_) => (
                self.pipelines.transition.clone(),
                self.transition_threads_per_threadgroup,
                4,
                4,
            ),
            None => (
                self.pipelines.message.clone(),
                self.message_threads_per_threadgroup,
                2,
                2,
            ),
        };
        if self.current_elements < minimum {
            return Err(MetalError::InvalidBytecodeCycleTableLength {
                minimum,
                got: self.current_elements,
            });
        }
        let message_pairs = self.current_elements / divisor;
        let threadgroups = self
            .max_threadgroups
            .min(message_pairs.div_ceil(threads_per_threadgroup));
        let params = Params {
            source_elements: self.current_elements as u32,
            message_pairs: message_pairs as u32,
            threadgroups: threadgroups as u32,
            reserved: 0,
        };

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in self.source_buffers().iter().enumerate() {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            if let Some(challenge) = challenge {
                for (index, buffer) in self.destination_buffers().iter().enumerate() {
                    encoder.set_buffer((BYTECODE_CYCLE_TABLES + index) as u64, Some(buffer), 0);
                }
                encoder.set_buffer(10, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 11, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(encoder, 12, &params);
            } else {
                encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 6, &params);
            }
            let dynamic_memory = BYTECODE_CYCLE_SAMPLES
                * (threads_per_threadgroup / SIMD_WIDTH)
                * size_of::<Fp128>();
            encoder.set_threadgroup_memory_length(0, dynamic_memory as u64);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: threadgroups as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_threadgroup as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = self.encode_reductions(encoder, threadgroups)?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;
        validate_completed_command(command_buffer)?;
        let message = self.read_reduced_message(final_in_a)?;
        if challenge.is_some() {
            self.current_elements /= 2;
            self.source_in_a = !self.source_in_a;
        }
        Ok(message)
    }

    pub(super) fn encode_reductions(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input_count: usize,
    ) -> Result<bool, MetalError> {
        encode_column_reductions(
            encoder,
            &self.pipelines.reduce,
            &self.buffers.partial_a,
            &self.buffers.partial_b,
            input_count,
            BYTECODE_CYCLE_SAMPLES,
            self.reduction_limits.thread_execution_width,
        )
    }

    pub(super) fn read_reduced_message(
        &self,
        final_in_a: bool,
    ) -> Result<[AkitaField; BYTECODE_CYCLE_SAMPLES], MetalError> {
        let final_buffer = if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the main dispatch and recursive reductions have completed
        // and leave four canonical fields at the selected buffer's front.
        let values = unsafe {
            slice::from_raw_parts(
                final_buffer.contents().cast::<Fp128>(),
                BYTECODE_CYCLE_SAMPLES,
            )
        };
        self.context
            .validate_inputs("bytecode cycle message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    pub(super) fn initial_table_buffers(&self) -> &[Buffer] {
        &self.buffers.tables_a
    }

    pub(super) fn partial_buffer(&self) -> &Buffer {
        &self.buffers.partial_a
    }

    fn source_buffers(&self) -> &[Buffer] {
        if self.source_in_a {
            &self.buffers.tables_a
        } else {
            &self.buffers.tables_b
        }
    }

    fn destination_buffers(&self) -> &[Buffer] {
        if self.source_in_a {
            &self.buffers.tables_b
        } else {
            &self.buffers.tables_a
        }
    }
}

fn write_fields(buffer: &Buffer, elements: usize, values: &[AkitaField]) {
    // SAFETY: the buffer has exactly the checked capacity and no command uses
    // it while the sequence is reset.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), elements) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
}

fn byte_length(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<Fp128>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<Params>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity test setup")]
mod tests {
    use super::*;
    use jolt_field::{One as _, Ring as _, Zero as _};

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn test_tables(elements: usize) -> Vec<AkitaField> {
        (0..BYTECODE_CYCLE_TABLES)
            .flat_map(|table| {
                (0..elements).map(move |index| {
                    let value = 19 + 97 * table as u64 + 131 * index as u64;
                    if table == 2 && index % 3 == 0 {
                        -field(value)
                    } else {
                        field(value)
                    }
                })
            })
            .collect()
    }

    fn table_views(tables: &[AkitaField], elements: usize) -> BytecodeCycleTables<'_> {
        assert_eq!(tables.len(), BYTECODE_CYCLE_TABLES * elements);
        let mut planes = tables.chunks_exact(elements);
        BytecodeCycleTables {
            combined: planes.next().unwrap(),
            fused_combined: planes.next().unwrap(),
            fused_inc: planes.next().unwrap(),
            ra0: planes.next().unwrap(),
            ra1: planes.next().unwrap(),
        }
    }

    fn table_views_mut(tables: &mut [AkitaField], elements: usize) -> BytecodeCycleTablesMut<'_> {
        assert_eq!(tables.len(), BYTECODE_CYCLE_TABLES * elements);
        let mut planes = tables.chunks_exact_mut(elements);
        BytecodeCycleTablesMut {
            combined: planes.next().unwrap(),
            fused_combined: planes.next().unwrap(),
            fused_inc: planes.next().unwrap(),
            ra0: planes.next().unwrap(),
            ra1: planes.next().unwrap(),
        }
    }

    fn grid_from_anchors(
        at_zero: AkitaField,
        at_one: AkitaField,
        leading: AkitaField,
    ) -> [AkitaField; BYTECODE_CYCLE_SAMPLES] {
        let second_difference = leading + leading;
        let delta_two = at_one - at_zero + second_difference;
        let at_two = at_one + delta_two;
        let delta_three = delta_two + second_difference;
        let at_three = at_two + delta_three;
        [
            at_zero,
            at_two,
            at_three,
            at_three + delta_three + second_difference,
        ]
    }

    fn q10(lo: [AkitaField; 5], hi: [AkitaField; 5]) -> [AkitaField; 4] {
        let ra = grid_from_anchors(
            lo[3] * lo[4],
            hi[3] * hi[4],
            (hi[3] - lo[3]) * (hi[4] - lo[4]),
        );
        let coefficient = grid_from_anchors(
            lo[0] + lo[2] * lo[1],
            hi[0] + hi[2] * hi[1],
            (hi[2] - lo[2]) * (hi[1] - lo[1]),
        );
        std::array::from_fn(|sample| ra[sample] * coefficient[sample])
    }

    fn cpu_message(tables: &[AkitaField], elements: usize) -> [AkitaField; 4] {
        let mut message = [AkitaField::zero(); 4];
        for pair in 0..elements / 2 {
            let lo = std::array::from_fn(|table| tables[table * elements + 2 * pair]);
            let hi = std::array::from_fn(|table| tables[table * elements + 2 * pair + 1]);
            for (acc, value) in message.iter_mut().zip(q10(lo, hi)) {
                *acc += value;
            }
        }
        message
    }

    fn bind_tables(
        tables: &[AkitaField],
        elements: usize,
        challenge: AkitaField,
    ) -> Vec<AkitaField> {
        let bound_elements = elements / 2;
        let mut bound = vec![AkitaField::zero(); BYTECODE_CYCLE_TABLES * bound_elements];
        for table in 0..BYTECODE_CYCLE_TABLES {
            for index in 0..bound_elements {
                let lo = tables[table * elements + 2 * index];
                let hi = tables[table * elements + 2 * index + 1];
                bound[table * bound_elements + index] = lo + challenge * (hi - lo);
            }
        }
        bound
    }

    fn assert_sequence_matches_cpu(
        context: &SolinasMetal,
        elements: usize,
        config: BytecodeCycleSequenceConfig,
    ) {
        let mut expected_tables = test_tables(elements);
        let mut sequence = context
            .prepare_bytecode_cycle_sequence(table_views(&expected_tables, elements), config)
            .unwrap();

        assert_eq!(
            sequence.message().unwrap(),
            cpu_message(&expected_tables, elements)
        );
        let rewind_challenge = field(5);
        let rewind_tables = bind_tables(&expected_tables, elements, rewind_challenge);
        let rewind_message = cpu_message(&rewind_tables, elements / 2);
        assert_eq!(
            sequence.bind_and_message(rewind_challenge).unwrap(),
            rewind_message
        );
        sequence.rewind_initial_state().unwrap();
        assert_eq!(
            sequence.bind_and_message(rewind_challenge).unwrap(),
            rewind_message
        );
        sequence.rewind_initial_state().unwrap();

        let mut current_elements = elements;
        for challenge in [
            AkitaField::zero(),
            AkitaField::one(),
            -AkitaField::one(),
            field(7),
            -field(13),
            field(29),
            field(43),
        ] {
            expected_tables = bind_tables(&expected_tables, current_elements, challenge);
            current_elements /= 2;
            assert_eq!(
                sequence.bind_and_message(challenge).unwrap(),
                cpu_message(&expected_tables, current_elements)
            );
            assert_eq!(sequence.current_elements(), current_elements);
        }

        let mut restored = vec![AkitaField::zero(); expected_tables.len()];
        sequence
            .read_current_tables(table_views_mut(&mut restored, current_elements))
            .unwrap();
        assert_eq!(restored, expected_tables);
        assert_eq!(sequence.round_device_buffer_allocations(), 0);
    }

    #[test]
    fn dense_q10_sequence_matches_cpu() {
        let context = SolinasMetal::for_akita().unwrap();
        assert_sequence_matches_cpu(
            &context,
            1 << 13,
            BytecodeCycleSequenceConfig {
                message_threads_per_threadgroup: Some(256),
                transition_threads_per_threadgroup: Some(64),
                max_threadgroups: 1 << 13,
            },
        );
        assert_sequence_matches_cpu(
            &context,
            1 << 13,
            BytecodeCycleSequenceConfig {
                message_threads_per_threadgroup: Some(128),
                transition_threads_per_threadgroup: Some(64),
                max_threadgroups: 17,
            },
        );
        assert_sequence_matches_cpu(
            &context,
            1 << 17,
            BytecodeCycleSequenceConfig {
                message_threads_per_threadgroup: Some(32),
                transition_threads_per_threadgroup: Some(32),
                max_threadgroups: 1_057,
            },
        );

        let wrong_context = SolinasMetal::for_offset_275().unwrap();
        assert!(matches!(
            wrong_context.prepare_bytecode_cycle_sequence(
                table_views(&test_tables(4), 4),
                BytecodeCycleSequenceConfig::default(),
            ),
            Err(MetalError::UnexpectedSolinasOffset {
                expected: AKITA_OFFSET_FFFFA7F7,
                got: 275,
            })
        ));
    }
}
