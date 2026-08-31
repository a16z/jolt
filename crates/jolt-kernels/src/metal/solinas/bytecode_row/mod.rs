use std::mem::size_of;

use jolt_field::One as _;
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_poly::EqPolynomial;
use metal::{objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};

#[cfg(test)]
use super::PipelineLimits;
use super::{
    buffer_from_slice, set_inline_bytes, validate_completed_command, BooleanityRows,
    BytecodeCycleSequence, BytecodeCycleSequenceConfig, BytecodeCycleTablesMut, Fp128, MetalError,
    SolinasMetal, BYTECODE_CYCLE_SAMPLES, BYTECODE_CYCLE_TABLES,
};

pub(crate) const BYTECODE_ROW_STAGES: usize = 9;
pub(crate) const BYTECODE_ROW_RA_ENTRIES: usize = 256;

const SIMD_WIDTH: usize = 32;
const FIRST_MESSAGE_PIPELINE: &str = "solinas_bytecode_row_first_message";
const BIND_ROOTS_PIPELINE: &str = "solinas_bytecode_row_bind_lo_roots";
const FIRST_BIND_PIPELINE: &str = "solinas_bytecode_row_first_bind_message";

pub(crate) struct BytecodeCycleRowInputs<'a> {
    pub stage_points: &'a [Vec<AkitaField>],
    pub stage_weights: &'a [AkitaField],
    pub entry_weight: AkitaField,
    pub ra0: &'a [AkitaField],
    pub ra1: &'a [AkitaField],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    lo_length: u32,
    hi_length: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RootBindParams {
    source_length: u32,
    output_length: u32,
    reserved: [u32; 2],
}

struct Pipelines {
    first_message: ComputePipelineState,
    bind_roots: ComputePipelineState,
    first_bind: ComputePipelineState,
}

struct RowBuffers {
    rows: BooleanityRows,
    eq_lo: Buffer,
    bound_eq_lo: Buffer,
    weighted_eq_hi: Buffer,
    ra0: Buffer,
    ra1: Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RowPhase {
    BeforeMessage,
    BeforeFirstBind,
    Dense,
}

pub(crate) struct BytecodeCycleRowSequence {
    context: SolinasMetal,
    pipelines: Pipelines,
    #[cfg(test)]
    first_message_limits: PipelineLimits,
    #[cfg(test)]
    first_bind_limits: PipelineLimits,
    row_buffers: Option<RowBuffers>,
    dense: BytecodeCycleSequence,
    params: Params,
    root_bind_params: RootBindParams,
    root_bind_elements: usize,
    entry_weight: AkitaField,
    first_message_threads: usize,
    root_bind_threads: usize,
    first_bind_threads: usize,
    initial_elements: usize,
    phase: RowPhase,
}

impl SolinasMetal {
    pub(crate) fn prepare_bytecode_cycle_row_sequence(
        &self,
        rows: BooleanityRows,
        inputs: BytecodeCycleRowInputs<'_>,
        config: BytecodeCycleSequenceConfig,
    ) -> Result<BytecodeCycleRowSequence, MetalError> {
        self.validate_booleanity_rows(&rows)?;
        let elements = rows.len();
        if elements < 16 || !elements.is_power_of_two() {
            return Err(MetalError::InvalidBytecodeCycleTableLength {
                minimum: 16,
                got: elements,
            });
        }
        let row_count = u32::try_from(elements).map_err(|_| MetalError::InputTooLong(elements))?;
        if inputs.stage_points.len() != BYTECODE_ROW_STAGES
            || inputs.stage_weights.len() != BYTECODE_ROW_STAGES
        {
            return Err(MetalError::BytecodeCycleRowStageCount {
                expected: BYTECODE_ROW_STAGES,
                points: inputs.stage_points.len(),
                weights: inputs.stage_weights.len(),
            });
        }
        for (stage, point) in inputs.stage_points.iter().enumerate() {
            if point.len() != elements.ilog2() as usize {
                return Err(MetalError::BytecodeCycleRowPointLength {
                    stage,
                    expected: elements.ilog2() as usize,
                    got: point.len(),
                });
            }
        }
        for (plane, values) in [("ra0", inputs.ra0), ("ra1", inputs.ra1)] {
            if values.len() != BYTECODE_ROW_RA_ENTRIES {
                return Err(MetalError::BytecodeCyclePlaneLength {
                    plane,
                    expected: BYTECODE_ROW_RA_ENTRIES,
                    got: values.len(),
                });
            }
        }

        let (_lo_bits, hi_bits, lo_length, hi_length) =
            row_split(elements.ilog2() as usize, config.max_threadgroups)?;
        let lo_length_u32 =
            u32::try_from(lo_length).map_err(|_| MetalError::InputTooLong(lo_length))?;
        let hi_length_u32 =
            u32::try_from(hi_length).map_err(|_| MetalError::InputTooLong(hi_length))?;
        let allocation = row_device_allocation(elements, lo_length, hi_length, config)?;
        for bytes in allocation.buffer_bytes {
            self.validate_buffer_length(bytes)?;
        }
        let device = self.device_info();
        let _allocation_span = tracing::info_span!(
            "MetalBytecodeReadRafCycle::allocation_plan",
            device_buffers = 17_u64,
            planned_device_bytes = allocation.total_bytes,
            current_device_bytes = device.current_allocated_size,
            recommended_device_bytes = device.recommended_max_working_set_size,
        )
        .entered();
        self.validate_additional_working_set(allocation.total_bytes)?;

        let pipelines = Pipelines {
            first_message: self.compile_named_pipeline(FIRST_MESSAGE_PIPELINE)?,
            bind_roots: self.compile_named_pipeline(BIND_ROOTS_PIPELINE)?,
            first_bind: self.compile_named_pipeline(FIRST_BIND_PIPELINE)?,
        };
        let first_message_limits = Self::limits(&pipelines.first_message);
        let bind_roots_limits = Self::limits(&pipelines.bind_roots);
        let first_bind_limits = Self::limits(&pipelines.first_bind);
        for (pipeline, limits) in [
            (FIRST_MESSAGE_PIPELINE, first_message_limits),
            (BIND_ROOTS_PIPELINE, bind_roots_limits),
            (FIRST_BIND_PIPELINE, first_bind_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedBytecodeCycleExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let first_message_threads = Self::resolve_threadgroup_width(
            config.message_threads_per_threadgroup,
            first_message_limits,
        )?;
        let first_bind_threads = Self::resolve_threadgroup_width(
            config.transition_threads_per_threadgroup,
            first_bind_limits,
        )?;
        let root_bind_threads = Self::resolve_threadgroup_width(None, bind_roots_limits)?;

        let eq_lo_capacity = allocation.eq_lo_elements;
        let weighted_eq_hi_capacity = allocation.weighted_eq_hi_elements;
        let mut eq_lo = Vec::with_capacity(eq_lo_capacity);
        let mut weighted_eq_hi = Vec::with_capacity(weighted_eq_hi_capacity);
        for (point, weight) in inputs.stage_points.iter().zip(inputs.stage_weights) {
            eq_lo.extend(EqPolynomial::<AkitaField>::evals(&point[hi_bits..], None));
            weighted_eq_hi.extend(EqPolynomial::<AkitaField>::evals(
                &point[..hi_bits],
                Some(*weight),
            ));
        }
        let eq_lo = fields_to_fp128(&eq_lo);
        let weighted_eq_hi = fields_to_fp128(&weighted_eq_hi);
        let ra0 = fields_to_fp128(inputs.ra0);
        let ra1 = fields_to_fp128(inputs.ra1);
        self.validate_inputs("bytecode row low roots", &eq_lo)?;
        self.validate_inputs("bytecode row weighted high roots", &weighted_eq_hi)?;
        self.validate_inputs("bytecode row RA0", &ra0)?;
        self.validate_inputs("bytecode row RA1", &ra1)?;

        let bound_root_elements = allocation.bound_eq_lo_elements;
        let _ = u32::try_from(bound_root_elements)
            .map_err(|_| MetalError::InputTooLong(bound_root_elements))?;
        let bound_root_bytes = byte_length(bound_root_elements)?;
        self.validate_buffer_length(bound_root_bytes)?;
        let dense = self.prepare_empty_bytecode_cycle_sequence_with_partial_capacity(
            elements / 2,
            config,
            hi_length,
        )?;
        Ok(BytecodeCycleRowSequence {
            context: self.clone(),
            pipelines,
            #[cfg(test)]
            first_message_limits,
            #[cfg(test)]
            first_bind_limits,
            row_buffers: Some(RowBuffers {
                rows,
                eq_lo: buffer_from_slice(&self.device, &eq_lo),
                bound_eq_lo: self
                    .device
                    .new_buffer(bound_root_bytes, MTLResourceOptions::StorageModeShared),
                weighted_eq_hi: buffer_from_slice(&self.device, &weighted_eq_hi),
                ra0: buffer_from_slice(&self.device, &ra0),
                ra1: buffer_from_slice(&self.device, &ra1),
            }),
            dense,
            params: Params {
                rows: row_count,
                lo_length: lo_length_u32,
                hi_length: hi_length_u32,
                reserved: 0,
            },
            root_bind_params: RootBindParams {
                source_length: lo_length_u32,
                output_length: lo_length_u32 / 2,
                reserved: [0; 2],
            },
            root_bind_elements: bound_root_elements,
            entry_weight: inputs.entry_weight,
            first_message_threads,
            root_bind_threads,
            first_bind_threads,
            initial_elements: elements,
            phase: RowPhase::BeforeMessage,
        })
    }
}

impl BytecodeCycleRowSequence {
    pub(crate) fn message(&mut self) -> Result<[AkitaField; BYTECODE_CYCLE_SAMPLES], MetalError> {
        if self.phase != RowPhase::BeforeMessage {
            return Err(MetalError::InvalidBytecodeCycleState(
                "row-derived message may run exactly once before the first bind",
            ));
        }
        let message = self.execute_row_round(None)?;
        self.phase = RowPhase::BeforeFirstBind;
        Ok(message)
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; BYTECODE_CYCLE_SAMPLES], MetalError> {
        match self.phase {
            RowPhase::BeforeMessage => Err(MetalError::InvalidBytecodeCycleState(
                "row-derived first bind requires the initial message",
            )),
            RowPhase::BeforeFirstBind => {
                let message = self.execute_row_round(Some(challenge))?;
                self.phase = RowPhase::Dense;
                self.row_buffers = None;
                Ok(message)
            }
            RowPhase::Dense => self.dense.bind_and_message(challenge),
        }
    }

    pub(crate) fn read_current_tables(
        &self,
        output: BytecodeCycleTablesMut<'_>,
    ) -> Result<(), MetalError> {
        if self.phase != RowPhase::Dense {
            return Err(MetalError::InvalidBytecodeCycleState(
                "row-derived factors are not dense before the first bind",
            ));
        }
        self.dense.read_current_tables(output)
    }

    pub(crate) const fn current_elements(&self) -> usize {
        match self.phase {
            RowPhase::BeforeMessage | RowPhase::BeforeFirstBind => self.initial_elements,
            RowPhase::Dense => self.dense.current_elements(),
        }
    }

    pub(crate) const fn is_dense(&self) -> bool {
        matches!(self.phase, RowPhase::Dense)
    }

    #[cfg(test)]
    pub(crate) const fn round_device_buffer_allocations() -> usize {
        0
    }

    #[cfg(test)]
    copy_field_getters! { pub(crate), {
        first_message_pipeline_limits => first_message_limits: PipelineLimits,
        first_bind_pipeline_limits => first_bind_limits: PipelineLimits,
    }}

    fn execute_row_round(
        &mut self,
        challenge: Option<AkitaField>,
    ) -> Result<[AkitaField; BYTECODE_CYCLE_SAMPLES], MetalError> {
        let row_buffers =
            self.row_buffers
                .as_ref()
                .ok_or(MetalError::InvalidBytecodeCycleState(
                    "row-derived buffers were released",
                ))?;
        let command_buffer = self.context.queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            if let Some(challenge) = challenge {
                let root_encoder = command_buffer.new_compute_command_encoder();
                root_encoder.set_compute_pipeline_state(&self.pipelines.bind_roots);
                root_encoder.set_buffer(0, Some(&row_buffers.eq_lo), 0);
                root_encoder.set_buffer(1, Some(&row_buffers.bound_eq_lo), 0);
                set_inline_bytes(root_encoder, 2, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(root_encoder, 3, &self.root_bind_params);
                root_encoder.dispatch_thread_groups(
                    MTLSize {
                        width: self.root_bind_elements.div_ceil(self.root_bind_threads) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.root_bind_threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                root_encoder.end_encoding();
            }

            let encoder = command_buffer.new_compute_command_encoder();
            if let Some(challenge) = challenge {
                encoder.set_compute_pipeline_state(&self.pipelines.first_bind);
                Self::encode_row_inputs(encoder, row_buffers, true);
                for (index, buffer) in self.dense.initial_table_buffers().iter().enumerate() {
                    encoder.set_buffer((5 + index) as u64, Some(buffer), 0);
                }
                encoder.set_buffer(10, Some(self.dense.partial_buffer()), 0);
                set_inline_bytes(encoder, 11, &Fp128::from_jolt_field(&challenge));
                let bound_entry = (AkitaField::one() - challenge) * self.entry_weight;
                set_inline_bytes(encoder, 12, &Fp128::from_jolt_field(&bound_entry));
                set_inline_bytes(encoder, 13, &self.params);
                self.encode_row_dispatch(encoder, self.first_bind_threads);
            } else {
                encoder.set_compute_pipeline_state(&self.pipelines.first_message);
                Self::encode_row_inputs(encoder, row_buffers, false);
                encoder.set_buffer(5, Some(self.dense.partial_buffer()), 0);
                set_inline_bytes(encoder, 6, &Fp128::from_jolt_field(&self.entry_weight));
                set_inline_bytes(encoder, 7, &self.params);
                self.encode_row_dispatch(encoder, self.first_message_threads);
            }
            let final_in_a = self
                .dense
                .encode_reductions(encoder, self.params.hi_length as usize)?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;
        validate_completed_command(command_buffer)?;
        self.dense.read_reduced_message(final_in_a)
    }

    fn encode_row_inputs(
        encoder: &metal::ComputeCommandEncoderRef,
        buffers: &RowBuffers,
        bound: bool,
    ) {
        encoder.set_buffer(0, Some(buffers.rows.buffer()), 0);
        encoder.set_buffer(
            1,
            Some(if bound {
                &buffers.bound_eq_lo
            } else {
                &buffers.eq_lo
            }),
            0,
        );
        encoder.set_buffer(2, Some(&buffers.weighted_eq_hi), 0);
        encoder.set_buffer(3, Some(&buffers.ra0), 0);
        encoder.set_buffer(4, Some(&buffers.ra1), 0);
    }

    fn encode_row_dispatch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        threads_per_threadgroup: usize,
    ) {
        let dynamic_elements =
            BYTECODE_ROW_STAGES + BYTECODE_CYCLE_SAMPLES * (threads_per_threadgroup / SIMD_WIDTH);
        encoder.set_threadgroup_memory_length(0, (dynamic_elements * size_of::<Fp128>()) as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.params.hi_length as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowDeviceAllocation {
    eq_lo_elements: usize,
    bound_eq_lo_elements: usize,
    weighted_eq_hi_elements: usize,
    buffer_bytes: [u64; 7],
    total_bytes: u64,
}

fn row_device_allocation(
    elements: usize,
    lo_length: usize,
    hi_length: usize,
    config: BytecodeCycleSequenceConfig,
) -> Result<RowDeviceAllocation, MetalError> {
    let eq_lo_elements = BYTECODE_ROW_STAGES
        .checked_mul(lo_length)
        .ok_or(MetalError::InputTooLong(lo_length))?;
    let bound_eq_lo_elements = BYTECODE_ROW_STAGES
        .checked_mul(lo_length / 2)
        .ok_or(MetalError::InputTooLong(lo_length))?;
    let weighted_eq_hi_elements = BYTECODE_ROW_STAGES
        .checked_mul(hi_length)
        .ok_or(MetalError::InputTooLong(hi_length))?;
    let dense_a_elements = BYTECODE_CYCLE_TABLES
        .checked_mul(elements / 2)
        .ok_or(MetalError::InputTooLong(elements))?;
    let dense_b_elements = BYTECODE_CYCLE_TABLES
        .checked_mul(elements / 4)
        .ok_or(MetalError::InputTooLong(elements))?;
    let partial_elements = 2usize
        .checked_mul(BYTECODE_CYCLE_SAMPLES)
        .and_then(|value| value.checked_mul(config.max_threadgroups))
        .ok_or(MetalError::InputTooLong(config.max_threadgroups))?;
    let ra_elements = 2usize
        .checked_mul(BYTECODE_ROW_RA_ENTRIES)
        .ok_or(MetalError::InputTooLong(BYTECODE_ROW_RA_ENTRIES))?;
    let buffer_bytes = [
        byte_length(eq_lo_elements)?,
        byte_length(bound_eq_lo_elements)?,
        byte_length(weighted_eq_hi_elements)?,
        byte_length(BYTECODE_ROW_RA_ENTRIES)?,
        byte_length(BYTECODE_ROW_RA_ENTRIES)?,
        byte_length(elements / 2)?,
        byte_length(partial_elements / 2)?,
    ];
    let total_elements = [
        eq_lo_elements,
        bound_eq_lo_elements,
        weighted_eq_hi_elements,
        ra_elements,
        dense_a_elements,
        dense_b_elements,
        partial_elements,
    ]
    .into_iter()
    .try_fold(0usize, |total, value| total.checked_add(value))
    .ok_or(MetalError::InputTooLong(elements))?;
    Ok(RowDeviceAllocation {
        eq_lo_elements,
        bound_eq_lo_elements,
        weighted_eq_hi_elements,
        buffer_bytes,
        total_bytes: byte_length(total_elements)?,
    })
}

fn row_split(
    log_t: usize,
    max_threadgroups: usize,
) -> Result<(usize, usize, usize, usize), MetalError> {
    if max_threadgroups == 0 {
        return Err(MetalError::InvalidBytecodeCycleThreadgroups(0));
    }
    let balanced_hi_bits = log_t - log_t / 2;
    let hi_bits = balanced_hi_bits.min(max_threadgroups.ilog2() as usize);
    let lo_bits = log_t - hi_bits;
    Ok((lo_bits, hi_bits, 1usize << lo_bits, 1usize << hi_bits))
}

fn fields_to_fp128(values: &[AkitaField]) -> Vec<Fp128> {
    values.iter().map(Fp128::from_jolt_field).collect()
}

fn byte_length(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<Fp128>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

const _: () = assert!(size_of::<Params>() == 16);
const _: () = assert!(size_of::<RootBindParams>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal row-derived parity setup")]
mod tests {
    use super::*;
    use crate::metal::solinas::BooleanityRow;
    use jolt_field::{Ring as _, Zero as _};

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
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

    fn q10(
        lo: [AkitaField; BYTECODE_CYCLE_TABLES],
        hi: [AkitaField; BYTECODE_CYCLE_TABLES],
    ) -> [AkitaField; BYTECODE_CYCLE_SAMPLES] {
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

    fn cpu_message(
        tables: &[Vec<AkitaField>; BYTECODE_CYCLE_TABLES],
    ) -> [AkitaField; BYTECODE_CYCLE_SAMPLES] {
        let mut message = [AkitaField::zero(); BYTECODE_CYCLE_SAMPLES];
        for pair in 0..tables[0].len() / 2 {
            let lo = std::array::from_fn(|table| tables[table][2 * pair]);
            let hi = std::array::from_fn(|table| tables[table][2 * pair + 1]);
            for (acc, value) in message.iter_mut().zip(q10(lo, hi)) {
                *acc += value;
            }
        }
        message
    }

    fn bind_tables(
        tables: &[Vec<AkitaField>; BYTECODE_CYCLE_TABLES],
        challenge: AkitaField,
    ) -> [Vec<AkitaField>; BYTECODE_CYCLE_TABLES] {
        std::array::from_fn(|table| {
            tables[table]
                .chunks_exact(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect()
        })
    }

    fn table_views_mut(
        tables: &mut [Vec<AkitaField>; BYTECODE_CYCLE_TABLES],
    ) -> BytecodeCycleTablesMut<'_> {
        let [combined, fused_combined, fused_inc, ra0, ra1] = tables;
        BytecodeCycleTablesMut {
            combined,
            fused_combined,
            fused_inc,
            ra0,
            ra1,
        }
    }

    #[test]
    fn row_split_caps_the_high_domain_by_the_group_budget() {
        assert_eq!(row_split(26, 1 << 13).unwrap(), (13, 13, 1 << 13, 1 << 13));
        assert_eq!(row_split(27, 1 << 13).unwrap(), (14, 13, 1 << 14, 1 << 13));
        assert_eq!(row_split(28, 1 << 13).unwrap(), (15, 13, 1 << 15, 1 << 13));
        assert_eq!(row_split(28, 1 << 14).unwrap(), (14, 14, 1 << 14, 1 << 14));
        assert_eq!(row_split(28, 10_000).unwrap(), (15, 13, 1 << 15, 1 << 13));
        assert!(matches!(
            row_split(26, 0),
            Err(MetalError::InvalidBytecodeCycleThreadgroups(0))
        ));
    }

    #[test]
    fn row_allocation_plan_charges_every_prepared_device_buffer() {
        let elements = 1usize << 26;
        let (_, _, lo_length, hi_length) = row_split(26, 1 << 13).unwrap();
        let config = BytecodeCycleSequenceConfig::default();
        let allocation = row_device_allocation(elements, lo_length, hi_length, config).unwrap();
        let expected_elements = BYTECODE_ROW_STAGES * lo_length
            + BYTECODE_ROW_STAGES * (lo_length / 2)
            + BYTECODE_ROW_STAGES * hi_length
            + 2 * BYTECODE_ROW_RA_ENTRIES
            + BYTECODE_CYCLE_TABLES * (elements / 2 + elements / 4)
            + 2 * BYTECODE_CYCLE_SAMPLES * config.max_threadgroups;
        assert_eq!(
            allocation.total_bytes,
            (expected_elements * size_of::<Fp128>()) as u64
        );
        assert_eq!(
            allocation.buffer_bytes[0],
            (BYTECODE_ROW_STAGES * lo_length * size_of::<Fp128>()) as u64
        );

        let tiny_group_plan = row_device_allocation(elements, elements, 1, config).unwrap();
        assert!(tiny_group_plan.total_bytes > allocation.total_bytes);
    }

    #[test]
    fn row_derived_sequence_matches_dense_cpu() {
        let context = SolinasMetal::for_akita().unwrap();
        let log_t = 14;
        let elements = 1usize << log_t;
        let stage_points = (0..BYTECODE_ROW_STAGES)
            .map(|stage| {
                (0..log_t)
                    .map(|index| field(5 + 41 * stage as u64 + 67 * index as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let stage_weights = (0..BYTECODE_ROW_STAGES)
            .map(|stage| {
                let value = field(101 + 73 * stage as u64);
                if stage % 3 == 1 {
                    -value
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let entry_weight = -field(0x9a37);
        let ra0 = (0..BYTECODE_ROW_RA_ENTRIES)
            .map(|index| field(1009 + 17 * index as u64))
            .collect::<Vec<_>>();
        let ra1 = (0..BYTECODE_ROW_RA_ENTRIES)
            .map(|index| -field(2003 + 29 * index as u64))
            .collect::<Vec<_>>();

        let mut rows = Vec::with_capacity(elements);
        let mut factors: [Vec<AkitaField>; BYTECODE_CYCLE_TABLES] =
            std::array::from_fn(|_| vec![AkitaField::zero(); elements]);
        for (stage, (point, weight)) in stage_points.iter().zip(&stage_weights).enumerate() {
            let scaled = EqPolynomial::<AkitaField>::evals(point, Some(*weight));
            let destination = usize::from(stage >= 5);
            for (acc, value) in factors[destination].iter_mut().zip(scaled) {
                *acc += value;
            }
        }
        factors[0][0] += entry_weight;

        let inc_values = [0, 1, -1, u64::MAX as i128, -(u64::MAX as i128)];
        for index in 0..elements {
            let mapped_pc = match index % 13 {
                0 => None,
                1 => Some(0),
                2 => Some(255),
                3 => Some(256),
                4 => Some(8191),
                _ => Some((37 * index % 8192) as u64),
            };
            let inc = inc_values[index % inc_values.len()];
            rows.push(BooleanityRow::new(index as u128, mapped_pc, None, inc).unwrap());
            factors[2][index] = if inc < 0 {
                -AkitaField::from_u64(inc.unsigned_abs() as u64)
            } else {
                AkitaField::from_u64(inc as u64)
            };
            if let Some(pc) = mapped_pc {
                factors[3][index] = ra0[((pc >> 8) & 0xff) as usize];
                factors[4][index] = ra1[(pc & 0xff) as usize];
            }
        }

        let resident_rows = context.prepare_booleanity_rows(&rows).unwrap();
        let mut sequence = context
            .prepare_bytecode_cycle_row_sequence(
                resident_rows,
                BytecodeCycleRowInputs {
                    stage_points: &stage_points,
                    stage_weights: &stage_weights,
                    entry_weight,
                    ra0: &ra0,
                    ra1: &ra1,
                },
                BytecodeCycleSequenceConfig {
                    message_threads_per_threadgroup: Some(32),
                    transition_threads_per_threadgroup: Some(32),
                    max_threadgroups: 1 << 5,
                },
            )
            .unwrap();
        assert!(!sequence.is_dense());
        assert_eq!(sequence.current_elements(), elements);
        let mut premature = std::array::from_fn(|_| Vec::new());
        assert!(sequence
            .read_current_tables(table_views_mut(&mut premature))
            .is_err());
        assert!(sequence.bind_and_message(field(3)).is_err());
        assert_eq!(sequence.message().unwrap(), cpu_message(&factors));
        assert!(sequence.message().is_err());

        let first_challenge = field(7);
        factors = bind_tables(&factors, first_challenge);
        assert_eq!(
            sequence.bind_and_message(first_challenge).unwrap(),
            cpu_message(&factors)
        );
        assert!(sequence.is_dense());
        assert_eq!(sequence.current_elements(), elements / 2);
        let mut restored =
            std::array::from_fn(|_| vec![AkitaField::zero(); sequence.current_elements()]);
        sequence
            .read_current_tables(table_views_mut(&mut restored))
            .unwrap();
        assert_eq!(restored, factors);

        for challenge in [
            AkitaField::zero(),
            AkitaField::one(),
            -AkitaField::one(),
            field(29),
            -field(43),
        ] {
            factors = bind_tables(&factors, challenge);
            assert_eq!(
                sequence.bind_and_message(challenge).unwrap(),
                cpu_message(&factors)
            );
        }
        assert_eq!(
            BytecodeCycleRowSequence::round_device_buffer_allocations(),
            0
        );
        assert_eq!(
            sequence
                .first_message_pipeline_limits()
                .thread_execution_width,
            32
        );
        assert_eq!(
            sequence.first_bind_pipeline_limits().thread_execution_width,
            32
        );
    }
}
