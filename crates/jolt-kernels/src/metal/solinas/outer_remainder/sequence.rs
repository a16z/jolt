use std::{mem::size_of, slice, time::Duration};

use jolt_field::AkitaField;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, MTLCommandBufferStatus, MTLSize,
};

use super::super::spartan_outer_uniskip::OuterResidualReleaseReceipt;
use super::super::{
    command_buffer_timestamp, Fp128, InstructionInputRows, MetalError, PipelineLimits,
    SolinasMetal, SpartanOuterUniskipRows,
};
use super::{
    api::{
        OuterRemainderDispatchCounts, OuterRemainderPhase, OuterRemainderSequenceConfig,
        OuterRemainderStorageInitializationStats, OuterRemainderStorageStats,
        OUTER_REMAINDER_A_LOOKUP_FIELDS, OUTER_REMAINDER_COLLAPSED_A_FIELDS,
        OUTER_REMAINDER_OPENINGS, OUTER_REMAINDER_PRODUCT_ENDPOINTS, OUTER_REMAINDER_STREAM_ROWS,
    },
    plan::{
        message_threadgroup_bytes, opening_output_count, opening_threadgroup_memory_lengths,
        to_u32, SIMD_WIDTH,
    },
    registers_claim::{
        next_completion_serial, OuterRegistersClaimCarrier, OuterRegistersClaimCarrierReceipt,
        COMPONENTS as REGISTERS_CLAIM_COMPONENTS,
    },
    storage::{
        write_fields, DenseBuffers, OuterRemainderSequenceStorage, RegistersClaimBuffers, Storage,
    },
};
use crate::metal::solinas::registers_claim_reduction::RegistersClaimLinearComponents;

const CANONICAL_PADDING_ONE_OPENING: usize = 30;

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
    columns: u32,
    e_in_length: u32,
    e_out_length: u32,
    blocks: u32,
    source_elements: u32,
    reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ReduceParams {
    input_count: u32,
    columns: u32,
    reserved: [u32; 2],
}

fn a_endpoint_coefficients(
    lagrange: &[AkitaField; OUTER_REMAINDER_STREAM_ROWS],
) -> [[AkitaField; 16]; 2] {
    let first_base = lagrange[0] + lagrange[5];
    let second_base = lagrange[4] + lagrange[8];
    let operation = lagrange[4] - lagrange[5];
    let first = [
        first_base,
        -lagrange[0] + lagrange[1] + lagrange[2],
        -lagrange[0] + lagrange[3],
        operation,
        operation,
        operation,
        AkitaField::zero(),
        AkitaField::zero(),
        lagrange[6],
        lagrange[7],
        lagrange[8],
        -lagrange[8],
        lagrange[9],
        -lagrange[9],
        AkitaField::zero(),
        AkitaField::zero(),
    ];
    let second = [
        second_base,
        lagrange[0],
        lagrange[0],
        lagrange[1] - lagrange[4],
        lagrange[2] - lagrange[4],
        lagrange[3] - lagrange[4],
        lagrange[6] - lagrange[8],
        lagrange[7] - lagrange[8],
        AkitaField::zero(),
        AkitaField::zero(),
        AkitaField::zero(),
        AkitaField::zero(),
        AkitaField::zero(),
        AkitaField::zero(),
        -lagrange[4],
        lagrange[5],
    ];
    [first, second]
}

fn a_lookup(coefficients: &[AkitaField; 16]) -> [AkitaField; OUTER_REMAINDER_COLLAPSED_A_FIELDS] {
    std::array::from_fn(|index| {
        let group = index / 32;
        let mask = index % 32;
        let mut value = if group == 0 {
            coefficients[0]
        } else {
            AkitaField::zero()
        };
        for bit in 0..5 {
            if mask & (1 << bit) != 0 {
                value += coefficients[1 + 5 * group + bit];
            }
        }
        value
    })
}

fn materialize_a_lookup(
    lagrange: &[AkitaField; OUTER_REMAINDER_STREAM_ROWS],
) -> [AkitaField; OUTER_REMAINDER_A_LOOKUP_FIELDS] {
    let endpoints = a_endpoint_coefficients(lagrange);
    let first = a_lookup(&endpoints[0]);
    let second = a_lookup(&endpoints[1]);
    std::array::from_fn(|index| {
        if index < OUTER_REMAINDER_STREAM_ROWS {
            lagrange[index]
        } else if index < OUTER_REMAINDER_STREAM_ROWS + OUTER_REMAINDER_COLLAPSED_A_FIELDS {
            first[index - OUTER_REMAINDER_STREAM_ROWS]
        } else {
            second[index - OUTER_REMAINDER_STREAM_ROWS - OUTER_REMAINDER_COLLAPSED_A_FIELDS]
        }
    })
}

fn collapsed_a_lookup(
    lagrange: &[AkitaField; OUTER_REMAINDER_STREAM_ROWS],
    challenge: AkitaField,
) -> [AkitaField; OUTER_REMAINDER_COLLAPSED_A_FIELDS] {
    let [first, second] = a_endpoint_coefficients(lagrange);
    let coefficients =
        std::array::from_fn(|index| first[index] + challenge * (second[index] - first[index]));
    a_lookup(&coefficients)
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
    product_uniskip_endpoints: Option<[AkitaField; OUTER_REMAINDER_PRODUCT_ENDPOINTS]>,
    registers_claim_carrier: Option<OuterRegistersClaimCarrier>,
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
            product_uniskip_endpoints: None,
            registers_claim_carrier: None,
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
        let a_lookup = materialize_a_lookup(stream_lagrange);
        write_fields(
            &self.storage.context,
            &self.storage.buffers.a_lookup,
            OUTER_REMAINDER_A_LOOKUP_FIELDS,
            &a_lookup,
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
            encoder.set_buffer(2, Some(&self.storage.buffers.a_lookup), 0);
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
        let collapsed_a = collapsed_a_lookup(stream_lagrange, challenge);
        write_fields(
            &self.storage.context,
            &self.storage.buffers.a_lookup,
            OUTER_REMAINDER_COLLAPSED_A_FIELDS,
            &collapsed_a,
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
            encoder.set_buffer(3, Some(&self.storage.buffers.a_lookup), 0);
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
        let explicit_rows = self.rows()?.explicit_rows();
        let output_count = opening_output_count(self.config.product_uniskip_carrier);
        let padding_weight = canonical_padding_weight(explicit_rows, e_in, e_out);
        if explicit_rows == 0 {
            if self.config.registers_claim_carrier {
                return Err(MetalError::InvalidOuterRemainderConfig(
                    "registers-claim carrier requires at least one explicit row",
                ));
            }
            self.product_uniskip_endpoints = self
                .config
                .product_uniskip_carrier
                .then_some([AkitaField::zero(); OUTER_REMAINDER_PRODUCT_ENDPOINTS]);
            self.phase = OuterRemainderPhase::OpeningsComplete;
            self.dispatch_counts.opening_scans += 1;
            let mut openings = [AkitaField::zero(); OUTER_REMAINDER_OPENINGS];
            openings[CANONICAL_PADDING_ONE_OPENING] = padding_weight;
            return Ok(openings);
        }
        self.write_weights(e_in, e_out)?;
        let active_e_out = explicit_rows.div_ceil(e_in.len());
        let blocks = active_e_out.min(self.storage.max_threadgroups);
        let params = OpeningParams {
            columns: to_u32(output_count)?,
            e_in_length: to_u32(e_in.len())?,
            e_out_length: to_u32(active_e_out)?,
            blocks: to_u32(blocks)?,
            source_elements: to_u32(explicit_rows)?,
            reserved: [0; 3],
        };
        let carrier_params = self
            .storage
            .buffers
            .registers_claim
            .as_ref()
            .map(|carrier| {
                if carrier.geometry.prefix_elements != e_in.len()
                    || carrier.geometry.suffix_elements != e_out.len()
                {
                    return Err(MetalError::OuterRemainderWeightShape {
                        phase: "registers-claim carrier",
                        expected: cycles,
                        e_in: e_in.len(),
                        e_out: e_out.len(),
                    });
                }
                Ok(OpeningParams {
                    columns: to_u32(output_count)?,
                    e_in_length: to_u32(e_in.len())?,
                    e_out_length: to_u32(e_out.len())?,
                    blocks: to_u32(carrier.geometry.blocks)?,
                    source_elements: to_u32(explicit_rows)?,
                    reserved: [0; 3],
                })
            })
            .transpose()?;
        let rows = self.rows()?;
        let threads = self.storage.threads.opening;
        let threadgroup_memory = opening_threadgroup_memory_lengths(
            self.config.binding_plan,
            threads,
            self.config.product_uniskip_carrier,
            self.config.registers_claim_carrier,
        )?;
        let queue = self.storage.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        if self.storage.buffers.registers_claim.is_some()
            && (self.storage.pipelines.registers_claim_build.is_none()
                || self.storage.pipelines.registers_claim_reduce.is_none()
                || self.storage.pipelines.registers_claim_dot.is_none())
        {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "registers-claim buffers with build, reduction, and dot pipelines",
                got: self.phase.name(),
            });
        }
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
                output_count,
            );
            if let (Some(carrier), Some(build), Some(reduce), Some(dot), Some(carrier_params)) = (
                self.storage.buffers.registers_claim.as_ref(),
                self.storage.pipelines.registers_claim_build.as_ref(),
                self.storage.pipelines.registers_claim_reduce.as_ref(),
                self.storage.pipelines.registers_claim_dot.as_ref(),
                carrier_params,
            ) {
                encoder.set_compute_pipeline_state(build);
                encoder.set_buffer(0, Some(rows.instruction_input_buffer()), 0);
                encoder.set_buffer(1, Some(rows.residual_buffer()), 0);
                encoder.set_buffer(2, Some(&self.storage.buffers.e_out), 0);
                encoder.set_buffer(3, Some(&carrier.partials), 0);
                encoder.set_buffer(4, Some(&carrier.rd_write_value), 0);
                set_inline_bytes(encoder, 5, &carrier_params);
                let build_threads = self.storage.threads.registers_claim_build;
                let high_per_block = carrier
                    .geometry
                    .suffix_elements
                    .div_ceil(carrier.geometry.blocks);
                encoder
                    .set_threadgroup_memory_length(0, (high_per_block * size_of::<Fp128>()) as u64);
                let low_groups = e_in.len().div_ceil(build_threads);
                dispatch(encoder, carrier.geometry.blocks * low_groups, build_threads);

                encoder.set_compute_pipeline_state(reduce);
                encoder.set_buffer(0, Some(&carrier.partials), 0);
                encoder.set_buffer(1, Some(&carrier.components), 0);
                set_inline_bytes(encoder, 2, &carrier_params);
                let elements = REGISTERS_CLAIM_COMPONENTS * e_in.len();
                let reduce_threads = self.storage.threads.registers_claim_reduce;
                dispatch(encoder, elements.div_ceil(reduce_threads), reduce_threads);

                encoder.set_compute_pipeline_state(dot);
                encoder.set_buffer(0, Some(&carrier.components), 0);
                encoder.set_buffer(1, Some(&self.storage.buffers.e_in), 0);
                encoder.set_buffer(2, Some(&self.storage.buffers.opening_output), 0);
                set_inline_bytes(encoder, 3, &carrier_params);
                let dot_threads = self.storage.threads.registers_claim_dot;
                encoder.set_threadgroup_memory_length(
                    0,
                    ((dot_threads / SIMD_WIDTH) * size_of::<Fp128>()) as u64,
                );
                dispatch(encoder, REGISTERS_CLAIM_COMPONENTS, dot_threads);
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        self.dispatch_counts.command_buffers += 1;
        self.finish_command(command_buffer)?;
        let output = self.storage.buffers.opening_output.clone();
        // SAFETY: the completed reduction initializes `output_count` fields in
        // the shared output buffer before this CPU-visible read.
        let values =
            unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), output_count) };
        self.storage
            .context
            .validate_inputs("outer openings and carriers", values)?;
        let mut openings = std::array::from_fn(|index| values[index].into_jolt_field());
        openings[CANONICAL_PADDING_ONE_OPENING] += padding_weight;
        self.product_uniskip_endpoints = self.config.product_uniskip_carrier.then(|| {
            std::array::from_fn(|index| values[OUTER_REMAINDER_OPENINGS + index].into_jolt_field())
        });
        if self.config.registers_claim_carrier {
            self.finish_registers_claim_carrier(explicit_rows)?;
            self.dispatch_counts.registers_claim_carriers += 1;
        }
        self.phase = OuterRemainderPhase::OpeningsComplete;
        self.dispatch_counts.opening_scans += 1;
        Ok(openings)
    }

    pub fn take_product_uniskip_endpoints(
        &mut self,
    ) -> Option<[AkitaField; OUTER_REMAINDER_PRODUCT_ENDPOINTS]> {
        self.product_uniskip_endpoints.take()
    }

    pub(crate) fn take_registers_claim_carrier(
        &mut self,
    ) -> Result<Option<OuterRegistersClaimCarrier>, MetalError> {
        self.require_phase(OuterRemainderPhase::OpeningsComplete)?;
        if !self.config.registers_claim_carrier {
            return Ok(None);
        }
        let Some(carrier) = self.registers_claim_carrier.take() else {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "unconsumed registers-claim carrier",
                got: self.phase.name(),
            });
        };
        self.storage.owned_bytes = self
            .storage
            .owned_bytes
            .checked_sub(carrier.receipt().rd_bytes)
            .ok_or(MetalError::InvalidOuterRemainderState {
                expected: "storage accounting that includes the registers rd carrier",
                got: self.phase.name(),
            })?;
        Ok(Some(carrier))
    }

    pub(crate) fn instruction_input_arena_release_receipt(
        &self,
    ) -> Result<OuterResidualReleaseReceipt, MetalError> {
        self.require_phase(OuterRemainderPhase::OpeningsComplete)?;
        Ok(OuterResidualReleaseReceipt {
            key: self.rows()?.residual_arena_key(),
        })
    }

    pub const fn opening_output_count(&self) -> usize {
        opening_output_count(self.config.product_uniskip_carrier)
    }

    pub fn into_instruction_input_rows(mut self) -> Result<InstructionInputRows, MetalError> {
        self.require_phase(OuterRemainderPhase::OpeningsComplete)?;
        if self.config.registers_claim_carrier && self.registers_claim_carrier.is_some() {
            return Err(MetalError::InvalidOuterRemainderState {
                expected: "registers-claim carrier detached before row transfer",
                got: self.phase.name(),
            });
        }
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

    pub(crate) fn context(&self) -> &SolinasMetal {
        &self.storage.context
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

    fn finish_registers_claim_carrier(&mut self, explicit_rows: usize) -> Result<(), MetalError> {
        let RegistersClaimBuffers {
            partials,
            components,
            rd_write_value,
            geometry,
        } = self.storage.buffers.registers_claim.take().ok_or(
            MetalError::InvalidOuterRemainderState {
                expected: "registers-claim buffers after carrier dispatch",
                got: self.phase.name(),
            },
        )?;
        // SAFETY: the completed carrier command fully initializes all 3P fields
        // before this shared-buffer read.
        let values = unsafe {
            slice::from_raw_parts(
                components.contents().cast::<Fp128>(),
                geometry.component_elements,
            )
        };
        self.storage
            .context
            .validate_inputs("outer registers-claim components", values)?;
        let table = |component: usize| {
            values[component * geometry.prefix_elements..(component + 1) * geometry.prefix_elements]
                .iter()
                .copied()
                .map(Fp128::into_jolt_field)
                .collect::<Vec<_>>()
        };
        let linear_components = RegistersClaimLinearComponents {
            rd_write_value: table(0),
            rs1_value: table(1),
            rs2_value: table(2),
        };
        let source = self.rows()?.residual_arena_key();
        let receipt = OuterRegistersClaimCarrierReceipt {
            rows: self.rows()?.len(),
            explicit_rows,
            prefix_elements: geometry.prefix_elements,
            suffix_elements: geometry.suffix_elements,
            blocks: geometry.blocks,
            device_registry_id: source.device_registry_id,
            source_generation: source.generation,
            source_compact_storage_id: source.compact_storage_id,
            source_residual_storage_id: source.storage_id,
            partial_storage_id: partials.as_ptr() as usize,
            component_storage_id: components.as_ptr() as usize,
            rd_storage_id: rd_write_value.as_ptr() as usize,
            partial_bytes: geometry.partial_bytes,
            component_bytes: geometry.component_bytes,
            rd_bytes: geometry.rd_bytes,
            completion_serial: next_completion_serial()?,
            row_scans: 2,
            command_buffers: 1,
            waits: 1,
            uploads: 0,
            prezero_dispatches: 0,
            complete_overwrite: true,
        };
        self.registers_claim_carrier = Some(OuterRegistersClaimCarrier::new(
            receipt,
            linear_components,
            rd_write_value,
        )?);
        let released = geometry
            .partial_bytes
            .checked_add(geometry.component_bytes)
            .ok_or(MetalError::InputTooLong(geometry.component_elements))?;
        self.storage.owned_bytes = self.storage.owned_bytes.checked_sub(released).ok_or(
            MetalError::InvalidOuterRemainderState {
                expected: "storage accounting that includes registers scratch",
                got: self.phase.name(),
            },
        )?;
        drop(partials);
        drop(components);
        Ok(())
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

fn canonical_padding_weight(
    explicit_rows: usize,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> AkitaField {
    let inner_sum = e_in
        .iter()
        .copied()
        .fold(AkitaField::zero(), |sum, weight| sum + weight);
    let outer_sum = e_out
        .iter()
        .copied()
        .fold(AkitaField::zero(), |sum, weight| sum + weight);
    let complete_blocks = explicit_rows / e_in.len();
    let partial_rows = explicit_rows % e_in.len();
    let complete_outer_sum = e_out[..complete_blocks]
        .iter()
        .copied()
        .fold(AkitaField::zero(), |sum, weight| sum + weight);
    let partial_inner_sum = e_in[..partial_rows]
        .iter()
        .copied()
        .fold(AkitaField::zero(), |sum, weight| sum + weight);
    let partial_weight = if partial_rows == 0 {
        AkitaField::zero()
    } else {
        e_out[complete_blocks] * partial_inner_sum
    };
    inner_sum * outer_sum - (inner_sum * complete_outer_sum + partial_weight)
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
const _: () = assert!(size_of::<OpeningParams>() == 32);
const _: () = assert!(size_of::<ReduceParams>() == 16);
