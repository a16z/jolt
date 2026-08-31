use std::{
    mem::{self, size_of},
    slice,
    time::{Duration, Instant},
};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer,
    ComputePipelineState, MTLResourceOptions, MTLSize,
};

use super::super::{
    buffer_from_slice, completed_command_gpu_time, encode_column_reductions,
    product_remainder::{in_place_bind_schedule, InPlaceBindRange},
    set_inline_bytes, Fp128, MetalError, ProductRemainderRows, ProductRemainderSourceKind,
    SolinasMetal,
};
use super::{
    finalize_openings, finish_bind, nontrivial_gamma_powers, InstructionClaimGeometry,
    InstructionClaimKernelConfig, InstructionClaimOpeningMode, InstructionClaimOpeningParams,
    InstructionClaimOpenings, InstructionClaimOperandPlanes, InstructionClaimPhaseParams,
    InstructionClaimReductionPlan, InstructionClaimStorageLayout, ALIASED_OPENING_PIPELINE,
    INSTRUCTION_CLAIM_ALIASED_OPENINGS, INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
    INSTRUCTION_CLAIM_SIMD_WIDTH, MATERIALIZE_PIPELINE, REDUCTION_PIPELINE, TRANSITION_PIPELINE,
};

const STAGE1_ROWS_MATERIALIZE_PIPELINE: &str = "solinas_instruction_claim_materialize_stage1_rows";
const STAGE1_LOOKUP_OPENING_PIPELINE: &str =
    "solinas_instruction_claim_open_stage1_lookup_operands";
const IN_PLACE_BIND_PIPELINE: &str = "solinas_instruction_claim_bind_range";
const IN_PLACE_COPY_PIPELINE: &str = "solinas_instruction_claim_copy_prefix";
const IN_PLACE_MESSAGE_PIPELINE: &str = "solinas_instruction_claim_bound_message";

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InstructionClaimBindRangeParams {
    source_offset: u32,
    destination_offset: u32,
    output_start: u32,
    output_count: u32,
}

const _: [(); 16] = [(); size_of::<InstructionClaimBindRangeParams>()];

impl InstructionClaimBindRangeParams {
    fn new(range: InPlaceBindRange) -> Result<Self, MetalError> {
        Ok(Self {
            source_offset: 0,
            destination_offset: 0,
            output_start: u32::try_from(range.start)
                .map_err(|_| MetalError::InputTooLong(range.start))?,
            output_count: u32::try_from(range.count)
                .map_err(|_| MetalError::InputTooLong(range.count))?,
        })
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionClaimTiming {
    pub wall: Duration,
    pub gpu_active: Duration,
}

struct InstructionClaimInitialMessageCommand {
    command_buffer: CommandBuffer,
    output: Buffer,
    submitted_at: Instant,
    sequence_identity: usize,
    generation: u64,
}

#[must_use = "a submitted instruction claim message must be joined"]
pub struct PendingInstructionClaimInitialMessage {
    sequence: Option<InstructionClaimSequence>,
    command: Option<InstructionClaimInitialMessageCommand>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingInstructionClaimInitialMessage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        visitor.exit();
    }
}

impl Drop for PendingInstructionClaimInitialMessage {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

impl PendingInstructionClaimInitialMessage {
    pub fn join(
        mut self,
    ) -> Result<
        (
            InstructionClaimSequence,
            [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
            InstructionClaimTiming,
        ),
        MetalError,
    > {
        let mut sequence = self
            .sequence
            .take()
            .ok_or(MetalError::InvalidInstructionClaimState(
                "the pending first message lost its resident sequence",
            ))?;
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidInstructionClaimState(
                "the pending first message lost its command buffer",
            ))?;
        let (message, timing) = sequence.complete_initial_message(command)?;
        Ok((sequence, message, timing))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InstructionClaimPhase {
    Raw,
    Materialized,
    CpuTail,
    Finished,
    TransitionRetired,
}

struct InstructionClaimPipelines {
    materialize: ComputePipelineState,
    transition: ComputePipelineState,
    in_place_bind: ComputePipelineState,
    in_place_copy: ComputePipelineState,
    in_place_message: ComputePipelineState,
    opening: ComputePipelineState,
    aliased_opening: ComputePipelineState,
    reduction: ComputePipelineState,
}

enum InstructionClaimRows {
    Standalone {
        lookup_output: Buffer,
        left_lookup_operand: Buffer,
        right_lookup_operand: Buffer,
        left_instruction_input: Buffer,
        right_instruction_input: Buffer,
    },
    Stage1(ProductRemainderRows),
}

impl InstructionClaimRows {
    fn materialize_pipeline(&self) -> &'static str {
        match self {
            Self::Standalone { .. } => MATERIALIZE_PIPELINE,
            Self::Stage1(_) => STAGE1_ROWS_MATERIALIZE_PIPELINE,
        }
    }

    fn aliased_opening_pipeline(&self) -> &'static str {
        match self {
            Self::Standalone { .. } => ALIASED_OPENING_PIPELINE,
            Self::Stage1(_) => STAGE1_LOOKUP_OPENING_PIPELINE,
        }
    }

    fn allocation_identities(&self) -> Vec<usize> {
        match self {
            Self::Standalone {
                lookup_output,
                left_lookup_operand,
                right_lookup_operand,
                left_instruction_input,
                right_instruction_input,
            } => vec![
                lookup_output.as_ptr() as usize,
                left_lookup_operand.as_ptr() as usize,
                right_lookup_operand.as_ptr() as usize,
                left_instruction_input.as_ptr() as usize,
                right_instruction_input.as_ptr() as usize,
            ],
            Self::Stage1(product) => product.allocation_identities(),
        }
    }
}

struct InstructionClaimBuffers {
    rows: InstructionClaimRows,
    gamma_powers: Buffer,
    state_a: Buffer,
    state_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct InstructionClaimSequence {
    context: SolinasMetal,
    pipelines: InstructionClaimPipelines,
    buffers: InstructionClaimBuffers,
    geometry: InstructionClaimGeometry,
    layout: InstructionClaimStorageLayout,
    config: InstructionClaimKernelConfig,
    gamma: AkitaField,
    opening_mode: InstructionClaimOpeningMode,
    current_elements: usize,
    rounds_bound: usize,
    generation: u64,
    source_in_a: bool,
    phase: InstructionClaimPhase,
    combined_claim: Option<AkitaField>,
    timing: InstructionClaimTiming,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionClaimSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_workspace"),
            self.layout.workspace_bytes(),
        );
        match &self.buffers.rows {
            InstructionClaimRows::Standalone { .. } => visitor.visit_simple(
                allocative::Key::new("device_operand_rows"),
                self.layout.resident_bytes() - self.layout.workspace_bytes(),
            ),
            InstructionClaimRows::Stage1(_) => {}
        }
        visitor.exit();
    }
}

pub struct InstructionClaimCpuTail {
    geometry: InstructionClaimGeometry,
    state: Vec<AkitaField>,
    scratch: Vec<AkitaField>,
    next_round: usize,
    sequence_identity: usize,
    generation: u64,
    wall: Duration,
}

impl SolinasMetal {
    pub fn prepare_instruction_claim_sequence(
        &self,
        planes: &InstructionClaimOperandPlanes,
        gamma: AkitaField,
        config: InstructionClaimKernelConfig,
    ) -> Result<InstructionClaimSequence, MetalError> {
        let rows = InstructionClaimRows::Standalone {
            lookup_output: buffer_from_slice(&self.device, planes.lookup_output()),
            left_lookup_operand: buffer_from_slice(&self.device, planes.left_lookup_operand()),
            right_lookup_operand: buffer_from_slice(&self.device, planes.right_lookup_operand()),
            left_instruction_input: buffer_from_slice(
                &self.device,
                planes.left_instruction_input(),
            ),
            right_instruction_input: buffer_from_slice(
                &self.device,
                planes.right_instruction_input(),
            ),
        };
        self.prepare_instruction_claim_sequence_from_rows(rows, planes.len(), gamma, config, true)
    }

    pub fn prepare_instruction_claim_sequence_with_stage1_rows(
        &self,
        product: ProductRemainderRows,
        gamma: AkitaField,
        config: InstructionClaimKernelConfig,
    ) -> Result<InstructionClaimSequence, MetalError> {
        if product.device_registry_id() != self.device_registry_id()
            || product.source_kind() != ProductRemainderSourceKind::SpartanStage1
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "resident instruction Stage-1 rows have the wrong source or Metal device",
            ));
        }
        let len = product.len();
        self.prepare_instruction_claim_sequence_from_rows(
            InstructionClaimRows::Stage1(product),
            len,
            gamma,
            config,
            false,
        )
    }

    fn prepare_instruction_claim_sequence_from_rows(
        &self,
        rows: InstructionClaimRows,
        row_count: usize,
        gamma: AkitaField,
        config: InstructionClaimKernelConfig,
        charge_operand_rows: bool,
    ) -> Result<InstructionClaimSequence, MetalError> {
        let config = config.validate()?;
        let geometry = InstructionClaimGeometry::new(row_count)?;
        let opening = geometry.opening();
        let maximum_buffer = usize::try_from(self.device.max_buffer_length()).unwrap_or(usize::MAX);
        let layout = InstructionClaimStorageLayout::new(
            geometry.rows(),
            opening.e_in_length(),
            opening.e_out_length(),
        )?
        .validate_max_buffer_length(maximum_buffer)?;
        let charged_bytes = if charge_operand_rows {
            layout.resident_bytes()
        } else {
            layout.workspace_bytes()
        };
        let resident_bytes =
            u64::try_from(charged_bytes).map_err(|_| MetalError::InputTooLong(charged_bytes))?;
        self.validate_additional_working_set(resident_bytes)?;

        let opening_mode = InstructionClaimOpeningMode::for_gamma(gamma);
        let materialize_pipeline = rows.materialize_pipeline();
        let aliased_opening_pipeline = rows.aliased_opening_pipeline();
        let materialize = self.compile_named_pipeline(materialize_pipeline)?;
        let transition = self.compile_named_pipeline(TRANSITION_PIPELINE)?;
        let in_place_bind = self.compile_named_pipeline(IN_PLACE_BIND_PIPELINE)?;
        let in_place_copy = self.compile_named_pipeline(IN_PLACE_COPY_PIPELINE)?;
        let in_place_message = self.compile_named_pipeline(IN_PLACE_MESSAGE_PIPELINE)?;
        let opening = self.compile_named_pipeline(opening_mode.pipeline())?;
        let aliased_opening = self.compile_named_pipeline(aliased_opening_pipeline)?;
        let reduction = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let materialize_limits = Self::limits(&materialize);
        let transition_limits = Self::limits(&transition);
        let in_place_bind_limits = Self::limits(&in_place_bind);
        let in_place_copy_limits = Self::limits(&in_place_copy);
        let in_place_message_limits = Self::limits(&in_place_message);
        let opening_limits = Self::limits(&opening);
        let aliased_opening_limits = Self::limits(&aliased_opening);
        let reduction_limits = Self::limits(&reduction);
        for (pipeline, limits) in [
            (materialize_pipeline, materialize_limits),
            (TRANSITION_PIPELINE, transition_limits),
            (IN_PLACE_BIND_PIPELINE, in_place_bind_limits),
            (IN_PLACE_COPY_PIPELINE, in_place_copy_limits),
            (IN_PLACE_MESSAGE_PIPELINE, in_place_message_limits),
            (opening_mode.pipeline(), opening_limits),
            (aliased_opening_pipeline, aliased_opening_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != INSTRUCTION_CLAIM_SIMD_WIDTH {
                return Err(MetalError::UnsupportedInstructionClaimExecutionWidth {
                    pipeline,
                    expected: INSTRUCTION_CLAIM_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        for (phase, requested, limits, dynamic) in [
            (
                "materialize",
                config.materialize_threads_per_threadgroup,
                materialize_limits,
                config.materialize_threadgroup_bytes()?,
            ),
            (
                "transition",
                config.transition_threads_per_threadgroup,
                transition_limits,
                config.transition_threadgroup_bytes()?,
            ),
            (
                "opening",
                config.opening_threads_per_threadgroup,
                opening_limits,
                config.opening_threadgroup_bytes(opening_mode.columns())?,
            ),
            (
                "aliased opening",
                config.opening_threads_per_threadgroup,
                aliased_opening_limits,
                config.opening_threadgroup_bytes(INSTRUCTION_CLAIM_ALIASED_OPENINGS)?,
            ),
        ] {
            let resolved = Self::resolve_threadgroup_width(Some(requested), limits)?;
            if resolved != requested {
                return Err(MetalError::InvalidInstructionClaimState(
                    "resolved threadgroup width differs from the checked configuration",
                ));
            }
            let total = u64::try_from(dynamic)
                .ok()
                .and_then(|bytes| bytes.checked_add(limits.static_threadgroup_memory_length))
                .ok_or(MetalError::InputTooLong(dynamic))?;
            if total > self.device.max_threadgroup_memory_length() {
                return Err(MetalError::InstructionClaimThreadgroupMemory {
                    phase,
                    requested: total,
                    maximum: self.device.max_threadgroup_memory_length(),
                });
            }
        }
        let reduction_threads =
            Self::resolve_threadgroup_width(Some(INSTRUCTION_CLAIM_SIMD_WIDTH), reduction_limits)?;
        if reduction_threads != INSTRUCTION_CLAIM_SIMD_WIDTH {
            return Err(MetalError::InvalidInstructionClaimState(
                "the recursive reduction must use one SIMD group",
            ));
        }
        for limits in [
            in_place_bind_limits,
            in_place_copy_limits,
            in_place_message_limits,
        ] {
            let resolved = Self::resolve_threadgroup_width(
                Some(config.transition_threads_per_threadgroup),
                limits,
            )?;
            if resolved != config.transition_threads_per_threadgroup {
                return Err(MetalError::InvalidInstructionClaimState(
                    "in-place transition threadgroup width differs from the dense path",
                ));
            }
        }

        let gamma_powers = nontrivial_gamma_powers(gamma)
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        self.validate_inputs("instruction claim gamma powers", &gamma_powers)?;
        let buffers = InstructionClaimBuffers {
            rows,
            gamma_powers: buffer_from_slice(&self.device, &gamma_powers),
            state_a: self.new_instruction_claim_buffer(layout.state_a_fields())?,
            state_b: self.new_instruction_claim_buffer(layout.state_b_fields())?,
            e_in: self.new_instruction_claim_buffer(layout.e_in_fields())?,
            e_out: self.new_instruction_claim_buffer(layout.e_out_fields())?,
            partial_a: self.new_instruction_claim_buffer(layout.partial_fields())?,
            partial_b: self.new_instruction_claim_buffer(layout.partial_fields())?,
        };

        Ok(InstructionClaimSequence {
            context: self.clone(),
            pipelines: InstructionClaimPipelines {
                materialize,
                transition,
                in_place_bind,
                in_place_copy,
                in_place_message,
                opening,
                aliased_opening,
                reduction,
            },
            buffers,
            geometry,
            layout,
            config,
            gamma,
            opening_mode,
            current_elements: geometry.rows(),
            rounds_bound: 0,
            generation: 0,
            source_in_a: true,
            phase: InstructionClaimPhase::Raw,
            combined_claim: None,
            timing: InstructionClaimTiming::default(),
        })
    }

    fn new_instruction_claim_buffer(&self, fields: usize) -> Result<Buffer, MetalError> {
        let bytes = fields
            .checked_mul(size_of::<Fp128>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(fields))?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl InstructionClaimSequence {
    pub(in crate::metal::solinas) fn encode_joint_stage1_materialize(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<InstructionClaimPhaseParams, MetalError> {
        if self.phase != InstructionClaimPhase::Raw {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint materialization requires a raw instruction sequence",
            ));
        }
        if !matches!(&self.buffers.rows, InstructionClaimRows::Stage1(_)) {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint materialization requires resident Stage-1 rows",
            ));
        }
        let params =
            InstructionClaimPhaseParams::materialize(self.geometry, e_in_length, e_out_length)?;
        encoder.set_buffer(3, Some(&self.buffers.gamma_powers), 0);
        encoder.set_buffer(7, Some(&self.buffers.state_a), 0);
        encoder.set_buffer(9, Some(&self.buffers.partial_a), 0);
        Ok(params)
    }

    pub(in crate::metal::solinas) fn encode_joint_initial_reductions(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        e_out_length: usize,
    ) -> Result<Buffer, MetalError> {
        let final_in_a = encode_reductions(
            encoder,
            &self.pipelines.reduction,
            &self.buffers.partial_a,
            &self.buffers.partial_b,
            e_out_length,
            INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
        )?;
        Ok(if final_in_a {
            self.buffers.partial_a.clone()
        } else {
            self.buffers.partial_b.clone()
        })
    }

    pub(in crate::metal::solinas) fn complete_joint_materialize(
        &mut self,
        wall: Duration,
        gpu_active: Duration,
    ) -> Result<(), MetalError> {
        if self.phase != InstructionClaimPhase::Raw {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint instruction state was materialized more than once",
            ));
        }
        self.phase = InstructionClaimPhase::Materialized;
        self.timing.wall += wall;
        self.timing.gpu_active += gpu_active;
        Ok(())
    }

    pub(in crate::metal::solinas) fn encode_joint_transition(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<Buffer, MetalError> {
        if self.phase != InstructionClaimPhase::Materialized || self.current_elements < 4 {
            return Err(MetalError::InvalidInstructionClaimState(
                "a joint transition needs a materialized instruction state",
            ));
        }
        if !self.source_in_a {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint in-place instruction state left state A",
            ));
        }
        let round = self.rounds_bound + 1;
        let params =
            InstructionClaimPhaseParams::transition(self.geometry, round, e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("joint instruction claim challenge", &[challenge])?;
        let state = &self.buffers.state_a;
        let scratch = &self.buffers.partial_b;
        let bound_elements = self.current_elements / 2;
        let schedule = in_place_bind_schedule(bound_elements, self.layout.partial_fields());
        if schedule.prefix == 0 {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint in-place instruction scratch is empty",
            ));
        }

        encoder.set_compute_pipeline_state(&self.pipelines.in_place_bind);
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(scratch), 0);
        set_inline_bytes(encoder, 2, &challenge);
        let prefix = InPlaceBindRange {
            start: 0,
            count: schedule.prefix,
        };
        let prefix_params = InstructionClaimBindRangeParams::new(prefix)?;
        set_inline_bytes(encoder, 3, &prefix_params);
        dispatch_linear(
            encoder,
            prefix.count,
            self.config.transition_threads_per_threadgroup,
        );
        encoder.memory_barrier_with_resources(&[&**state, &**scratch]);

        encoder.set_compute_pipeline_state(&self.pipelines.in_place_copy);
        encoder.set_buffer(0, Some(scratch), 0);
        encoder.set_buffer(1, Some(state), 0);
        let prefix_count = u32::try_from(schedule.prefix)
            .map_err(|_| MetalError::InputTooLong(schedule.prefix))?;
        set_inline_bytes(encoder, 2, &prefix_count);
        dispatch_linear(
            encoder,
            schedule.prefix,
            self.config.transition_threads_per_threadgroup,
        );
        encoder.memory_barrier_with_resources(&[&**state, &**scratch]);

        encoder.set_compute_pipeline_state(&self.pipelines.in_place_bind);
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(state), 0);
        set_inline_bytes(encoder, 2, &challenge);
        for range in schedule.direct {
            let range_params = InstructionClaimBindRangeParams::new(range)?;
            set_inline_bytes(encoder, 3, &range_params);
            dispatch_linear(
                encoder,
                range.count,
                self.config.transition_threads_per_threadgroup,
            );
            encoder.memory_barrier_with_resources(&[&**state]);
        }

        encoder.set_compute_pipeline_state(&self.pipelines.in_place_message);
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
        set_inline_bytes(encoder, 4, &params);
        encoder
            .set_threadgroup_memory_length(0, self.config.transition_threadgroup_bytes()? as u64);
        dispatch_blocks(
            encoder,
            e_out.len(),
            self.config.transition_threads_per_threadgroup,
        );
        let final_in_a = encode_reductions(
            encoder,
            &self.pipelines.reduction,
            &self.buffers.partial_a,
            &self.buffers.partial_b,
            e_out.len(),
            INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
        )?;
        Ok(if final_in_a {
            self.buffers.partial_a.clone()
        } else {
            self.buffers.partial_b.clone()
        })
    }

    pub(in crate::metal::solinas) fn complete_joint_transition(
        &mut self,
        wall: Duration,
        gpu_active: Duration,
    ) -> Result<(), MetalError> {
        if self.phase != InstructionClaimPhase::Materialized || self.current_elements < 4 {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint instruction transition completed in the wrong phase",
            ));
        }
        self.current_elements /= 2;
        self.rounds_bound += 1;
        self.timing.wall += wall;
        self.timing.gpu_active += gpu_active;
        Ok(())
    }

    pub(in crate::metal::solinas) fn bind_and_message_in_place_timed(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
            InstructionClaimTiming,
        ),
        MetalError,
    > {
        let started = Instant::now();
        let (message, gpu_active) = autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let output = self.encode_joint_transition(encoder, challenge, e_in, e_out);
            encoder.end_encoding();
            let output = output?;
            finish_command::<INSTRUCTION_CLAIM_MESSAGE_COLUMNS>(
                &self.context,
                command_buffer,
                &output,
                "in-place instruction claim transition message",
            )
        })?;
        let timing = InstructionClaimTiming {
            wall: started.elapsed(),
            gpu_active,
        };
        self.complete_joint_transition(timing.wall, timing.gpu_active)?;
        Ok((message, timing))
    }

    pub(in crate::metal::solinas) const fn joint_materialize_threads_per_threadgroup(
        &self,
    ) -> usize {
        self.config.materialize_threads_per_threadgroup
    }

    pub(in crate::metal::solinas) fn joint_stage1_allocation_identities(
        &self,
    ) -> Option<[usize; 2]> {
        let InstructionClaimRows::Stage1(product) = &self.buffers.rows else {
            return None;
        };
        let identities = product.allocation_identities();
        (identities.len() == 2).then(|| [identities[0], identities[1]])
    }

    pub(in crate::metal::solinas) const fn joint_rows(&self) -> usize {
        self.geometry.rows()
    }

    pub(in crate::metal::solinas) fn joint_device_registry_id(&self) -> u64 {
        self.context.device_registry_id()
    }

    #[cfg(test)]
    pub(in crate::metal::solinas) const fn joint_state_b_buffer(&self) -> &Buffer {
        &self.buffers.state_b
    }

    pub(in crate::metal::solinas) fn release_joint_alternate(&mut self) -> Result<u64, MetalError> {
        if self.phase != InstructionClaimPhase::Raw
            || self.current_elements != self.geometry.rows()
            || !self.source_in_a
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint alternate release requires a raw sequence",
            ));
        }
        let expected_bytes = self
            .layout
            .state_b_fields()
            .checked_mul(size_of::<Fp128>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(self.layout.state_b_fields()))?;
        if self.buffers.state_b.length() != expected_bytes {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint instruction alternate has already been released or has the wrong size",
            ));
        }
        let tombstone = self
            .context
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        let alternate = mem::replace(&mut self.buffers.state_b, tombstone);
        let released_bytes = alternate.length();
        drop(alternate);
        Ok(released_bytes)
    }

    copy_field_getters! { pub(crate), { joint_gamma => gamma: AkitaField }}

    pub fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS], MetalError> {
        self.message_timed(e_in, e_out).map(|(message, _)| message)
    }

    pub fn message_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
            InstructionClaimTiming,
        ),
        MetalError,
    > {
        let command = self.submit_initial_message_command(e_in, e_out)?;
        let (message, timing) = self.complete_initial_message(command)?;
        Ok((message, timing))
    }

    pub fn submit_initial_message(
        self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<PendingInstructionClaimInitialMessage, MetalError> {
        let command = self.submit_initial_message_command(e_in, e_out)?;
        Ok(PendingInstructionClaimInitialMessage {
            sequence: Some(self),
            command: Some(command),
        })
    }

    fn submit_initial_message_command(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<InstructionClaimInitialMessageCommand, MetalError> {
        if self.phase != InstructionClaimPhase::Raw {
            return Err(MetalError::InvalidInstructionClaimState(
                "the materialization message was already emitted",
            ));
        }
        let submitted_at = Instant::now();
        let params =
            InstructionClaimPhaseParams::materialize(self.geometry, e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer().to_owned();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.materialize);
            match &self.buffers.rows {
                InstructionClaimRows::Standalone {
                    lookup_output,
                    left_lookup_operand,
                    right_lookup_operand,
                    left_instruction_input,
                    right_instruction_input,
                } => {
                    encoder.set_buffer(0, Some(lookup_output), 0);
                    encoder.set_buffer(1, Some(left_lookup_operand), 0);
                    encoder.set_buffer(2, Some(right_lookup_operand), 0);
                    encoder.set_buffer(3, Some(left_instruction_input), 0);
                    encoder.set_buffer(4, Some(right_instruction_input), 0);
                    encoder.set_buffer(5, Some(&self.buffers.gamma_powers), 0);
                    encoder.set_buffer(6, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(7, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(8, Some(&self.buffers.state_a), 0);
                    encoder.set_buffer(9, Some(&self.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 10, &params);
                }
                InstructionClaimRows::Stage1(product) => {
                    let (compact, residual) = product.stage1_buffers().ok_or(
                        MetalError::InvalidInstructionClaimState(
                            "resident instruction rows lost their Stage-1 allocations",
                        ),
                    )?;
                    encoder.set_buffer(0, Some(compact), 0);
                    encoder.set_buffer(1, Some(residual), 0);
                    encoder.set_buffer(2, Some(&self.buffers.gamma_powers), 0);
                    encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(5, Some(&self.buffers.state_a), 0);
                    encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 7, &params);
                }
            }
            encoder.set_threadgroup_memory_length(
                0,
                self.config.materialize_threadgroup_bytes()? as u64,
            );
            dispatch_blocks(
                encoder,
                e_out.len(),
                self.config.materialize_threads_per_threadgroup,
            );
            let final_in_a = encode_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            )?;
            encoder.end_encoding();
            let output = if final_in_a {
                self.buffers.partial_a.clone()
            } else {
                self.buffers.partial_b.clone()
            };
            command_buffer.commit();
            Ok(InstructionClaimInitialMessageCommand {
                command_buffer,
                output,
                submitted_at,
                sequence_identity: self.buffers.gamma_powers.as_ptr() as usize,
                generation: self.generation,
            })
        })
    }

    fn complete_initial_message(
        &mut self,
        command: InstructionClaimInitialMessageCommand,
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
            InstructionClaimTiming,
        ),
        MetalError,
    > {
        if self.phase != InstructionClaimPhase::Raw
            || command.sequence_identity != self.buffers.gamma_powers.as_ptr() as usize
            || command.generation != self.generation
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "the pending first message belongs to a different sequence generation",
            ));
        }
        command.command_buffer.wait_until_completed();
        let gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        let values = unsafe {
            // SAFETY: the completed reduction wrote two fields at the front
            // of the selected shared output buffer.
            slice::from_raw_parts(
                command.output.contents().cast::<Fp128>(),
                INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            )
        };
        self.context
            .validate_inputs("instruction claim first message", values)?;
        let message = values
            .iter()
            .copied()
            .map(Fp128::into_jolt_field)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| {
                MetalError::InvalidInstructionClaimState(
                    "the first-message output column count changed",
                )
            })?;
        let timing = InstructionClaimTiming {
            wall: command.submitted_at.elapsed(),
            gpu_active,
        };
        self.phase = InstructionClaimPhase::Materialized;
        self.timing.wall += timing.wall;
        self.timing.gpu_active += timing.gpu_active;
        Ok((message, timing))
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS], MetalError> {
        self.bind_and_message_timed(challenge, e_in, e_out)
            .map(|(message, _)| message)
    }

    pub fn bind_and_message_timed(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
            InstructionClaimTiming,
        ),
        MetalError,
    > {
        if self.phase != InstructionClaimPhase::Materialized || self.current_elements < 4 {
            return Err(MetalError::InvalidInstructionClaimState(
                "a transition needs a materialized state of at least four elements",
            ));
        }
        let started = Instant::now();
        let round = self.rounds_bound + 1;
        let params =
            InstructionClaimPhaseParams::transition(self.geometry, round, e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("instruction claim challenge", &[challenge])?;
        let (message, gpu_active) = autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.transition);
            encoder.set_buffer(0, Some(self.source_buffer()), 0);
            encoder.set_buffer(1, Some(self.destination_buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 5, &challenge);
            set_inline_bytes(encoder, 6, &params);
            encoder.set_threadgroup_memory_length(
                0,
                self.config.transition_threadgroup_bytes()? as u64,
            );
            dispatch_blocks(
                encoder,
                e_out.len(),
                self.config.transition_threads_per_threadgroup,
            );
            let final_in_a = encode_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            )?;
            encoder.end_encoding();
            finish_command::<INSTRUCTION_CLAIM_MESSAGE_COLUMNS>(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                "instruction claim transition message",
            )
        })?;
        self.current_elements /= 2;
        self.rounds_bound = round;
        self.source_in_a = !self.source_in_a;
        let timing = InstructionClaimTiming {
            wall: started.elapsed(),
            gpu_active,
        };
        self.timing.wall += timing.wall;
        self.timing.gpu_active += timing.gpu_active;
        Ok((message, timing))
    }

    pub fn finish(&mut self, challenge: AkitaField) -> Result<AkitaField, MetalError> {
        let started = Instant::now();
        if self.phase != InstructionClaimPhase::Materialized
            || self.current_elements != 2
            || self.rounds_bound + 1 != self.geometry.log_t()
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "finish requires every round message and the last challenge",
            ));
        }
        let values = unsafe {
            // SAFETY: the active shared buffer contains the two final values,
            // and the preceding transition command completed before returning.
            slice::from_raw_parts(self.source_buffer().contents().cast::<Fp128>(), 2)
        };
        self.context
            .validate_inputs("instruction claim final state", values)?;
        let combined = finish_bind(
            [values[0].into_jolt_field(), values[1].into_jolt_field()],
            challenge,
        );
        self.combined_claim = Some(combined);
        self.phase = InstructionClaimPhase::Finished;
        self.timing.wall += started.elapsed();
        Ok(combined)
    }

    pub fn handoff_to_cpu(&mut self) -> Result<InstructionClaimCpuTail, MetalError> {
        let started = Instant::now();
        if self.phase != InstructionClaimPhase::Materialized {
            return Err(MetalError::InvalidInstructionClaimState(
                "a CPU handoff requires a materialized device state",
            ));
        }
        let values = unsafe {
            // SAFETY: the active shared buffer contains `current_elements`
            // initialized fields and the previous command completed.
            slice::from_raw_parts(
                self.source_buffer().contents().cast::<Fp128>(),
                self.current_elements,
            )
        };
        self.context
            .validate_inputs("instruction claim CPU handoff", values)?;
        let state = values
            .iter()
            .copied()
            .map(Fp128::into_jolt_field)
            .collect::<Vec<_>>();
        let scratch = Vec::with_capacity(state.len() / 2);
        self.phase = InstructionClaimPhase::CpuTail;
        self.timing.wall += started.elapsed();
        Ok(InstructionClaimCpuTail {
            geometry: self.geometry,
            state,
            scratch,
            next_round: self.rounds_bound + 1,
            sequence_identity: self.buffers.gamma_powers.as_ptr() as usize,
            generation: self.generation,
            wall: Duration::ZERO,
        })
    }

    pub fn finish_cpu_tail(
        &mut self,
        tail: InstructionClaimCpuTail,
        challenge: AkitaField,
    ) -> Result<AkitaField, MetalError> {
        let started = Instant::now();
        if self.phase != InstructionClaimPhase::CpuTail {
            return Err(MetalError::InvalidInstructionClaimState(
                "the sequence has no active CPU tail",
            ));
        }
        if tail.sequence_identity != self.buffers.gamma_powers.as_ptr() as usize
            || tail.geometry != self.geometry
            || tail.generation != self.generation
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "the CPU tail belongs to a different resident sequence or generation",
            ));
        }
        if tail.state.len() != 2 || tail.next_round != self.geometry.log_t() {
            return Err(MetalError::InvalidInstructionClaimState(
                "the CPU tail has not emitted every remaining round message",
            ));
        }
        let combined = finish_bind([tail.state[0], tail.state[1]], challenge);
        self.current_elements = 2;
        self.rounds_bound = self.geometry.log_t() - 1;
        self.combined_claim = Some(combined);
        self.phase = InstructionClaimPhase::Finished;
        self.timing.wall += tail.wall + started.elapsed();
        Ok(combined)
    }

    pub fn openings(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<InstructionClaimOpenings<AkitaField>, MetalError> {
        self.openings_timed(e_in, e_out)
            .map(|(openings, _)| openings)
    }

    pub fn openings_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(InstructionClaimOpenings<AkitaField>, InstructionClaimTiming), MetalError> {
        let combined_claim =
            self.combined_claim
                .ok_or(MetalError::InvalidInstructionClaimState(
                    "openings require the final combined claim",
                ))?;
        let started = Instant::now();
        let columns = self.opening_mode.columns();
        let params = InstructionClaimOpeningParams::new(
            self.geometry,
            e_in.len(),
            e_out.len(),
            self.opening_mode,
        )?;
        self.write_weights(e_in, e_out)?;
        let (values, gpu_active) = self.execute_opening(columns, params)?;
        let core = [values[0], values[1], values[2], values[3]];
        let right_input =
            (self.opening_mode == InstructionClaimOpeningMode::AllColumns).then(|| values[4]);
        let openings = finalize_openings(
            self.opening_mode,
            self.gamma,
            combined_claim,
            core,
            right_input,
        )?;
        let timing = InstructionClaimTiming {
            wall: started.elapsed(),
            gpu_active,
        };
        self.timing.wall += timing.wall;
        self.timing.gpu_active += timing.gpu_active;
        Ok((openings, timing))
    }

    pub fn aliased_openings(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; INSTRUCTION_CLAIM_ALIASED_OPENINGS], MetalError> {
        self.aliased_openings_timed(e_in, e_out)
            .map(|(openings, _)| openings)
    }

    pub fn aliased_openings_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_CLAIM_ALIASED_OPENINGS],
            InstructionClaimTiming,
        ),
        MetalError,
    > {
        if !matches!(
            self.phase,
            InstructionClaimPhase::Finished | InstructionClaimPhase::TransitionRetired
        ) {
            return Err(MetalError::InvalidInstructionClaimState(
                "aliased openings require the final combined claim",
            ));
        }
        let started = Instant::now();
        let params =
            InstructionClaimOpeningParams::aliased(self.geometry, e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        let (values, gpu_active) = autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.aliased_opening);
            match &self.buffers.rows {
                InstructionClaimRows::Standalone {
                    left_lookup_operand,
                    right_lookup_operand,
                    ..
                } => {
                    encoder.set_buffer(0, Some(left_lookup_operand), 0);
                    encoder.set_buffer(1, Some(right_lookup_operand), 0);
                    encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 5, &params);
                }
                InstructionClaimRows::Stage1(product) => {
                    let (compact, residual) = product.stage1_buffers().ok_or(
                        MetalError::InvalidInstructionClaimState(
                            "resident instruction openings lost their Stage-1 allocations",
                        ),
                    )?;
                    encoder.set_buffer(0, Some(compact), 0);
                    encoder.set_buffer(1, Some(residual), 0);
                    encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 5, &params);
                }
            }
            encoder.set_threadgroup_memory_length(
                0,
                self.config
                    .opening_threadgroup_bytes(INSTRUCTION_CLAIM_ALIASED_OPENINGS)?
                    as u64,
            );
            dispatch_blocks(
                encoder,
                e_out.len(),
                self.config.opening_threads_per_threadgroup,
            );
            let final_in_a = encode_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                INSTRUCTION_CLAIM_ALIASED_OPENINGS,
            )?;
            encoder.end_encoding();
            finish_command::<INSTRUCTION_CLAIM_ALIASED_OPENINGS>(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                "instruction claim aliased openings",
            )
        })?;
        let timing = InstructionClaimTiming {
            wall: started.elapsed(),
            gpu_active,
        };
        self.timing.wall += timing.wall;
        self.timing.gpu_active += timing.gpu_active;
        Ok((values, timing))
    }

    fn execute_opening(
        &self,
        columns: usize,
        params: InstructionClaimOpeningParams,
    ) -> Result<(Vec<AkitaField>, Duration), MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.opening);
            let InstructionClaimRows::Standalone {
                lookup_output,
                left_lookup_operand,
                right_lookup_operand,
                left_instruction_input,
                right_instruction_input,
            } = &self.buffers.rows
            else {
                return Err(MetalError::InvalidInstructionClaimState(
                    "resident product rows support the alias opening path only",
                ));
            };
            match self.opening_mode {
                InstructionClaimOpeningMode::CoreAndRecover => {
                    encoder.set_buffer(0, Some(lookup_output), 0);
                    encoder.set_buffer(1, Some(left_lookup_operand), 0);
                    encoder.set_buffer(2, Some(right_lookup_operand), 0);
                    encoder.set_buffer(3, Some(left_instruction_input), 0);
                    encoder.set_buffer(4, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(5, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 7, &params);
                }
                InstructionClaimOpeningMode::AllColumns => {
                    encoder.set_buffer(0, Some(lookup_output), 0);
                    encoder.set_buffer(1, Some(left_lookup_operand), 0);
                    encoder.set_buffer(2, Some(right_lookup_operand), 0);
                    encoder.set_buffer(3, Some(left_instruction_input), 0);
                    encoder.set_buffer(4, Some(right_instruction_input), 0);
                    encoder.set_buffer(5, Some(&self.buffers.e_in), 0);
                    encoder.set_buffer(6, Some(&self.buffers.e_out), 0);
                    encoder.set_buffer(7, Some(&self.buffers.partial_a), 0);
                    set_inline_bytes(encoder, 8, &params);
                }
            }
            encoder.set_threadgroup_memory_length(
                0,
                self.config.opening_threadgroup_bytes(columns)? as u64,
            );
            dispatch_blocks(
                encoder,
                params.e_out_length as usize,
                self.config.opening_threads_per_threadgroup,
            );
            let final_in_a = encode_reductions(
                encoder,
                &self.pipelines.reduction,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                params.e_out_length as usize,
                columns,
            )?;
            encoder.end_encoding();
            finish_command_vec(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                columns,
                "instruction claim openings",
            )
        })
    }

    pub fn reset(&mut self) {
        self.current_elements = self.geometry.rows();
        self.rounds_bound = 0;
        self.generation = self.generation.wrapping_add(1);
        self.source_in_a = true;
        self.phase = InstructionClaimPhase::Raw;
        self.combined_claim = None;
        self.timing = InstructionClaimTiming::default();
    }

    copy_field_getters! { pub, {
        current_elements: usize,
        storage_layout => layout: InstructionClaimStorageLayout,
        timing: InstructionClaimTiming,
    }}

    pub fn resident_buffer_count(&self) -> usize {
        self.buffers.rows.allocation_identities().len() + 7
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn allocation_identities(&self) -> Vec<usize> {
        let mut identities = self.buffers.rows.allocation_identities();
        identities.extend([
            self.buffers.gamma_powers.as_ptr() as usize,
            self.buffers.state_a.as_ptr() as usize,
            self.buffers.state_b.as_ptr() as usize,
            self.buffers.e_in.as_ptr() as usize,
            self.buffers.e_out.as_ptr() as usize,
            self.buffers.partial_a.as_ptr() as usize,
            self.buffers.partial_b.as_ptr() as usize,
        ]);
        identities
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn read_current_state(&self) -> Result<Vec<AkitaField>, MetalError> {
        if matches!(
            self.phase,
            InstructionClaimPhase::Raw | InstructionClaimPhase::TransitionRetired
        ) {
            return Err(MetalError::InvalidInstructionClaimState(
                "the combined state is not materialized",
            ));
        }
        let values = unsafe {
            // SAFETY: the active shared buffer contains `current_elements`
            // initialized fields and all prior commands completed.
            slice::from_raw_parts(
                self.source_buffer().contents().cast::<Fp128>(),
                self.current_elements,
            )
        };
        self.context
            .validate_inputs("instruction claim resident state", values)?;
        Ok(values.iter().copied().map(Fp128::into_jolt_field).collect())
    }

    fn write_weights(&self, e_in: &[AkitaField], e_out: &[AkitaField]) -> Result<(), MetalError> {
        write_fields(&self.buffers.e_in, self.layout.e_in_fields(), e_in, "e_in")?;
        write_fields(
            &self.buffers.e_out,
            self.layout.e_out_fields(),
            e_out,
            "e_out",
        )
    }

    pub(crate) fn retire_transition_state(&mut self) -> Result<usize, MetalError> {
        if self.phase != InstructionClaimPhase::Finished {
            return Err(MetalError::InvalidInstructionClaimState(
                "transition retirement requires the final combined claim",
            ));
        }
        let tombstone = self
            .context
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        let state_a = mem::replace(&mut self.buffers.state_a, tombstone.clone());
        let state_b = mem::replace(&mut self.buffers.state_b, tombstone);
        let retired_bytes = usize::try_from(state_a.length().saturating_add(state_b.length()))
            .map_err(|_| MetalError::InputTooLong(usize::MAX))?;
        drop(state_a);
        drop(state_b);
        self.phase = InstructionClaimPhase::TransitionRetired;
        Ok(retired_bytes)
    }

    fn source_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.state_a
        } else {
            &self.buffers.state_b
        }
    }

    fn destination_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.state_b
        } else {
            &self.buffers.state_a
        }
    }
}

impl InstructionClaimCpuTail {
    pub const fn current_elements(&self) -> usize {
        self.state.len()
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS], MetalError> {
        let started = Instant::now();
        let params = InstructionClaimPhaseParams::transition(
            self.geometry,
            self.next_round,
            e_in.len(),
            e_out.len(),
        )?;
        let source_elements = params.source_elements as usize;
        if self.state.len() != source_elements {
            return Err(MetalError::InvalidInstructionClaimState(
                "the CPU tail source length differs from its round geometry",
            ));
        }
        self.scratch.clear();
        self.scratch.resize(source_elements / 2, AkitaField::zero());
        let mut endpoints = [AkitaField::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS];
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = [AkitaField::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let pair = x_out * e_in.len() + x_in;
                let source = 4 * pair;
                let destination = 2 * pair;
                let low = finish_bind([self.state[source], self.state[source + 1]], challenge);
                let high = finish_bind([self.state[source + 2], self.state[source + 3]], challenge);
                self.scratch[destination] = low;
                self.scratch[destination + 1] = high;
                inner[0] += inner_weight * low;
                inner[1] += inner_weight * (high + high - low);
            }
            for (endpoint, inner) in endpoints.iter_mut().zip(inner) {
                *endpoint += outer_weight * inner;
            }
        }
        std::mem::swap(&mut self.state, &mut self.scratch);
        self.next_round += 1;
        self.wall += started.elapsed();
        Ok(endpoints)
    }
}

fn dispatch_blocks(
    encoder: &metal::ComputeCommandEncoderRef,
    blocks: usize,
    threads_per_threadgroup: usize,
) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: blocks as u64,
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

fn dispatch_linear(
    encoder: &metal::ComputeCommandEncoderRef,
    elements: usize,
    threads_per_threadgroup: usize,
) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: elements.div_ceil(threads_per_threadgroup) as u64,
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

fn encode_reductions(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    partial_a: &Buffer,
    partial_b: &Buffer,
    input_count: usize,
    columns: usize,
) -> Result<bool, MetalError> {
    let _ = InstructionClaimReductionPlan::new(input_count, columns)?;
    encode_column_reductions(
        encoder,
        pipeline,
        partial_a,
        partial_b,
        input_count,
        columns,
        INSTRUCTION_CLAIM_SIMD_WIDTH,
    )
}

fn finish_command<const COLUMNS: usize>(
    context: &SolinasMetal,
    command_buffer: &metal::CommandBufferRef,
    output: &Buffer,
    label: &'static str,
) -> Result<([AkitaField; COLUMNS], Duration), MetalError> {
    let (values, duration) = finish_command_vec(context, command_buffer, output, COLUMNS, label)?;
    let values: [AkitaField; COLUMNS] = values.try_into().map_err(|_| {
        MetalError::InvalidInstructionClaimState("the reduced output column count changed")
    })?;
    Ok((values, duration))
}

fn finish_command_vec(
    context: &SolinasMetal,
    command_buffer: &metal::CommandBufferRef,
    output: &Buffer,
    columns: usize,
    label: &'static str,
) -> Result<(Vec<AkitaField>, Duration), MetalError> {
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let gpu_active = completed_command_gpu_time(command_buffer)?;
    let values = unsafe {
        // SAFETY: the completed reduction leaves `columns` fields at the
        // front of the selected shared output buffer.
        slice::from_raw_parts(output.contents().cast::<Fp128>(), columns)
    };
    context.validate_inputs(label, values)?;
    Ok((
        values.iter().copied().map(Fp128::into_jolt_field).collect(),
        gpu_active,
    ))
}

fn write_fields(
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
    name: &'static str,
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(super::InstructionClaimShapeError::StorageLength {
            name,
            expected: capacity,
            got: values.len(),
        }
        .into());
    }
    let output = unsafe {
        // SAFETY: the shared buffer holds `capacity` fields and no device
        // command remains active when the next equality prefix is written.
        slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity)
    };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}
