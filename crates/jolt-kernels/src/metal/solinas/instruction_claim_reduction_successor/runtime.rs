use std::{
    ffi::c_void,
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::AkitaField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
use metal::{objc::rc::autoreleasepool, Buffer, CommandBuffer, MTLCommandBufferStatus, MTLSize};

use super::super::{
    command_buffer_timestamp,
    instruction_claim_reduction::{
        InstructionClaimPhaseParams, InstructionClaimSequence, INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
    },
    product_remainder::{
        ProductRemainderPhaseParams, ProductRemainderSequence, PRODUCT_REMAINDER_MESSAGE_COLUMNS,
    },
    Fp128, MetalError, SolinasMetal,
};

const PIPELINE: &str = "solinas_product_instruction_materialize_stage1_message";
const MESSAGE_COLUMNS: usize =
    PRODUCT_REMAINDER_MESSAGE_COLUMNS + INSTRUCTION_CLAIM_MESSAGE_COLUMNS;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProductInstructionRoundStats {
    pub(crate) wall: Duration,
    pub(crate) gpu_active: Duration,
    pub(crate) joint: bool,
}

struct CachedInstructionRound {
    round: usize,
    challenge: AkitaField,
    e_in: Vec<AkitaField>,
    e_out: Vec<AkitaField>,
    message: [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
}

pub(crate) struct ProductInstructionRoundService {
    context: SolinasMetal,
    product: ProductRemainderSequence,
    instruction: InstructionClaimSequence,
    instruction_weights: GruenSplitEqPolynomial<AkitaField>,
    cached_instruction: Option<CachedInstructionRound>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ProductInstructionRoundService {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("product"), &self.product);
        visitor.visit_field(allocative::Key::new("instruction"), &self.instruction);
        visitor.exit();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProductInstructionInitialMessageStats {
    pub(crate) wall: Duration,
    pub(crate) submit_wall: Duration,
    pub(crate) overlap_wall: Duration,
    pub(crate) join_wall: Duration,
    pub(crate) gpu_active: Duration,
    pub(crate) completed_before_join: bool,
    pub(crate) threads_per_threadgroup: usize,
    pub(crate) threadgroup_bytes: usize,
}

struct ProductInstructionInitialMessageCommand {
    command_buffer: CommandBuffer,
    product_output: Buffer,
    instruction_output: Buffer,
    submitted_at: Instant,
    submit_wall: Duration,
    threads_per_threadgroup: usize,
    threadgroup_bytes: usize,
}

#[must_use = "a submitted joint Product/Instruction message must be joined"]
pub(crate) struct PendingProductInstructionInitialMessage {
    product: Option<ProductRemainderSequence>,
    instruction: Option<InstructionClaimSequence>,
    command: Option<ProductInstructionInitialMessageCommand>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingProductInstructionInitialMessage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(product) = &self.product {
            visitor.visit_field(allocative::Key::new("product"), product);
        }
        if let Some(instruction) = &self.instruction {
            visitor.visit_field(allocative::Key::new("instruction"), instruction);
        }
        visitor.exit();
    }
}

impl Drop for PendingProductInstructionInitialMessage {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
        }
    }
}

impl PendingProductInstructionInitialMessage {
    pub(crate) fn join(mut self) -> Result<ProductInstructionInitialResult, MetalError> {
        let mut product = self
            .product
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "joint first message lost its product sequence",
            ))?;
        let mut instruction =
            self.instruction
                .take()
                .ok_or(MetalError::InvalidInstructionClaimState(
                    "joint first message lost its instruction sequence",
                ))?;
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "joint first message lost its command buffer",
            ))?;
        let completed_before_join =
            command.command_buffer.status() == MTLCommandBufferStatus::Completed;
        let join_started = Instant::now();
        let overlap_wall = join_started
            .saturating_duration_since(command.submitted_at)
            .saturating_sub(command.submit_wall);
        command.command_buffer.wait_until_completed();
        let status = command.command_buffer.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(status));
        }
        let start = command_buffer_timestamp(&command.command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(&command.command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        let product_values = unsafe {
            // SAFETY: command completion makes the two reduced product fields visible.
            slice::from_raw_parts(
                command.product_output.contents().cast::<Fp128>(),
                PRODUCT_REMAINDER_MESSAGE_COLUMNS,
            )
        };
        let instruction_values = unsafe {
            // SAFETY: command completion makes the two reduced instruction fields visible.
            slice::from_raw_parts(
                command.instruction_output.contents().cast::<Fp128>(),
                INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            )
        };
        product
            .context()
            .validate_inputs("joint product first message", product_values)?;
        product
            .context()
            .validate_inputs("joint instruction first message", instruction_values)?;
        let product_message = std::array::from_fn(|index| product_values[index].into_jolt_field());
        let instruction_message =
            std::array::from_fn(|index| instruction_values[index].into_jolt_field());
        let stats = ProductInstructionInitialMessageStats {
            wall: command.submitted_at.elapsed(),
            submit_wall: command.submit_wall,
            overlap_wall,
            join_wall: join_started.elapsed(),
            gpu_active: Duration::from_secs_f64(end - start),
            completed_before_join,
            threads_per_threadgroup: command.threads_per_threadgroup,
            threadgroup_bytes: command.threadgroup_bytes,
        };
        product.complete_joint_materialize()?;
        instruction.complete_joint_materialize(stats.wall, stats.gpu_active)?;
        Ok((
            product,
            product_message,
            instruction,
            instruction_message,
            stats,
        ))
    }
}

impl ProductInstructionRoundService {
    pub(crate) fn new(
        product: ProductRemainderSequence,
        instruction: InstructionClaimSequence,
        tau_low: &[AkitaField],
    ) -> Result<Self, MetalError> {
        if product.device_registry_id() != instruction.joint_device_registry_id()
            || product.storage_layout().rows() != instruction.joint_rows()
            || product.current_elements() != instruction.current_elements()
            || product.storage_layout().rows().ilog2() as usize != tau_low.len()
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint round service received mismatched resident sequences or equality point",
            ));
        }
        Ok(Self {
            context: product.context().clone(),
            product,
            instruction,
            instruction_weights: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            cached_instruction: None,
        })
    }

    pub(crate) const fn product_current_elements(&self) -> usize {
        self.product.current_elements()
    }

    pub(crate) const fn instruction_current_elements(&self) -> usize {
        self.instruction.current_elements()
    }

    pub(crate) const fn instruction_gamma(&self) -> AkitaField {
        self.instruction.joint_gamma()
    }

    pub(crate) fn instruction_allocation_identities(&self) -> Option<[usize; 2]> {
        self.instruction.joint_stage1_allocation_identities()
    }

    pub(crate) const fn instruction_workspace_bytes(&self) -> usize {
        self.instruction.storage_layout().workspace_bytes()
    }

    pub(crate) fn read_product_current_state(
        &self,
    ) -> Result<(Vec<AkitaField>, Vec<AkitaField>), MetalError> {
        self.product.read_current_state()
    }

    #[cfg(test)]
    fn read_instruction_current_state(&self) -> Result<Vec<AkitaField>, MetalError> {
        self.instruction.read_current_state()
    }

    pub(crate) fn product_bind_and_message(
        &mut self,
        round: usize,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
            ProductInstructionRoundStats,
        ),
        MetalError,
    > {
        if self.cached_instruction.is_some()
            || self.product.current_elements() != self.instruction.current_elements()
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint round service was advanced before instruction consumed its message",
            ));
        }
        self.instruction_weights.bind(challenge);
        let instruction_e_in = self.instruction_weights.e_in_current().to_vec();
        let instruction_e_out = self.instruction_weights.e_out_current().to_vec();
        let started = Instant::now();
        let (product_output, instruction_output, command_buffer) = autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer().to_owned();
            let encoder = command_buffer.new_compute_command_encoder();
            let encoded = (|| {
                let product_output = self
                    .product
                    .encode_joint_transition(encoder, challenge, e_in, e_out)?;
                let instruction_output = self.instruction.encode_joint_transition(
                    encoder,
                    challenge,
                    &instruction_e_in,
                    &instruction_e_out,
                )?;
                Ok::<_, MetalError>((product_output, instruction_output))
            })();
            encoder.end_encoding();
            let (product_output, instruction_output) = encoded?;
            command_buffer.commit();
            Ok::<_, MetalError>((product_output, instruction_output, command_buffer))
        })?;
        command_buffer.wait_until_completed();
        let status = command_buffer.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(MetalError::CommandFailed(status));
        }
        let start = command_buffer_timestamp(&command_buffer, "GPUStartTime")?;
        let end = command_buffer_timestamp(&command_buffer, "GPUEndTime")?;
        if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
            return Err(MetalError::InvalidGpuTimestamps { start, end });
        }
        let product_values = unsafe {
            // SAFETY: the completed command reduced the product columns into this shared buffer.
            slice::from_raw_parts(
                product_output.contents().cast::<Fp128>(),
                PRODUCT_REMAINDER_MESSAGE_COLUMNS,
            )
        };
        let instruction_values = unsafe {
            // SAFETY: the completed command reduced the instruction columns into this shared buffer.
            slice::from_raw_parts(
                instruction_output.contents().cast::<Fp128>(),
                INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            )
        };
        self.context
            .validate_inputs("joint product transition message", product_values)?;
        self.context
            .validate_inputs("joint instruction transition message", instruction_values)?;
        let product_message = std::array::from_fn(|index| product_values[index].into_jolt_field());
        let instruction_message =
            std::array::from_fn(|index| instruction_values[index].into_jolt_field());
        let stats = ProductInstructionRoundStats {
            wall: started.elapsed(),
            gpu_active: Duration::from_secs_f64(end - start),
            joint: true,
        };
        self.product.complete_joint_transition()?;
        self.instruction
            .complete_joint_transition(stats.wall, stats.gpu_active)?;
        self.cached_instruction = Some(CachedInstructionRound {
            round,
            challenge,
            e_in: instruction_e_in,
            e_out: instruction_e_out,
            message: instruction_message,
        });
        Ok((product_message, stats))
    }

    pub(crate) fn instruction_bind_and_message(
        &mut self,
        round: usize,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
            ProductInstructionRoundStats,
        ),
        MetalError,
    > {
        if let Some(cached) = self.cached_instruction.take() {
            if cached.round != round
                || cached.challenge != challenge
                || cached.e_in != e_in
                || cached.e_out != e_out
            {
                return Err(MetalError::InvalidInstructionClaimState(
                    "cached joint instruction message has the wrong round, bind, or equality tables",
                ));
            }
            return Ok((
                cached.message,
                ProductInstructionRoundStats {
                    joint: true,
                    ..ProductInstructionRoundStats::default()
                },
            ));
        }
        let (message, timing) = self
            .instruction
            .bind_and_message_timed(challenge, e_in, e_out)?;
        Ok((
            message,
            ProductInstructionRoundStats {
                wall: timing.wall,
                gpu_active: timing.gpu_active,
                joint: false,
            },
        ))
    }

    pub(crate) fn finish_instruction(
        &mut self,
        challenge: AkitaField,
    ) -> Result<AkitaField, MetalError> {
        if self.cached_instruction.is_some() {
            return Err(MetalError::InvalidInstructionClaimState(
                "instruction finish observed an unconsumed joint message",
            ));
        }
        self.instruction.finish(challenge)
    }

    pub(crate) fn product_openings(
        &mut self,
        after_cpu_tail: bool,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; 8], Duration), MetalError> {
        if after_cpu_tail {
            self.product.openings_after_cpu_tail_timed(e_in, e_out)
        } else {
            self.product.openings_timed(e_in, e_out)
        }
    }

    pub(crate) fn instruction_aliased_openings(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; 2],
            super::super::instruction_claim_reduction::InstructionClaimTiming,
        ),
        MetalError,
    > {
        self.instruction.aliased_openings_timed(e_in, e_out)
    }
}

impl SolinasMetal {
    pub(crate) fn submit_product_instruction_initial_message(
        &self,
        product: ProductRemainderSequence,
        instruction: InstructionClaimSequence,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<PendingProductInstructionInitialMessage, MetalError> {
        if product.device_registry_id() != self.device_registry_id()
            || instruction.joint_device_registry_id() != self.device_registry_id()
            || product.storage_layout().rows() != instruction.joint_rows()
            || product.joint_stage1_allocation_identities()
                != instruction.joint_stage1_allocation_identities()
        {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint materialization source shape, device, or allocation differs",
            ));
        }
        let product_threads = product.joint_materialize_threads_per_threadgroup();
        let instruction_threads = instruction.joint_materialize_threads_per_threadgroup();
        if product_threads != instruction_threads {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint materialization threadgroup widths differ",
            ));
        }
        let pipeline = self.compile_named_pipeline(PIPELINE)?;
        let limits = Self::limits(&pipeline);
        let threads = Self::resolve_threadgroup_width(Some(product_threads), limits)?;
        let threadgroup_bytes = MESSAGE_COLUMNS
            .checked_mul(threads / 32)
            .and_then(|fields| fields.checked_mul(size_of::<Fp128>()))
            .ok_or(MetalError::InputTooLong(threads))?;
        let total_threadgroup_bytes = u64::try_from(threadgroup_bytes)
            .ok()
            .and_then(|dynamic| dynamic.checked_add(limits.static_threadgroup_memory_length))
            .ok_or(MetalError::InputTooLong(threadgroup_bytes))?;
        if total_threadgroup_bytes > self.device.max_threadgroup_memory_length() {
            return Err(MetalError::InvalidInstructionClaimState(
                "joint materialization exceeds threadgroup memory",
            ));
        }

        let submitted_at = Instant::now();
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer().to_owned();
            let encoder = command_buffer.new_compute_command_encoder();
            let encoded = (|| {
                encoder.set_compute_pipeline_state(&pipeline);
                let product_params =
                    product.encode_joint_stage1_materialize(encoder, e_in, e_out)?;
                let instruction_params = instruction.encode_joint_stage1_materialize(
                    encoder,
                    e_in.len(),
                    e_out.len(),
                )?;
                validate_matching_params(product_params, instruction_params)?;
                set_inline_bytes(encoder, 10, &product_params);
                encoder.set_threadgroup_memory_length(0, threadgroup_bytes as u64);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: e_out.len() as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                let product_output =
                    product.encode_joint_initial_reductions(encoder, e_out.len())?;
                let instruction_output =
                    instruction.encode_joint_initial_reductions(encoder, e_out.len())?;
                Ok::<_, MetalError>((product_output, instruction_output))
            })();
            encoder.end_encoding();
            let (product_output, instruction_output) = encoded?;
            command_buffer.commit();
            Ok(PendingProductInstructionInitialMessage {
                product: Some(product),
                instruction: Some(instruction),
                command: Some(ProductInstructionInitialMessageCommand {
                    command_buffer,
                    product_output,
                    instruction_output,
                    submitted_at,
                    submit_wall: submitted_at.elapsed(),
                    threads_per_threadgroup: threads,
                    threadgroup_bytes,
                }),
            })
        })
    }
}

fn validate_matching_params(
    product: ProductRemainderPhaseParams,
    instruction: InstructionClaimPhaseParams,
) -> Result<(), MetalError> {
    if product.source_elements != instruction.source_elements
        || product.e_in_length != instruction.e_in_length
        || product.e_out_length != instruction.e_out_length
    {
        return Err(MetalError::InvalidInstructionClaimState(
            "joint materialization equality geometry differs",
        ));
    }
    Ok(())
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<c_void>(),
    );
}

type ProductInstructionInitialResult = (
    ProductRemainderSequence,
    [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
    InstructionClaimSequence,
    [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
    ProductInstructionInitialMessageStats,
);

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "the test uses fixed valid Metal geometry"
)]
mod tests {
    use jolt_field::AkitaField;
    use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};

    use super::ProductInstructionRoundService;

    use super::super::super::{
        instruction_claim_reduction::InstructionClaimKernelConfig, ProductRemainderSequenceConfig,
        SolinasMetal, SpartanOuterUniskipRow,
    };

    #[test]
    fn stage1_joint_materialization_matches_both_standalone_sequences() {
        let Ok(context) = SolinasMetal::for_akita() else {
            return;
        };
        let rows = 1usize << 8;
        let source = (0..rows)
            .map(|index| {
                let mut words = [0u64; 20];
                let signed = index as i128 - 131;
                let magnitude = signed.unsigned_abs();
                let right_lookup =
                    (u128::from(index as u64) << 69) | u128::from(19 * index as u64 + 3);
                words[0] = 17 * index as u64 + 1;
                words[1] = magnitude as u64;
                words[2] = (magnitude >> 64) as u64;
                words[13] = 23 * index as u64 + 5;
                words[14] = right_lookup as u64;
                words[15] = (right_lookup >> 64) as u64;
                words[18] = 29 * index as u64 + 7;
                words[19] = u64::from(signed >= 0) << 17;
                SpartanOuterUniskipRow::from_words(words)
            })
            .collect::<Vec<_>>();
        let resident = context
            .prepare_spartan_outer_uniskip_rows(&source)
            .expect("Stage-1 rows should prepare")
            .share_product_remainder_rows()
            .expect("Stage-1 rows should expose the shared product view");
        let lagrange = [
            AkitaField::from_u64(5),
            AkitaField::from_u64(7),
            AkitaField::from_u64(11),
        ];
        let gamma = AkitaField::from_u64(13);
        let point = (0..rows.ilog2())
            .map(|index| AkitaField::from_u64(101 + 2 * u64::from(index)))
            .collect::<Vec<_>>();
        let mut instruction_gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let e_in = instruction_gruen.e_in_current();
        let e_out = instruction_gruen.e_out_current();

        let product = context
            .prepare_product_remainder_sequence_with_rows(
                resident.clone(),
                lagrange,
                e_in.len(),
                2 * e_out.len(),
                ProductRemainderSequenceConfig::default(),
            )
            .expect("joint product sequence should prepare");
        let instruction = context
            .prepare_instruction_claim_sequence_with_stage1_rows(
                resident.clone(),
                gamma,
                InstructionClaimKernelConfig::default(),
            )
            .expect("joint instruction sequence should prepare");
        let mut product_control = context
            .prepare_product_remainder_sequence_with_rows(
                resident.clone(),
                lagrange,
                e_in.len(),
                2 * e_out.len(),
                ProductRemainderSequenceConfig::default(),
            )
            .expect("control product sequence should prepare");
        let mut instruction_control = context
            .prepare_instruction_claim_sequence_with_stage1_rows(
                resident,
                gamma,
                InstructionClaimKernelConfig::default(),
            )
            .expect("control instruction sequence should prepare");

        let expected_product = product_control
            .restart_message_timed(e_in, e_out)
            .expect("control product materialization should execute")
            .0;
        let expected_instruction = instruction_control
            .message(e_in, e_out)
            .expect("control instruction materialization should execute");
        let pending = context
            .submit_product_instruction_initial_message(product, instruction, e_in, e_out)
            .expect("joint materialization should submit");
        let (product, product_message, instruction, instruction_message, stats) =
            pending.join().expect("joint materialization should join");

        assert_eq!(product_message, expected_product);
        assert_eq!(instruction_message, expected_instruction);
        assert_eq!(
            product.read_current_state().unwrap(),
            product_control.read_current_state().unwrap()
        );
        assert_eq!(
            instruction.read_current_state().unwrap(),
            instruction_control.read_current_state().unwrap()
        );
        assert_eq!(stats.threads_per_threadgroup, 128);
        assert_eq!(stats.threadgroup_bytes, 256);
        assert!(stats.gpu_active > std::time::Duration::ZERO);

        let mut service = ProductInstructionRoundService::new(product, instruction, &point)
            .expect("joint round service should adopt both sequences");
        for round in 1..point.len() {
            let challenge = AkitaField::from_u64(311 + 2 * round as u64);
            instruction_gruen.bind(challenge);
            let instruction_e_in = instruction_gruen.e_in_current();
            let instruction_e_out = instruction_gruen.e_out_current();
            let product_head = &point[..point.len() - round - 1];
            let product_split = product_head.len().div_ceil(2);
            let (product_out_point, product_in_point) = product_head.split_at(product_split);
            let product_e_in = EqPolynomial::evals(product_in_point, None);
            let product_e_out = EqPolynomial::evals(product_out_point, None);
            let expected_product_next = product_control
                .bind_and_message(challenge, &product_e_in, &product_e_out)
                .expect("control product transition should execute");
            let expected_instruction_next = instruction_control
                .bind_and_message(challenge, instruction_e_in, instruction_e_out)
                .expect("control instruction transition should execute");
            let (product_next, product_stats) = service
                .product_bind_and_message(round, challenge, &product_e_in, &product_e_out)
                .expect("joint product transition should execute");
            let (instruction_next, instruction_stats) = service
                .instruction_bind_and_message(round, challenge, instruction_e_in, instruction_e_out)
                .expect("cached instruction transition should be consumed");
            assert_eq!(product_next, expected_product_next);
            assert_eq!(instruction_next, expected_instruction_next);
            assert_eq!(
                service.read_product_current_state().unwrap(),
                product_control.read_current_state().unwrap()
            );
            assert_eq!(
                service.read_instruction_current_state().unwrap(),
                instruction_control.read_current_state().unwrap()
            );
            assert!(product_stats.joint);
            assert!(product_stats.gpu_active > std::time::Duration::ZERO);
            assert!(instruction_stats.joint);
            assert_eq!(instruction_stats.wall, std::time::Duration::ZERO);
            assert_eq!(instruction_stats.gpu_active, std::time::Duration::ZERO);
        }
    }
}
