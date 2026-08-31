use std::{
    mem::size_of,
    slice,
    time::{Duration, Instant},
};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::{Field as _, One as _};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
use metal::{objc::rc::autoreleasepool, Buffer, CommandBuffer, MTLSize};

use super::super::{
    completed_command_gpu_time,
    instruction_claim_reduction::{
        InstructionClaimPhaseParams, InstructionClaimSequence, INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
    },
    product_remainder::{
        ProductRemainderPhaseParams, ProductRemainderSequence, PRODUCT_REMAINDER_MESSAGE_COLUMNS,
    },
    set_inline_bytes, Fp128, MetalError, SolinasMetal,
};

const PIPELINE: &str = "solinas_product_instruction_materialize_stage1_message";
const CACHED_PIPELINE: &str = "solinas_product_instruction_materialize_stage1_message_cached";
const MESSAGE_COLUMNS: usize =
    PRODUCT_REMAINDER_MESSAGE_COLUMNS + INSTRUCTION_CLAIM_MESSAGE_COLUMNS;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProductInstructionRoundStats {
    pub(crate) wall: Duration,
    pub(crate) gpu_active: Duration,
    pub(crate) joint: bool,
}

pub(crate) struct ProductInstructionOpenings {
    pub(crate) values: [AkitaField; 8],
    pub(crate) left_lookup_operand: Option<AkitaField>,
    pub(crate) gpu_active: Duration,
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
    pending_initial: Option<ProductInstructionInitialMessageCommand>,
    product_initial: Option<[AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS]>,
    instruction_initial: Option<[AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS]>,
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

struct ProductInstructionInitialMessageCommand {
    command_buffer: CommandBuffer,
    state_b_fill: Option<CommandBuffer>,
    product_output: Buffer,
    instruction_output: Buffer,
    released_product_alternate_bytes: u64,
    released_instruction_alternate_bytes: u64,
    terminal_cache: bool,
    submitted_at: Instant,
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
            if let Some(state_b_fill) = &command.state_b_fill {
                state_b_fill.wait_until_completed();
            }
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
        let (product_message, instruction_message) =
            complete_initial_message(&mut product, &mut instruction, command, false)?;
        Ok((product, product_message, instruction, instruction_message))
    }

    pub(crate) fn into_round_service(
        mut self,
        tau_low: &[AkitaField],
    ) -> Result<ProductInstructionRoundService, MetalError> {
        let product = self
            .product
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "deferred joint first message lost its product sequence",
            ))?;
        let instruction =
            self.instruction
                .take()
                .ok_or(MetalError::InvalidInstructionClaimState(
                    "deferred joint first message lost its instruction sequence",
                ))?;
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "deferred joint first message lost its command buffer",
            ))?;
        let mut service = ProductInstructionRoundService::new(product, instruction, tau_low)?;
        service.pending_initial = Some(command);
        Ok(service)
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
            pending_initial: None,
            product_initial: None,
            instruction_initial: None,
            cached_instruction: None,
        })
    }

    fn complete_pending_initial(&mut self) -> Result<(), MetalError> {
        let Some(command) = self.pending_initial.take() else {
            return Ok(());
        };
        let (product, instruction) =
            complete_initial_message(&mut self.product, &mut self.instruction, command, true)?;
        self.product_initial = Some(product);
        self.instruction_initial = Some(instruction);
        Ok(())
    }

    pub(crate) fn take_product_initial_message(
        &mut self,
    ) -> Result<[AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], MetalError> {
        self.complete_pending_initial()?;
        self.product_initial
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "deferred joint Product first message was already consumed",
            ))
    }

    pub(crate) fn take_instruction_initial_message(
        &mut self,
    ) -> Result<[AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS], MetalError> {
        self.complete_pending_initial()?;
        self.instruction_initial
            .take()
            .ok_or(MetalError::InvalidInstructionClaimState(
                "deferred joint Instruction first message was already consumed",
            ))
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

    pub(crate) fn read_product_current_state(
        &self,
    ) -> Result<(Vec<AkitaField>, Vec<AkitaField>), MetalError> {
        self.product.read_current_state()
    }

    pub(crate) fn retire_product_transition_state_after_cpu_tail_copy(
        &mut self,
    ) -> Result<usize, MetalError> {
        self.product.retire_transition_state_after_cpu_tail_copy()
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
        self.complete_pending_initial()?;
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
        let gpu_active = completed_command_gpu_time(&command_buffer)?;
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
            gpu_active,
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
        self.complete_pending_initial()?;
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
            .bind_and_message_in_place_timed(challenge, e_in, e_out)?;
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
    ) -> Result<(AkitaField, usize), MetalError> {
        self.complete_pending_initial()?;
        if self.cached_instruction.is_some() {
            return Err(MetalError::InvalidInstructionClaimState(
                "instruction finish observed an unconsumed joint message",
            ));
        }
        let claim = self.instruction.finish(challenge)?;
        let retired_bytes = self.instruction.retire_transition_state()?;
        Ok((claim, retired_bytes))
    }

    pub(crate) fn product_openings(
        &mut self,
        after_cpu_tail: bool,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        terminal_factors: [AkitaField; 2],
        lagrange_weights: [AkitaField; 3],
    ) -> Result<ProductInstructionOpenings, MetalError> {
        self.complete_pending_initial()?;
        if let Some(cached) = self.product.terminal_cache_openings_timed(e_in, e_out)? {
            let inverse =
                lagrange_weights[0]
                    .inverse()
                    .ok_or(MetalError::InvalidProductRemainderState(
                    "terminal cache cannot reconstruct Product openings with a zero coefficient",
                ))?;
            let raw = cached.values;
            let left_instruction_input =
                (terminal_factors[0] - lagrange_weights[1] * raw[0] - lagrange_weights[2] * raw[2])
                    * inverse;
            let right_instruction_input = (terminal_factors[1]
                - lagrange_weights[1] * raw[4]
                - lagrange_weights[2] * (AkitaField::one() - raw[5]))
                * inverse;
            Ok(ProductInstructionOpenings {
                values: [
                    left_instruction_input,
                    right_instruction_input,
                    raw[2],
                    raw[3],
                    raw[0],
                    raw[4],
                    raw[5],
                    raw[6],
                ],
                left_lookup_operand: Some(raw[1]),
                gpu_active: cached.gpu_active,
            })
        } else {
            let (values, gpu_active) = if after_cpu_tail {
                self.product.openings_after_cpu_tail_timed(e_in, e_out)?
            } else {
                self.product.openings_timed(e_in, e_out)?
            };
            Ok(ProductInstructionOpenings {
                values,
                left_lookup_operand: None,
                gpu_active,
            })
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
        self.complete_pending_initial()?;
        self.instruction.aliased_openings_timed(e_in, e_out)
    }
}

fn complete_initial_message(
    product: &mut ProductRemainderSequence,
    instruction: &mut InstructionClaimSequence,
    command: ProductInstructionInitialMessageCommand,
    deferred: bool,
) -> Result<
    (
        [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
        [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
    ),
    MetalError,
> {
    let span = tracing::info_span!(
        "MetalProductInstruction::initial_join",
        deferred,
        terminal_cache = command.terminal_cache,
        released_product_alternate_bytes = command.released_product_alternate_bytes,
        released_instruction_alternate_bytes = command.released_instruction_alternate_bytes,
        terminal_cache_capacity_bytes = tracing::field::Empty,
        terminal_cache_dense_bytes = tracing::field::Empty,
        terminal_cache_exception_count = tracing::field::Empty,
        terminal_cache_overflow_groups = tracing::field::Empty,
        terminal_cache_active_logical_bytes = tracing::field::Empty,
        wait_wall_ns = tracing::field::Empty,
        command_wall_ns = tracing::field::Empty,
        gpu_active_ns = tracing::field::Empty,
    );
    let _entered = span.enter();
    let wait_started = Instant::now();
    command.command_buffer.wait_until_completed();
    let mut gpu_active = completed_command_gpu_time(&command.command_buffer)?;
    if let Some(state_b_fill) = &command.state_b_fill {
        state_b_fill.wait_until_completed();
        gpu_active += completed_command_gpu_time(state_b_fill)?;
    }
    let wait_wall = wait_started.elapsed();
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
    let command_wall = command.submitted_at.elapsed();
    if let Some(stats) = product.complete_joint_terminal_cache()? {
        let _ = span.record("terminal_cache_capacity_bytes", stats.capacity_bytes);
        let _ = span.record("terminal_cache_dense_bytes", stats.dense_bytes);
        let _ = span.record(
            "terminal_cache_exception_count",
            u64::try_from(stats.exception_count).unwrap_or(u64::MAX),
        );
        let _ = span.record(
            "terminal_cache_overflow_groups",
            u64::try_from(stats.overflow_groups).unwrap_or(u64::MAX),
        );
        let _ = span.record(
            "terminal_cache_active_logical_bytes",
            stats.active_logical_bytes,
        );
    }
    product.complete_joint_materialize()?;
    instruction.complete_joint_materialize(command_wall, gpu_active)?;
    let _ = span.record(
        "wait_wall_ns",
        u64::try_from(wait_wall.as_nanos()).unwrap_or(u64::MAX),
    );
    let _ = span.record(
        "command_wall_ns",
        u64::try_from(command_wall.as_nanos()).unwrap_or(u64::MAX),
    );
    let _ = span.record(
        "gpu_active_ns",
        u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX),
    );
    Ok((product_message, instruction_message))
}

impl SolinasMetal {
    pub(crate) fn submit_product_instruction_initial_message(
        &self,
        mut product: ProductRemainderSequence,
        mut instruction: InstructionClaimSequence,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
        terminal_cache: bool,
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
        let released_product_alternate_bytes = if terminal_cache {
            product.enable_joint_terminal_cache(e_in.len(), e_out.len())?;
            0
        } else {
            product.release_joint_alternate()?
        };
        let released_instruction_alternate_bytes = instruction.release_joint_alternate()?;
        let pipeline_name = if terminal_cache {
            CACHED_PIPELINE
        } else {
            PIPELINE
        };
        let pipeline = self.compile_named_pipeline(pipeline_name)?;
        let limits = Self::limits(&pipeline);
        let threads = Self::resolve_threadgroup_width(Some(product_threads), limits)?;
        let threadgroup_bytes = MESSAGE_COLUMNS
            .checked_mul(threads / 32)
            .and_then(|fields| fields.checked_mul(size_of::<Fp128>()))
            .ok_or(MetalError::InputTooLong(threads))?;
        let cache_threadgroup_bytes = usize::from(terminal_cache) * 2 * size_of::<u32>();
        let total_threadgroup_bytes = threadgroup_bytes
            .checked_add(cache_threadgroup_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
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
                if terminal_cache {
                    let cache_params = product.encode_joint_terminal_cache(encoder)?.ok_or(
                        MetalError::InvalidProductRemainderState(
                            "cached joint materialization lost its cache layout",
                        ),
                    )?;
                    if cache_params.rows != product_params.source_elements
                        || cache_params.e_in_length != product_params.e_in_length
                        || cache_params.e_out_length != product_params.e_out_length
                    {
                        return Err(MetalError::InvalidProductRemainderState(
                            "terminal cache equality geometry differs from Product materialization",
                        ));
                    }
                    set_inline_bytes(encoder, 15, &cache_params);
                    encoder.set_threadgroup_memory_length(1, cache_threadgroup_bytes as u64);
                } else {
                    set_inline_bytes(encoder, 10, &product_params);
                }
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
                    state_b_fill: None,
                    product_output,
                    instruction_output,
                    released_product_alternate_bytes,
                    released_instruction_alternate_bytes,
                    terminal_cache,
                    submitted_at,
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

type ProductInstructionInitialResult = (
    ProductRemainderSequence,
    [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
    InstructionClaimSequence,
    [AkitaField; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
);

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "the test uses fixed valid Metal geometry"
)]
mod tests {
    use super::ProductInstructionRoundService;
    use jolt_field::Prime128OffsetA7F7 as AkitaField;
    use jolt_field::{Ring as _, Zero as _};
    use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};
    use metal::Buffer;

    use super::super::super::{
        instruction_claim_reduction::InstructionClaimKernelConfig, ProductRemainderSequenceConfig,
        SolinasMetal, SpartanOuterUniskipRow,
    };

    fn fill_buffer_bytes(buffer: &Buffer, value: u8) {
        let length = usize::try_from(buffer.length()).unwrap();
        // SAFETY: the test owns the shared buffer and no command is using it.
        unsafe { std::ptr::write_bytes(buffer.contents().cast::<u8>(), value, length) };
    }

    fn buffer_bytes_are(buffer: &Buffer, expected: u8) -> bool {
        let length = usize::try_from(buffer.length()).unwrap();
        // SAFETY: all commands touching this shared buffer have completed.
        let bytes = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), length) };
        bytes.iter().all(|&value| value == expected)
    }

    #[test]
    fn stage1_joint_materialization_matches_both_standalone_sequences() {
        let Ok(context) = SolinasMetal::for_akita() else {
            return;
        };
        let rows = 1usize << 9;
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
                if index % 19 == 0 {
                    words[13] |= 1u64 << 43;
                }
                words[14] = right_lookup as u64;
                words[15] = (right_lookup >> 64) as u64;
                words[18] = 29 * index as u64 + 7;
                if index % 23 == 0 {
                    words[18] |= 1u64 << 47;
                }
                words[19] = u64::from(index % 2 == 0) << 5
                    | u64::from(index % 3 == 0) << 9
                    | u64::from(index % 5 == 0) << 14
                    | u64::from(signed >= 0) << 17
                    | u64::from(index % 7 == 0) << 25
                    | u64::from(index % 11 == 0) << 26;
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
        let opening_capacity = 1usize << (point.len() / 2);

        let product_config = ProductRemainderSequenceConfig {
            async_state_b_fill: true,
            ..ProductRemainderSequenceConfig::default()
        };
        let product = context
            .prepare_product_remainder_sequence_with_rows(
                resident.clone(),
                lagrange,
                opening_capacity,
                rows / opening_capacity,
                product_config,
            )
            .expect("joint product sequence should prepare");
        let product_state_b = product.joint_state_b_buffer().clone();
        fill_buffer_bytes(&product_state_b, 0xa5);
        let instruction = context
            .prepare_instruction_claim_sequence_with_stage1_rows(
                resident.clone(),
                gamma,
                InstructionClaimKernelConfig::default(),
            )
            .expect("joint instruction sequence should prepare");
        let instruction_state_b = instruction.joint_state_b_buffer().clone();
        fill_buffer_bytes(&instruction_state_b, 0xa5);
        let mut product_control = context
            .prepare_product_remainder_sequence_with_rows(
                resident.clone(),
                lagrange,
                opening_capacity,
                rows / opening_capacity,
                ProductRemainderSequenceConfig::default(),
            )
            .expect("control product sequence should prepare");
        let mut instruction_control = context
            .prepare_instruction_claim_sequence_with_stage1_rows(
                resident.clone(),
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

        let deferred_product = context
            .prepare_product_remainder_sequence_with_rows(
                resident.clone(),
                lagrange,
                opening_capacity,
                rows / opening_capacity,
                product_config,
            )
            .expect("deferred product sequence should prepare");
        let deferred_instruction = context
            .prepare_instruction_claim_sequence_with_stage1_rows(
                resident,
                gamma,
                InstructionClaimKernelConfig::default(),
            )
            .expect("deferred instruction sequence should prepare");
        let deferred = context
            .submit_product_instruction_initial_message(
                deferred_product,
                deferred_instruction,
                e_in,
                e_out,
                true,
            )
            .expect("deferred materialization should submit");
        let mut deferred = deferred
            .into_round_service(&point)
            .expect("deferred service should adopt the active command");
        assert_eq!(
            deferred
                .take_instruction_initial_message()
                .expect("Instruction may join the deferred command first"),
            expected_instruction
        );
        assert_eq!(
            deferred
                .take_product_initial_message()
                .expect("Product may consume its endpoint second"),
            expected_product
        );
        assert_eq!(
            deferred.read_product_current_state().unwrap(),
            product_control.read_current_state().unwrap()
        );
        assert_eq!(
            deferred.read_instruction_current_state().unwrap(),
            instruction_control.read_current_state().unwrap()
        );

        let opening_split = point.len().div_ceil(2);
        let (opening_out_point, opening_in_point) = point.split_at(opening_split);
        let opening_e_in = EqPolynomial::evals(opening_in_point, None);
        let opening_e_out = EqPolynomial::evals(opening_out_point, None);
        let cached_openings = deferred
            .product
            .terminal_cache_openings_timed(&opening_e_in, &opening_e_out)
            .expect("terminal cache opening should execute")
            .expect("cached materialization should retain its terminal cache")
            .values;
        let expected_product_openings = product_control
            .openings_after_cpu_tail_timed(&opening_e_in, &opening_e_out)
            .expect("source Product openings should execute")
            .0;
        let mut expected_left_lookup = AkitaField::zero();
        for (x_out, &outer_weight) in opening_e_out.iter().enumerate() {
            for (x_in, &inner_weight) in opening_e_in.iter().enumerate() {
                let row = x_out * opening_e_in.len() + x_in;
                expected_left_lookup +=
                    outer_weight * inner_weight * AkitaField::from_u64(source[row].words()[13]);
            }
        }
        assert_eq!(cached_openings[0], expected_product_openings[4]);
        assert_eq!(cached_openings[1], expected_left_lookup);
        assert_eq!(cached_openings[2], expected_product_openings[2]);
        assert_eq!(cached_openings[3], expected_product_openings[3]);
        assert_eq!(cached_openings[4], expected_product_openings[5]);
        assert_eq!(cached_openings[5], expected_product_openings[6]);
        assert_eq!(cached_openings[6], expected_product_openings[7]);
        assert_eq!(cached_openings[7], AkitaField::zero());

        let pending = context
            .submit_product_instruction_initial_message(product, instruction, e_in, e_out, false)
            .expect("joint materialization should submit");
        let command = pending
            .command
            .as_ref()
            .expect("joint materialization should retain its command");
        assert_eq!(
            command.released_product_alternate_bytes,
            product_state_b.length()
        );
        assert_eq!(
            command.released_instruction_alternate_bytes,
            instruction_state_b.length()
        );
        let (product, product_message, instruction, instruction_message) =
            pending.join().expect("joint materialization should join");

        assert!(buffer_bytes_are(&product_state_b, 0xa5));
        assert!(buffer_bytes_are(&instruction_state_b, 0xa5));
        assert_eq!(product.joint_state_b_buffer().length(), 1);
        assert_eq!(instruction.joint_state_b_buffer().length(), 1);
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
            assert!(buffer_bytes_are(&product_state_b, 0xa5));
            assert!(buffer_bytes_are(&instruction_state_b, 0xa5));
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
