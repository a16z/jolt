use std::time::Instant;

use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::{InstructionClaimReductionPublic, JoltDerivedId};
use jolt_field::AkitaField;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::solinas::instruction_claim_reduction::{
    finalize_aliased_openings, round_polynomial_from_q_endpoints, InstructionClaimAliasedOpenings,
    InstructionClaimKernelConfig, InstructionClaimSequence,
    PendingInstructionClaimInitialMessage,
};
use super::solinas::MetalError;
use super::spartan_product::{MetalInstructionClaimAliasOutput, MetalInstructionClaimHandoff};
use crate::optimized::instruction_claim_reduction::OptimizedInstructionClaimReduction;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimReductionMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: InstructionClaimKernelConfig,
}

impl Default for InstructionClaimReductionMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            dispatch: InstructionClaimKernelConfig::default(),
        }
    }
}

impl PrepareKernel<AkitaField, InstructionClaimReduction<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionClaimReduction<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionClaimReduction<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let rounds = inputs.relation.rounds();
        let cycles = 1usize
            .checked_shl(rounds as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "instruction claim trace length overflows usize",
            })?;
        if cycles
            < self
                .config
                .instruction_claim_reduction
                .trace_cutoff_elements
        {
            drop(session.take::<MetalInstructionClaimHandoff>());
            return OptimizedInstructionClaimReduction.prepare(session, witness, inputs);
        }
        let Some(handoff) = session.take::<MetalInstructionClaimHandoff>() else {
            return OptimizedInstructionClaimReduction.prepare(session, witness, inputs);
        };
        if handoff.rows.log_t != rounds
            || handoff.rows.product.len() != cycles
            || handoff.rows.lookup.len() != cycles
            || handoff.rows.device_registry_id != self.context.device_registry_id()
        {
            return Err(KernelError::InvariantViolation {
                reason: "instruction claim handoff has the wrong shape or Metal device",
            });
        }
        let row_storage_id = handoff.rows.product.allocation_identity();
        let lookup_storage_id = handoff.rows.lookup.allocation_identity();
        let host = MetalInstructionClaimHost::new(inputs.relation.tau_low());
        let prepare_span = tracing::info_span!(
            "MetalInstructionClaimReduction::prepare",
            cycles,
            rounds,
            resident_rows_storage_id = row_storage_id as u64,
            lookup_rows_storage_id = lookup_storage_id as u64,
            row_upload_bytes = 0u64,
            round_device_buffer_allocations = 0u64,
            workspace_bytes = tracing::field::Empty,
        );
        let _entered = prepare_span.enter();
        let sequence = match self
            .context
            .prepare_instruction_claim_sequence_with_product_rows(
                handoff.rows.product,
                handoff.rows.lookup,
                inputs.challenges.gamma,
                self.config.instruction_claim_reduction.dispatch,
            ) {
            Ok(sequence) => sequence,
            Err(error) if error.is_capacity_error() => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "instruction claim workspace was not admitted; using optimized CPU"
                );
                return OptimizedInstructionClaimReduction.prepare(session, witness, inputs);
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let _ = prepare_span.record(
            "workspace_bytes",
            sequence.storage_layout().workspace_bytes(),
        );
        let (e_in, e_out) = host.current_weights();
        let submit_span = tracing::info_span!(
            "MetalInstructionClaimReduction::first_message_submit",
            resident_rows_storage_id = row_storage_id as u64,
            lookup_rows_storage_id = lookup_storage_id as u64,
            command_committed = tracing::field::Empty,
            submit_wall_ns = tracing::field::Empty,
        );
        let _submit_entered = submit_span.enter();
        let pending_initial = match sequence.submit_initial_message(e_in, e_out) {
            Ok(pending) => pending,
            Err(error) if instruction_prepare_fallback_reason(&error).is_some() => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "instruction claim first submission failed before round absorption; using optimized CPU"
                );
                return OptimizedInstructionClaimReduction.prepare(session, witness, inputs);
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let _ = submit_span.record("command_committed", true);
        let _ = submit_span.record(
            "submit_wall_ns",
            duration_nanos(pending_initial.submit_wall()),
        );
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .instruction_claim_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(Box::new(MetalInstructionClaimKernel {
            host,
            pending_initial: Some(pending_initial),
            sequence: None,
            gamma: inputs.challenges.gamma,
            combined_claim: None,
            aliases: handoff.aliases,
            row_storage_id,
        }))
    }
}

fn instruction_prepare_fallback_reason(error: &MetalError) -> Option<&'static str> {
    if error.is_capacity_error() {
        return Some("capacity");
    }
    match error {
        MetalError::CommandFailed(_) => Some("command_failed"),
        MetalError::GpuTimestampLookup { .. } => Some("gpu_timestamp_lookup"),
        MetalError::InvalidGpuTimestamps { .. } => Some("invalid_gpu_timestamps"),
        _ => None,
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

fn metal_round_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn metal_output_error(message: impl ToString) -> SumcheckKernelError<AkitaField> {
    SumcheckKernelError::ComputeBackend {
        backend: "metal",
        message: message.to_string(),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

struct MetalInstructionClaimHost {
    rounds: usize,
    split_eq: GruenSplitEqPolynomial<AkitaField>,
    challenges: Vec<AkitaField>,
}

impl MetalInstructionClaimHost {
    fn new(tau_low: &[AkitaField]) -> Self {
        Self {
            rounds: tau_low.len(),
            split_eq: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            challenges: Vec::with_capacity(tau_low.len()),
        }
    }

    fn current_weights(&self) -> (&[AkitaField], &[AkitaField]) {
        (self.split_eq.e_in_current(), self.split_eq.e_out_current())
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
    }

    fn polynomial(
        &self,
        endpoints: [AkitaField; 2],
        previous_claim: AkitaField,
    ) -> UnivariatePoly<AkitaField> {
        round_polynomial_from_q_endpoints(
            previous_claim,
            endpoints,
            self.split_eq.current_linear_evals().into(),
        )
    }

    fn opening_weights(&self) -> (Vec<AkitaField>, Vec<AkitaField>) {
        let point = self.challenges.iter().rev().copied().collect::<Vec<_>>();
        let split = point.len() / 2;
        let (out_point, in_point) = point.split_at(split);
        (
            EqPolynomial::evals(in_point, None),
            EqPolynomial::evals(out_point, None),
        )
    }
}

struct MetalInstructionClaimKernel {
    host: MetalInstructionClaimHost,
    pending_initial: Option<PendingInstructionClaimInitialMessage>,
    sequence: Option<InstructionClaimSequence>,
    gamma: AkitaField,
    combined_claim: Option<AkitaField>,
    aliases: super::spartan_product::MetalInstructionClaimAliasSlot,
    row_storage_id: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionClaimKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(pending) = &self.pending_initial {
            visitor.visit_field(allocative::Key::new("pending_initial"), pending);
        }
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for MetalInstructionClaimKernel {
    fn num_rounds(&self) -> usize {
        self.host.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let endpoints = if let Some(challenge) = bind {
            self.host.bind(challenge);
            if self.host.challenges.len() != round {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim round order drifted".to_string(),
                });
            }
            if self.pending_initial.is_some() {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim first message was not joined".to_string(),
                });
            }
            let sequence = self.sequence.as_mut().ok_or_else(|| {
                SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim resident sequence is missing".to_string(),
                }
            })?;
            let source_elements = sequence.current_elements();
            let span = tracing::info_span!(
                "MetalInstructionClaimReduction::bind_and_message",
                round,
                source_elements,
                resident_rows_storage_id = self.row_storage_id as u64,
                dispatch_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
            );
            let _entered = span.enter();
            let started = Instant::now();
            let (e_in, e_out) = self.host.current_weights();
            let (message, timing) = sequence
                .bind_and_message_timed(challenge, e_in, e_out)
                .map_err(metal_round_error)?;
            let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
            let _ = span.record("gpu_active_ns", duration_nanos(timing.gpu_active));
            message
        } else {
            if round != 0 || !self.host.challenges.is_empty() {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim first message was requested out of order"
                        .to_string(),
                });
            }
            if self.sequence.is_some() {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim first message was already consumed".to_string(),
                });
            }
            let pending = self.pending_initial.take().ok_or_else(|| {
                SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim first message is missing".to_string(),
                }
            })?;
            let span = tracing::info_span!(
                "MetalInstructionClaimReduction::first_message_join",
                resident_rows_storage_id = self.row_storage_id as u64,
                command_completed = tracing::field::Empty,
                completed_before_join = tracing::field::Empty,
                submit_wall_ns = tracing::field::Empty,
                overlap_wall_ns = tracing::field::Empty,
                join_wall_ns = tracing::field::Empty,
                lifecycle_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
            );
            let _entered = span.enter();
            let (sequence, message, stats) = pending.join().map_err(metal_round_error)?;
            let _ = span.record("command_completed", true);
            let _ = span.record("completed_before_join", stats.completed_before_join);
            let _ = span.record("submit_wall_ns", duration_nanos(stats.submit_wall));
            let _ = span.record("overlap_wall_ns", duration_nanos(stats.overlap_wall));
            let _ = span.record("join_wall_ns", duration_nanos(stats.join_wall));
            let _ = span.record("lifecycle_wall_ns", duration_nanos(stats.wall));
            let _ = span.record("gpu_active_ns", duration_nanos(stats.gpu_active));
            self.sequence = Some(sequence);
            message
        };
        Ok(self.host.polynomial(endpoints, previous_claim))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.host.bind(bind);
        if self.host.challenges.len() != self.host.rounds {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "instruction claim did not receive every bind challenge".to_string(),
            });
        }
        let sequence = self.sequence.as_mut().ok_or_else(|| {
            SumcheckError::ComputeBackend {
                backend: "metal",
                message: "instruction claim resident sequence is missing".to_string(),
            }
        })?;
        self.combined_claim = Some(sequence.finish(bind).map_err(metal_round_error)?);
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalInstructionClaimKernel {
    type Relation = InstructionClaimReduction<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<InstructionClaimReductionOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>>
    {
        let remaining = self.host.rounds.saturating_sub(self.host.challenges.len());
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let combined_claim = self.combined_claim.ok_or_else(|| {
            metal_output_error("instruction claim final combined value is missing")
        })?;
        let (e_in, e_out) = self.host.opening_weights();
        let span = tracing::info_span!(
            "MetalInstructionClaimReduction::output_claims",
            resident_rows_storage_id = self.row_storage_id as u64,
            row_upload_bytes = 0u64,
            dispatch_wall_ns = tracing::field::Empty,
            gpu_active_ns = tracing::field::Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let sequence = self
            .sequence
            .as_mut()
            .ok_or_else(|| metal_output_error("instruction claim resident sequence is missing"))?;
        let (lookup_operands, timing) = sequence
            .aliased_openings_timed(&e_in, &e_out)
            .map_err(metal_output_error)?;
        let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
        let _ = span.record("gpu_active_ns", duration_nanos(timing.gpu_active));
        let published = self
            .aliases
            .try_borrow_mut()
            .map_err(|_| metal_output_error("instruction claim alias slot is already borrowed"))?
            .take()
            .ok_or_else(|| metal_output_error("product remainder did not publish its aliases"))?;
        validate_alias_output(&published, self.row_storage_id, &self.host.challenges)?;
        let openings = finalize_aliased_openings(
            self.gamma,
            combined_claim,
            lookup_operands,
            InstructionClaimAliasedOpenings {
                lookup_output: published.values.lookup_output,
                left_instruction_input: published.values.left_instruction_input,
                right_instruction_input: published.values.right_instruction_input,
            },
        )
        .map_err(metal_output_error)?;
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output: openings.lookup_output,
            left_lookup_operand: openings.left_lookup_operand,
            right_lookup_operand: openings.right_lookup_operand,
            left_instruction_input: openings.left_instruction_input,
            right_instruction_input: openings.right_instruction_input,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let remaining = self.host.rounds.saturating_sub(self.host.challenges.len());
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let id = JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self.host.split_eq.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

fn validate_alias_output(
    published: &MetalInstructionClaimAliasOutput,
    row_storage_id: usize,
    challenges: &[AkitaField],
) -> Result<(), SumcheckKernelError<AkitaField>> {
    if published.row_storage_id != row_storage_id || published.challenges != challenges {
        return Err(metal_output_error(
            "product and instruction claim alias identities or points differ",
        ));
    }
    Ok(())
}
