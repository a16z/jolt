use std::sync::{Arc, Mutex};

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
    finalize_aliased_openings, nontrivial_gamma_powers, round_polynomial_from_q_endpoints,
    InstructionClaimAliasedOpenings, InstructionClaimKernelConfig, InstructionClaimSequence,
    PendingInstructionClaimInitialMessage,
};
use super::solinas::{MetalError, ProductInstructionRoundService, ProductInstructionRoundStats};
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
        let source_identities = handoff.rows.product.allocation_identities();
        if handoff.rows.log_t != rounds
            || handoff.rows.product.len() != cycles
            || handoff.rows.device_registry_id != self.context.device_registry_id()
            || source_identities.len() != 2
        {
            return Err(KernelError::InvariantViolation {
                reason: "instruction claim handoff has the wrong shape or Metal device",
            });
        }
        let row_storage_id = handoff.rows.product.allocation_identity();
        let host = MetalInstructionClaimHost::new(inputs.relation.tau_low());
        let joint_prefetched = handoff.prefetched_initial.is_some();
        let prepare_span = tracing::info_span!(
            "MetalInstructionClaimReduction::prepare",
            cycles,
            rounds,
            joint_prefetched,
        );
        let _entered = prepare_span.enter();
        let (state, initial_endpoints) = if let Some(prefetched) = handoff.prefetched_initial {
            let service =
                prefetched
                    .service
                    .lock()
                    .map_err(|_| KernelError::InvariantViolation {
                        reason: "joint Product/Instruction service lock is poisoned",
                    })?;
            let valid = service.instruction_gamma() == inputs.challenges.gamma
                && service.instruction_current_elements() == cycles
                && service.instruction_allocation_identities()
                    == Some(source_identities[..2].try_into().map_err(|_| {
                        KernelError::InvariantViolation {
                            reason: "instruction claim source identity count changed",
                        }
                    })?);
            drop(service);
            if !valid {
                return Err(KernelError::InvariantViolation {
                    reason: "prefetched instruction state disagrees with its relation or rows",
                });
            }
            (
                MetalInstructionClaimState::Joint(prefetched.service),
                prefetched.endpoints,
            )
        } else {
            let prepared = self
                .context
                .prepare_instruction_claim_sequence_with_stage1_rows(
                    handoff.rows.product,
                    inputs.challenges.gamma,
                    self.config.instruction_claim_reduction.dispatch,
                );
            match prepared {
                Ok(sequence) => (
                    MetalInstructionClaimState::Standalone(Box::new(sequence)),
                    None,
                ),
                Err(error) if error.is_capacity_error() => {
                    tracing::warn!(
                        target: "jolt::metal",
                        error = %error,
                        "instruction claim workspace was not admitted; using optimized CPU"
                    );
                    return OptimizedInstructionClaimReduction.prepare(session, witness, inputs);
                }
                Err(error) => return Err(metal_prepare_error(error)),
            }
        };
        let (e_in, e_out) = host.current_weights();
        let submit_span = tracing::info_span!(
            "MetalInstructionClaimReduction::first_message_submit",
            joint_prefetched,
        );
        let _submit_entered = submit_span.enter();
        let (pending_initial, state) = match (initial_endpoints.is_some(), state) {
            (true, state) => (None, Some(state)),
            (false, MetalInstructionClaimState::Joint(service)) => {
                (None, Some(MetalInstructionClaimState::Joint(service)))
            }
            (false, MetalInstructionClaimState::Standalone(sequence)) => {
                let pending = match (*sequence).submit_initial_message(e_in, e_out) {
                    Ok(pending) => pending,
                    Err(error) if instruction_prepare_fallback_reason(&error).is_some() => {
                        tracing::warn!(
                            target: "jolt::metal",
                            error = %error,
                            "instruction claim first submission failed before round absorption; using optimized CPU"
                        );
                        return OptimizedInstructionClaimReduction
                            .prepare(session, witness, inputs);
                    }
                    Err(error) => return Err(metal_prepare_error(error)),
                };
                (Some(pending), None)
            }
        };
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .instruction_claim_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(Box::new(MetalInstructionClaimKernel {
            host,
            pending_initial,
            initial_endpoints,
            state,
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

enum MetalInstructionClaimState {
    Standalone(Box<InstructionClaimSequence>),
    Joint(Arc<Mutex<ProductInstructionRoundService>>),
}

impl MetalInstructionClaimState {
    fn take_initial_message(&mut self) -> Result<[AkitaField; 2], SumcheckError<AkitaField>> {
        match self {
            Self::Standalone(_) => Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "standalone Instruction first message was already consumed".to_string(),
            }),
            Self::Joint(service) => service
                .lock()
                .map_err(|_| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "joint Product/Instruction service lock is poisoned".to_string(),
                })?
                .take_instruction_initial_message()
                .map_err(metal_round_error),
        }
    }

    fn current_elements(&self) -> Result<usize, SumcheckError<AkitaField>> {
        match self {
            Self::Standalone(sequence) => Ok(sequence.current_elements()),
            Self::Joint(service) => service
                .lock()
                .map(|service| service.instruction_current_elements())
                .map_err(|_| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "joint Product/Instruction service lock is poisoned".to_string(),
                }),
        }
    }

    fn bind_and_message(
        &mut self,
        round: usize,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; 2], ProductInstructionRoundStats), SumcheckError<AkitaField>> {
        match self {
            Self::Standalone(sequence) => {
                let (message, timing) = sequence
                    .bind_and_message_timed(challenge, e_in, e_out)
                    .map_err(metal_round_error)?;
                Ok((
                    message,
                    ProductInstructionRoundStats {
                        wall: timing.wall,
                        gpu_active: timing.gpu_active,
                        joint: false,
                    },
                ))
            }
            Self::Joint(service) => service
                .lock()
                .map_err(|_| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "joint Product/Instruction service lock is poisoned".to_string(),
                })?
                .instruction_bind_and_message(round, challenge, e_in, e_out)
                .map_err(metal_round_error),
        }
    }

    fn finish(&mut self, challenge: AkitaField) -> Result<(AkitaField, usize), MetalError> {
        match self {
            Self::Standalone(sequence) => {
                let claim = sequence.finish(challenge)?;
                let retired_bytes = sequence.retire_transition_state()?;
                Ok((claim, retired_bytes))
            }
            Self::Joint(service) => service
                .lock()
                .map_err(|_| {
                    MetalError::InvalidInstructionClaimState(
                        "joint Product/Instruction service lock is poisoned",
                    )
                })?
                .finish_instruction(challenge),
        }
    }

    fn openings(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<
        (
            [AkitaField; 2],
            super::solinas::instruction_claim_reduction::InstructionClaimTiming,
        ),
        MetalError,
    > {
        match self {
            Self::Standalone(sequence) => sequence.aliased_openings_timed(e_in, e_out),
            Self::Joint(service) => service
                .lock()
                .map_err(|_| {
                    MetalError::InvalidInstructionClaimState(
                        "joint Product/Instruction service lock is poisoned",
                    )
                })?
                .instruction_aliased_openings(e_in, e_out),
        }
    }
}

struct MetalInstructionClaimKernel {
    host: MetalInstructionClaimHost,
    pending_initial: Option<PendingInstructionClaimInitialMessage>,
    initial_endpoints: Option<[AkitaField; 2]>,
    state: Option<MetalInstructionClaimState>,
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
        if let Some(MetalInstructionClaimState::Standalone(sequence)) = &self.state {
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
            let state = self
                .state
                .as_mut()
                .ok_or_else(|| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim resident sequence is missing".to_string(),
                })?;
            let source_elements = state.current_elements()?;
            let span = tracing::info_span!(
                "MetalInstructionClaimReduction::bind_and_message",
                round,
                source_elements,
                gpu_active_ns = tracing::field::Empty,
            );
            let _entered = span.enter();
            let (e_in, e_out) = self.host.current_weights();
            let (message, stats) = state.bind_and_message(round, challenge, e_in, e_out)?;
            let _ = span.record("gpu_active_ns", duration_nanos(stats.gpu_active));
            message
        } else {
            if round != 0 || !self.host.challenges.is_empty() {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "instruction claim first message was requested out of order"
                        .to_string(),
                });
            }
            let span = tracing::info_span!("MetalInstructionClaimReduction::first_message_join");
            let _entered = span.enter();
            if let Some(message) = self.initial_endpoints.take() {
                if self.pending_initial.is_some() || self.state.is_none() {
                    return Err(SumcheckError::ComputeBackend {
                        backend: "metal",
                        message: "joint instruction first-message ownership drifted".to_string(),
                    });
                }
                message
            } else if let Some(state) = &mut self.state {
                if self.pending_initial.is_some() {
                    return Err(SumcheckError::ComputeBackend {
                        backend: "metal",
                        message: "Instruction first-message ownership is ambiguous".to_string(),
                    });
                }
                state.take_initial_message()?
            } else {
                let pending =
                    self.pending_initial
                        .take()
                        .ok_or_else(|| SumcheckError::ComputeBackend {
                            backend: "metal",
                            message: "instruction claim first message is missing".to_string(),
                        })?;
                let (sequence, message, _stats) = pending.join().map_err(metal_round_error)?;
                self.state = Some(MetalInstructionClaimState::Standalone(Box::new(sequence)));
                message
            }
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
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| SumcheckError::ComputeBackend {
                backend: "metal",
                message: "instruction claim resident sequence is missing".to_string(),
            })?;
        let (combined_claim, retired_bytes) = state.finish(bind).map_err(metal_round_error)?;
        tracing::info!(
            target: "jolt::metal",
            retired_bytes,
            "retired Instruction transition state after final bind"
        );
        self.combined_claim = Some(combined_claim);
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
        let span = tracing::info_span!(
            "MetalInstructionClaimReduction::output_claims",
            gpu_active_ns = tracing::field::Empty,
        );
        let _entered = span.enter();
        let published = self
            .aliases
            .lock()
            .map_err(|_| metal_output_error("instruction claim alias slot lock is poisoned"))?
            .take()
            .ok_or_else(|| metal_output_error("product remainder did not publish its aliases"))?;
        validate_alias_output(&published, self.row_storage_id, &self.host.challenges)?;
        let aliases = InstructionClaimAliasedOpenings {
            lookup_output: published.values.lookup_output,
            left_instruction_input: published.values.left_instruction_input,
            right_instruction_input: published.values.right_instruction_input,
        };
        let (lookup_operands, gpu_active) =
            if let Some(left_lookup_operand) = published.values.left_lookup_operand {
                let powers = nontrivial_gamma_powers(self.gamma);
                let inverse = powers[1].inverse().ok_or_else(|| {
                    metal_output_error(
                        "cached instruction aliases cannot recover right lookup at zero gamma",
                    )
                })?;
                let right_lookup_operand = (combined_claim
                    - aliases.lookup_output
                    - powers[0] * left_lookup_operand
                    - powers[2] * aliases.left_instruction_input
                    - powers[3] * aliases.right_instruction_input)
                    * inverse;
                (
                    [left_lookup_operand, right_lookup_operand],
                    std::time::Duration::ZERO,
                )
            } else {
                let (e_in, e_out) = self.host.opening_weights();
                let state = self.state.as_mut().ok_or_else(|| {
                    metal_output_error("instruction claim resident sequence is missing")
                })?;
                let (values, timing) = state.openings(&e_in, &e_out).map_err(metal_output_error)?;
                (values, timing.gpu_active)
            };
        let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
        let openings =
            finalize_aliased_openings(self.gamma, combined_claim, lookup_operands, aliases)
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
