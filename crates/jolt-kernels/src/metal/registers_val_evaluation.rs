use std::mem::size_of;

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::registers::rd_inc_val_evaluation;
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::registers_val_evaluation::{
    RegistersValEvaluation, RegistersValEvaluationOutputClaims,
};
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::backend::MetalBackend;
use super::solinas::{
    MetalError, PendingRegistersValFirstMessage, RegistersValDenseConfig,
    RegistersValFirstMessageConfig, RegistersValFirstMessageInvocation,
    RegistersValFirstTransitionInvocation, RegistersValInstructionSourceLease,
    RegistersValSequence, RegistersValTransitionConfig,
};
use crate::optimized::registers_read_write::{RegisterCycleRow, SharedRdIndices};
use crate::optimized::registers_val_evaluation::{
    OptimizedRegistersValEvaluation, ValEvaluationKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RegistersValEvaluationSource {
    #[default]
    WitnessUpload,
    Stage1Resident,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValEvaluationMetalConfig {
    pub source: RegistersValEvaluationSource,
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub first_message: RegistersValFirstMessageConfig,
    pub first_transition: RegistersValTransitionConfig,
    pub dense_transition: RegistersValDenseConfig,
}

impl Default for RegistersValEvaluationMetalConfig {
    fn default() -> Self {
        Self {
            source: RegistersValEvaluationSource::WitnessUpload,
            trace_cutoff_elements: 1 << 25,
            cutoff_elements: 1 << 16,
            first_message: RegistersValFirstMessageConfig::default(),
            first_transition: RegistersValTransitionConfig::default(),
            dense_transition: RegistersValDenseConfig::default(),
        }
    }
}

enum RegistersValState {
    Submitted(PendingRegistersValFirstMessage),
    FirstJoined(RegistersValFirstMessageInvocation),
    Native(RegistersValFirstTransitionInvocation),
    Dense(RegistersValSequence),
    CpuTail,
    Finished,
    Failed,
}

impl RegistersValState {
    fn current_elements(&self) -> Option<usize> {
        match self {
            Self::Submitted(pending) => pending.cycles(),
            Self::FirstJoined(invocation) => Some(invocation.cycles()),
            Self::Native(invocation) => Some(invocation.current_elements()),
            Self::Dense(sequence) => Some(sequence.current_elements()),
            Self::CpuTail | Self::Finished | Self::Failed => None,
        }
    }
}

struct MetalRegistersValEvaluationKernel {
    cpu: ValEvaluationKernel<AkitaField>,
    state: RegistersValState,
    host_tail: Vec<[AkitaField; 2]>,
    cutoff_elements: usize,
    first_transition: RegistersValTransitionConfig,
    dense_transition: RegistersValDenseConfig,
    next_round: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRegistersValEvaluationKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("cpu"), &self.cpu);
        visitor.visit_simple(
            allocative::Key::new("host_tail"),
            crate::backend::vec_heap_bytes(&self.host_tail),
        );
        visitor.exit();
    }
}

impl PrepareKernel<AkitaField, RegistersValEvaluation<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RegistersValEvaluation<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RegistersValEvaluation<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let config = self.config.registers_val_evaluation;
        let log_t = inputs.relation.trace_dimensions().log_t();
        let cycles = 1usize << log_t;
        let cpu_inputs = || ProverInputs {
            relation: inputs.relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let resident_from_stage1 = config.source == RegistersValEvaluationSource::Stage1Resident;
        let resident_requested = resident_from_stage1
            && cycles >= config.trace_cutoff_elements
            && (26..=28).contains(&log_t);
        let resident_route = "instruction_rows_v1";
        if !resident_requested {
            if session
                .state::<RegistersValInstructionSourceLease>()
                .is_some()
            {
                return Err(KernelError::InvariantViolation {
                    reason: "registers value found an unexpected Stage-1 source lease",
                });
            }
            if resident_from_stage1
                || cycles < config.trace_cutoff_elements
                || cycles <= config.cutoff_elements
            {
                return OptimizedRegistersValEvaluation.prepare(session, witness, cpu_inputs());
            }
        }
        let point = &inputs.points.registers_val;
        if point.len() != REGISTER_ADDRESS_BITS + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = point.split_at(REGISTER_ADDRESS_BITS);
        let split_bits = r_cycle.len() / 2;
        let split_handoff = cycles >> split_bits.saturating_sub(1);
        let cutoff_elements = config.cutoff_elements.max(split_handoff);

        if resident_requested {
            let lease = session.take::<RegistersValInstructionSourceLease>().ok_or(
                KernelError::InvariantViolation {
                    reason: "RegistersVal stage1 route is missing its instruction-source lease",
                },
            )?;
            if cutoff_elements >= cycles {
                return Err(KernelError::InvariantViolation {
                    reason: "registers value resident route cannot hand off before dispatch",
                });
            }
            let receipt = lease.receipt();
            let prepare_span = tracing::info_span!(
                "MetalRegistersValEvaluation::prepare",
                cycles,
                cutoff_elements,
                source = "instruction_rows_v1",
                row_layout = "column_major_packed_u64_v3",
                source_generation = receipt.generation(),
                source_device_registry_id = receipt.device_registry_id(),
                source_ready_serial = receipt.completion_serial(),
                instruction_rows_storage_id = receipt.instruction_rows_storage_id(),
                resident_source_bytes = receipt.instruction_rows_bytes(),
                inc_upload_bytes = 0,
                rd_upload_bytes = 0,
            );
            let invocation = {
                let _guard = prepare_span.enter();
                self.context
                    .prepare_registers_val_first_message_instruction_rows(
                        lease,
                        r_address,
                        r_cycle,
                        config.first_message,
                    )
            }
            .map_err(metal_prepare_error)?;
            if invocation.resident_source_receipt() != Some(receipt) {
                return Err(KernelError::InvariantViolation {
                    reason: "registers value invocation lost its instruction-source receipt",
                });
            }
            record_route(cycles, resident_route, resident_route, "none");
            return Ok(self.finish_metal_registers_val_prepare(
                invocation,
                r_cycle,
                cutoff_elements,
                config,
            ));
        }

        let inc: Vec<AkitaField> = witness.oracle_table(rd_inc_val_evaluation().polynomial_id())?;
        if inc.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", rd_inc_val_evaluation()),
                expected: cycles,
                got: inc.len(),
            });
        }
        let rd = match session.take::<SharedRdIndices>() {
            Some(SharedRdIndices(rd)) if rd.len() == cycles => rd,
            _ => collect_bundles::<RegisterCycleRow>(witness, cycles)?
                .iter()
                .map(|row| row.rd.map(|(index, ..)| index))
                .collect(),
        };
        if cutoff_elements >= cycles {
            return Ok(Box::new(ValEvaluationKernel::new_ready(
                inc, rd, r_address, r_cycle,
            )?));
        }
        let rd_device = rd
            .iter()
            .map(|index| index.unwrap_or(u8::MAX))
            .collect::<Vec<_>>();

        let prepare_span = tracing::info_span!(
            "MetalRegistersValEvaluation::prepare",
            cycles,
            cutoff_elements,
            inc_upload_bytes = cycles * size_of::<AkitaField>(),
            rd_upload_bytes = cycles,
        );
        let invocation = {
            let _guard = prepare_span.enter();
            self.context.prepare_registers_val_first_message(
                &inc,
                &rd_device,
                r_address,
                r_cycle,
                config.first_message,
            )
        };
        let invocation = match invocation {
            Ok(invocation) => invocation,
            Err(MetalError::BufferTooLong { .. } | MetalError::WorkingSetTooLarge { .. }) => {
                return Ok(Box::new(ValEvaluationKernel::new_ready(
                    inc, rd, r_address, r_cycle,
                )?));
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        record_route(cycles, "witness_upload", "witness_upload", "none");
        let pending = invocation.submit();
        drop(inc);
        drop(rd);
        drop(rd_device);
        let cpu = ValEvaluationKernel::new_offloaded(r_cycle);
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .registers_val_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Box::new(MetalRegistersValEvaluationKernel {
            cpu,
            state: RegistersValState::Submitted(pending),
            host_tail: vec![[AkitaField::zero(); 2]; cutoff_elements],
            cutoff_elements,
            first_transition: config.first_transition,
            dense_transition: config.dense_transition,
            next_round: 0,
        }))
    }
}

impl MetalBackend {
    #[cfg_attr(
        not(any(test, feature = "test-utils")),
        expect(
            clippy::unused_self,
            reason = "test builds record backend dispatch counters"
        )
    )]
    fn finish_metal_registers_val_prepare(
        &self,
        invocation: RegistersValFirstMessageInvocation,
        r_cycle: &[AkitaField],
        cutoff_elements: usize,
        config: RegistersValEvaluationMetalConfig,
    ) -> Box<dyn SumcheckKernel<AkitaField, Relation = RegistersValEvaluation<AkitaField>>> {
        let pending = invocation.submit();
        let cpu = ValEvaluationKernel::new_offloaded(r_cycle);
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .registers_val_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Box::new(MetalRegistersValEvaluationKernel {
            cpu,
            state: RegistersValState::Submitted(pending),
            host_tail: vec![[AkitaField::zero(); 2]; cutoff_elements],
            cutoff_elements,
            first_transition: config.first_transition,
            dense_transition: config.dense_transition,
            next_round: 0,
        })
    }
}

fn record_route(
    cycles: usize,
    requested: &'static str,
    selected: &'static str,
    fallback_reason: &'static str,
) {
    let _span = tracing::info_span!(
        "MetalRegistersValEvaluation::route",
        cycles,
        requested,
        selected,
        fallback_reason,
    )
    .entered();
}

impl MetalRegistersValEvaluationKernel {
    fn validate_round(
        &self,
        bind: Option<&AkitaField>,
        round: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if round != self.next_round || round >= self.cpu.num_rounds() {
            return Err(metal_error(
                "registers value received an out-of-order round",
            ));
        }
        if (round == 0) != bind.is_none() {
            return Err(metal_error(
                "registers value round has the wrong bind shape",
            ));
        }
        if matches!(
            &self.state,
            RegistersValState::Finished | RegistersValState::Failed
        ) {
            return Err(metal_error(
                "registers value received a round after terminal state",
            ));
        }
        Ok(())
    }

    fn should_restore(&self) -> bool {
        matches!(
            &self.state,
            RegistersValState::Native(invocation)
                if invocation.current_elements() <= self.cutoff_elements
        ) || matches!(
            &self.state,
            RegistersValState::Dense(sequence)
                if sequence.current_elements() <= self.cutoff_elements
        )
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let state = std::mem::replace(&mut self.state, RegistersValState::Failed);
        let elements = state
            .current_elements()
            .ok_or_else(|| metal_error("registers value has no resident state to read back"))?;
        if elements > self.host_tail.len() {
            return Err(metal_error(
                "registers value device state exceeds the prepared CPU tail",
            ));
        }
        let rows = &mut self.host_tail[..elements];
        let _span = tracing::info_span!(
            "MetalRegistersValEvaluation::readback",
            elements,
            bytes = elements * size_of::<[AkitaField; 2]>(),
        )
        .entered();
        match state {
            RegistersValState::Submitted(_) | RegistersValState::FirstJoined(_) => {
                return Err(metal_error(
                    "registers value cannot restore before its first bind",
                ));
            }
            RegistersValState::Native(invocation) => invocation
                .read_dense_state_into(rows)
                .map_err(metal_runtime_error)?,
            RegistersValState::Dense(sequence) => sequence
                .read_current_dense_state_into(rows)
                .map_err(metal_runtime_error)?,
            RegistersValState::CpuTail
            | RegistersValState::Finished
            | RegistersValState::Failed => {
                return Err(metal_error(
                    "registers value cannot read back from a terminal host state",
                ));
            }
        }
        self.cpu.metal_restore_dense(rows)?;
        self.state = RegistersValState::CpuTail;
        Ok(())
    }

    fn device_message(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
    ) -> Result<[AkitaField; 3], SumcheckError<AkitaField>> {
        match bind {
            None => {
                let state = std::mem::replace(&mut self.state, RegistersValState::Failed);
                let RegistersValState::Submitted(pending) = state else {
                    return Err(metal_error(
                        "registers value first message has an invalid device state",
                    ));
                };
                let source_elements = pending.cycles().ok_or_else(|| {
                    metal_error("registers value submitted message lost its cycle count")
                })?;
                let span = tracing::info_span!(
                    "MetalRegistersValEvaluation::first_message_join",
                    source_elements,
                    gpu_active_ns = tracing::field::Empty,
                );
                let _entered = span.enter();
                let (invocation, gpu_active) = pending.join().map_err(metal_runtime_error)?;
                let _ = span.record(
                    "gpu_active_ns",
                    u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX),
                );
                let message = invocation.read_message().map_err(metal_runtime_error)?;
                self.state = RegistersValState::FirstJoined(invocation);
                Ok(message)
            }
            Some(challenge) => {
                if !matches!(
                    &self.state,
                    RegistersValState::FirstJoined(_)
                        | RegistersValState::Native(_)
                        | RegistersValState::Dense(_)
                ) {
                    return Err(metal_error(
                        "registers value bind has an invalid device state",
                    ));
                }
                let bound_lt_lo = self.cpu.metal_bind_offloaded(challenge)?;
                let state = std::mem::replace(&mut self.state, RegistersValState::Failed);
                match state {
                    RegistersValState::FirstJoined(invocation) => {
                        let transition = invocation
                            .into_first_transition(bound_lt_lo, self.first_transition)
                            .map_err(metal_runtime_error)?;
                        let _span = tracing::info_span!(
                            "MetalRegistersValEvaluation::native_transition",
                            source_elements = transition.source_cycles(),
                        )
                        .entered();
                        transition.execute(challenge).map_err(metal_runtime_error)?;
                        let message = transition.read_message().map_err(metal_runtime_error)?;
                        self.state = RegistersValState::Native(transition);
                        Ok(message)
                    }
                    RegistersValState::Native(invocation) => {
                        let mut sequence = invocation
                            .into_sequence(self.dense_transition)
                            .map_err(metal_runtime_error)?;
                        let source_elements = sequence.current_elements();
                        let _span = tracing::info_span!(
                            "MetalRegistersValEvaluation::dense_transition",
                            round,
                            source_elements,
                        )
                        .entered();
                        let message = sequence
                            .bind_and_message(challenge, bound_lt_lo)
                            .map_err(metal_runtime_error)?;
                        self.state = RegistersValState::Dense(sequence);
                        Ok(message)
                    }
                    RegistersValState::Dense(mut sequence) => {
                        let source_elements = sequence.current_elements();
                        let _span = tracing::info_span!(
                            "MetalRegistersValEvaluation::dense_transition",
                            round,
                            source_elements,
                        )
                        .entered();
                        let message = sequence
                            .bind_and_message(challenge, bound_lt_lo)
                            .map_err(metal_runtime_error)?;
                        self.state = RegistersValState::Dense(sequence);
                        Ok(message)
                    }
                    RegistersValState::Submitted(_)
                    | RegistersValState::CpuTail
                    | RegistersValState::Finished
                    | RegistersValState::Failed => Err(metal_error(
                        "registers value bind changed to an invalid device state",
                    )),
                }
            }
        }
    }
}

impl ProveRounds<AkitaField> for MetalRegistersValEvaluationKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.validate_round(bind.as_ref(), round)?;
        if self.should_restore() {
            self.restore_cpu_tail()?;
        }
        let polynomial = if matches!(&self.state, RegistersValState::CpuTail) {
            let _span =
                tracing::info_span!("MetalRegistersValEvaluation::cpu_tail", round).entered();
            self.cpu.prove_round(bind, round, previous_claim)?
        } else {
            let message = self.device_message(bind, round)?;
            self.cpu.metal_message(message, previous_claim)?
        };
        self.next_round += 1;
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.cpu.num_rounds()
            || matches!(
                &self.state,
                RegistersValState::Finished | RegistersValState::Failed
            )
        {
            return Err(metal_error(
                "registers value reached finish in an invalid state",
            ));
        }
        if matches!(
            &self.state,
            RegistersValState::Native(_) | RegistersValState::Dense(_)
        ) {
            self.restore_cpu_tail()?;
        }
        if !matches!(&self.state, RegistersValState::CpuTail) {
            return Err(metal_error(
                "registers value finish has no restored CPU tail",
            ));
        }
        let _span = tracing::info_span!("MetalRegistersValEvaluation::cpu_tail").entered();
        self.cpu.finish_rounds(bind)?;
        self.state = RegistersValState::Finished;
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalRegistersValEvaluationKernel {
    type Relation = RegistersValEvaluation<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>>
    {
        if !matches!(&self.state, RegistersValState::Finished) {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers value output requested before finish",
            });
        }
        self.cpu.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        if !matches!(&self.state, RegistersValState::Finished) {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers value derived table requested before finish",
            });
        }
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    metal_runtime_error(error).into()
}

fn metal_runtime_error(error: MetalError) -> SumcheckError<AkitaField> {
    metal_error(error.to_string())
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal registers-value parity setup")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    use jolt_field::{One as _, Ring as _};
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage5::registers_val_evaluation::{
        RegistersValEvaluationInputClaims, RegistersValEvaluationOutputClaims,
    };
    use jolt_witness::testing::with_sample_backend_at_geometry;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn production_kernel_matches_optimized_cpu_through_handoff() {
        for log_t in [9, 10, 11] {
            with_sample_backend_at_geometry(log_t, 13, 8, |witness| {
                let relation = RegistersValEvaluation::new(TraceDimensions::new(log_t));
                let registers_val_point = point(REGISTER_ADDRESS_BITS + log_t, 19);
                let table = JoltWitnessOracle::<AkitaField>::oracle_table(
                    witness,
                    JoltPolynomialId::Virtual(JoltVirtualPolynomial::RegistersVal),
                )
                .unwrap();
                let input_claim = Polynomial::new(table).evaluate(&registers_val_point);
                let claims = RegistersValEvaluationInputClaims {
                    registers_val: input_claim,
                };
                let points = RegistersValEvaluationInputClaims {
                    registers_val: registers_val_point,
                };
                let no_challenges = NoChallenges::default();
                let inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &no_challenges,
                };
                let mut expected = OptimizedRegistersValEvaluation
                    .prepare(&mut ProofSession::default(), witness, inputs())
                    .unwrap();
                let metal = MetalBackend::new(super::super::MetalConfig {
                    registers_val_evaluation: RegistersValEvaluationMetalConfig {
                        source: RegistersValEvaluationSource::WitnessUpload,
                        trace_cutoff_elements: 2,
                        cutoff_elements: 16,
                        first_message: RegistersValFirstMessageConfig {
                            threads_per_threadgroup: Some(32),
                        },
                        first_transition: RegistersValTransitionConfig {
                            threads_per_threadgroup: Some(32),
                        },
                        dense_transition: RegistersValDenseConfig {
                            threads_per_threadgroup: Some(64),
                        },
                    },
                    ..Default::default()
                })
                .unwrap();
                assert_eq!(metal.registers_val_sequences(), 0);
                let mut actual = <MetalBackend as PrepareKernel<
                    AkitaField,
                    RegistersValEvaluation<AkitaField>,
                >>::prepare(
                    &metal, &mut ProofSession::default(), witness, inputs()
                )
                .unwrap();
                assert_eq!(metal.registers_val_sequences(), 1);
                assert!(actual.output_claims(&claims).is_err());

                let challenges = point(log_t, 211);
                assert!(actual
                    .prove_round(Some(challenges[0]), 0, input_claim)
                    .is_err());
                assert!(actual.prove_round(None, 1, input_claim).is_err());
                let mut claim = input_claim;
                for round in 0..log_t {
                    let bind = round.checked_sub(1).map(|previous| challenges[previous]);
                    let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                    let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                    assert_eq!(actual_poly, expected_poly, "round {round}");
                    assert_eq!(
                        actual_poly.evaluate(AkitaField::zero())
                            + actual_poly.evaluate(AkitaField::one()),
                        claim,
                        "round {round} running claim",
                    );
                    claim = expected_poly.evaluate(challenges[round]);
                }
                let final_bind = challenges[log_t - 1];
                expected.finish_rounds(final_bind).unwrap();
                actual.finish_rounds(final_bind).unwrap();
                let output_points = relation
                    .derive_opening_points(&challenges, &points)
                    .unwrap();
                expected
                    .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                    .unwrap();
                actual
                    .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                    .unwrap();
                let expected_outputs: RegistersValEvaluationOutputClaims<AkitaField> =
                    expected.output_claims(&claims).unwrap();
                assert_eq!(actual.output_claims(&claims).unwrap(), expected_outputs);
                assert!(actual.finish_rounds(final_bind).is_err());

                let cpu_only = MetalBackend::new(super::super::MetalConfig {
                    registers_val_evaluation: RegistersValEvaluationMetalConfig {
                        trace_cutoff_elements: 2,
                        cutoff_elements: 1 << log_t,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .unwrap();
                let _ = <MetalBackend as PrepareKernel<
                    AkitaField,
                    RegistersValEvaluation<AkitaField>,
                >>::prepare(
                    &cpu_only, &mut ProofSession::default(), witness, inputs()
                )
                .unwrap();
                assert_eq!(cpu_only.registers_val_sequences(), 0);
            });
        }
    }
}
