use std::mem::size_of;

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::registers::rd_inc_val_evaluation;
use jolt_field::AkitaField;
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
    MetalError, RegistersValDenseConfig, RegistersValFirstMessageConfig,
    RegistersValFirstMessageInvocation, RegistersValFirstTransitionInvocation,
    RegistersValSequence, RegistersValTransitionConfig,
};
use crate::optimized::registers_read_write::{RegisterCycleRow, SharedRdIndices};
use crate::optimized::registers_val_evaluation::{
    OptimizedRegistersValEvaluation, ValEvaluationKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValEvaluationMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub first_message: RegistersValFirstMessageConfig,
    pub first_transition: RegistersValTransitionConfig,
    pub dense_transition: RegistersValDenseConfig,
}

impl Default for RegistersValEvaluationMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            cutoff_elements: 1 << 16,
            first_message: RegistersValFirstMessageConfig::default(),
            first_transition: RegistersValTransitionConfig::default(),
            dense_transition: RegistersValDenseConfig::default(),
        }
    }
}

enum RegistersValDeviceState {
    First(RegistersValFirstMessageInvocation),
    Native(RegistersValFirstTransitionInvocation),
    Dense(RegistersValSequence),
}

impl RegistersValDeviceState {
    fn current_elements(&self) -> usize {
        match self {
            Self::First(invocation) => invocation.cycles(),
            Self::Native(invocation) => invocation.current_elements(),
            Self::Dense(sequence) => sequence.current_elements(),
        }
    }
}

struct MetalRegistersValEvaluationKernel {
    cpu: ValEvaluationKernel<AkitaField>,
    device: Option<RegistersValDeviceState>,
    host_tail: Vec<[AkitaField; 2]>,
    cutoff_elements: usize,
    first_transition: RegistersValTransitionConfig,
    dense_transition: RegistersValDenseConfig,
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
        if cycles < config.trace_cutoff_elements || cycles <= config.cutoff_elements {
            return OptimizedRegistersValEvaluation.prepare(session, witness, inputs);
        }
        let point = &inputs.points.registers_val;
        if point.len() != REGISTER_ADDRESS_BITS + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = point.split_at(REGISTER_ADDRESS_BITS);
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
        let rd_device = rd
            .iter()
            .map(|index| index.unwrap_or(u8::MAX))
            .collect::<Vec<_>>();
        let split_bits = r_cycle.len() / 2;
        let split_handoff = cycles >> split_bits.saturating_sub(1);
        let cutoff_elements = config.cutoff_elements.max(split_handoff);

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
        drop(inc);
        drop(rd);
        drop(rd_device);
        let cpu = ValEvaluationKernel::new_offloaded(r_cycle);
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .registers_val_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Box::new(MetalRegistersValEvaluationKernel {
            cpu,
            device: Some(RegistersValDeviceState::First(invocation)),
            host_tail: vec![[AkitaField::zero(); 2]; cutoff_elements],
            cutoff_elements,
            first_transition: config.first_transition,
            dense_transition: config.dense_transition,
        }))
    }
}

impl MetalRegistersValEvaluationKernel {
    fn should_restore(&self) -> bool {
        self.device.as_ref().is_some_and(|device| {
            !matches!(device, RegistersValDeviceState::First(_))
                && device.current_elements() <= self.cutoff_elements
        })
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let device = self.device.take().ok_or_else(|| {
            metal_error("registers value device state disappeared before readback")
        })?;
        let elements = device.current_elements();
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
        match device {
            RegistersValDeviceState::First(_) => {
                return Err(metal_error(
                    "registers value cannot restore before its first bind",
                ));
            }
            RegistersValDeviceState::Native(invocation) => invocation
                .read_dense_state_into(rows)
                .map_err(metal_runtime_error)?,
            RegistersValDeviceState::Dense(sequence) => sequence
                .read_current_dense_state_into(rows)
                .map_err(metal_runtime_error)?,
        }
        self.cpu.metal_restore_dense(rows)
    }

    fn device_message(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
    ) -> Result<[AkitaField; 3], SumcheckError<AkitaField>> {
        match bind {
            None => {
                if round != 0 {
                    return Err(metal_error(
                        "registers value received an unbound noninitial round",
                    ));
                }
                let Some(RegistersValDeviceState::First(invocation)) = self.device.as_ref() else {
                    return Err(metal_error(
                        "registers value first message has an invalid device state",
                    ));
                };
                let _span = tracing::info_span!(
                    "MetalRegistersValEvaluation::first_message",
                    source_elements = invocation.cycles(),
                )
                .entered();
                invocation.execute().map_err(metal_runtime_error)?;
                invocation.read_message().map_err(metal_runtime_error)
            }
            Some(challenge) => {
                let bound_lt_lo = self.cpu.metal_bind_offloaded(challenge)?;
                let device = self.device.take().ok_or_else(|| {
                    metal_error("registers value device state disappeared before bind")
                })?;
                match device {
                    RegistersValDeviceState::First(invocation) => {
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
                        self.device = Some(RegistersValDeviceState::Native(transition));
                        Ok(message)
                    }
                    RegistersValDeviceState::Native(invocation) => {
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
                        self.device = Some(RegistersValDeviceState::Dense(sequence));
                        Ok(message)
                    }
                    RegistersValDeviceState::Dense(mut sequence) => {
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
                        self.device = Some(RegistersValDeviceState::Dense(sequence));
                        Ok(message)
                    }
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
        if self.should_restore() {
            self.restore_cpu_tail()?;
        }
        if self.device.is_some() {
            let message = self.device_message(bind, round)?;
            return self.cpu.metal_message(message, previous_claim);
        }
        let _span = tracing::info_span!("MetalRegistersValEvaluation::cpu_tail", round).entered();
        self.cpu.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.device.is_some() {
            self.restore_cpu_tail()?;
        }
        let _span = tracing::info_span!("MetalRegistersValEvaluation::cpu_tail").entered();
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalRegistersValEvaluationKernel {
    type Relation = RegistersValEvaluation<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
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
        let log_t = 10;
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

            let challenges = point(log_t, 211);
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
        });
    }
}
