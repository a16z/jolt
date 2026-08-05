use jolt_field::AkitaField;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::JoltWitnessPlane;

use super::instruction_read_raf::MetalBackend;
use super::solinas::{
    BooleanityRows, BytecodeCycleRowInputs, BytecodeCycleRowSequence, BytecodeCycleSequenceConfig,
    BytecodeCycleTablesMut,
};
use crate::optimized::bytecode_read_raf::{
    prepare_metal_bytecode_cycle_shell, BytecodeCycleAlgebra, BytecodeCycleDenseState, CycleKernel,
    OptimizedBytecodeReadRafCycle,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: BytecodeCycleSequenceConfig,
    pub cpu_tail_algebra: BytecodeCycleAlgebra,
}

impl Default for BytecodeReadRafMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            cutoff_elements: 1 << 16,
            dispatch: BytecodeCycleSequenceConfig::default(),
            cpu_tail_algebra: BytecodeCycleAlgebra::Q10Accum,
        }
    }
}

impl PrepareKernel<AkitaField, BytecodeReadRafCycle<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BytecodeReadRafCycle<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafCycle<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let config = self.config.bytecode_read_raf_cycle;
        let cpu_inputs = || ProverInputs {
            relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let fallback = |session: &mut ProofSession| {
            OptimizedBytecodeReadRafCycle::new(config.cpu_tail_algebra).prepare(
                session,
                witness,
                cpu_inputs(),
            )
        };

        if trace_elements < config.trace_cutoff_elements
            || config.cutoff_elements > trace_elements / 2
            || relation.degree() != 4
            || dimensions.num_committed_ra_polys() != 2
            || relation.committed_chunk_bits() != 8
        {
            return fallback(session);
        }
        let Some(rows) = session.state::<BooleanityRows>().cloned() else {
            return fallback(session);
        };
        if rows.len() != trace_elements || self.context.validate_booleanity_rows(&rows).is_err() {
            return fallback(session);
        }

        let (cpu, metadata) =
            prepare_metal_bytecode_cycle_shell(cpu_inputs(), config.cpu_tail_algebra)?;
        let sequence = self
            .context
            .prepare_bytecode_cycle_row_sequence(
                rows,
                BytecodeCycleRowInputs {
                    stage_points: &metadata.stage_points,
                    stage_weights: &metadata.stage_weights,
                    entry_weight: metadata.entry_weight,
                    ra0: &metadata.ra0,
                    ra1: &metadata.ra1,
                },
                config.dispatch,
            )
            .map_err(|error| metal_error(error.to_string()))?;
        Ok(Box::new(MetalBytecodeReadRafKernel::new(
            cpu,
            sequence,
            config.cutoff_elements,
        )))
    }
}

pub(crate) struct MetalBytecodeReadRafKernel {
    cpu: CycleKernel<AkitaField>,
    sequence: Option<BytecodeCycleRowSequence>,
    host_tail: Option<[Vec<AkitaField>; 5]>,
    cutoff_elements: usize,
}

impl MetalBytecodeReadRafKernel {
    fn new(
        cpu: CycleKernel<AkitaField>,
        sequence: BytecodeCycleRowSequence,
        cutoff_elements: usize,
    ) -> Self {
        Self {
            cpu,
            sequence: Some(sequence),
            host_tail: Some(std::array::from_fn(|_| {
                vec![AkitaField::zero(); cutoff_elements]
            })),
            cutoff_elements,
        }
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let sequence = self
            .sequence
            .take()
            .ok_or_else(|| metal_error("bytecode cycle sequence disappeared before readback"))?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "bytecode cycle CPU handoff requires materialized factor tables",
            ));
        }
        let elements = sequence.current_elements();
        let mut tables = self
            .host_tail
            .take()
            .ok_or_else(|| metal_error("bytecode cycle host tail was already consumed"))?;
        if elements > self.cutoff_elements {
            return Err(metal_error(
                "bytecode cycle readback exceeds the preallocated host tail",
            ));
        }
        let [combined, fused_combined, fused_inc, ra0, ra1] = &mut tables;
        sequence
            .read_current_tables(BytecodeCycleTablesMut {
                combined: &mut combined[..elements],
                fused_combined: &mut fused_combined[..elements],
                fused_inc: &mut fused_inc[..elements],
                ra0: &mut ra0[..elements],
                ra1: &mut ra1[..elements],
            })
            .map_err(|error| metal_error(error.to_string()))?;
        for table in &mut tables {
            table.truncate(elements);
        }
        let [combined, fused_combined, fused_inc, ra0, ra1] = tables;
        self.cpu.metal_restore_dense(BytecodeCycleDenseState {
            combined,
            fused_combined,
            fused_inc,
            ra0,
            ra1,
        })
    }
}

impl ProveRounds<AkitaField> for MetalBytecodeReadRafKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.sequence.as_ref().is_some_and(|sequence| {
            sequence.is_dense() && sequence.current_elements() <= self.cutoff_elements
        }) {
            self.restore_cpu_tail()?;
            return self.cpu.prove_round(bind, round, previous_claim);
        }

        if let Some(sequence) = self.sequence.as_mut() {
            let evals = match bind {
                None => sequence
                    .message()
                    .map_err(|error| metal_error(error.to_string()))?,
                Some(challenge) => {
                    let evals = sequence
                        .bind_and_message(challenge)
                        .map_err(|error| metal_error(error.to_string()))?;
                    self.cpu.metal_commit_bind(sequence.current_elements())?;
                    evals
                }
            };
            return self.cpu.metal_message(evals, previous_claim);
        }

        self.cpu.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalBytecodeReadRafKernel {
    type Relation = BytecodeReadRafCycle<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal bytecode parity setup")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, READ_RAF_CYCLE_STAGES,
    };

    use super::*;
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::harness::probe_input_claim;
    use crate::optimized::instruction_read_raf::{
        collect_instruction_cycle_rows, InstructionCycleRow,
    };

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn production_kernel_matches_optimized_cpu_through_handoff() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, _| {
            let dimensions = BytecodeReadRafDimensions::new(log_t, 13, 2);
            let relation = BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                dimensions,
                r_address: point(13, 19),
                stage_cycle_points: std::array::from_fn(|stage| point(log_t, 41 + stage as u64)),
                entry_bytecode_index: 17,
                committed_chunk_bits: 8,
                val_stages: (0..NUM_BYTECODE_VAL_STAGES)
                    .map(|stage| AkitaField::from_u64(101 + stage as u64))
                    .collect(),
            });
            assert_eq!(relation.stage_cycle_points().len(), READ_RAF_CYCLE_STAGES);
            let claims = BytecodeReadRafInputClaims::<AkitaField>::default();
            let points = BytecodeReadRafInputClaims::<Vec<AkitaField>>::default();
            let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let (mut shell, _) =
                prepare_metal_bytecode_cycle_shell(inputs(), BytecodeCycleAlgebra::Q10Accum)
                    .unwrap();
            assert_eq!(shell.metal_elements().unwrap(), 1 << log_t);
            assert_eq!(shell.metal_rounds_bound(), 0);
            let _ = shell
                .metal_message(
                    std::array::from_fn(|index| AkitaField::from_u64(7 + index as u64)),
                    AkitaField::zero(),
                )
                .unwrap();
            assert!(shell.metal_commit_bind((1 << log_t) / 4).is_err());
            shell.metal_commit_bind((1 << log_t) / 2).unwrap();
            assert_eq!(shell.metal_rounds_bound(), 1);
            assert_eq!(shell.metal_elements().unwrap(), (1 << log_t) / 2);

            let mut expected = OptimizedBytecodeReadRafCycle::new(BytecodeCycleAlgebra::Q10Accum)
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(super::super::MetalConfig {
                bytecode_read_raf_cycle: BytecodeReadRafMetalConfig {
                    trace_cutoff_elements: 2,
                    cutoff_elements: 4,
                    dispatch: BytecodeCycleSequenceConfig {
                        message_threads_per_threadgroup: Some(32),
                        transition_threads_per_threadgroup: Some(32),
                        max_threadgroups: 1 << 13,
                    },
                    cpu_tail_algebra: BytecodeCycleAlgebra::Q10Accum,
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let mut session = ProofSession::default();
            session.park(resident);
            let mut actual = <MetalBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafCycle<AkitaField>,
            >>::prepare(&metal, &mut session, witness, inputs())
            .unwrap();
            assert!(session.state::<BooleanityRows>().is_some());

            let claim = probe_input_claim(expected.as_mut());
            let mut round_challenges = point(log_t, 211);
            round_challenges[0] = AkitaField::zero();
            round_challenges[1] = AkitaField::one();
            round_challenges[2] = -AkitaField::one();
            let mut claim = claim;
            let mut nonzero_round = false;
            for round in 0..log_t {
                let bind = round
                    .checked_sub(1)
                    .map(|previous| round_challenges[previous]);
                let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                assert_eq!(actual_poly, expected_poly, "round {round}");
                nonzero_round |= expected_poly
                    .coefficients()
                    .iter()
                    .any(|coefficient| *coefficient != AkitaField::zero());
                claim = expected_poly.evaluate(round_challenges[round]);
            }
            assert!(nonzero_round, "all round polynomials were zero");
            let final_bind = round_challenges[log_t - 1];
            expected.finish_rounds(final_bind).unwrap();
            actual.finish_rounds(final_bind).unwrap();
            assert_eq!(
                actual.output_claims(&claims).unwrap(),
                expected.output_claims(&claims).unwrap()
            );
        });
    }
}
