use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::COMMITTED_BYTECODE_LANE_CAPACITY;
use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::BytecodeReductionIntermediateClaims;
use jolt_claims::protocols::jolt::{
    BytecodeClaimReductionLayout, PrecommittedReductionLayout, TracePolynomialOrder,
};
use jolt_field::JoltField;
use jolt_riscv::JoltInstructionRow;
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, BytecodeReductionCyclePhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, BytecodeReductionAddressPhaseOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::common::context::CudaKernelContext;
use super::common::device::require_fr_slice;
use super::common::precommitted_reduction::{
    lane_cycle_eq, reclaim_carry, DeviceAddressReductionKernel, DeviceCycleReductionKernel,
    DevicePrecommittedTables, DeviceRowPlan,
};
use super::{require_context, CudaBackend};
use crate::committed_program::build_committed_bytecode_coeffs;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

fn bytecode_tables<F: JoltField>(
    context: &'static CudaKernelContext,
    layout: &BytecodeClaimReductionLayout,
    weights: &BytecodeReductionWeights<F>,
    bytecode: &[JoltInstructionRow],
) -> Result<DevicePrecommittedTables<F>, KernelError<F>> {
    let reduction = layout.precommitted();
    let len = 1usize << reduction.poly_opening_round_permutation_be().len();
    let cycle_len = 1usize << layout.log_rows();
    if len != COMMITTED_BYTECODE_LANE_CAPACITY * cycle_len {
        return Err(KernelError::TableSizeMismatch {
            table: "committed bytecode grid".to_owned(),
            expected: COMMITTED_BYTECODE_LANE_CAPACITY * cycle_len,
            got: len,
        });
    }
    if weights.lane_weights.len() < COMMITTED_BYTECODE_LANE_CAPACITY {
        return Err(KernelError::TableSizeMismatch {
            table: "bytecode lane weights".to_owned(),
            expected: COMMITTED_BYTECODE_LANE_CAPACITY,
            got: weights.lane_weights.len(),
        });
    }
    let coefficients = build_committed_bytecode_coeffs::<F>(bytecode, layout.trace_order())?;
    let value = context.upload(require_fr_slice(&coefficients)?)?;
    let eq_cycle = context.eq_evals(require_fr_slice(&weights.r_bc)?)?;
    let lane_weights = context
        .upload(&require_fr_slice(&weights.lane_weights)?[..COMMITTED_BYTECODE_LANE_CAPACITY])?;
    let eq = lane_cycle_eq(
        context,
        &lane_weights,
        &eq_cycle,
        cycle_len,
        COMMITTED_BYTECODE_LANE_CAPACITY,
        layout.trace_order() == TracePolynomialOrder::CycleMajor,
        len,
    )?;
    DevicePrecommittedTables::from_rows(
        context,
        reduction,
        len,
        &[
            DeviceRowPlan {
                source: &value,
                source_row: 0,
                permute: true,
            },
            DeviceRowPlan {
                source: &eq,
                source_row: 0,
                permute: true,
            },
        ],
    )
}

impl<F: JoltField> SumcheckKernel<F>
    for DeviceCycleReductionKernel<F, BytecodeReductionCyclePhase<F>>
{
    type Relation = BytecodeReductionCyclePhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReductionCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(if self.has_address_phase() {
            BytecodeReductionCyclePhaseOutputClaims::Intermediate(
                BytecodeReductionIntermediateClaims {
                    intermediate: self.tables().intermediate_claim()?,
                },
            )
        } else {
            BytecodeReductionCyclePhaseOutputClaims::Final(
                BytecodeReductionAddressPhaseOutputClaims {
                    bytecode: self.tables().final_claim()?,
                },
            )
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<BytecodeReductionAddressPhase<F>>(session);
    }
}

impl<F: JoltField> SumcheckKernel<F>
    for DeviceAddressReductionKernel<F, BytecodeReductionAddressPhase<F>>
{
    type Relation = BytecodeReductionAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReductionAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(BytecodeReductionAddressPhaseOutputClaims {
            bytecode: self.final_claim()?,
        })
    }
}

impl<F: JoltField> PrepareKernel<F, BytecodeReductionCyclePhase<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReductionCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReductionCyclePhase<F>>>, KernelError<F>>
    {
        let context = require_context::<F>()?;
        let layout = inputs.relation.layout();
        let program = witness.program_preprocessing();
        let tables = bytecode_tables(
            context,
            layout,
            inputs.relation.weights(),
            &program.bytecode.bytecode,
        )?;
        Ok(Box::new(DeviceCycleReductionKernel::<
            F,
            BytecodeReductionCyclePhase<F>,
        >::new(
            layout.precommitted().clone(), tables
        )))
    }
}

impl<F: JoltField> PrepareKernel<F, BytecodeReductionAddressPhase<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, BytecodeReductionAddressPhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = BytecodeReductionAddressPhase<F>>>,
        KernelError<F>,
    > {
        Ok(Box::new(reclaim_carry::<
            F,
            BytecodeReductionAddressPhase<F>,
        >(
            session,
            "committed-bytecode address phase found no parked cycle-phase carry",
        )?))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::COMMITTED_BYTECODE_LANE_CAPACITY;
    use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::{
        BytecodeReductionAddressPhaseInputClaims, BytecodeReductionCyclePhaseChallenges,
        BytecodeReductionCyclePhaseInputClaims,
    };
    use jolt_claims::protocols::jolt::{
        BytecodeClaimReductionLayout, PrecommittedReductionLayout, TracePolynomialOrder,
    };
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::Fr;
    use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;
    use jolt_verifier::stages::{CommittedProgramSchedule, PrecommittedSchedule};
    use proptest::prelude::*;

    use super::*;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        committed_program_plane, drive, precommitted_cycle_variables,
        precommitted_round_challenges, precommitted_synthetic_point, CommittedProgramFixture,
    };
    use crate::reference::precommitted_reduction::ReferencePrecommittedAddress;
    use crate::reference::ReferenceBackend;

    const LOG_K_CHUNK: usize = 4;
    const BYTECODE_ROWS: usize = 50;
    const IMAGE_WORDS: usize = 1024;
    const MIN_BYTECODE_ADDRESS: u64 = 0x8000_0000;

    const CONFIGS: [usize; 2] = [8, 16];

    fn schedule(log_t: usize, bytecode_len: usize) -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            LOG_K_CHUNK,
            Some(4096),
            Some(4096),
            Some(CommittedProgramSchedule {
                bytecode_len,
                program_image_len_words: IMAGE_WORDS,
                program_image_start_index: 37,
            }),
        )
        .expect("committed precommitted schedule")
    }

    fn layout(schedule: &PrecommittedSchedule) -> &BytecodeClaimReductionLayout {
        schedule
            .bytecode
            .as_ref()
            .expect("committed bytecode layout present")
    }

    fn weights(layout: &BytecodeClaimReductionLayout, seed: u64) -> BytecodeReductionWeights<Fr> {
        BytecodeReductionWeights {
            r_bc: precommitted_synthetic_point(layout.log_rows(), seed ^ 0x31),
            lane_weights: precommitted_synthetic_point(
                COMMITTED_BYTECODE_LANE_CAPACITY,
                seed ^ 0x71,
            ),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]

        #[test]
        fn bytecode_reduction_cycle_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            for log_t in CONFIGS {
                let fixture =
                    committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, seed);
                let CommittedProgramFixture { plane, bytecode_len, .. } = fixture;
                let schedule = schedule(log_t, bytecode_len);
                let layout = layout(&schedule);
                let reduction = layout.precommitted();
                let input_claim = precommitted_synthetic_point(1, claim_seed)[0];
                let relation =
                    BytecodeReductionCyclePhase::<Fr>::new(layout, weights(layout, seed));
                let claims = BytecodeReductionCyclePhaseInputClaims::default();
                let points = BytecodeReductionCyclePhaseInputClaims::default();
                let challenge_set = BytecodeReductionCyclePhaseChallenges {
                    eta: precommitted_synthetic_point(1, seed ^ 0x1F)[0],
                };
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };
                let challenges =
                    precommitted_round_challenges(reduction.cycle_phase_total_rounds(), seed);

                let mut expected_kernel = ReferenceBackend
                    .prepare(&mut ProofSession::default(), &plane, make_inputs())
                    .expect("reference prepare");
                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), &plane, make_inputs())
                    .expect("cuda prepare");

                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(
                    got,
                    expected,
                    "round polynomials diverged at log_T {}",
                    log_t
                );

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged at log_T {}",
                    log_t
                );
            }
        }

        #[test]
        fn bytecode_reduction_address_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            for log_t in CONFIGS {
                let fixture =
                    committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, seed);
                let CommittedProgramFixture { plane, bytecode_len, .. } = fixture;
                let schedule = schedule(log_t, bytecode_len);
                let layout = layout(&schedule);
                let reduction = layout.precommitted();
                let input_claim = precommitted_synthetic_point(1, claim_seed)[0];
                let weights = weights(layout, seed);
                let eta = precommitted_synthetic_point(1, seed ^ 0x1F)[0];

                let mut expected_session = ProofSession::default();
                let mut got_session = ProofSession::default();
                for (cuda, session) in [(false, &mut expected_session), (true, &mut got_session)] {
                    let cycle_relation =
                        BytecodeReductionCyclePhase::<Fr>::new(layout, weights.clone());
                    let cycle_claims = BytecodeReductionCyclePhaseInputClaims::default();
                    let cycle_points = BytecodeReductionCyclePhaseInputClaims::default();
                    let cycle_challenges = BytecodeReductionCyclePhaseChallenges { eta };
                    let cycle_inputs = ProverInputs {
                        relation: &cycle_relation, claims: &cycle_claims,
                        points: &cycle_points, challenges: &cycle_challenges,
                    };
                    let mut cycle_kernel = if cuda {
                        CudaBackend.prepare(session, &plane, cycle_inputs)
                    } else {
                        ReferenceBackend.prepare(session, &plane, cycle_inputs)
                    }.expect("cycle prepare");
                    let _ = drive(
                        &mut *cycle_kernel,
                        input_claim,
                        &precommitted_round_challenges(reduction.cycle_phase_total_rounds(), seed),
                    );
                    let _ = cycle_kernel
                        .output_claims(&cycle_claims)
                        .expect("cycle output claims");
                    cycle_kernel.park_residue(session);
                }

                let relation = BytecodeReductionAddressPhase::<Fr>::new(
                    layout,
                    Some(weights),
                    precommitted_cycle_variables(reduction, seed),
                );
                let claims = BytecodeReductionAddressPhaseInputClaims::default();
                let points = BytecodeReductionAddressPhaseInputClaims::default();
                let challenge_set = NoChallenges::default();
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };
                let challenges = precommitted_round_challenges(
                    reduction.address_phase_total_rounds(),
                    seed ^ 0xA5A5,
                );

                let oracle =
                    ReferencePrecommittedAddress::new("bytecode reduction address carry missing");
                let mut expected_kernel = oracle
                    .prepare(&mut expected_session, &plane, make_inputs())
                    .expect("reference prepare");
                let mut got_kernel = CudaBackend
                    .prepare(&mut got_session, &plane, make_inputs())
                    .expect("cuda prepare");

                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(
                    got,
                    expected,
                    "round polynomials diverged at log_T {}",
                    log_t
                );

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged at log_T {}",
                    log_t
                );
            }
        }
    }
}
