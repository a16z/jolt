use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::COMMITTED_BYTECODE_LANE_CAPACITY;
use jolt_claims::protocols::jolt::{
    BytecodeClaimReductionLayout, PrecommittedReductionLayout, TracePolynomialOrder,
};
use jolt_field::{Field, Fr};
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
    fold_chunk_weights, lane_cycle_eq, reclaim_carry, scatter_sparse_row,
    DeviceAddressReductionKernel, DeviceCycleReductionKernel, DevicePrecommittedTables,
    DeviceRowPlan,
};
use super::{require_context, CudaBackend};
use crate::committed_program::for_each_active_lane_value;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

struct SparseChunks {
    indices: Vec<Vec<u32>>,
    values: Vec<Vec<Fr>>,
}

fn sparse_chunk_rows<F: Field>(
    bytecode: &[JoltInstructionRow],
    chunk_count: usize,
    chunk_cycle_len: usize,
    order: TracePolynomialOrder,
) -> Result<SparseChunks, KernelError<F>> {
    let mut indices = vec![Vec::new(); chunk_count];
    let mut values: Vec<Vec<Fr>> = vec![Vec::new(); chunk_count];
    let mut overflow = None;
    for (cycle, instruction) in bytecode.iter().enumerate() {
        let chunk = cycle / chunk_cycle_len;
        let chunk_cycle = cycle % chunk_cycle_len;
        if chunk >= chunk_count {
            return Err(KernelError::InvalidGeometry {
                reason: format!(
                    "bytecode row {cycle} falls outside {chunk_count} chunks of {chunk_cycle_len}"
                ),
            });
        }
        for_each_active_lane_value::<Fr>(instruction, |lane, value| {
            let index = order.address_cycle_to_index(
                lane,
                chunk_cycle,
                COMMITTED_BYTECODE_LANE_CAPACITY,
                chunk_cycle_len,
            );
            match u32::try_from(index) {
                Ok(index) => {
                    indices[chunk].push(index);
                    values[chunk].push(value);
                }
                Err(_) => overflow = Some(index),
            }
        });
    }
    if let Some(index) = overflow {
        return Err(KernelError::InvalidGeometry {
            reason: format!("committed bytecode chunk index {index} exceeds a u32"),
        });
    }
    Ok(SparseChunks { indices, values })
}

fn bytecode_tables<F: Field>(
    context: &'static CudaKernelContext,
    layout: &BytecodeClaimReductionLayout,
    weights: &BytecodeReductionWeights<F>,
    bytecode: &[JoltInstructionRow],
) -> Result<DevicePrecommittedTables<F>, KernelError<F>> {
    let reduction = layout.precommitted();
    let num_vars = reduction.poly_opening_round_permutation_be().len();
    let len = 1usize << num_vars;
    let chunk_count = layout.chunk_count();
    let chunk_cycle_len = 1usize << layout.log_bytecode_chunk_size();
    if len != COMMITTED_BYTECODE_LANE_CAPACITY * chunk_cycle_len {
        return Err(KernelError::TableSizeMismatch {
            table: "committed bytecode chunk grid".to_owned(),
            expected: COMMITTED_BYTECODE_LANE_CAPACITY * chunk_cycle_len,
            got: len,
        });
    }
    if weights.chunk_rbc_weights.len() != chunk_count {
        return Err(KernelError::TableSizeMismatch {
            table: "bytecode chunk weights".to_owned(),
            expected: chunk_count,
            got: weights.chunk_rbc_weights.len(),
        });
    }
    if weights.lane_weights.len() < COMMITTED_BYTECODE_LANE_CAPACITY {
        return Err(KernelError::TableSizeMismatch {
            table: "bytecode lane weights".to_owned(),
            expected: COMMITTED_BYTECODE_LANE_CAPACITY,
            got: weights.lane_weights.len(),
        });
    }

    let sparse =
        sparse_chunk_rows::<F>(bytecode, chunk_count, chunk_cycle_len, layout.trace_order())?;
    let mut chunks = context.alloc(chunk_count * len)?;
    for chunk in 0..chunk_count {
        scatter_sparse_row(
            context,
            &sparse.indices[chunk],
            &sparse.values[chunk],
            len,
            chunk,
            &mut chunks,
        )?;
    }
    drop(sparse);

    let chunk_weights = context.upload(require_fr_slice(&weights.chunk_rbc_weights)?)?;
    let value = fold_chunk_weights(context, &chunks, &chunk_weights, chunk_count, len)?;
    let eq_cycle = context.eq_evals(require_fr_slice(&weights.r_bc)?)?;
    let lane_weights = context
        .upload(&require_fr_slice(&weights.lane_weights)?[..COMMITTED_BYTECODE_LANE_CAPACITY])?;
    let eq = lane_cycle_eq(
        context,
        &lane_weights,
        &eq_cycle,
        chunk_cycle_len,
        COMMITTED_BYTECODE_LANE_CAPACITY,
        layout.trace_order() == TracePolynomialOrder::CycleMajor,
        len,
    )?;

    let mut rows = Vec::with_capacity(2 + chunk_count);
    rows.push(DeviceRowPlan {
        source: &value,
        source_row: 0,
        permute: true,
    });
    rows.push(DeviceRowPlan {
        source: &eq,
        source_row: 0,
        permute: true,
    });
    for chunk in 0..chunk_count {
        rows.push(DeviceRowPlan {
            source: &chunks,
            source_row: chunk,
            permute: true,
        });
    }
    DevicePrecommittedTables::from_rows(context, reduction, len, &rows)
}

impl<F: Field> SumcheckKernel<F> for DeviceCycleReductionKernel<F, BytecodeReductionCyclePhase<F>> {
    type Relation = BytecodeReductionCyclePhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReductionCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(if self.has_address_phase() {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: Some(self.tables().intermediate_claim()?),
                chunks: Vec::new(),
            }
        } else {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: None,
                chunks: self.tables().final_aux_claims()?,
            }
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<BytecodeReductionAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F>
    for DeviceAddressReductionKernel<F, BytecodeReductionAddressPhase<F>>
{
    type Relation = BytecodeReductionAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReductionAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(BytecodeReductionAddressPhaseOutputClaims {
            chunks: self.final_aux_claims()?,
        })
    }
}

impl<F: Field> PrepareKernel<F, BytecodeReductionCyclePhase<F>> for CudaBackend {
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

impl<F: Field> PrepareKernel<F, BytecodeReductionAddressPhase<F>> for CudaBackend {
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
    use crate::committed_program::build_committed_bytecode_chunk_coeffs;
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

    const CONFIGS: [(usize, usize); 3] = [(8, 1), (8, 2), (16, 2)];

    fn schedule(log_t: usize, chunk_count: usize, bytecode_len: usize) -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            LOG_K_CHUNK,
            Some(4096),
            Some(4096),
            Some(CommittedProgramSchedule {
                bytecode_len,
                bytecode_chunk_count: chunk_count,
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
            r_bc: precommitted_synthetic_point(layout.log_bytecode_chunk_size(), seed ^ 0x31),
            chunk_rbc_weights: precommitted_synthetic_point(layout.chunk_count(), seed ^ 0x53),
            lane_weights: precommitted_synthetic_point(
                COMMITTED_BYTECODE_LANE_CAPACITY,
                seed ^ 0x71,
            ),
        }
    }

    #[test]
    fn fixture_bytecode_geometry_exercises_both_phases_and_both_permutation_branches() {
        let mut dominance = 0usize;
        let mut plain = 0usize;
        for (log_t, chunk_count) in CONFIGS {
            let fixture =
                committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, 7);
            let schedule = schedule(log_t, chunk_count, fixture.bytecode_len);
            let layout = layout(&schedule);
            let reduction = layout.precommitted();
            let vars = reduction.poly_opening_round_permutation_be().len();
            assert_eq!(
                1usize << vars,
                COMMITTED_BYTECODE_LANE_CAPACITY * (fixture.bytecode_len / chunk_count),
                "log_T {log_t} chunks {chunk_count}: the chunk grid width must match the layout",
            );
            assert!(
                reduction.num_address_phase_rounds() > 0,
                "log_T {log_t} chunks {chunk_count}: no active address rounds, so the \
                 address-phase test would be vacuous",
            );
            assert_eq!(
                reduction.cycle_phase_rounds().len() + reduction.address_phase_rounds().len(),
                vars,
                "log_T {log_t} chunks {chunk_count}: the two phases must bind every variable",
            );
            if log_t + LOG_K_CHUNK < vars {
                dominance += 1;
            } else {
                plain += 1;
            }
        }
        assert!(
            dominance > 0 && plain > 0,
            "the configs cover {dominance} dominance and {plain} plain cases; both permutation \
             branches must be exercised or a kernel that implemented only one would pass",
        );
    }

    #[test]
    fn fixture_bytecode_chunks_and_weights_discriminate() {
        let fixture = committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, 7);
        let schedule = schedule(8, 2, fixture.bytecode_len);
        let layout = layout(&schedule);
        let program = jolt_witness::ProgramSource::program_preprocessing(&fixture.plane);
        let chunks = build_committed_bytecode_chunk_coeffs::<Fr>(
            &program.bytecode.bytecode,
            layout.chunk_count(),
            layout.trace_order(),
        )
        .expect("chunk coefficients");
        assert_eq!(chunks.len(), 2, "this fixture is the two-chunk config");
        assert_ne!(
            chunks[0], chunks[1],
            "the two chunk grids are identical, so a fold that dropped one chunk, or applied the \
             wrong chunk weight, would still pass",
        );
        for (index, chunk) in chunks.iter().enumerate() {
            let nonzero = chunk
                .iter()
                .filter(|value| **value != Fr::from(0u64))
                .count();
            assert!(
                nonzero > 1,
                "chunk {index} has {nonzero} nonzero lane entries, so the lane-weight fold cannot \
                 discriminate",
            );
        }
        let weights = weights(layout, 7);
        assert_ne!(
            weights.chunk_rbc_weights[0], weights.chunk_rbc_weights[1],
            "the two chunk weights are equal, so swapping the chunks would still pass",
        );
        let varying_lanes = weights
            .lane_weights
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count();
        assert!(
            varying_lanes > weights.lane_weights.len() / 2,
            "only {varying_lanes} adjacent lane weights differ, so a kernel that mixed up lanes \
             could pass",
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]

        #[test]
        fn bytecode_reduction_cycle_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            for (log_t, chunk_count) in CONFIGS {
                let fixture =
                    committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, seed);
                let CommittedProgramFixture { plane, bytecode_len, .. } = fixture;
                let schedule = schedule(log_t, chunk_count, bytecode_len);
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
                    "round polynomials diverged at log_T {} chunks {}",
                    log_t,
                    chunk_count
                );

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged at log_T {} chunks {}",
                    log_t,
                    chunk_count
                );
            }
        }

        #[test]
        fn bytecode_reduction_address_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            for (log_t, chunk_count) in CONFIGS {
                let fixture =
                    committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, seed);
                let CommittedProgramFixture { plane, bytecode_len, .. } = fixture;
                let schedule = schedule(log_t, chunk_count, bytecode_len);
                let layout = layout(&schedule);
                let reduction = layout.precommitted();
                let input_claim = precommitted_synthetic_point(1, claim_seed)[0];
                let weights = weights(layout, seed);
                let eta = precommitted_synthetic_point(1, seed ^ 0x1F)[0];

                let mut expected_session = ProofSession::default();
                let mut got_session = ProofSession::default();
                for session in [&mut expected_session, &mut got_session] {
                    let cycle_relation =
                        BytecodeReductionCyclePhase::<Fr>::new(layout, weights.clone());
                    let cycle_claims = BytecodeReductionCyclePhaseInputClaims::default();
                    let cycle_points = BytecodeReductionCyclePhaseInputClaims::default();
                    let cycle_challenges = BytecodeReductionCyclePhaseChallenges { eta };
                    let mut cycle_kernel = ReferenceBackend
                        .prepare(
                            session,
                            &plane,
                            ProverInputs {
                                relation: &cycle_relation,
                                claims: &cycle_claims,
                                points: &cycle_points,
                                challenges: &cycle_challenges,
                            },
                        )
                        .expect("reference cycle prepare");
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
                    "round polynomials diverged at log_T {} chunks {}",
                    log_t,
                    chunk_count
                );

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged at log_T {} chunks {}",
                    log_t,
                    chunk_count
                );
            }
        }
    }
}
