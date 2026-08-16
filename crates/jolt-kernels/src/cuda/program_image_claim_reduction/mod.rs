use jolt_claims::protocols::jolt::{PrecommittedReductionLayout, ProgramImageClaimReductionLayout};
use jolt_field::Field;
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    ProgramImageReductionCyclePhase, ProgramImageReductionCyclePhaseOutputClaims,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    ProgramImageReductionAddressPhase, ProgramImageReductionAddressPhaseOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::common::context::CudaKernelContext;
use super::common::device::require_fr_slice;
use super::common::precommitted_reduction::{
    reclaim_carry, shifted_block_eq, DeviceAddressReductionKernel, DeviceCycleReductionKernel,
    DevicePrecommittedTables, DeviceRowPlan,
};
use super::{require_context, CudaBackend};
use crate::committed_program::program_image_words_padded;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

fn program_image_tables<F: Field>(
    context: &'static CudaKernelContext,
    layout: &ProgramImageClaimReductionLayout,
    r_addr_rw: &[F],
    bytecode_words: &[u64],
) -> Result<DevicePrecommittedTables<F>, KernelError<F>> {
    let reduction = layout.precommitted();
    let num_vars = reduction.poly_opening_round_permutation_be().len();
    let words = program_image_words_padded(bytecode_words);
    let len = words.len();
    if len != 1usize << num_vars {
        return Err(KernelError::TableSizeMismatch {
            table: "program image words".to_owned(),
            expected: 1usize << num_vars,
            got: len,
        });
    }
    let ram_domain = 1usize << r_addr_rw.len();
    let start_index = layout.start_index();
    if start_index >= ram_domain || len > ram_domain {
        return Err(KernelError::InvalidGeometry {
            reason: format!(
                "program image block [{start_index}, +{len}) cannot index the RAM domain {ram_domain}"
            ),
        });
    }

    let host_value: Vec<F> = words.iter().map(|word| F::from_u64(*word)).collect();
    let value = context.upload(require_fr_slice(&host_value)?)?;
    let eq = shifted_block_eq(
        context,
        require_fr_slice(r_addr_rw)?,
        start_index,
        ram_domain - 1,
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

impl<F: Field> SumcheckKernel<F>
    for DeviceCycleReductionKernel<F, ProgramImageReductionCyclePhase<F>>
{
    type Relation = ProgramImageReductionCyclePhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<ProgramImageReductionCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: self.scalar_claim()?,
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<ProgramImageReductionAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F>
    for DeviceAddressReductionKernel<F, ProgramImageReductionAddressPhase<F>>
{
    type Relation = ProgramImageReductionAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<ProgramImageReductionAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(ProgramImageReductionAddressPhaseOutputClaims {
            program_image: self.final_claim()?,
        })
    }
}

impl<F: Field> PrepareKernel<F, ProgramImageReductionCyclePhase<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, ProgramImageReductionCyclePhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = ProgramImageReductionCyclePhase<F>>>,
        KernelError<F>,
    > {
        let context = require_context::<F>()?;
        let layout = inputs.relation.layout();
        let program = witness.program_preprocessing();
        let tables = program_image_tables(
            context,
            layout,
            inputs.relation.r_addr_rw(),
            &program.ram.bytecode_words,
        )?;
        Ok(Box::new(DeviceCycleReductionKernel::<
            F,
            ProgramImageReductionCyclePhase<F>,
        >::new(
            layout.precommitted().clone(), tables
        )))
    }
}

impl<F: Field> PrepareKernel<F, ProgramImageReductionAddressPhase<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, ProgramImageReductionAddressPhase<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = ProgramImageReductionAddressPhase<F>>>,
        KernelError<F>,
    > {
        Ok(Box::new(reclaim_carry::<
            F,
            ProgramImageReductionAddressPhase<F>,
        >(
            session,
            "program-image address phase found no parked cycle-phase carry",
        )?))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::{
        ProgramImageReductionAddressPhaseInputClaims, ProgramImageReductionCyclePhaseInputClaims,
    };
    use jolt_claims::protocols::jolt::{
        PrecommittedReductionLayout, ProgramImageClaimReductionLayout, TracePolynomialOrder,
    };
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::Fr;
    use jolt_verifier::stages::{CommittedProgramSchedule, PrecommittedSchedule};
    use proptest::prelude::*;

    use super::*;
    use crate::committed_program::program_image_words_padded;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        committed_program_plane, drive, precommitted_cycle_variables,
        precommitted_round_challenges, precommitted_synthetic_point, CommittedProgramFixture,
    };
    use crate::reference::precommitted_reduction::ReferencePrecommittedAddress;
    use crate::reference::ReferenceBackend;

    const LOG_K_CHUNK: usize = 4;
    const BYTECODE_ROWS: usize = 50;
    const BYTECODE_CHUNKS: usize = 2;
    const IMAGE_WORDS: usize = 1024;
    const MIN_BYTECODE_ADDRESS: u64 = 0x8000_0000;
    const RAM_LOG_K: usize = 12;

    const CONFIGS: [(usize, usize); 2] = [(8, 37), (8, (1 << RAM_LOG_K) - 512)];

    fn schedule(log_t: usize, start_index: usize, bytecode_len: usize) -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            LOG_K_CHUNK,
            Some(4096),
            Some(4096),
            Some(CommittedProgramSchedule {
                bytecode_len,
                bytecode_chunk_count: BYTECODE_CHUNKS,
                program_image_len_words: IMAGE_WORDS,
                program_image_start_index: start_index,
            }),
        )
        .expect("committed precommitted schedule")
    }

    fn layout(schedule: &PrecommittedSchedule) -> &ProgramImageClaimReductionLayout {
        schedule
            .program_image
            .as_ref()
            .expect("program image layout present")
    }

    #[test]
    fn fixture_program_image_geometry_covers_the_address_phase_and_the_wrapping_block() {
        let fixture = committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, 7);
        let mut wrapping = 0usize;
        for (log_t, start_index) in CONFIGS {
            let schedule = schedule(log_t, start_index, fixture.bytecode_len);
            let layout = layout(&schedule);
            let reduction = layout.precommitted();
            let vars = reduction.poly_opening_round_permutation_be().len();
            assert_eq!(
                1usize << vars,
                program_image_words_padded(&fixture.image_words).len(),
                "log_T {log_t} start {start_index}: the image table width must match the layout",
            );
            assert_ne!(
                start_index, 0,
                "a zero start index makes the shifted eq slice the plain eq table, so a kernel \
                 that ignored the shift would pass",
            );
            assert!(
                reduction.num_address_phase_rounds() > 0,
                "log_T {log_t} start {start_index}: no active address rounds, so the address-phase \
                 test would be vacuous",
            );
            assert_eq!(
                reduction.cycle_phase_rounds().len() + reduction.address_phase_rounds().len(),
                vars,
                "log_T {log_t} start {start_index}: the two phases must bind every variable",
            );
            if start_index + (1usize << vars) > (1usize << RAM_LOG_K) {
                wrapping += 1;
            }
        }
        assert!(
            wrapping > 0,
            "no config puts the padded image block across the top of the RAM domain, so the \
             wrapping branch of the shifted eq slice is untested",
        );
    }

    #[test]
    fn fixture_program_image_words_discriminate() {
        let fixture = committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, 7);
        let words = program_image_words_padded(&fixture.image_words);
        assert_eq!(
            words.len(),
            IMAGE_WORDS,
            "the fixture image is already padded"
        );
        let varying = words.windows(2).filter(|pair| pair[0] != pair[1]).count();
        assert!(
            varying > words.len() / 2,
            "only {varying} adjacent image words differ, so a kernel that permuted the image \
             wrongly could pass",
        );
        assert!(
            words.iter().any(|word| *word != 0),
            "the image is all zeros, so any value table would pass",
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2))]

        #[test]
        fn program_image_reduction_cycle_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            for (log_t, start_index) in CONFIGS {
                let CommittedProgramFixture { plane, bytecode_len, .. } =
                    committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, seed);
                let schedule = schedule(log_t, start_index, bytecode_len);
                let layout = layout(&schedule);
                let reduction = layout.precommitted();
                let input_claim = precommitted_synthetic_point(1, claim_seed)[0];
                let r_addr_rw = precommitted_synthetic_point(RAM_LOG_K, seed ^ 0x13);
                let relation = ProgramImageReductionCyclePhase::<Fr>::new(layout, r_addr_rw);
                let claims = ProgramImageReductionCyclePhaseInputClaims::default();
                let points = ProgramImageReductionCyclePhaseInputClaims::default();
                let challenge_set = NoChallenges::default();
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
                    "round polynomials diverged at log_T {} start {}",
                    log_t,
                    start_index
                );

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged at log_T {} start {}",
                    log_t,
                    start_index
                );
            }
        }

        #[test]
        fn program_image_reduction_address_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            for (log_t, start_index) in CONFIGS {
                let CommittedProgramFixture { plane, bytecode_len, .. } =
                    committed_program_plane(BYTECODE_ROWS, IMAGE_WORDS, MIN_BYTECODE_ADDRESS, seed);
                let schedule = schedule(log_t, start_index, bytecode_len);
                let layout = layout(&schedule);
                let reduction = layout.precommitted();
                let input_claim = precommitted_synthetic_point(1, claim_seed)[0];
                let r_addr_rw = precommitted_synthetic_point(RAM_LOG_K, seed ^ 0x13);

                let mut expected_session = ProofSession::default();
                let mut got_session = ProofSession::default();
                for session in [&mut expected_session, &mut got_session] {
                    let cycle_relation =
                        ProgramImageReductionCyclePhase::<Fr>::new(layout, r_addr_rw.clone());
                    let cycle_claims = ProgramImageReductionCyclePhaseInputClaims::default();
                    let cycle_points = ProgramImageReductionCyclePhaseInputClaims::default();
                    let cycle_challenges = NoChallenges::default();
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

                let relation = ProgramImageReductionAddressPhase::<Fr>::new(
                    layout,
                    Some(r_addr_rw),
                    precommitted_cycle_variables(reduction, seed),
                );
                let claims = ProgramImageReductionAddressPhaseInputClaims::default();
                let points = ProgramImageReductionAddressPhaseInputClaims::default();
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
                    ReferencePrecommittedAddress::new("program image address carry missing");
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
                    "round polynomials diverged at log_T {} start {}",
                    log_t,
                    start_index
                );

                let expected_claims = expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged at log_T {} start {}",
                    log_t,
                    start_index
                );
            }
        }
    }
}
