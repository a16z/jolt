use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::ram_val_check_advice_opening;
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedClaimReduction,
    PrecommittedReductionLayout,
};
use jolt_field::Field;
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    TrustedAdviceCyclePhase, TrustedAdviceCyclePhaseOutputClaims, UntrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhaseOutputClaims,
};
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhase, TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhase,
    UntrustedAdviceAddressPhaseOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::common::context::CudaKernelContext;
use super::common::device::require_fr_slice;
use super::common::precommitted_reduction::{
    permuted_eq_point, reclaim_carry, DeviceAddressReductionKernel, DeviceCycleReductionKernel,
    DevicePrecommittedTables, DeviceRowPlan,
};
use super::{require_context, CudaBackend};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

fn advice_tables<F: Field>(
    context: &'static CudaKernelContext,
    witness: &dyn JoltWitnessPlane<F>,
    kind: JoltAdviceKind,
    reduction: &PrecommittedClaimReduction,
    r_val: &[F],
) -> Result<DevicePrecommittedTables<F>, KernelError<F>> {
    let num_vars = reduction.poly_opening_round_permutation_be().len();
    if r_val.len() != num_vars {
        return Err(KernelError::InvalidGeometry {
            reason: format!(
                "advice reference point has {} variables, schedule expects {num_vars}",
                r_val.len()
            ),
        });
    }
    let len = 1usize << num_vars;
    let table = witness.oracle_table(ram_val_check_advice_opening(kind).polynomial_id())?;
    if table.len() != len {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{kind:?} advice"),
            expected: len,
            got: table.len(),
        });
    }
    let value = context.upload(require_fr_slice(&table)?)?;
    let eq = context.eq_evals(&permuted_eq_point(reduction, r_val)?)?;
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
                permute: false,
            },
        ],
    )
}

fn advice_kernel<F: Field, R>(
    session_witness: &dyn JoltWitnessPlane<F>,
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
    r_val: &[F],
) -> Result<DeviceCycleReductionKernel<F, R>, KernelError<F>> {
    let context = require_context::<F>()?;
    let reduction = layout.precommitted().clone();
    let tables = advice_tables(context, session_witness, kind, &reduction, r_val)?;
    Ok(DeviceCycleReductionKernel::new(reduction, tables))
}

fn reference_point<F: Field>(
    point: Option<&[F]>,
    label: &'static str,
) -> Result<Vec<F>, KernelError<F>> {
    point
        .map(<[F]>::to_vec)
        .ok_or(KernelError::InvariantViolation { reason: label })
}

impl<F: Field> SumcheckKernel<F> for DeviceCycleReductionKernel<F, TrustedAdviceCyclePhase<F>> {
    type Relation = TrustedAdviceCyclePhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<TrustedAdviceCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(TrustedAdviceCyclePhaseOutputClaims {
            trusted: self.scalar_claim()?,
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<TrustedAdviceAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F> for DeviceCycleReductionKernel<F, UntrustedAdviceCyclePhase<F>> {
    type Relation = UntrustedAdviceCyclePhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<UntrustedAdviceCyclePhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(UntrustedAdviceCyclePhaseOutputClaims {
            untrusted: self.scalar_claim()?,
        })
    }

    fn park_residue(self: Box<Self>, session: &mut ProofSession) {
        self.park_carry::<UntrustedAdviceAddressPhase<F>>(session);
    }
}

impl<F: Field> SumcheckKernel<F> for DeviceAddressReductionKernel<F, TrustedAdviceAddressPhase<F>> {
    type Relation = TrustedAdviceAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<TrustedAdviceAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(TrustedAdviceAddressPhaseOutputClaims {
            trusted: self.final_claim()?,
        })
    }
}

impl<F: Field> SumcheckKernel<F>
    for DeviceAddressReductionKernel<F, UntrustedAdviceAddressPhase<F>>
{
    type Relation = UntrustedAdviceAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<UntrustedAdviceAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        Ok(UntrustedAdviceAddressPhaseOutputClaims {
            untrusted: self.final_claim()?,
        })
    }
}

impl<F: Field> PrepareKernel<F, TrustedAdviceCyclePhase<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, TrustedAdviceCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = TrustedAdviceCyclePhase<F>>>, KernelError<F>>
    {
        let r_val = reference_point(
            inputs.relation.reference_opening_point(),
            "trusted-advice cycle phase carries no reference opening point",
        )?;
        Ok(Box::new(advice_kernel::<F, TrustedAdviceCyclePhase<F>>(
            witness,
            JoltAdviceKind::Trusted,
            inputs.relation.layout(),
            &r_val,
        )?))
    }
}

impl<F: Field> PrepareKernel<F, UntrustedAdviceCyclePhase<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, UntrustedAdviceCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceCyclePhase<F>>>, KernelError<F>>
    {
        let r_val = reference_point(
            inputs.relation.reference_opening_point(),
            "untrusted-advice cycle phase carries no reference opening point",
        )?;
        Ok(Box::new(advice_kernel::<F, UntrustedAdviceCyclePhase<F>>(
            witness,
            JoltAdviceKind::Untrusted,
            inputs.relation.layout(),
            &r_val,
        )?))
    }
}

impl<F: Field> PrepareKernel<F, TrustedAdviceAddressPhase<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, TrustedAdviceAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = TrustedAdviceAddressPhase<F>>>, KernelError<F>>
    {
        Ok(Box::new(reclaim_carry::<F, TrustedAdviceAddressPhase<F>>(
            session,
            "trusted-advice address phase found no parked cycle-phase carry",
        )?))
    }
}

impl<F: Field> PrepareKernel<F, UntrustedAdviceAddressPhase<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, UntrustedAdviceAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = UntrustedAdviceAddressPhase<F>>>, KernelError<F>>
    {
        Ok(Box::new(
            reclaim_carry::<F, UntrustedAdviceAddressPhase<F>>(
                session,
                "untrusted-advice address phase found no parked cycle-phase carry",
            )?,
        ))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
        TrustedAdviceAddressPhaseInputClaims, TrustedAdviceCyclePhaseInputClaims,
        UntrustedAdviceAddressPhaseInputClaims, UntrustedAdviceCyclePhaseInputClaims,
    };
    use jolt_claims::protocols::jolt::{
        AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedReductionLayout,
        TracePolynomialOrder,
    };
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::Fr;
    use jolt_verifier::stages::PrecommittedSchedule;
    use proptest::prelude::*;

    use super::*;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        advice_plane, drive, precommitted_cycle_variables, precommitted_round_challenges,
        precommitted_synthetic_point, AdviceFixture,
    };
    use crate::reference::precommitted_reduction::ReferencePrecommittedAddress;
    use crate::reference::ReferenceBackend;

    const LOG_T: usize = 8;
    const LOG_K_CHUNK: usize = 4;
    const ADVICE_BYTES: usize = 4096;

    fn schedule() -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            LOG_T,
            LOG_K_CHUNK,
            Some(ADVICE_BYTES),
            Some(ADVICE_BYTES),
            None,
        )
        .expect("advice precommitted schedule")
    }

    fn layout(
        schedule: &PrecommittedSchedule,
        kind: JoltAdviceKind,
    ) -> &AdviceClaimReductionLayout {
        schedule.advice(kind).expect("advice layout present")
    }

    #[test]
    fn fixture_advice_geometry_exercises_both_phases_and_both_round_kinds() {
        let schedule = schedule();
        for kind in [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted] {
            let reduction = layout(&schedule, kind).precommitted();
            let vars = reduction.poly_opening_round_permutation_be().len();
            assert_eq!(
                1usize << vars,
                ADVICE_BYTES / 8,
                "{kind:?}: the advice table width must match the declared advice buffer",
            );
            assert!(
                reduction.cycle_phase_rounds().len() < reduction.cycle_phase_total_rounds(),
                "{kind:?}: every cycle round is active ({}/{}), so the inactive-round path — the \
                 constant claim/2 message and the running scale halving — is never exercised and a \
                 kernel that ignored `scale` would pass",
                reduction.cycle_phase_rounds().len(),
                reduction.cycle_phase_total_rounds(),
            );
            assert!(
                reduction.num_address_phase_rounds() > 0,
                "{kind:?}: no active address rounds, so the address-phase tests would be vacuous — \
                 that is the DEFAULT outcome at every gate scale and only fails to hold here \
                 because LOG_T is {LOG_T}",
            );
            assert!(
                reduction.address_phase_rounds().len() < reduction.address_phase_total_rounds(),
                "{kind:?}: every address round is active, so the address phase never exercises the \
                 inactive-round path",
            );
            assert_eq!(
                reduction.cycle_phase_rounds().len() + reduction.address_phase_rounds().len(),
                vars,
                "{kind:?}: the two phases' active rounds must bind every variable, else the final \
                 claim is taken from a partially bound table",
            );
        }
        let permutation = layout(&schedule, JoltAdviceKind::Trusted)
            .precommitted()
            .poly_opening_round_permutation_be()
            .to_vec();
        let identity: Vec<usize> = (0..permutation.len()).rev().collect();
        assert_ne!(
            permutation, identity,
            "the opening-round permutation is the identity relabeling, so a kernel that skipped \
             the coefficient permute would pass",
        );
    }

    #[test]
    fn fixture_advice_columns_discriminate_the_two_kinds() {
        let AdviceFixture {
            trusted, untrusted, ..
        } = advice_plane(ADVICE_BYTES, 7);
        assert_ne!(
            trusted, untrusted,
            "the two advice columns are equal, so a kernel reading the wrong kind would pass",
        );
        let varying = trusted.windows(2).filter(|pair| pair[0] != pair[1]).count();
        assert!(
            varying > trusted.len() / 2,
            "only {varying} of {} adjacent trusted-advice coefficients differ, so a kernel that \
             permuted coefficients wrongly could still pass",
            trusted.len() - 1,
        );
    }

    fn park_trusted_cycle(
        session: &mut ProofSession,
        plane: &dyn JoltWitnessPlane<Fr>,
        layout: &AdviceClaimReductionLayout,
        r_val: &[Fr],
        seed: u64,
        input_claim: Fr,
    ) {
        let relation = TrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val.to_vec()));
        let claims = TrustedAdviceCyclePhaseInputClaims::default();
        let points = TrustedAdviceCyclePhaseInputClaims::default();
        let challenges = NoChallenges::default();
        let mut kernel = ReferenceBackend
            .prepare(
                session,
                plane,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .expect("reference trusted-advice cycle prepare");
        let rounds = layout.precommitted().cycle_phase_total_rounds();
        let _ = drive(
            &mut *kernel,
            input_claim,
            &precommitted_round_challenges(rounds, seed),
        );
        let _ = kernel.output_claims(&claims).expect("cycle output claims");
        kernel.park_residue(session);
    }

    fn park_untrusted_cycle(
        session: &mut ProofSession,
        plane: &dyn JoltWitnessPlane<Fr>,
        layout: &AdviceClaimReductionLayout,
        r_val: &[Fr],
        seed: u64,
        input_claim: Fr,
    ) {
        let relation = UntrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val.to_vec()));
        let claims = UntrustedAdviceCyclePhaseInputClaims::default();
        let points = UntrustedAdviceCyclePhaseInputClaims::default();
        let challenges = NoChallenges::default();
        let mut kernel = ReferenceBackend
            .prepare(
                session,
                plane,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .expect("reference untrusted-advice cycle prepare");
        let rounds = layout.precommitted().cycle_phase_total_rounds();
        let _ = drive(
            &mut *kernel,
            input_claim,
            &precommitted_round_challenges(rounds, seed),
        );
        let _ = kernel.output_claims(&claims).expect("cycle output claims");
        kernel.park_residue(session);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]

        #[test]
        fn trusted_advice_cycle_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let AdviceFixture { plane, .. } = advice_plane(ADVICE_BYTES, seed);
            let schedule = schedule();
            let layout = layout(&schedule, JoltAdviceKind::Trusted);
            let reduction = layout.precommitted();
            let vars = reduction.poly_opening_round_permutation_be().len();
            let r_val = precommitted_synthetic_point(vars, seed);
            let input_claim = precommitted_synthetic_point(1, claim_seed)[0];
            let relation = TrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val));
            let claims = TrustedAdviceCyclePhaseInputClaims::default();
            let points = TrustedAdviceCyclePhaseInputClaims::default();
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
            prop_assert_eq!(got, expected, "round polynomials diverged");

            let expected_claims = expected_kernel
                .output_claims(&claims)
                .expect("reference claims");
            let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
            prop_assert_eq!(
                got_claims.opening_values(),
                expected_claims.opening_values(),
                "output claims diverged"
            );
        }

        #[test]
        fn untrusted_advice_cycle_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let AdviceFixture { plane, .. } = advice_plane(ADVICE_BYTES, seed);
            let schedule = schedule();
            let layout = layout(&schedule, JoltAdviceKind::Untrusted);
            let reduction = layout.precommitted();
            let vars = reduction.poly_opening_round_permutation_be().len();
            let r_val = precommitted_synthetic_point(vars, seed);
            let input_claim = precommitted_synthetic_point(1, claim_seed)[0];
            let relation = UntrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val));
            let claims = UntrustedAdviceCyclePhaseInputClaims::default();
            let points = UntrustedAdviceCyclePhaseInputClaims::default();
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
            prop_assert_eq!(got, expected, "round polynomials diverged");

            let expected_claims = expected_kernel
                .output_claims(&claims)
                .expect("reference claims");
            let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
            prop_assert_eq!(
                got_claims.opening_values(),
                expected_claims.opening_values(),
                "output claims diverged"
            );
        }

        #[test]
        fn trusted_advice_address_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let AdviceFixture { plane, .. } = advice_plane(ADVICE_BYTES, seed);
            let schedule = schedule();
            let layout = layout(&schedule, JoltAdviceKind::Trusted);
            let reduction = layout.precommitted();
            let vars = reduction.poly_opening_round_permutation_be().len();
            let r_val = precommitted_synthetic_point(vars, seed);
            let input_claim = precommitted_synthetic_point(1, claim_seed)[0];

            let mut expected_session = ProofSession::default();
            let mut got_session = ProofSession::default();
            park_trusted_cycle(&mut expected_session, &plane, layout, &r_val, seed, input_claim);
            park_trusted_cycle(&mut got_session, &plane, layout, &r_val, seed, input_claim);

            let relation = TrustedAdviceAddressPhase::<Fr>::new(
                layout,
                Some(r_val),
                precommitted_cycle_variables(reduction, seed),
            );
            let claims = TrustedAdviceAddressPhaseInputClaims::default();
            let points = TrustedAdviceAddressPhaseInputClaims::default();
            let challenge_set = NoChallenges::default();
            let make_inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };
            let challenges =
                precommitted_round_challenges(reduction.address_phase_total_rounds(), seed ^ 0xA5A5);

            let oracle = ReferencePrecommittedAddress::new("advice address carry missing");
            let mut expected_kernel = oracle
                .prepare(&mut expected_session, &plane, make_inputs())
                .expect("reference prepare");
            let mut got_kernel = CudaBackend
                .prepare(&mut got_session, &plane, make_inputs())
                .expect("cuda prepare");

            let expected = drive(&mut *expected_kernel, input_claim, &challenges);
            let got = drive(&mut *got_kernel, input_claim, &challenges);
            prop_assert_eq!(got, expected, "round polynomials diverged");

            let expected_claims = expected_kernel
                .output_claims(&claims)
                .expect("reference claims");
            let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
            prop_assert_eq!(
                got_claims.opening_values(),
                expected_claims.opening_values(),
                "output claims diverged"
            );
        }

        #[test]
        fn untrusted_advice_address_matches_reference_round_for_round(
            seed in any::<u64>(),
            claim_seed in any::<u64>(),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let AdviceFixture { plane, .. } = advice_plane(ADVICE_BYTES, seed);
            let schedule = schedule();
            let layout = layout(&schedule, JoltAdviceKind::Untrusted);
            let reduction = layout.precommitted();
            let vars = reduction.poly_opening_round_permutation_be().len();
            let r_val = precommitted_synthetic_point(vars, seed);
            let input_claim = precommitted_synthetic_point(1, claim_seed)[0];

            let mut expected_session = ProofSession::default();
            let mut got_session = ProofSession::default();
            park_untrusted_cycle(&mut expected_session, &plane, layout, &r_val, seed, input_claim);
            park_untrusted_cycle(&mut got_session, &plane, layout, &r_val, seed, input_claim);

            let relation = UntrustedAdviceAddressPhase::<Fr>::new(
                layout,
                Some(r_val),
                precommitted_cycle_variables(reduction, seed),
            );
            let claims = UntrustedAdviceAddressPhaseInputClaims::default();
            let points = UntrustedAdviceAddressPhaseInputClaims::default();
            let challenge_set = NoChallenges::default();
            let make_inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };
            let challenges =
                precommitted_round_challenges(reduction.address_phase_total_rounds(), seed ^ 0xA5A5);

            let oracle = ReferencePrecommittedAddress::new("advice address carry missing");
            let mut expected_kernel = oracle
                .prepare(&mut expected_session, &plane, make_inputs())
                .expect("reference prepare");
            let mut got_kernel = CudaBackend
                .prepare(&mut got_session, &plane, make_inputs())
                .expect("cuda prepare");

            let expected = drive(&mut *expected_kernel, input_claim, &challenges);
            let got = drive(&mut *got_kernel, input_claim, &challenges);
            prop_assert_eq!(got, expected, "round polynomials diverged");

            let expected_claims = expected_kernel
                .output_claims(&claims)
                .expect("reference claims");
            let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
            prop_assert_eq!(
                got_claims.opening_values(),
                expected_claims.opening_values(),
                "output claims diverged"
            );
        }
    }
}
