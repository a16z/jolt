use jolt_field::Field;
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::CudaBackend;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[expect(
    clippy::todo,
    reason = "phase 1 stub: the equivalence test drives this until the kernels land"
)]
impl<F: Field> PrepareKernel<F, IncClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, IncClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = IncClaimReduction<F>>>, KernelError<F>> {
        todo!("CUDA increment claim-reduction kernel")
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::increments::{
        ram_inc_reduced, rd_inc_reduced,
    };
    use jolt_claims::protocols::jolt::relations::claim_reductions::increments::{
        IncClaimReductionChallenges, IncClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_r1cs_witness,
    };
    use crate::reference::views::dense_view;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    #[test]
    fn fixture_increment_columns_are_nonzero_and_differ() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let zero = Fr::from_u64(0);
            let ram_inc = dense_view::<Fr>(witness, ram_inc_reduced())
                .expect("the fixture serves the RAM increment column");
            let rd_inc = dense_view::<Fr>(witness, rd_inc_reduced())
                .expect("the fixture serves the register increment column");

            for (name, column) in [("ram_inc", &ram_inc), ("rd_inc", &rd_inc)] {
                assert!(
                    column.iter().any(|value| *value != zero),
                    "increment column {name} is zero at every cycle, so its whole gamma-weighted \
                     term drops out and a wrong eq pairing on it would still pass",
                );
                assert!(
                    column.contains(&zero),
                    "increment column {name} is nonzero at every cycle, so the fixture never \
                     exercises an idle cycle",
                );
            }
            assert!(
                ram_inc.iter().zip(rd_inc.iter()).any(|(ram, rd)| ram != rd),
                "the two increment columns are equal at every cycle, so swapping their eq weights \
                 would not change any round polynomial",
            );
        });
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn inc_claim_reduction_matches_reference_round_for_round(
            seed in any::<u64>(),
            ram_read_write_cycle in arb_point(LOG_T),
            ram_val_check_cycle in arb_point(LOG_T),
            registers_read_write_cycle in arb_point(LOG_T),
            registers_val_evaluation_cycle in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = IncClaimReduction::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    ram_read_write_cycle.clone(),
                    ram_val_check_cycle.clone(),
                    registers_read_write_cycle.clone(),
                    registers_val_evaluation_cycle.clone(),
                );
                let claims = IncClaimReductionInputClaims::default();
                let points = IncClaimReductionInputClaims::default();
                let challenge_set = IncClaimReductionChallenges { gamma };
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };

                let input_claim = reference_input_claim(witness, make_inputs);
                let mut expected_kernel = ReferenceBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("reference prepare");
                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("cuda prepare");

                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "round polynomials diverged");

                let expected_claims =
                    expected_kernel.output_claims(&claims).expect("reference claims");
                let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                prop_assert_eq!(
                    got_claims.opening_values(),
                    expected_claims.opening_values(),
                    "output claims diverged"
                );
                Ok(())
            })?;
        }
    }
}
