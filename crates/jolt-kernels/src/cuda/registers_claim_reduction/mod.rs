use jolt_field::Field;
use jolt_verifier::stages::stage3::outputs::RegistersClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::CudaBackend;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[expect(
    clippy::todo,
    reason = "phase 1 stub: the equivalence test must fail here until the kernels land, and \
              the expectation becomes an error the moment they do"
)]
impl<F: Field> PrepareKernel<F, RegistersClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, RegistersClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersClaimReduction<F>>>, KernelError<F>>
    {
        todo!("CUDA registers claim-reduction kernel")
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::registers::{
        rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced,
    };
    use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
        RegistersClaimReductionChallenges, RegistersClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltOpeningId, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage3::outputs::RegistersClaimReduction;
    use jolt_witness::JoltWitnessPlane;
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

    fn column_ids() -> [JoltOpeningId; 3] {
        [
            rd_write_value_reduced(),
            rs1_value_reduced(),
            rs2_value_reduced(),
        ]
    }

    #[test]
    fn fixture_registers_claim_reduction_columns_vary() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let plane: &dyn JoltWitnessPlane<Fr> = witness;
            let zero = Fr::from_u64(0);
            for id in column_ids() {
                let column = dense_view::<Fr>(plane, id).expect("the fixture serves the column");
                assert!(
                    column.iter().any(|value| *value != zero),
                    "{id:?} is zero at every cycle, so its gamma power could be anything",
                );
                assert!(
                    column.iter().any(|value| *value != column[0]),
                    "{id:?} is constant across the fixture, so a mis-indexed read would pass",
                );
            }
        });
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn registers_claim_reduction_matches_reference_round_for_round(
            seed in any::<u64>(),
            product_uniskip_tau_low in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = RegistersClaimReduction::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    product_uniskip_tau_low.clone(),
                );
                let claims = RegistersClaimReductionInputClaims::default();
                let points = RegistersClaimReductionInputClaims::default();
                let challenge_set = RegistersClaimReductionChallenges { gamma };
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
                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let expected_claims =
                    expected_kernel.output_claims(&claims).expect("reference claims");

                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("cuda prepare");
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "round polynomials diverged");

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
