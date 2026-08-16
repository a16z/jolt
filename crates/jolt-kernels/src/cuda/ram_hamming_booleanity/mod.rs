use jolt_field::Field;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_witness::JoltWitnessPlane;

use super::CudaBackend;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[expect(
    clippy::todo,
    reason = "phase 1 stub: the equivalence test drives this until the kernels land"
)]
impl<F: Field> PrepareKernel<F, RamHammingBooleanity<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, RamHammingBooleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamHammingBooleanity<F>>>, KernelError<F>>
    {
        todo!("CUDA RAM Hamming-booleanity kernel")
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::ram::ram_hamming_weight;
    use jolt_claims::protocols::jolt::relations::ram::RamHammingBooleanityInputClaims;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, TraceDimensions};
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, drive, reference_input_claim, with_ram_witness};
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
    fn fixture_hamming_weight_takes_both_boolean_values() {
        with_ram_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let zero = Fr::from_u64(0);
            let one = Fr::from_u64(1);
            let weights = dense_view::<Fr>(witness, ram_hamming_weight())
                .expect("the fixture serves the RAM Hamming-weight column");

            assert!(
                weights.iter().all(|value| *value == zero || *value == one),
                "the Hamming-weight indicator is not boolean on the fixture",
            );
            assert!(
                weights.contains(&one),
                "no fixture cycle touches RAM, so the booleanity summand is identically zero and \
                 a wrong eq point would still pass",
            );
            assert!(
                weights.contains(&zero),
                "every fixture cycle touches RAM, so the booleanity summand is identically zero \
                 and a wrong eq point would still pass",
            );
        });
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn ram_hamming_booleanity_matches_reference_round_for_round(
            seed in any::<u64>(),
            stage1_cycle_binding in arb_point(LOG_T),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_ram_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = RamHammingBooleanity::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    stage1_cycle_binding.clone(),
                );
                let claims = RamHammingBooleanityInputClaims::default();
                let points = RamHammingBooleanityInputClaims::default();
                let challenge_set = NoChallenges::default();
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
