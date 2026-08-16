use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::CudaBackend;
use crate::uniskip::UniskipKernel;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[expect(
    clippy::todo,
    reason = "phase 1 stub: the equivalence tests drive it until the kernels land, and the \
              expectation becomes an unfulfilled-expectation error the moment they do"
)]
impl<F: Field> UniskipKernel<F, ProductRemainder<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _log_t: usize,
        _tau_low: &[F],
        _witness: &dyn JoltWitnessPlane<F>,
    ) -> Result<(), KernelError<F>> {
        todo!("the CUDA Spartan product uni-skip prepare")
    }

    fn first_round_poly(
        &self,
        _session: &mut ProofSession,
        _late_tau: &[F],
    ) -> Result<UnivariatePoly<F>, KernelError<F>> {
        todo!("the CUDA Spartan product uni-skip first-round polynomial")
    }
}

#[expect(
    clippy::todo,
    reason = "phase 1 stub: the equivalence tests drive it until the kernels land, and the \
              expectation becomes an unfulfilled-expectation error the moment they do"
)]
impl<F: Field> PrepareKernel<F, ProductRemainder<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, ProductRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ProductRemainder<F>>>, KernelError<F>> {
        todo!("the CUDA Spartan product remainder prepare")
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::{
        branch_flag_product, jump_flag_product, left_instruction_input_product,
        lookup_output_product, next_is_noop_product, right_instruction_input_product,
        virtual_instruction_product, write_lookup_output_to_rd_product, SpartanProductDimensions,
    };
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage2::product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainder,
        ProductRemainderInputClaims,
    };
    use jolt_witness::JoltWitnessPlane;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, probe_input_claim, with_r1cs_witness,
    };
    use crate::reference::spartan_product::ReferenceProductRemainder;
    use crate::reference::views::dense_view;
    use crate::reference::ReferenceBackend;
    use crate::uniskip::UniskipKernel;
    use crate::{PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    #[test]
    fn fixture_r1cs_columns_exercise_every_product_factor() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let zero = Fr::from_u64(0);
            let one = Fr::from_u64(1);

            for (label, opening) in [
                ("jump_flag", jump_flag_product()),
                (
                    "write_lookup_output_to_rd",
                    write_lookup_output_to_rd_product(),
                ),
                ("branch_flag", branch_flag_product()),
                ("next_is_noop", next_is_noop_product()),
                ("virtual_instruction", virtual_instruction_product()),
            ] {
                let column = dense_view::<Fr>(witness, opening)
                    .expect("the fixture serves every product flag column");
                assert!(
                    column.contains(&one),
                    "{label}: the flag is off at every fixture cycle, so dropping its lane would \
                     not change the round polynomials",
                );
                assert!(
                    column.contains(&zero),
                    "{label}: the flag is on at every fixture cycle, so the fixture cannot tell \
                     the lane from a constant",
                );
            }

            for (label, opening) in [
                ("left_instruction_input", left_instruction_input_product()),
                ("right_instruction_input", right_instruction_input_product()),
                ("lookup_output", lookup_output_product()),
            ] {
                let column = dense_view::<Fr>(witness, opening)
                    .expect("the fixture serves every product operand column");
                assert!(
                    column.iter().any(|value| *value != zero),
                    "{label}: zero at every fixture cycle, so its Lagrange weight could be \
                     anything",
                );
                assert!(
                    column.iter().any(|value| *value != column[0]),
                    "{label}: constant across the fixture, so a swapped operand column would go \
                     unnoticed",
                );
            }
        });
    }

    fn prepared(
        witness: &dyn JoltWitnessPlane<Fr>,
        tau_low: &[Fr],
        tau_high: Fr,
        uniskip_challenge: Fr,
        cuda: bool,
    ) -> Box<dyn SumcheckKernel<Fr, Relation = ProductRemainder<Fr>>> {
        let mut session = ProofSession::default();
        if cuda {
            UniskipKernel::<Fr, ProductRemainder<Fr>>::prepare(
                &CudaBackend,
                &mut session,
                LOG_T,
                tau_low,
                witness,
            )
            .expect("cuda uni-skip prepare");
        } else {
            UniskipKernel::<Fr, ProductRemainder<Fr>>::prepare(
                &ReferenceBackend,
                &mut session,
                LOG_T,
                tau_low,
                witness,
            )
            .expect("reference uni-skip prepare");
        }

        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(LOG_T),
            uniskip_challenge,
            tau_high,
            tau_low.to_vec(),
        );
        let claims = product_remainder_input_values_from_uniskip_output(Fr::from_u64(0));
        let points = ProductRemainderInputClaims {
            product_uniskip: Vec::new(),
        };
        let challenges = NoChallenges::default();
        let inputs = ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        if cuda {
            PrepareKernel::<Fr, ProductRemainder<Fr>>::prepare(
                &CudaBackend,
                &mut session,
                witness,
                inputs,
            )
            .expect("cuda remainder prepare")
        } else {
            ReferenceProductRemainder
                .prepare(&mut session, witness, inputs)
                .expect("reference remainder prepare")
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn spartan_product_uniskip_first_round_poly_matches_reference(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
            tau_high in any::<u64>().prop_map(fr),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let mut expected_session = ProofSession::default();
                UniskipKernel::<Fr, ProductRemainder<Fr>>::prepare(
                    &ReferenceBackend, &mut expected_session, LOG_T, &tau_low, witness,
                ).expect("reference uni-skip prepare");
                let expected = UniskipKernel::<Fr, ProductRemainder<Fr>>::first_round_poly(
                    &ReferenceBackend, &mut expected_session, &[tau_high],
                ).expect("reference uni-skip first-round polynomial");

                let mut got_session = ProofSession::default();
                UniskipKernel::<Fr, ProductRemainder<Fr>>::prepare(
                    &CudaBackend, &mut got_session, LOG_T, &tau_low, witness,
                ).expect("cuda uni-skip prepare");
                let got = UniskipKernel::<Fr, ProductRemainder<Fr>>::first_round_poly(
                    &CudaBackend, &mut got_session, &[tau_high],
                ).expect("cuda uni-skip first-round polynomial");

                prop_assert_eq!(
                    got.coefficients().to_vec(),
                    expected.coefficients().to_vec(),
                    "the uni-skip first-round polynomial diverged"
                );
                Ok(())
            })?;
        }

        #[test]
        fn spartan_product_remainder_matches_reference_round_for_round(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
            tau_high in any::<u64>().prop_map(fr),
            uniskip_challenge in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let input_claim = probe_input_claim(
                    &mut *prepared(witness, &tau_low, tau_high, uniskip_challenge, false),
                );
                let mut expected_kernel =
                    prepared(witness, &tau_low, tau_high, uniskip_challenge, false);
                let mut got_kernel =
                    prepared(witness, &tau_low, tau_high, uniskip_challenge, true);

                let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                let got = drive(&mut *got_kernel, input_claim, &challenges);
                prop_assert_eq!(got, expected, "round polynomials diverged");

                let wire = product_remainder_input_values_from_uniskip_output(input_claim);
                let expected_claims =
                    expected_kernel.output_claims(&wire).expect("reference claims");
                let got_claims = got_kernel.output_claims(&wire).expect("cuda claims");
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
