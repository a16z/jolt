use jolt_field::Field;
use jolt_verifier::stages::stage3::outputs::SpartanShift;
use jolt_witness::JoltWitnessPlane;

use super::CudaBackend;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[expect(
    clippy::todo,
    reason = "phase 1 stub: the equivalence test drives this until the kernels land"
)]
impl<F: Field> PrepareKernel<F, SpartanShift<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, SpartanShift<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = SpartanShift<F>>>, KernelError<F>> {
        todo!("CUDA Spartan shift kernel")
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::{
        is_first_in_sequence_shift, is_noop_shift, is_virtual_shift, pc_shift, unexpanded_pc_shift,
    };
    use jolt_claims::protocols::jolt::relations::spartan::{
        SpartanShiftChallenges, SpartanShiftInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::Fr;
    use jolt_verifier::stages::stage3::outputs::SpartanShift;
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
    fn fixture_shift_columns_are_non_constant() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let ids = [
                ("unexpanded_pc", unexpanded_pc_shift()),
                ("pc", pc_shift()),
                ("is_virtual", is_virtual_shift()),
                ("is_first_in_sequence", is_first_in_sequence_shift()),
                ("is_noop", is_noop_shift()),
            ];
            for (name, id) in ids {
                let table = dense_view::<Fr>(witness, id).expect("the fixture serves the column");
                let first = table[0];
                assert!(
                    table.iter().any(|value| *value != first),
                    "shift column {name} is constant across the fixture, so a wrong eq+1 weight \
                     on it would not change any round polynomial",
                );
            }
        });
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn spartan_shift_matches_reference_round_for_round(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
            product_point in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = SpartanShift::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    tau_low.clone(),
                    product_point.clone(),
                );
                let claims = SpartanShiftInputClaims::default();
                let points = SpartanShiftInputClaims::default();
                let challenge_set = SpartanShiftChallenges { gamma };
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
