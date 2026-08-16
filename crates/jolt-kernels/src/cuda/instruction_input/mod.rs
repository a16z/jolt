use jolt_field::Field;
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::JoltWitnessPlane;

use super::CudaBackend;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[expect(
    clippy::todo,
    reason = "phase 1 stub: the equivalence test must fail here until the kernels land, and \
              the expectation becomes an error the moment they do"
)]
impl<F: Field> PrepareKernel<F, InstructionInput<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        todo!("CUDA instruction input-virtualization kernel")
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::{
        imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
        rs1_value, rs2_value, unexpanded_pc,
    };
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionInputChallenges, InstructionInputInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltOpeningId, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage3::outputs::InstructionInput;
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

    fn flag_ids() -> [JoltOpeningId; 4] {
        [
            left_operand_is_rs1(),
            left_operand_is_pc(),
            right_operand_is_rs2(),
            right_operand_is_imm(),
        ]
    }

    fn value_ids() -> [JoltOpeningId; 4] {
        [rs1_value(), unexpanded_pc(), rs2_value(), imm()]
    }

    #[test]
    fn fixture_instruction_input_columns_vary() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let plane: &dyn JoltWitnessPlane<Fr> = witness;
            let zero = Fr::from_u64(0);
            let one = Fr::from_u64(1);
            for id in flag_ids() {
                let column = dense_view::<Fr>(plane, id).expect("the fixture serves the flag");
                assert!(
                    column.contains(&zero) && column.contains(&one),
                    "{id:?} takes one value across the fixture, so dropping its term would not \
                     change the round polynomials",
                );
            }
            for id in value_ids() {
                let column = dense_view::<Fr>(plane, id).expect("the fixture serves the operand");
                assert!(
                    column.iter().any(|value| *value != zero),
                    "{id:?} is zero at every cycle, so its coefficient could be anything",
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
        fn instruction_input_matches_reference_round_for_round(
            seed in any::<u64>(),
            product_remainder_point in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = InstructionInput::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    product_remainder_point.clone(),
                );
                let claims = InstructionInputInputClaims::default();
                let points = InstructionInputInputClaims::default();
                let challenge_set = InstructionInputChallenges { gamma };
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
