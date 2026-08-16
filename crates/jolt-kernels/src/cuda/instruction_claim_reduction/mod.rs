use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
    InstructionClaimReductionInputClaims, InstructionClaimReductionOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::common::prefix_suffix::{
    eq_pair, prefix_rounds_ceil, PrefixSuffixGroup, PrefixSuffixRounds,
};
use super::{require_context, CudaBackend};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod witness;

const COLUMNS: usize = 5;

pub struct InstructionClaimReductionKernel<F: Field> {
    rounds: PrefixSuffixRounds<F>,
    total: usize,
    bound: usize,
}

impl<F: Field> ProveRounds<F> for InstructionClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.total
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if bind.is_some() {
            self.bound += 1;
        }
        self.rounds.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bound += 1;
        self.rounds.finish_rounds(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for InstructionClaimReductionKernel<F> {
    type Relation = InstructionClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &InstructionClaimReductionInputClaims<F>,
    ) -> Result<InstructionClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.total - self.bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let claims =
            self.rounds
                .column_claims()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason:
                        "CUDA instruction claim reduction failed to read back its column claims",
                })?;
        let [lookup_output, left_lookup_operand, right_lookup_operand, left_instruction_input, right_instruction_input] =
            claims.as_slice()
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA instruction claim reduction produces one claim per reduced column",
            });
        };
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output: *lookup_output,
            left_lookup_operand: *left_lookup_operand,
            right_lookup_operand: *right_lookup_operand,
            left_instruction_input: *left_instruction_input,
            right_instruction_input: *right_instruction_input,
        })
    }
}

impl<F: Field> PrepareKernel<F, InstructionClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let log_t = relation.symbolic().rounds();
        let tau_low = relation.tau_low();
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "the instruction reduction Spartan point has the wrong variable count",
            });
        }
        let prefix_rounds = prefix_rounds_ceil(log_t).ok_or(KernelError::Unsupported {
            reason: "the CUDA instruction claim reduction needs at least two cycle variables to \
                     split the prefix-suffix sumcheck",
        })?;

        let cycles = 1usize << log_t;
        let rows = collect_bundles::<witness::InstructionClaimReductionWitness>(witness, cycles)?;
        let columns = witness::device_columns(context, &rows)?;
        drop(rows);

        let gamma = inputs.challenges.gamma;
        let mut powers = Vec::with_capacity(COLUMNS);
        let mut power = F::one();
        for _ in 0..COLUMNS {
            powers.push(power);
            power *= gamma;
        }
        let group = PrefixSuffixGroup {
            pairs: vec![eq_pair(tau_low, prefix_rounds)?],
            columns: powers.into_iter().enumerate().collect(),
            constant: F::zero(),
        };

        Ok(Box::new(InstructionClaimReductionKernel {
            rounds: PrefixSuffixRounds::new(context, columns, vec![group], prefix_rounds)?,
            total: log_t,
            bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::instruction::{
        left_instruction_input_reduced, left_lookup_operand_reduced, lookup_output_reduced,
        right_instruction_input_reduced, right_lookup_operand_reduced,
    };
    use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
        InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltOpeningId, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
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

    fn column_ids() -> [JoltOpeningId; 5] {
        [
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            left_instruction_input_reduced(),
            right_instruction_input_reduced(),
        ]
    }

    #[test]
    fn fixture_instruction_claim_reduction_columns_vary() {
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
        fn instruction_claim_reduction_matches_reference_round_for_round(
            seed in any::<u64>(),
            tau_low in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            with_r1cs_witness(LOG_T, RAM_K, one_hot(), seed, |witness| {
                let relation = InstructionClaimReduction::<Fr>::new(
                    TraceDimensions::new(LOG_T),
                    tau_low.clone(),
                );
                let claims = InstructionClaimReductionInputClaims::default();
                let points = InstructionClaimReductionInputClaims::default();
                let challenge_set = InstructionClaimReductionChallenges { gamma };
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
