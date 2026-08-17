use jolt_claims::protocols::jolt::geometry::spartan::{
    is_first_in_sequence_shift, is_noop_shift, is_virtual_shift, pc_shift, unexpanded_pc_shift,
};
use jolt_claims::protocols::jolt::relations::spartan::{
    SpartanShiftInputClaims, SpartanShiftOutputClaims,
};
use jolt_claims::{OutputClaims, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage3::outputs::SpartanShift;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::common::trace_columns::cached_bundles;

use super::common::prefix_suffix::{
    eq_plus_one_pairs, prefix_rounds_ceil, PrefixSuffixGroup, PrefixSuffixRounds,
};
use super::CudaBackend;
use crate::cuda::require_context;
use crate::reference::ReferenceBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod columns;
pub(crate) mod witness;

use witness::SpartanShiftWitness;

const GAMMA_POWERS: usize = 5;

pub struct SpartanShiftKernel<F: Field> {
    rounds: PrefixSuffixRounds<F>,
    log_t: usize,
    bound: usize,
}

impl<F: Field> ProveRounds<F> for SpartanShiftKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
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

impl<F: Field> SumcheckKernel<F> for SpartanShiftKernel<F> {
    type Relation = SpartanShift<F>;

    fn output_claims(
        &mut self,
        _inputs: &SpartanShiftInputClaims<F>,
    ) -> Result<SpartanShiftOutputClaims<F>, SumcheckKernelError<F>> {
        if self.bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t.saturating_sub(self.bound),
            });
        }
        let claims =
            self.rounds
                .column_claims()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "the CUDA Spartan shift column claim readback failed",
                })?;
        if claims.len() != columns::COLUMNS {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "the CUDA Spartan shift driver returned the wrong number of column claims",
            });
        }
        let slots = [
            (unexpanded_pc_shift(), columns::UNEXPANDED_PC),
            (pc_shift(), columns::PC),
            (is_virtual_shift(), columns::IS_VIRTUAL),
            (is_first_in_sequence_shift(), columns::IS_FIRST_IN_SEQUENCE),
            (is_noop_shift(), columns::IS_NOOP),
        ];
        SpartanShiftOutputClaims::from_opening_values(|id| {
            slots
                .iter()
                .find(|(slot, _)| slot == id)
                .map(|&(_, column)| claims[column])
        })
        .map_err(SumcheckKernelError::from)
    }
}

impl<F: Field> PrepareKernel<F, SpartanShift<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, SpartanShift<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = SpartanShift<F>>>, KernelError<F>> {
        let context = require_context::<F>()?;
        let relation = inputs.relation;
        let log_t = relation.symbolic().rounds();
        let tau_low = relation.product_uniskip_tau_low();
        let product_point = relation.product_remainder_opening_point();
        if tau_low.len() != log_t || product_point.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "the Spartan shift eq+1 points span the cycle variables",
            });
        }
        let Some(prefix_rounds) = prefix_rounds_ceil(log_t) else {
            return ReferenceBackend.prepare(session, witness, inputs);
        };

        let rows = cached_bundles::<SpartanShiftWitness, _>(session, witness, 1usize << log_t)?;
        let packed = witness::pack(&rows);
        let device_columns = columns::upload(context, &packed)?;

        let mut powers = [F::one(); GAMMA_POWERS];
        for index in 1..GAMMA_POWERS {
            powers[index] = powers[index - 1] * inputs.challenges.gamma;
        }
        let groups = vec![
            PrefixSuffixGroup {
                pairs: eq_plus_one_pairs(tau_low, prefix_rounds)?,
                columns: vec![
                    (columns::UNEXPANDED_PC, powers[0]),
                    (columns::PC, powers[1]),
                    (columns::IS_VIRTUAL, powers[2]),
                    (columns::IS_FIRST_IN_SEQUENCE, powers[3]),
                ],
                constant: F::zero(),
            },
            PrefixSuffixGroup {
                pairs: eq_plus_one_pairs(product_point, prefix_rounds)?,
                columns: vec![(columns::IS_NOOP, -powers[4])],
                constant: powers[4],
            },
        ];

        Ok(Box::new(SpartanShiftKernel {
            rounds: PrefixSuffixRounds::new(context, device_columns, groups, prefix_rounds)?,
            log_t,
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
    use jolt_claims::protocols::jolt::geometry::spartan::{
        is_first_in_sequence_shift, is_noop_shift, is_virtual_shift, pc_shift, unexpanded_pc_shift,
    };
    use jolt_claims::protocols::jolt::relations::spartan::{
        SpartanShiftChallenges, SpartanShiftInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, TraceDimensions};
    use jolt_claims::OutputClaims;
    use jolt_field::Fr;
    use jolt_poly::EqPlusOnePrefixSuffix;
    use jolt_verifier::stages::stage3::outputs::SpartanShift;
    use proptest::prelude::*;

    use super::{eq_plus_one_pairs, prefix_rounds_ceil, CudaBackend};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_r1cs_witness,
    };
    use crate::reference::views::dense_view;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    #[test]
    fn prefix_rounds_match_the_host_midpoint_split() {
        for log_t in 2usize..12 {
            let point: Vec<Fr> = (0..log_t).map(|index| fr(index as u64 * 13 + 5)).collect();
            let expected = EqPlusOnePrefixSuffix::<Fr>::new(&point);
            let got = eq_plus_one_pairs(&point, prefix_rounds_ceil(log_t).expect("prefix rounds"))
                .expect("eq+1 pairs");
            assert_eq!(got[0].prefix, expected.prefix_0, "log_t {log_t}");
            assert_eq!(got[0].suffix, expected.suffix_0, "log_t {log_t}");
            assert_eq!(got[1].prefix, expected.prefix_1, "log_t {log_t}");
            assert_eq!(got[1].suffix, expected.suffix_1, "log_t {log_t}");
        }
    }

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

    #[test]
    fn a_one_round_trace_is_served_below_the_prefix_suffix_minimum() {
        let Some(_) = shared_context() else {
            return;
        };
        const TINY: usize = 1;
        let point = vec![fr(7)];
        let challenges = vec![fr(11)];
        with_r1cs_witness(TINY, RAM_K, one_hot(), 3, |witness| {
            let relation =
                SpartanShift::<Fr>::new(TraceDimensions::new(TINY), point.clone(), point.clone());
            let claims = SpartanShiftInputClaims::default();
            let points = SpartanShiftInputClaims::default();
            let challenge_set = SpartanShiftChallenges { gamma: fr(5) };
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
                .expect("cuda prepare below the prefix-suffix minimum");

            let expected = drive(&mut *expected_kernel, input_claim, &challenges);
            let got = drive(&mut *got_kernel, input_claim, &challenges);
            assert_eq!(got, expected, "round polynomials diverged at log_T = 1");
            assert_eq!(
                got_kernel
                    .output_claims(&claims)
                    .expect("cuda claims")
                    .opening_values(),
                expected_kernel
                    .output_claims(&claims)
                    .expect("reference claims")
                    .opening_values(),
                "output claims diverged at log_T = 1",
            );
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
