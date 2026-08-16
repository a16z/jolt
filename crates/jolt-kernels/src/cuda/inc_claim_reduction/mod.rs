use jolt_claims::protocols::jolt::relations::claim_reductions::increments::{
    IncClaimReductionInputClaims, IncClaimReductionOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::common::prefix_suffix::{
    eq_pair, prefix_rounds_floor, PrefixSuffixGroup, PrefixSuffixRounds,
};
use super::{require_context, CudaBackend};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod witness;

const RAM_COLUMN: usize = 0;

const RD_COLUMN: usize = 1;

const GROUP_COLUMNS: [usize; 4] = [RAM_COLUMN, RAM_COLUMN, RD_COLUMN, RD_COLUMN];

pub struct IncClaimReductionKernel<F: Field> {
    rounds: PrefixSuffixRounds<F>,
    total: usize,
    bound: usize,
}

impl<F: Field> ProveRounds<F> for IncClaimReductionKernel<F> {
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

impl<F: Field> SumcheckKernel<F> for IncClaimReductionKernel<F> {
    type Relation = IncClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &IncClaimReductionInputClaims<F>,
    ) -> Result<IncClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.total - self.bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let claims =
            self.rounds
                .column_claims()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA increment claim reduction failed to read back its column claims",
                })?;
        let [ram_inc, rd_inc] = claims.as_slice() else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA increment claim reduction produces one claim per increment column",
            });
        };
        Ok(IncClaimReductionOutputClaims {
            ram_inc: *ram_inc,
            rd_inc: *rd_inc,
        })
    }
}

impl<F: Field> PrepareKernel<F, IncClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, IncClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = IncClaimReduction<F>>>, KernelError<F>> {
        let context = require_context()?;
        let relation = inputs.relation;
        let log_t = relation.symbolic().rounds();
        let cycle_points = relation.cycle_points();
        if cycle_points.iter().any(|point| point.len() != log_t) {
            return Err(KernelError::InvariantViolation {
                reason: "an increment reduction cycle point has the wrong variable count",
            });
        }
        let prefix_rounds = prefix_rounds_floor(log_t).ok_or(KernelError::Unsupported {
            reason: "the CUDA increment claim reduction needs at least two cycle variables to \
                     split the prefix-suffix sumcheck",
        })?;

        let cycles = 1usize << log_t;
        let rows = collect_bundles::<witness::IncClaimReductionWitness>(witness, cycles)?;
        let columns = witness::device_columns(context, &rows)?;
        drop(rows);

        let gamma = inputs.challenges.gamma;
        let mut power = F::one();
        let mut groups = Vec::with_capacity(GROUP_COLUMNS.len());
        for (point, column) in cycle_points.into_iter().zip(GROUP_COLUMNS) {
            groups.push(PrefixSuffixGroup {
                pairs: vec![eq_pair(point, prefix_rounds)?],
                columns: vec![(column, power)],
                constant: F::zero(),
            });
            power *= gamma;
        }

        Ok(Box::new(IncClaimReductionKernel {
            rounds: PrefixSuffixRounds::new(context, columns, groups, prefix_rounds)?,
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
