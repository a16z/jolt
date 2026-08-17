use jolt_claims::protocols::jolt::relations::ram::{
    RamHammingBooleanityInputClaims, RamHammingBooleanityOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::common::trace_columns::cached_bundles;

use super::{require_context, CudaBackend};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use hamming_weight::DeviceHammingWeight;

pub(crate) mod hamming_weight;
pub(crate) mod witness;

pub struct RamHammingBooleanityKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: RamHammingBooleanity<F>,
    weights: DeviceHammingWeight,
    eq: DeviceSplitEq<F>,
    rounds_bound: usize,
    final_claim: Option<F>,
}

impl<F: Field> RamHammingBooleanityKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda RAM Hamming-booleanity bind",
        };
        self.weights
            .bind(self.context, challenge)
            .map_err(|_| failed())?;
        self.eq.bind(challenge);
        self.rounds_bound += 1;
        if self.rounds_bound == self.relation.symbolic().rounds() {
            self.final_claim = Some(self.weights.final_claim().map_err(|_| failed())?);
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for RamHammingBooleanityKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        let (constant, quadratic) = self
            .weights
            .round_coefficients(self.context, &self.eq)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda RAM Hamming-booleanity round",
            })?;
        let mut coefficients = self
            .eq
            .gruen_poly_deg_3(constant, quadratic, previous_claim)
            .into_coefficients();
        coefficients.resize(self.relation.degree() + 1, F::from_u64(0));
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for RamHammingBooleanityKernel<F> {
    type Relation = RamHammingBooleanity<F>;

    fn output_claims(
        &mut self,
        _inputs: &RamHammingBooleanityInputClaims<F>,
    ) -> Result<RamHammingBooleanityOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let ram_hamming_weight =
            self.final_claim
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "CUDA RAM Hamming booleanity never read back its bound claim",
                })?;
        Ok(RamHammingBooleanityOutputClaims { ram_hamming_weight })
    }
}

impl<F: Field> PrepareKernel<F, RamHammingBooleanity<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamHammingBooleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamHammingBooleanity<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        if relation.stage1_cycle_binding().len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "stage-1 cycle binding has the wrong variable count",
            });
        }

        let cycles = 1usize << log_t;
        let rows =
            cached_bundles::<witness::RamHammingBooleanityWitness, _>(session, witness, cycles)?;
        let column = witness::packed_weights(&rows);
        drop(rows);
        let weights = DeviceHammingWeight::new(context, &column)?;
        drop(column);

        let eq_point: Vec<F> = relation
            .stage1_cycle_binding()
            .iter()
            .rev()
            .copied()
            .collect();
        let eq = DeviceSplitEq::new(context, &eq_point, BindingOrder::LowToHigh)?;

        Ok(Box::new(RamHammingBooleanityKernel {
            context,
            relation: relation.clone(),
            weights,
            eq,
            rounds_bound: 0,
            final_claim: None,
        }))
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
