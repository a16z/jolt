//! CUDA reduction of hamming-weight claims across all one-hot families.
//!
//! Like optimized, the kernel folds cycle equality weights into per-polynomial
//! address masses and combines each polynomial's claim weights into one table.
//! The resulting sum of products runs over the small address-chunk domain.
//!
//! CUDA obtains masses from shared resident one-hot shards, assembles them on
//! GPU 0, and keeps both the weight-table binds and quadratic round reductions
//! on that device. Only the reduced message lanes and final claims return to
//! the host.

#[cfg(feature = "allocative")]
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::Visitor;
use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::{
    HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    HammingWeightClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{HammingWeightClaimReductionPublic, JoltDerivedId};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::{require_context, CudaBackend};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device_columns::ANY_SPAN;
use crate::cuda::common::one_hot_fold::OneHotShards;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use reduction::DeviceHammingWeightReduction;

mod reduction;

pub struct HammingWeightClaimReductionKernel<F: JoltField> {
    context: &'static CudaKernelContext,
    relation: HammingWeightClaimReduction<F>,
    state: DeviceHammingWeightReduction,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: JoltField> Allocative for HammingWeightClaimReductionKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: JoltField> HammingWeightClaimReductionKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.state.bind(self.context, challenge).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda hamming reduction bind",
            }
        })?;
        self.rounds_bound += 1;
        Ok(())
    }
}

impl<F: JoltField> ProveRounds<F> for HammingWeightClaimReductionKernel<F> {
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
        let (at_one, at_infinity) = self.state.round_lanes(self.context).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda hamming reduction round",
            }
        })?;
        let at_zero = previous_claim - at_one;
        Ok(UnivariatePoly::from_evals_toom(&[
            at_zero,
            at_one,
            at_infinity,
        ]))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for HammingWeightClaimReductionKernel<F> {
    type Relation = HammingWeightClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &HammingWeightClaimReductionInputClaims<F>,
    ) -> Result<HammingWeightClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let layout = self.relation.dimensions().layout;
        let claims: Vec<F> =
            self.state
                .reduced_claims()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA hamming reduction claim readback failed",
                })?;
        if claims.len() != layout.total() {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA hamming reduction produces one claim per checked polynomial",
            });
        }
        let mut claims = claims.into_iter();
        Ok(HammingWeightClaimReductionOutputClaims {
            instruction_ra: claims.by_ref().take(layout.instruction()).collect(),
            bytecode_ra: claims.by_ref().take(layout.bytecode()).collect(),
            ram_ra: claims.take(layout.ram()).collect(),
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &HammingWeightClaimReduction<F>,
        input_points: &HammingWeightClaimReductionInputClaims<Vec<F>>,
        output_points: &HammingWeightClaimReductionOutputClaims<Vec<F>>,
        challenges: &HammingWeightClaimReductionChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let layout = relation.dimensions().layout;
        let booleanity_id = JoltDerivedId::from(HammingWeightClaimReductionPublic::EqBooleanity);
        let booleanity =
            relation.derive_output_term(&booleanity_id, input_points, output_points, challenges)?;

        let mut expected = F::zero();
        let mut power = F::one();
        for index in 0..layout.total() {
            let id =
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqVirtualization(index));
            let virtualization =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            expected += power;
            power *= challenges.gamma;
            expected += power * booleanity;
            power *= challenges.gamma;
            expected += power * virtualization;
            power *= challenges.gamma;
        }

        let weights: Vec<F> =
            self.state
                .weight_claims()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA hamming reduction weight readback failed",
                })?;
        let got = weights
            .into_iter()
            .fold(F::zero(), |accumulator, value| accumulator + value);
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift {
                id: booleanity_id.into(),
                expected,
                got,
            });
        }
        Ok(())
    }
}

impl<F: JoltField> PrepareKernel<F, HammingWeightClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, HammingWeightClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let layout = dimensions.layout;
        if relation.r_address().len() != dimensions.log_k_chunk
            || relation.virtualization_points().len() != layout.total()
        {
            return Err(KernelError::InvariantViolation {
                reason: "hamming reduction point shapes disagree with the layout",
            });
        }

        let cycles = 1usize << relation.r_cycle().len();
        let families = [layout.instruction(), layout.bytecode(), layout.ram()];
        let shards = OneHotShards::new::<F>(
            session,
            witness,
            cycles,
            families,
            dimensions.log_k_chunk,
            ANY_SPAN,
        )?;

        let state = DeviceHammingWeightReduction::new(
            context,
            &shards,
            relation.r_cycle(),
            relation.r_address(),
            relation.virtualization_points(),
            inputs.challenges.gamma,
        )?;
        drop(shards);

        Ok(Box::new(HammingWeightClaimReductionKernel {
            context,
            relation: relation.clone(),
            state,
            rounds_bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {

    use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltRelationId};
    use jolt_claims::OutputClaims;
    use jolt_field::Fr;
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
        HammingWeightClaimReduction, HammingWeightClaimReductionChallenges,
        HammingWeightClaimReductionInputClaims,
    };
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_one_hot_witness, ONE_HOT_BYTECODE_LEN,
    };
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const CONFIGS: [(usize, usize); 2] = [(4, 9), (8, 9)];

    fn one_hot(chunk_bits: usize) -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: chunk_bits as u8,
            lookups_ra_virtual_log_k_chunk: if chunk_bits == 4 { 16 } else { 32 },
        }
    }

    fn virtualization_points(total: usize, chunk_bits: usize, seed: u64) -> Vec<Vec<Fr>> {
        (0..total)
            .map(|leg| {
                (0..chunk_bits)
                    .map(|i| fr(seed ^ (0x51ED_270B * leg as u64 + i as u64 + 1)))
                    .collect()
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn hamming_weight_claim_reduction_matches_reference_round_for_round(
            seed in any::<u64>(),
            r_cycle in arb_point(LOG_T),
            r_address in arb_point(8),
            challenges in arb_point(8),
            gamma in any::<u64>().prop_map(fr),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            for (chunk_bits, ram_log_k) in CONFIGS {
                with_one_hot_witness(
                    LOG_T,
                    ONE_HOT_BYTECODE_LEN,
                    1usize << ram_log_k,
                    one_hot(chunk_bits),
                    seed,
                    |witness, bytecode_len| {
                        let layout = formula_dimensions_from_parts(
                            one_hot(chunk_bits),
                            LOG_T,
                            bytecode_len,
                            1usize << ram_log_k,
                            JoltRelationId::HammingWeightClaimReduction,
                        )
                        .expect("formula dimensions")
                        .ra_layout;
                        let dimensions =
                            HammingWeightClaimReductionDimensions::new(layout, chunk_bits);
                        let relation = HammingWeightClaimReduction::<Fr>::new(
                            dimensions,
                            r_cycle.clone(),
                            r_address[..chunk_bits].to_vec(),
                            virtualization_points(layout.total(), chunk_bits, seed),
                        );

                        let claims = HammingWeightClaimReductionInputClaims::default();
                        let points = HammingWeightClaimReductionInputClaims::default();
                        let challenge_set = HammingWeightClaimReductionChallenges { gamma };
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

                        let expected = drive(
                            &mut *expected_kernel,
                            input_claim,
                            &challenges[..chunk_bits],
                        );
                        let got =
                            drive(&mut *got_kernel, input_claim, &challenges[..chunk_bits]);
                        prop_assert_eq!(
                            got,
                            expected,
                            "round polynomials diverged at chunk_bits {} ram_log_k {}",
                            chunk_bits,
                            ram_log_k
                        );

                        let expected_claims = expected_kernel
                            .output_claims(&claims)
                            .expect("reference claims");
                        let got_claims =
                            got_kernel.output_claims(&claims).expect("cuda claims");
                        prop_assert_eq!(
                            got_claims.opening_values(),
                            expected_claims.opening_values(),
                            "output claims diverged at chunk_bits {} ram_log_k {}",
                            chunk_bits,
                            ram_log_k
                        );
                        Ok(())
                    },
                )?;
            }
        }
    }
}
