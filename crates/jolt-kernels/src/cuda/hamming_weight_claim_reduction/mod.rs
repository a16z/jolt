use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::{
    HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    HammingWeightClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{HammingWeightClaimReductionPublic, JoltDerivedId};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::{require_context, CudaBackend};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device_columns::{device_trace_columns, ANY_SPAN};
use crate::cuda::common::one_hot_fold::DeviceOneHotColumns;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use reduction::DeviceHammingWeightReduction;

mod reduction;

pub struct HammingWeightClaimReductionKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: HammingWeightClaimReduction<F>,
    state: DeviceHammingWeightReduction,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for HammingWeightClaimReductionKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

impl<F: Field> HammingWeightClaimReductionKernel<F> {
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

impl<F: Field> ProveRounds<F> for HammingWeightClaimReductionKernel<F> {
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

impl<F: Field> SumcheckKernel<F> for HammingWeightClaimReductionKernel<F> {
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
                id: booleanity_id,
                expected,
                got,
            });
        }
        Ok(())
    }
}

impl<F: Field> PrepareKernel<F, HammingWeightClaimReduction<F>> for CudaBackend {
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
        let columns =
            device_trace_columns::<F>(context, session, witness, cycles, families, ANY_SPAN)?;
        let device_columns =
            DeviceOneHotColumns::from_device(columns, families, dimensions.log_k_chunk, cycles)?;

        let state = DeviceHammingWeightReduction::new(
            context,
            &device_columns,
            relation.r_cycle(),
            relation.r_address(),
            relation.virtualization_points(),
            inputs.challenges.gamma,
        )?;
        drop(device_columns);

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
    use std::collections::BTreeSet;

    use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltRelationId};
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
        arb_point, drive, fr, hot_addresses, one_hot_fixture_bytecode_is_cold,
        one_hot_fixture_is_padding, ram_fixture_is_cold, reference_input_claim,
        with_one_hot_witness, ONE_HOT_BYTECODE_LEN,
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

    fn families(
        layout: jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout,
    ) -> [(&'static str, Vec<JoltCommittedPolynomial>); 3] {
        [
            (
                "instruction",
                (0..layout.instruction())
                    .map(JoltCommittedPolynomial::InstructionRa)
                    .collect(),
            ),
            (
                "bytecode",
                (0..layout.bytecode())
                    .map(JoltCommittedPolynomial::BytecodeRa)
                    .collect(),
            ),
            (
                "ram",
                (0..layout.ram())
                    .map(JoltCommittedPolynomial::RamRa)
                    .collect(),
            ),
        ]
    }

    #[test]
    fn fixture_hamming_folds_accumulate_and_discriminate_chunks() {
        let cycles = 1usize << LOG_T;
        for (chunk_bits, ram_log_k) in CONFIGS {
            let addresses = 1usize << chunk_bits;
            with_one_hot_witness(
                LOG_T,
                ONE_HOT_BYTECODE_LEN,
                1usize << ram_log_k,
                one_hot(chunk_bits),
                7,
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

                    let points = virtualization_points(layout.total(), chunk_bits, 7);
                    for (left, point) in points.iter().enumerate() {
                        for other in points.iter().skip(left + 1) {
                            assert_ne!(
                                point, other,
                                "chunk_bits {chunk_bits}: two virtualization points coincide, so \
                                 swapping their EqVirtualization tables would not change any \
                                 round polynomial",
                            );
                        }
                    }

                    for (family, polynomials) in families(layout) {
                        assert!(
                            !polynomials.is_empty(),
                            "chunk_bits {chunk_bits} ram_log_k {ram_log_k}: the {family} family \
                             is empty, so this config cannot exercise it",
                        );
                        let mut per_chunk = Vec::new();
                        for polynomial in polynomials {
                            let hot = hot_addresses(witness, polynomial, addresses, cycles);
                            let mut counts = vec![0usize; addresses];
                            for address in hot.iter().flatten() {
                                counts[*address] += 1;
                            }
                            let occupied = counts.iter().filter(|count| **count > 0).count();
                            assert!(
                                occupied > 1,
                                "chunk_bits {chunk_bits} ram_log_k {ram_log_k} {polynomial:?}: \
                                 the fold lands on {occupied} address(es), so its G table cannot \
                                 detect a wrong address",
                            );
                            assert!(
                                counts.iter().any(|count| *count > 1),
                                "chunk_bits {chunk_bits} ram_log_k {ram_log_k} {polynomial:?}: no \
                                 address is hot at two cycles, so a fold that OVERWRITES instead \
                                 of accumulating would still pass",
                            );
                            per_chunk.push(hot);
                        }
                        if per_chunk.len() > 1 {
                            let disagree = (0..cycles).any(|cycle| {
                                let mut seen = BTreeSet::new();
                                for hot in &per_chunk {
                                    if let Some(address) = hot[cycle] {
                                        let _ = seen.insert(address);
                                    }
                                }
                                seen.len() > 1
                            });
                            assert!(
                                disagree,
                                "chunk_bits {chunk_bits} ram_log_k {ram_log_k}: every {family} \
                                 chunk is hot at the same address on every cycle, so a wrong \
                                 chunk shift would still pass",
                            );
                        }
                    }

                    let instruction_cold = (0..cycles)
                        .filter(|&cycle| one_hot_fixture_is_padding(LOG_T, cycle))
                        .count();
                    let bytecode_cold = (0..cycles)
                        .filter(|&cycle| one_hot_fixture_bytecode_is_cold(LOG_T, cycle))
                        .count();
                    let ram_cold = (0..cycles)
                        .filter(|&cycle| ram_fixture_is_cold(LOG_T, cycle))
                        .count();
                    let disagree = (0..cycles)
                        .filter(|&cycle| {
                            one_hot_fixture_bytecode_is_cold(LOG_T, cycle)
                                != ram_fixture_is_cold(LOG_T, cycle)
                        })
                        .count();
                    assert!(instruction_cold > 0, "the fixture has no padding tail");
                    assert!(
                        bytecode_cold > 0 && bytecode_cold < cycles,
                        "{bytecode_cold} of {cycles} cycles are bytecode-cold, so one of the two \
                         fold paths is unexercised",
                    );
                    assert!(
                        ram_cold > 0 && ram_cold < cycles,
                        "{ram_cold} of {cycles} cycles are RAM-cold, so one of the two fold paths \
                         is unexercised",
                    );
                    assert!(
                        disagree > 0,
                        "the bytecode and RAM cold sets coincide, so a fold reading the wrong \
                         family's cold flag would still pass",
                    );
                },
            );
        }
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
