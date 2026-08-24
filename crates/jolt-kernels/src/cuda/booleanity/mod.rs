use jolt_claims::protocols::jolt::relations::booleanity::{
    BooleanityInputClaims, BooleanityOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{try_eq_mle, BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::booleanity::Booleanity;
use jolt_witness::JoltWitnessPlane;

use super::{require_context, CudaBackend};
use crate::cuda::common::context::context_for;
use crate::cuda::common::device_columns::{device_trace_columns, ANY_SPAN};
use crate::cuda::common::devices::witness_windows;
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use one_hot::{BooleanityShard, DeviceBooleanityRa, ShardedBooleanityRa};

pub(crate) mod address;
pub(crate) mod masses;
pub(crate) mod one_hot;

pub struct BooleanityCycleKernel<F: Field> {
    relation: Booleanity<F>,
    one_hot: ShardedBooleanityRa<F>,
    eq: DeviceSplitEq<F>,
    gamma: F,
    rounds_bound: usize,
    finals: Option<Vec<F>>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for BooleanityCycleKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("eq"), self.eq.device_bytes());
        visitor.visit_simple(
            allocative::Key::new("finals"),
            self.finals.as_ref().map_or(0, |v| v.len() * size_of::<F>()),
        );
        visitor.exit();
    }
}

impl<F: Field> BooleanityCycleKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda booleanity cycle-phase bind",
        };
        let bound = self.rounds_bound;
        self.one_hot.bind(challenge, bound).map_err(|_| failed())?;
        self.eq.bind(challenge);
        self.rounds_bound += 1;
        if self.rounds_bound == self.relation.symbolic().rounds() {
            self.finals = Some(self.one_hot.final_claims().map_err(|_| failed())?);
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for BooleanityCycleKernel<F> {
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
        let (constant, quadratic) = self.one_hot.round_coefficients(&self.eq).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda booleanity cycle-phase round",
            }
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

impl<F: Field> SumcheckKernel<F> for BooleanityCycleKernel<F> {
    type Relation = Booleanity<F>;

    fn output_claims(
        &mut self,
        _inputs: &BooleanityInputClaims<F>,
    ) -> Result<BooleanityOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let finals = self
            .finals
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA booleanity never read back its bound claims",
            })?;
        let layout = self.relation.dimensions().layout;
        if finals.len() != layout.total() {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA booleanity produces one claim per checked polynomial",
            });
        }

        let inverse = self
            .gamma
            .inverse()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "the booleanity batching challenge must be invertible",
            })?;
        let mut scale = F::one();
        let mut unscaled = Vec::with_capacity(finals.len());
        for &value in finals {
            unscaled.push(value * scale);
            scale *= inverse;
        }

        let mut claims = unscaled.into_iter();
        Ok(BooleanityOutputClaims {
            instruction_ra: claims.by_ref().take(layout.instruction()).collect(),
            bytecode_ra: claims.by_ref().take(layout.bytecode()).collect(),
            ram_ra: claims.take(layout.ram()).collect(),
        })
    }
}

impl<F: Field> PrepareKernel<F, Booleanity<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, Booleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        if relation.r_address().len() != dimensions.log_k_chunk
            || relation.reference_address().len() != dimensions.log_k_chunk
        {
            return Err(KernelError::InvariantViolation {
                reason: "a booleanity address point has the wrong variable count",
            });
        }
        if relation.reference_cycle().len() != dimensions.log_t {
            return Err(KernelError::InvariantViolation {
                reason: "the booleanity reference cycle point has the wrong variable count",
            });
        }

        let unsupported = || KernelError::Unsupported {
            reason: "the CUDA booleanity kernel supports only the BN254 scalar field",
        };
        let address_scalar = try_eq_mle(relation.r_address(), relation.reference_address())
            .map_err(|_| KernelError::InvariantViolation {
                reason: "booleanity address point and reference length mismatch",
            })?;

        let layout = dimensions.layout;
        let families = [layout.instruction(), layout.bytecode(), layout.ram()];
        let log_t = dimensions.log_t;
        let cycles = 1usize << log_t;
        let windows = witness_windows(cycles);
        let shards = windows.len();
        let mut one_hot_shards = Vec::with_capacity(shards);
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a booleanity window names an absent device",
            })?;
            let columns = device_trace_columns::<F>(
                device, session, witness, cycles, window, families, ANY_SPAN,
            )?;
            one_hot_shards.push(BooleanityShard {
                ordinal,
                one_hot: DeviceBooleanityRa::from_device(
                    device,
                    columns,
                    window.len,
                    dimensions.log_k_chunk,
                    families,
                    relation.r_address(),
                    inputs.challenges.gamma,
                )
                .map_err(|_| KernelError::Unsupported {
                    reason: "the CUDA booleanity kernel packs each family's committed chunk \
                             indices into one word per cycle",
                })?,
                eq: DeviceSplitEq::new_window_with_scaling(
                    device,
                    relation.reference_cycle(),
                    BindingOrder::LowToHigh,
                    address_scalar,
                    ordinal,
                    shards,
                )
                .map_err(|_| unsupported())?,
            });
        }
        let eq = DeviceSplitEq::new_with_scaling(
            require_context()?,
            relation.reference_cycle(),
            BindingOrder::LowToHigh,
            address_scalar,
        )
        .map_err(|_| unsupported())?;

        Ok(Box::new(BooleanityCycleKernel {
            relation: relation.clone(),
            one_hot: ShardedBooleanityRa::new(one_hot_shards, log_t)?,
            eq,
            gamma: inputs.challenges.gamma,
            rounds_bound: 0,
            finals: None,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltRelationId};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_verifier::stages::stage6b::booleanity::{
        Booleanity, BooleanityCyclePhaseChallenges, BooleanityInputClaims,
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

    const CONFIGS: [(usize, usize); 4] = [(4, 13), (8, 13), (4, 9), (4, 3)];

    fn one_hot(chunk_bits: usize) -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: chunk_bits as u8,
            lookups_ra_virtual_log_k_chunk: if chunk_bits == 4 { 16 } else { 32 },
        }
    }

    #[test]
    fn fixture_booleanity_columns_are_one_hot_per_family() {
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
                        JoltRelationId::Booleanity,
                    )
                    .expect("formula dimensions")
                    .ra_layout;

                    let families: [(&str, Vec<JoltCommittedPolynomial>); 3] = [
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
                    ];

                    for (family, polynomials) in families {
                        assert!(
                            !polynomials.is_empty(),
                            "chunk_bits {chunk_bits} ram_log_k {ram_log_k}: the {family} family is \
                             empty, so this config cannot exercise it",
                        );
                        for polynomial in polynomials {
                            let hot = hot_addresses(witness, polynomial, addresses, cycles);
                            for (cycle, address) in hot.iter().enumerate() {
                                let cold = match family {
                                    "instruction" => false,
                                    "bytecode" => one_hot_fixture_bytecode_is_cold(LOG_T, cycle),
                                    _ => ram_fixture_is_cold(LOG_T, cycle),
                                };
                                assert_eq!(
                                    address.is_none(),
                                    cold,
                                    "chunk_bits {chunk_bits} ram_log_k {ram_log_k} \
                                     {polynomial:?} cycle {cycle}: the committed column disagrees \
                                     with the fixture on whether this cycle is cold",
                                );
                            }
                            let distinct: std::collections::BTreeSet<usize> =
                                hot.iter().flatten().copied().collect();
                            assert!(
                                distinct.len() > 1,
                                "chunk_bits {chunk_bits} ram_log_k {ram_log_k} {polynomial:?}: \
                                 the hot address is constant across every hot cycle, so this \
                                 chunk cannot detect a wrong shift",
                            );
                        }
                    }

                    let padding = (0..cycles)
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
                    assert!(padding > 0, "the fixture has no padding tail");
                    assert!(
                        bytecode_cold > 0 && bytecode_cold < cycles,
                        "{bytecode_cold} of {cycles} cycles are bytecode-cold, so one of the two \
                         paths is unexercised",
                    );
                    assert!(
                        ram_cold > 0 && ram_cold < cycles,
                        "{ram_cold} of {cycles} cycles are RAM-cold, so one of the two paths is \
                         unexercised",
                    );
                    assert!(
                        disagree > 0,
                        "the bytecode and RAM cold sets coincide, so a kernel reading the wrong \
                         family's cold flag would still pass",
                    );
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn booleanity_cycle_matches_reference(
            seed in any::<u64>(),
            r_address in arb_point(8),
            reference_address in arb_point(8),
            reference_cycle in arb_point(LOG_T),
            challenges in arb_point(LOG_T),
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
                            JoltRelationId::Booleanity,
                        )
                        .expect("formula dimensions")
                        .ra_layout;
                        let dimensions =
                            BooleanityDimensions::new(layout, LOG_T, chunk_bits);
                        let relation = Booleanity::<Fr>::new(
                            dimensions,
                            r_address[..chunk_bits].to_vec(),
                            reference_address[..chunk_bits].to_vec(),
                            reference_cycle.clone(),
                        );
                        let claims = BooleanityInputClaims {
                            address_phase: Fr::from_u64(0),
                        };
                        let points = BooleanityInputClaims {
                            address_phase: [
                                &r_address[..chunk_bits],
                                reference_cycle.as_slice(),
                            ]
                            .concat(),
                        };
                        let challenge_set = BooleanityCyclePhaseChallenges { gamma };
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
                        let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
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
