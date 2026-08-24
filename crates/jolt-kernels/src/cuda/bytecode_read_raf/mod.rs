use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{try_eq_mle, IdentityPolynomial, MultilinearEvaluation, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::JoltWitnessPlane;

use super::CudaBackend;
use crate::cuda::common::context::context_for;
use crate::cuda::common::device_columns::device_trace_columns;
use crate::cuda::common::devices::witness_windows;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use coefficient::DeviceCoefficient;
use one_hot::{BytecodeShard, DeviceBytecodeRa, ShardedBytecodeRa};

pub(crate) mod address;
pub(crate) mod coefficient;
pub(crate) mod one_hot;
pub(crate) mod pushforward;

pub struct BytecodeReadRafCycleKernel<F: Field> {
    relation: BytecodeReadRafCycle<F>,
    one_hot: ShardedBytecodeRa,
    rounds_bound: usize,
    finals: Option<Vec<F>>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for BytecodeReadRafCycleKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("finals"),
            self.finals.as_ref().map_or(0, |v| v.len() * size_of::<F>()),
        );
        visitor.exit();
    }
}

impl<F: Field> BytecodeReadRafCycleKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda bytecode read-RAF cycle-phase bind",
        };
        self.one_hot
            .bind(challenge, self.rounds_bound)
            .map_err(|_| failed())?;
        self.rounds_bound += 1;
        if self.rounds_bound == self.relation.symbolic().rounds() {
            self.finals = Some(self.one_hot.final_claims().map_err(|_| failed())?);
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for BytecodeReadRafCycleKernel<F> {
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
        let evals =
            self.one_hot
                .round_evals()
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda bytecode read-RAF cycle-phase round",
                })?;
        let mut toom = Vec::with_capacity(evals.len() + 1);
        toom.push(previous_claim - evals[0]);
        toom.extend_from_slice(&evals);
        let mut coefficients = UnivariatePoly::from_evals_toom(&toom).into_coefficients();
        coefficients.resize(self.relation.degree() + 1, F::from_u64(0));
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for BytecodeReadRafCycleKernel<F> {
    type Relation = BytecodeReadRafCycle<F>;

    fn output_claims(
        &mut self,
        _inputs: &BytecodeReadRafInputClaims<F>,
    ) -> Result<BytecodeReadRafOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let finals = self
            .finals
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA bytecode read-RAF never read back its bound claims",
            })?;
        if finals.len() != self.relation.dimensions().num_committed_ra_polys() {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA bytecode read-RAF produces one claim per committed RA chunk",
            });
        }
        Ok(BytecodeReadRafOutputClaims {
            bytecode_ra: finals.clone(),
        })
    }
}

fn eq_at_index<F: Field>(point: &[F], index: usize) -> Result<F, KernelError<F>> {
    let bits: Vec<F> = (0..point.len())
        .map(|bit| {
            if (index >> (point.len() - 1 - bit)) & 1 == 1 {
                F::one()
            } else {
                F::from_u64(0)
            }
        })
        .collect();
    try_eq_mle(point, &bits).map_err(|_| KernelError::InvariantViolation {
        reason: "bytecode entry index does not span the bytecode address point",
    })
}

impl<F: Field> PrepareKernel<F, BytecodeReadRafCycle<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let addresses = 1usize << dimensions.log_k();
        if relation.r_address().len() != dimensions.log_k() {
            return Err(KernelError::InvariantViolation {
                reason: "the bytecode read-RAF address point has the wrong variable count",
            });
        }
        if relation
            .stage_cycle_points()
            .iter()
            .any(|point| point.len() != dimensions.log_t())
        {
            return Err(KernelError::InvariantViolation {
                reason: "a bytecode read-RAF stage cycle point has the wrong variable count",
            });
        }
        if relation.entry_bytecode_index() >= addresses {
            return Err(KernelError::InvariantViolation {
                reason: "the bytecode entry index falls outside the padded bytecode domain",
            });
        }
        let chunks =
            committed_address_chunks(relation.r_address(), relation.committed_chunk_bits());
        if chunks.len() != dimensions.num_committed_ra_polys() {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address chunk count disagrees with the committed RA count",
            });
        }
        if relation.symbolic().degree() != chunks.len() + 1 {
            return Err(KernelError::Unsupported {
                reason: "the CUDA bytecode read-RAF cycle kernel proves the committed RA product \
                         against one multilinear public coefficient, so its degree is the chunk \
                         count plus one",
            });
        }

        let values = relation.stage_values_at_r_address()?;
        let points = relation.stage_cycle_points();
        let stages = points.len();
        if values.len() != stages || stages < 3 {
            return Err(KernelError::Unsupported {
                reason: "the CUDA bytecode read-RAF cycle kernel folds one staged value per stage \
                         cycle point, with the two RAF terms riding stages one and three",
            });
        }

        let gamma = inputs.challenges.gamma;
        let mut powers = Vec::with_capacity(stages + 3);
        let mut power = F::one();
        for _ in 0..stages + 3 {
            powers.push(power);
            power *= gamma;
        }
        let identity = IdentityPolynomial::new(dimensions.log_k()).evaluate(relation.r_address());
        let mut weights: Vec<F> = (0..stages)
            .map(|stage| powers[stage] * values[stage])
            .collect();
        weights[0] += powers[stages] * identity;
        weights[2] += powers[stages + 1] * identity;
        let entry = powers[stages + 2]
            * eq_at_index(relation.r_address(), relation.entry_bytecode_index())?;

        let log_t = dimensions.log_t();
        let cycles = 1usize << log_t;

        let unsupported = || KernelError::Unsupported {
            reason: "the CUDA bytecode read-RAF kernel supports only the BN254 scalar field",
        };
        let windows = witness_windows(cycles);
        let shards = windows.len();
        let mut one_hot_shards = Vec::with_capacity(shards);
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a bytecode read-RAF cycle window names an absent device",
            })?;
            let columns =
                device_trace_columns::<F>(device, session, witness, cycles, window, [0, 1, 0], 0)?;
            one_hot_shards.push(BytecodeShard {
                ordinal,
                one_hot: DeviceBytecodeRa::new(
                    device,
                    columns.pc,
                    window.len,
                    relation.committed_chunk_bits(),
                    &chunks,
                )
                .map_err(|_| KernelError::Unsupported {
                    reason: "the CUDA bytecode read-RAF kernel packs every committed chunk index \
                             into one word per cycle and evaluates at most eight round-polynomial \
                             lanes",
                })?,
                coefficient: DeviceCoefficient::new_window(
                    device,
                    points.as_slice(),
                    &weights,
                    entry,
                    log_t,
                    ordinal,
                    shards,
                )
                .map_err(|_| unsupported())?,
            });
        }

        Ok(Box::new(BytecodeReadRafCycleKernel {
            relation: relation.clone(),
            one_hot: ShardedBytecodeRa::new(one_hot_shards, log_t)?,
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
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltRelationId};
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle,
        BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
        READ_RAF_CYCLE_STAGES,
    };
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, hot_addresses, one_hot_fixture_bytecode_is_cold,
        reference_input_claim, with_one_hot_witness,
    };
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const CONFIGS: [(usize, usize, usize, usize); 4] = [
        (8, 20, 13, 1),
        (4, 20, 13, 2),
        (4, 600, 9, 3),
        (4, 5000, 13, 4),
    ];

    fn one_hot(chunk_bits: usize) -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: chunk_bits as u8,
            lookups_ra_virtual_log_k_chunk: if chunk_bits == 4 { 16 } else { 32 },
        }
    }

    #[test]
    fn fixture_bytecode_chunks_span_the_production_width() {
        let cycles = 1usize << LOG_T;
        for (chunk_bits, bytecode_rows, ram_log_k, chunks) in CONFIGS {
            with_one_hot_witness(
                LOG_T,
                bytecode_rows,
                1usize << ram_log_k,
                one_hot(chunk_bits),
                7,
                |witness, bytecode_len| {
                    let dimensions = formula_dimensions_from_parts(
                        one_hot(chunk_bits),
                        LOG_T,
                        bytecode_len,
                        1usize << ram_log_k,
                        JoltRelationId::BytecodeReadRaf,
                    )
                    .expect("formula dimensions")
                    .bytecode_read_raf;
                    assert_eq!(
                        dimensions.num_committed_ra_polys(),
                        chunks,
                        "chunk_bits {chunk_bits} bytecode_rows {bytecode_rows}: the fixture no \
                         longer produces the chunk count this config exists to cover, so the \
                         lane-count coverage claim is stale",
                    );

                    for chunk in 0..chunks {
                        let hot = hot_addresses(
                            witness,
                            JoltCommittedPolynomial::BytecodeRa(chunk),
                            1usize << chunk_bits,
                            cycles,
                        );
                        for (cycle, address) in hot.iter().enumerate() {
                            assert_eq!(
                                address.is_none(),
                                one_hot_fixture_bytecode_is_cold(LOG_T, cycle),
                                "chunk_bits {chunk_bits} bytecode_rows {bytecode_rows} chunk \
                                 {chunk} cycle {cycle}: the committed column disagrees with the \
                                 fixture on whether this cycle is cold",
                            );
                        }
                        let distinct: std::collections::BTreeSet<usize> =
                            hot.iter().flatten().copied().collect();
                        assert!(
                            distinct.len() > 1,
                            "chunk_bits {chunk_bits} bytecode_rows {bytecode_rows} chunk {chunk}: \
                             the hot address is constant across every hot cycle, so this chunk \
                             cannot detect a wrong shift",
                        );
                    }
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn bytecode_read_raf_cycle_matches_reference(
            seed in any::<u64>(),
            r_address in arb_point(16),
            stage_cycles in proptest::collection::vec(arb_point(LOG_T), READ_RAF_CYCLE_STAGES),
            val_stages in proptest::collection::vec(
                any::<u64>().prop_map(fr),
                NUM_BYTECODE_VAL_STAGES,
            ),
            challenges in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            for (chunk_bits, bytecode_rows, ram_log_k, chunks) in CONFIGS {
                with_one_hot_witness(
                    LOG_T,
                    bytecode_rows,
                    1usize << ram_log_k,
                    one_hot(chunk_bits),
                    seed,
                    |witness, bytecode_len| {
                        let dimensions = formula_dimensions_from_parts(
                            one_hot(chunk_bits),
                            LOG_T,
                            bytecode_len,
                            1usize << ram_log_k,
                            JoltRelationId::BytecodeReadRaf,
                        )
                        .expect("formula dimensions")
                        .bytecode_read_raf;
                        prop_assert_eq!(
                            dimensions.num_committed_ra_polys(),
                            chunks,
                            "chunk_bits {} bytecode_rows {}: the fixture no longer produces the \
                             chunk count this config exists to cover",
                            chunk_bits,
                            bytecode_rows
                        );
                        let log_k = dimensions.log_k();
                        let address = r_address[..log_k].to_vec();
                        let stage_cycle_points: [Vec<Fr>; READ_RAF_CYCLE_STAGES] =
                            core::array::from_fn(|stage| stage_cycles[stage].clone());
                        let relation = BytecodeReadRafCycle::<Fr>::committed(
                            BytecodeReadRafCommittedCycleInputs {
                                dimensions,
                                r_address: address.clone(),
                                stage_cycle_points,
                                entry_bytecode_index: (1usize << log_k) - 2,
                                committed_chunk_bits: chunk_bits,
                                val_stages: val_stages.clone(),
                            },
                        );

                        let claims = BytecodeReadRafInputClaims {
                            address_phase: Fr::from_u64(0),
                        };
                        let points = BytecodeReadRafInputClaims {
                            address_phase: address,
                        };
                        let challenge_set =
                            BytecodeReadRafCyclePhaseCommittedChallenges { gamma };
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
                            "round polynomials diverged at chunk_bits {} bytecode_rows {} \
                             ram_log_k {}",
                            chunk_bits,
                            bytecode_rows,
                            ram_log_k
                        );

                        let expected_claims = expected_kernel
                            .output_claims(&claims)
                            .expect("reference claims");
                        let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
                        prop_assert_eq!(
                            got_claims.opening_values(),
                            expected_claims.opening_values(),
                            "output claims diverged at chunk_bits {} bytecode_rows {} \
                             ram_log_k {}",
                            chunk_bits,
                            bytecode_rows,
                            ram_log_k
                        );
                        Ok(())
                    },
                )?;
            }
        }
    }
}
