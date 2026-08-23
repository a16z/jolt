use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionRaVirtualizationInputClaims, InstructionRaVirtualizationOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use super::{require_context, CudaBackend};
use crate::cuda::common::context::context_for;
use crate::cuda::common::device_columns::windowed_trace_columns;
use crate::cuda::common::devices::witness_windows;
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use one_hot::{DevicePackedRa, PackedRaShard, ShardedPackedRa};

pub(crate) mod one_hot;

const COMMITTED_PER_VIRTUAL: usize = 4;

pub struct InstructionRaVirtualizationKernel<F: Field> {
    relation: InstructionRaVirtualization<F>,
    one_hot: ShardedPackedRa<F>,
    eq: DeviceSplitEq<F>,
    virtual_polys: usize,
    unscale: Vec<F>,
    rounds_bound: usize,
    finals: Option<Vec<F>>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for InstructionRaVirtualizationKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("eq"), self.eq.device_bytes());
        visitor.visit_simple(
            allocative::Key::new("unscale"),
            self.unscale.len() * size_of::<F>(),
        );
        visitor.visit_simple(
            allocative::Key::new("finals"),
            self.finals.as_ref().map_or(0, |v| v.len() * size_of::<F>()),
        );
        visitor.exit();
    }
}

impl<F: Field> InstructionRaVirtualizationKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda instruction RA virtualization bind",
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

impl<F: Field> ProveRounds<F> for InstructionRaVirtualizationKernel<F> {
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
        let evals = self
            .one_hot
            .round_evals(self.virtual_polys, &self.eq)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda instruction RA virtualization round",
            })?;
        let mut coefficients = self
            .eq
            .gruen_poly_from_evals(&evals, previous_claim)
            .into_coefficients();
        coefficients.resize(self.relation.degree() + 1, F::from_u64(0));
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for InstructionRaVirtualizationKernel<F> {
    type Relation = InstructionRaVirtualization<F>;

    fn output_claims(
        &mut self,
        _inputs: &InstructionRaVirtualizationInputClaims<F>,
    ) -> Result<InstructionRaVirtualizationOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let finals = self
            .finals
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA instruction RA virtualization never read back its bound claims",
            })?;
        Ok(InstructionRaVirtualizationOutputClaims {
            committed_instruction_ra: finals
                .iter()
                .zip(&self.unscale)
                .map(|(claim, unscale)| *claim * *unscale)
                .collect(),
        })
    }
}

impl<F: Field> PrepareKernel<F, InstructionRaVirtualization<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionRaVirtualization<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        if dimensions.num_committed_per_virtual() != COMMITTED_PER_VIRTUAL {
            return Err(KernelError::Unsupported {
                reason: "CUDA instruction RA virtualization supports four committed RA \
                         polynomials per virtual RA, the count both production one-hot \
                         configs produce",
            });
        }
        if relation.instruction_address().len()
            != dimensions.num_committed_ra_polys() * relation.committed_chunk_bits()
        {
            return Err(KernelError::InvariantViolation {
                reason: "the instruction address point does not split evenly into one \
                         committed-width chunk per committed RA polynomial",
            });
        }
        if relation.instruction_read_raf_cycle().len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "the instruction read-RAF cycle point has the wrong variable count",
            });
        }

        let unsupported = || KernelError::Unsupported {
            reason: "the CUDA instruction RA virtualization kernel supports only the BN254 \
                     scalar field",
        };
        let committed = dimensions.num_committed_ra_polys();
        let virtual_polys = dimensions.num_virtual_ra_polys();
        let chunk_bits = relation.committed_chunk_bits();

        let mut gamma_powers = Vec::with_capacity(virtual_polys);
        let mut power = F::from_u64(1);
        for _ in 0..virtual_polys {
            gamma_powers.push(power);
            power *= inputs.challenges.gamma;
        }
        let mut seeds = Vec::with_capacity(committed);
        let mut unscale = Vec::with_capacity(committed);
        for index in 0..committed {
            if index.is_multiple_of(COMMITTED_PER_VIRTUAL) {
                let gamma = gamma_powers[index / COMMITTED_PER_VIRTUAL];
                seeds.push(gamma);
                unscale.push(gamma.inverse().ok_or(KernelError::InvariantViolation {
                    reason: "the RA virtualization batching scalar must be invertible to undo \
                             the per-group pre-scaling on the produced openings",
                })?);
            } else {
                seeds.push(F::from_u64(1));
                unscale.push(F::from_u64(1));
            }
        }

        let log_t = dimensions.log_t();
        let cycles = 1usize << log_t;
        let windows = witness_windows(cycles);
        let shards = windows.len();
        let mut one_hot_shards = Vec::with_capacity(shards);
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "an instruction RA virtualization window names an absent device",
            })?;
            let columns = windowed_trace_columns::<F>(
                device,
                session,
                witness,
                cycles,
                window,
                [1, 0, 0],
                0,
            )?;
            one_hot_shards.push(PackedRaShard {
                ordinal,
                one_hot: DevicePackedRa::new(
                    device,
                    columns.lookup,
                    window.len,
                    chunk_bits,
                    relation.instruction_address(),
                    &seeds,
                )
                .map_err(|_| unsupported())?,
                eq: DeviceSplitEq::new_window(
                    device,
                    relation.instruction_read_raf_cycle(),
                    BindingOrder::LowToHigh,
                    ordinal,
                    shards,
                )?,
            });
        }
        let eq = DeviceSplitEq::new(
            require_context()?,
            relation.instruction_read_raf_cycle(),
            BindingOrder::LowToHigh,
        )?;

        Ok(Box::new(InstructionRaVirtualizationKernel {
            relation: relation.clone(),
            one_hot: ShardedPackedRa::new(one_hot_shards, log_t)?,
            eq,
            virtual_polys,
            unscale,
            rounds_bound: 0,
            finals: None,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionRaVirtualizationDimensions;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId,
    };
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage6b::instruction_ra_virtualization::{
        InstructionRaVirtualization, InstructionRaVirtualizationChallenges,
        InstructionRaVirtualizationInputClaims,
    };
    use jolt_witness::JoltWitnessOracle;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_instruction_witness,
    };
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 5;

    const ADDRESS_BITS: usize = 128;

    const CONFIGS: [(usize, usize); 2] = [(4, 16), (8, 32)];

    fn one_hot(log_k_chunk: usize, virtual_chunk_bits: usize) -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: log_k_chunk as u8,
            lookups_ra_virtual_log_k_chunk: virtual_chunk_bits as u8,
        }
    }

    fn dimensions(
        log_k_chunk: usize,
        virtual_chunk_bits: usize,
    ) -> InstructionRaVirtualizationDimensions {
        InstructionRaVirtualizationDimensions::try_from((
            LOG_T,
            ADDRESS_BITS / virtual_chunk_bits,
            virtual_chunk_bits / log_k_chunk,
        ))
        .expect("production RA virtualization dimensions")
    }

    #[test]
    fn fixture_committed_ra_columns_are_one_hot_and_vary_per_chunk() {
        let cycles = 1usize << LOG_T;
        for (log_k_chunk, virtual_chunk_bits) in CONFIGS {
            let addresses = 1usize << log_k_chunk;
            let chunks = ADDRESS_BITS / log_k_chunk;
            with_instruction_witness(LOG_T, one_hot(log_k_chunk, virtual_chunk_bits), 7, |w| {
                for chunk in 0..chunks {
                    let grid = JoltWitnessOracle::<Fr>::oracle_table(
                        w,
                        JoltPolynomialId::Committed(JoltCommittedPolynomial::InstructionRa(chunk)),
                    )
                    .expect("committed instruction ra column");
                    assert_eq!(grid.len(), addresses * cycles);

                    let mut hot = Vec::with_capacity(cycles);
                    for cycle in 0..cycles {
                        let mut found = None;
                        for address in 0..addresses {
                            if grid[address * cycles + cycle] != Fr::from_u64(0) {
                                assert_eq!(
                                    grid[address * cycles + cycle],
                                    Fr::from_u64(1),
                                    "chunk {chunk} cycle {cycle}: one-hot entry is not 1",
                                );
                                assert!(
                                    found.is_none(),
                                    "chunk {chunk} cycle {cycle}: two hot addresses",
                                );
                                found = Some(address);
                            }
                        }
                        hot.push(found.unwrap_or_else(|| {
                            panic!("chunk {chunk} cycle {cycle}: no hot address")
                        }));
                    }

                    let distinct: std::collections::BTreeSet<usize> = hot.iter().copied().collect();
                    assert!(
                        distinct.len() > 1,
                        "log_k_chunk {log_k_chunk} chunk {chunk}: hot address is constant \
                         across all {cycles} cycles, so this chunk cannot detect a wrong shift",
                    );
                }
            });
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn instruction_ra_virtualization_matches_reference(
            seed in any::<u64>(),
            instruction_address in arb_point(ADDRESS_BITS),
            instruction_read_raf_cycle in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            for (log_k_chunk, virtual_chunk_bits) in CONFIGS {
                let dimensions = dimensions(log_k_chunk, virtual_chunk_bits);
                let relation = InstructionRaVirtualization::<Fr>::new(
                    dimensions,
                    instruction_address.clone(),
                    instruction_read_raf_cycle.clone(),
                    log_k_chunk,
                );
                let claims = InstructionRaVirtualizationInputClaims {
                    instruction_ra: vec![Fr::from_u64(0); dimensions.num_virtual_ra_polys()],
                };
                let points = InstructionRaVirtualizationInputClaims {
                    instruction_ra: (0..dimensions.num_virtual_ra_polys())
                        .map(|virtual_index| {
                            (0..virtual_chunk_bits + LOG_T)
                                .map(|bit| fr((virtual_index * 131 + bit * 17 + 5) as u64))
                                .collect()
                        })
                        .collect(),
                };
                let challenge_set = InstructionRaVirtualizationChallenges { gamma };

                with_instruction_witness(
                    LOG_T,
                    one_hot(log_k_chunk, virtual_chunk_bits),
                    seed,
                    |witness| {
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
                            "round polynomials diverged at log_k_chunk {}",
                            log_k_chunk
                        );

                        let expected_claims = expected_kernel
                            .output_claims(&claims)
                            .expect("reference claims");
                        let got_claims =
                            got_kernel.output_claims(&claims).expect("cuda claims");
                        prop_assert_eq!(
                            got_claims.opening_values(),
                            expected_claims.opening_values(),
                            "output claims diverged at log_k_chunk {}",
                            log_k_chunk
                        );
                        Ok(())
                    },
                )?;
            }
        }
    }
}
