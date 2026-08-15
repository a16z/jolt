use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::ram::{
    RamRaVirtualizationInputClaims, RamRaVirtualizationOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::{require_context, CudaBackend};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use one_hot::DevicePackedRamRa;

pub(crate) mod one_hot;
pub(crate) mod witness;

pub struct RamRaVirtualizationKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: RamRaVirtualization<F>,
    one_hot: DevicePackedRamRa,
    eq: DeviceSplitEq<F>,
    rounds_bound: usize,
    finals: Option<Vec<F>>,
}

impl<F: Field> RamRaVirtualizationKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda RAM RA virtualization bind",
        };
        self.one_hot
            .bind(self.context, challenge)
            .map_err(|_| failed())?;
        self.eq.bind(challenge);
        self.rounds_bound += 1;
        if self.rounds_bound == self.relation.symbolic().rounds() {
            self.finals = Some(
                self.one_hot
                    .final_claims(self.context)
                    .map_err(|_| failed())?,
            );
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for RamRaVirtualizationKernel<F> {
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
            .round_evals(self.context, &self.eq)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda RAM RA virtualization round",
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

impl<F: Field> SumcheckKernel<F> for RamRaVirtualizationKernel<F> {
    type Relation = RamRaVirtualization<F>;

    fn output_claims(
        &mut self,
        _inputs: &RamRaVirtualizationInputClaims<F>,
    ) -> Result<RamRaVirtualizationOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let finals = self
            .finals
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA RAM RA virtualization never read back its bound claims",
            })?;
        Ok(RamRaVirtualizationOutputClaims {
            ram_ra: finals.clone(),
        })
    }
}

impl<F: Field> PrepareKernel<F, RamRaVirtualization<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamRaVirtualization<F>>>, KernelError<F>> {
        let context = require_context()?;
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let chunk_bits = relation.committed_chunk_bits();
        let chunks = committed_address_chunks(relation.ram_reduced_address(), chunk_bits);
        if chunks.len() != dimensions.num_committed_ra_polys() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address chunk count disagrees with the committed RA count",
            });
        }
        if relation.ram_reduced_cycle().len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "the RAM reduced cycle point has the wrong variable count",
            });
        }

        let unsupported = || KernelError::Unsupported {
            reason: "the CUDA RAM RA virtualization kernel supports only the BN254 scalar field",
        };
        let address_point: Vec<F> = chunks.concat();

        let cycles = 1usize << dimensions.log_t();
        let rows = collect_bundles::<witness::RamRaVirtualizationWitness>(witness, cycles)?;
        let words = witness::packed_words(&rows).map_err(|_| KernelError::Unsupported {
            reason: "the CUDA RAM RA virtualization kernel packs each remapped RAM word address \
                     into one 32-bit word, reserving the all-ones word for a cold cycle",
        })?;
        drop(rows);
        let packed = context
            .upload_u32_slice(&words)
            .map_err(|_| unsupported())?;
        drop(words);

        let one_hot = DevicePackedRamRa::new(context, packed, cycles, chunk_bits, &address_point)
            .map_err(|_| KernelError::Unsupported {
            reason: "the CUDA RAM RA virtualization kernel packs the remapped address \
                             into one 32-bit word and evaluates at most eight round-polynomial \
                             lanes",
        })?;
        let eq = DeviceSplitEq::new(
            context,
            relation.ram_reduced_cycle(),
            BindingOrder::LowToHigh,
        )?;

        Ok(Box::new(RamRaVirtualizationKernel {
            context,
            relation: relation.clone(),
            one_hot,
            eq,
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
    use jolt_claims::protocols::jolt::geometry::ram::RamRaVirtualizationDimensions;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId,
    };
    use jolt_claims::{NoChallenges, OutputClaims};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage6b::ram_ra_virtualization::{
        RamRaVirtualization, RamRaVirtualizationInputClaims,
    };
    use jolt_witness::JoltWitnessOracle;
    use proptest::prelude::*;

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, ram_fixture_is_cold, reference_input_claim, with_ram_witness,
    };
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const CONFIGS: [(usize, usize); 6] = [(4, 13), (8, 13), (4, 17), (4, 9), (4, 8), (4, 3)];

    fn one_hot(chunk_bits: usize) -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: chunk_bits as u8,
            lookups_ra_virtual_log_k_chunk: if chunk_bits == 4 { 16 } else { 32 },
        }
    }

    const fn chunks(chunk_bits: usize, ram_log_k: usize) -> usize {
        ram_log_k.div_ceil(chunk_bits)
    }

    #[test]
    fn fixture_ram_ra_columns_are_one_hot_with_cold_cycles() {
        let cycles = 1usize << LOG_T;
        for (chunk_bits, ram_log_k) in CONFIGS {
            let addresses = 1usize << chunk_bits;
            let count = chunks(chunk_bits, ram_log_k);
            with_ram_witness(LOG_T, 1usize << ram_log_k, one_hot(chunk_bits), 7, |w| {
                let mut cold_agreement = vec![None; cycles];
                for chunk in 0..count {
                    let grid = JoltWitnessOracle::<Fr>::oracle_table(
                        w,
                        JoltPolynomialId::Committed(JoltCommittedPolynomial::RamRa(chunk)),
                    )
                    .expect("committed ram ra column");
                    assert_eq!(grid.len(), addresses * cycles);

                    let mut hot = Vec::new();
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
                        match cold_agreement[cycle] {
                            None => cold_agreement[cycle] = Some(found.is_some()),
                            Some(previous) => assert_eq!(
                                previous,
                                found.is_some(),
                                "cycle {cycle}: chunk {chunk} disagrees with an earlier chunk on \
                                 whether the cycle is cold, but a cold cycle has no hot address \
                                 in ANY chunk",
                            ),
                        }
                        if let Some(address) = found {
                            hot.push(address);
                        }
                    }

                    let distinct: std::collections::BTreeSet<usize> = hot.iter().copied().collect();
                    assert!(
                        distinct.len() > 1,
                        "chunk_bits {chunk_bits} ram_log_k {ram_log_k} chunk {chunk}: hot address \
                         is constant across all {} hot cycles, so this chunk cannot detect a \
                         wrong shift",
                        hot.len(),
                    );
                }

                for (cycle, hot) in cold_agreement.iter().enumerate() {
                    assert_eq!(
                        *hot,
                        Some(!ram_fixture_is_cold(LOG_T, cycle)),
                        "chunk_bits {chunk_bits} ram_log_k {ram_log_k} cycle {cycle}: the \
                         committed columns disagree with the fixture on whether this cycle has a \
                         hot address",
                    );
                }
                let cold = cold_agreement
                    .iter()
                    .filter(|hot| *hot == &Some(false))
                    .count();
                assert!(
                    cold > 0 && cold < cycles,
                    "chunk_bits {chunk_bits} ram_log_k {ram_log_k}: {cold} of {cycles} cycles are \
                     cold, so one of the two paths is unexercised",
                );
            });
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn ram_ra_virtualization_matches_reference(
            seed in any::<u64>(),
            ram_reduced_address in arb_point(32),
            ram_reduced_cycle in arb_point(LOG_T),
            challenges in arb_point(LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };

            for (chunk_bits, ram_log_k) in CONFIGS {
                let address = ram_reduced_address[..ram_log_k].to_vec();
                let dimensions = RamRaVirtualizationDimensions::new(
                    LOG_T,
                    chunks(chunk_bits, ram_log_k),
                );
                let relation = RamRaVirtualization::<Fr>::new(
                    dimensions,
                    address.clone(),
                    ram_reduced_cycle.clone(),
                    chunk_bits,
                );
                let claims = RamRaVirtualizationInputClaims {
                    ram_ra_reduced: Fr::from_u64(0),
                };
                let points = RamRaVirtualizationInputClaims {
                    ram_ra_reduced: [address.as_slice(), ram_reduced_cycle.as_slice()].concat(),
                };
                let challenge_set = NoChallenges::<Fr>::default();

                with_ram_witness(
                    LOG_T,
                    1usize << ram_log_k,
                    one_hot(chunk_bits),
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
