//! CUDA reduction of RAM and register increment claims.
//!
//! As in optimized, four equality-weighted terms combine into two paired
//! weight tables, one per signed increment column. Split equality factors
//! stay small; the first bind promotes two half-size increment tables. Once
//! the low factors are exhausted, each pair becomes one dense high table.
//!
//! CUDA keeps large tables in aligned resident device windows and gathers the
//! small dense tail on GPU 0. Native first-round products accumulate before
//! field reduction; later rounds bind and evaluate field tables on device.
//! One- and two-cycle domains use optimized directly.

use std::marker::PhantomData;

use jolt_claims::protocols::jolt::geometry::claim_reductions::increments::{
    ram_inc_reduced, rd_inc_reduced,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::{
    IncClaimReductionInputClaims, IncClaimReductionOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::{Fr, JoltField, Zero};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_into, require_fr};
use crate::cuda::common::devices::{fan_out, witness_windows, DeviceTask};
use crate::cuda::common::error::{backend, CudaError};
use crate::cuda::witness::session_memory_window;
use crate::cuda::{require_context, CudaBackend};
use crate::optimized::inc_claim_reduction::OptimizedIncClaimReduction;
use crate::optimized::support::{round_poly_from_skipped_evals, RoundProgress};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

mod device;
use device::{device, invariant, IncrementShard};

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct IncClaimReductionKernel<F: JoltField> {
    shards: Vec<IncrementShard>,
    progress: RoundProgress,
    field: PhantomData<F>,
}

impl<F: JoltField> PrepareKernel<F, IncClaimReduction<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, IncClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = IncClaimReduction<F>>>, KernelError<F>> {
        let _ = require_context()?;
        let log_t = inputs.relation.symbolic().rounds();
        let points = inputs.relation.cycle_points();
        if points.iter().any(|point| point.len() != log_t) {
            return Err(invariant("increment cycle point has the wrong variable count").into());
        }
        if log_t < 2 {
            return OptimizedIncClaimReduction.prepare(session, witness, inputs);
        }
        let cycles = 1usize
            .checked_shl(
                u32::try_from(log_t).map_err(|_| invariant("increment domain is too large"))?,
            )
            .ok_or(invariant("increment domain is too large"))?;
        let _ = CudaKernelContext::count_of(cycles)?;
        for id in [ram_inc_reduced(), rd_inc_reduced()] {
            let rows = witness.shape(id.polynomial_id())?.rows();
            if rows != cycles {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{id:?}"),
                    expected: cycles,
                    got: rows,
                });
            }
        }
        let mut shards = Vec::new();
        for (ordinal, window) in witness_windows(cycles).iter().enumerate() {
            let context = device(ordinal)?;
            let (_, atoms) = session_memory_window(context, session, witness, cycles, window)?;
            shards.push(IncrementShard::new(
                context,
                window,
                atoms,
                points,
                inputs.challenges.gamma,
            )?);
        }
        Ok(Box::new(IncClaimReductionKernel {
            shards,
            progress: RoundProgress::new(log_t),
            field: PhantomData,
        }))
    }
}

impl<F: JoltField> IncClaimReductionKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), CudaError> {
        if self.progress.bound() >= self.progress.total() {
            return Err(invariant("increment reduction is already fully bound"));
        }
        let challenge = require_fr(challenge)?;
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
            .shards
            .iter_mut()
            .map(|shard| {
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || shard.bind(challenge));
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        if self.shards.len() > 1 && self.shards.iter().all(IncrementShard::ready_to_consolidate) {
            self.shards = vec![IncrementShard::consolidate(&self.shards)?];
        }
        self.progress.advance();
        Ok(())
    }

    fn message(&self, previous_claim: F) -> Result<UnivariatePoly<F>, CudaError> {
        let tasks: Vec<DeviceTask<'_, [Fr; 2], CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, [Fr; 2], CudaError> = Box::new(move || shard.message());
                task
            })
            .collect();
        let sum = fan_out(tasks)?
            .into_iter()
            .fold([Fr::zero(); 2], |[a, b], [c, d]| [a + c, b + d]);
        let sum = [convert(sum[0])?, convert(sum[1])?];
        Ok(round_poly_from_skipped_evals(&sum, previous_claim))
    }
}

impl<F: JoltField> ProveRounds<F> for IncClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r).map_err(backend("CUDA increment bind"))?;
        }
        self.message(previous_claim)
            .map_err(backend("CUDA increment message"))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind).map_err(backend("CUDA increment finish"))
    }
}

impl<F: JoltField> SumcheckKernel<F> for IncClaimReductionKernel<F> {
    type Relation = IncClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &IncClaimReductionInputClaims<F>,
    ) -> Result<IncClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let failed = || SumcheckKernelError::InvariantViolation {
            reason: "CUDA increment final claims unavailable",
        };
        let [shard] = self.shards.as_slice() else {
            return Err(failed());
        };
        let [ram, rd] = shard.final_values().map_err(|_| failed())?;
        Ok(IncClaimReductionOutputClaims {
            ram_inc: convert(ram).map_err(|_| failed())?,
            rd_inc: convert(rd).map_err(|_| failed())?,
        })
    }
}

fn convert<F: JoltField>(value: Fr) -> Result<F, CudaError> {
    fr_into(value).ok_or(invariant("increment reduction requires BN254"))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
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
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
    use proptest::prelude::*;

    use super::device::{device, IncrementShard};
    use super::{CudaBackend, IncClaimReductionKernel};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::devices::witness_windows;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, with_r1cs_witness,
    };
    use crate::optimized::support::RoundProgress;
    use crate::reference::views::dense_view;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};
    use jolt_poly::{EqPolynomial, Polynomial};
    use jolt_witness::backend::cuda::DeviceMemoryColumns;
    use std::marker::PhantomData;
    use std::sync::Arc;

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

    #[test]
    fn tiny_domains_match_reference() {
        let Some(_) = shared_context() else {
            return;
        };
        for log_t in [0usize, 1] {
            let point = vec![fr(7); log_t];
            let challenges = vec![fr(11); log_t];
            with_r1cs_witness(log_t, RAM_K, one_hot(), 3, |witness| {
                let relation = IncClaimReduction::<Fr>::new(
                    TraceDimensions::new(log_t),
                    point.clone(),
                    point.clone(),
                    point.clone(),
                    point.clone(),
                );
                let claims = IncClaimReductionInputClaims::default();
                let points = IncClaimReductionInputClaims::default();
                let challenge_set = IncClaimReductionChallenges { gamma: fr(5) };
                let make_inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenge_set,
                };

                let mut expected_kernel = ReferenceBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("reference prepare");
                let mut got_kernel = CudaBackend
                    .prepare(&mut ProofSession::default(), witness, make_inputs())
                    .expect("cuda prepare on a tiny domain");

                if log_t > 0 {
                    let input_claim = reference_input_claim(witness, make_inputs);
                    let expected = drive(&mut *expected_kernel, input_claim, &challenges);
                    let got = drive(&mut *got_kernel, input_claim, &challenges);
                    assert_eq!(got, expected, "round polynomials diverged on a tiny domain");
                } else {
                    assert_eq!(got_kernel.num_rounds(), 0);
                }
                assert_eq!(
                    got_kernel
                        .output_claims(&claims)
                        .expect("cuda claims")
                        .opening_values(),
                    expected_kernel
                        .output_claims(&claims)
                        .expect("reference claims")
                        .opening_values(),
                    "output claims diverged on a tiny domain",
                );
            });
        }
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

    #[test]
    fn signed_increments_match_multilinear_evaluations() {
        let Some(_) = shared_context() else {
            return;
        };
        for log_t in [2usize, 3, 13] {
            let len = 1usize << log_t;
            let windows = witness_windows(len);
            let ram: Vec<i128> = (0..len)
                .map(|i| {
                    let k = (i / 4) as i128;
                    [i128::MIN + k, -1 - k, 0, i128::MAX - k][i % 4]
                })
                .collect();
            let rd: Vec<i128> = (0..len)
                .map(|i| {
                    let k = (i / 4) as i128;
                    [1 + k, i128::MAX - k, -(1i128 << 99) + k, 0][i % 4]
                })
                .collect();
            for case in 0..4 {
                let points: [Vec<Fr>; 4] = std::array::from_fn(|k| {
                    (0..log_t)
                        .map(|i| {
                            Fr::from_u64(match case {
                                0 => ((i + k) % 2) as u64,
                                1 => 1,
                                2 => (k * 5 + i + 3) as u64,
                                _ => (i + 5) as u64,
                            })
                        })
                        .collect()
                });
                let gamma = match case {
                    1 => Fr::from_u64(0),
                    3 => -Fr::from_u64(1),
                    _ => Fr::from_u64(13),
                };
                let mut power = Fr::from_u64(1);
                let mut weights: Vec<_> = points
                    .iter()
                    .map(|p| {
                        let values = EqPolynomial::<Fr>::evals(p, Some(power));
                        power *= gamma;
                        Polynomial::new(values)
                    })
                    .collect();
                let mut columns = [
                    Polynomial::new(ram.iter().map(|x| Fr::from_i128(*x)).collect()),
                    Polynomial::new(rd.iter().map(|x| Fr::from_i128(*x)).collect()),
                ];
                let mut shards = Vec::new();
                for (ordinal, window) in windows.iter().enumerate() {
                    let context = device(ordinal).unwrap();
                    let words = |column: &[i128]| {
                        column[window.start..window.end()]
                            .iter()
                            .flat_map(|x| [*x as u64, (*x >> 64) as u64])
                            .collect::<Vec<_>>()
                    };
                    let atoms = Arc::new(DeviceMemoryColumns {
                        rd_pre_value: context.stream().alloc_zeros::<u64>(0).unwrap(),
                        rs1_address: context.stream().alloc_zeros::<u32>(0).unwrap(),
                        rs2_address: context.stream().alloc_zeros::<u32>(0).unwrap(),
                        rd_address: context.stream().alloc_zeros::<u32>(0).unwrap(),
                        ram_inc: context.upload_u64_slice(&words(&ram)).unwrap(),
                        rd_inc: context.upload_u64_slice(&words(&rd)).unwrap(),
                    });
                    shards.push(
                        IncrementShard::new(
                            context,
                            window,
                            atoms,
                            points.each_ref().map(Vec::as_slice),
                            gamma,
                        )
                        .unwrap(),
                    );
                }
                let mut kernel = IncClaimReductionKernel::<Fr> {
                    shards,
                    progress: RoundProgress::new(log_t),
                    field: PhantomData,
                };
                for round in 0..log_t {
                    let mut expected = [Fr::from_u64(0); 3];
                    for row in 0..columns[0].len() / 2 {
                        for (x, sum) in expected.iter_mut().enumerate() {
                            let x = Fr::from_u64(x as u64);
                            let value = |p: &Polynomial<Fr>| {
                                p.evals()[2 * row]
                                    + x * (p.evals()[2 * row + 1] - p.evals()[2 * row])
                            };
                            for k in 0..4 {
                                *sum += value(&weights[k]) * value(&columns[k / 2]);
                            }
                        }
                    }
                    let actual = kernel.message(expected[0] + expected[1]).unwrap();
                    for (x, expected) in expected.into_iter().enumerate() {
                        assert_eq!(
                            actual.evaluate(Fr::from_u64(x as u64)),
                            expected,
                            "log_t {log_t}, case {case}, round {round}"
                        );
                    }
                    let r = Fr::from_u64(if case < 2 {
                        ((round + 1) % 2) as u64
                    } else {
                        (round + 17) as u64
                    });
                    kernel.bind(r).unwrap();
                    for p in weights.iter_mut().chain(columns.iter_mut()) {
                        let _ = p.bind_low_to_high_in_place(r);
                    }
                }
                let claims = kernel
                    .output_claims(&IncClaimReductionInputClaims::default())
                    .unwrap();
                assert_eq!(claims.ram_inc, columns[0].evals()[0]);
                assert_eq!(claims.rd_inc, columns[1].evals()[0]);
            }
        }
    }
}
