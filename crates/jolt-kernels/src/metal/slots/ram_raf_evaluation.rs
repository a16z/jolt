//! Metal RAM RAF evaluation (stage 2): device twin of the optimized slot
//! (which runs the naive round loop over `O(T + K)`-built tables),
//! byte-identical round polynomials by construction.
//!
//! Table construction ([`build_raf_tables`]: the analytic `unmap` column and
//! the cycle-folded `ra` off the session-shared RAM access columns) is the
//! optimized kernel's. The summand `unmap(k) · ra_folded(k)` is a single
//! table pair, so the rounds REUSE `jk_table_pairs_round` with
//! `num_tables = 1` — no new shader. The naive tier computes `s(1)`
//! directly where this kernel recovers it from the previous claim; on
//! honest claims those coincide, so the wire polynomials are byte-equal
//! (`byte_diff` pins it end to end).

use jolt_claims::protocols::jolt::{JoltDerivedId, RamRafEvaluationPublic};
use jolt_field::{Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, DeviceRound, Partials, RoundTable};
use crate::metal::field::fr_to_u32_limbs;
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::ram_raf_evaluation::{build_raf_tables, RafTables};
#[cfg(feature = "parallel")]
use crate::optimized::support::merge_evals;
use crate::optimized::support::round_poly_from_skipped_evals;
use crate::optimized::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

use jolt_claims::protocols::jolt::relations::ram::RamRafEvaluationOutputClaims;

const KIND: &str = "ram_raf_evaluation";

/// Slot front: device kernel above the [`metal_gate`] threshold, the
/// optimized fallback below it or on any device failure.
pub struct MetalRamRafEvaluation {
    pub fallback: OptimizedBackend,
}

impl PrepareKernel<Fr, RamRafEvaluation<Fr>> for MetalRamRafEvaluation {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, RamRafEvaluation<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = RamRafEvaluation<Fr>>>, KernelError<Fr>> {
        if metal_gate(KIND, 1usize << inputs.relation.ram_log_k()) {
            match MetalContext::global() {
                Ok(context) => {
                    // Structural errors (including the non-default-config
                    // Unsupported) propagate — the fallback would fail
                    // identically; only device failures fall back.
                    let tables = build_raf_tables(session, witness, &inputs)?;
                    match MetalRafKernel::new(context, inputs.relation.ram_log_k(), tables) {
                        Ok(kernel) => return Ok(Box::new(kernel)),
                        Err(error) => tracing::warn!(
                            slot = KIND,
                            %error,
                            "device prepare failed; using the optimized fallback"
                        ),
                    }
                }
                Err(error) => tracing::warn!(
                    slot = KIND,
                    %error,
                    "no device context; using the optimized fallback"
                ),
            }
        }
        self.fallback.prepare(session, witness, inputs)
    }
}

struct MetalRafKernel {
    rounds: usize,
    rounds_bound: usize,
    /// Current logical table length (both tables stay same-length).
    len: usize,
    ra_folded: RoundTable,
    unmap: RoundTable,
    partials: Partials,
    device: DeviceRound,
}

impl MetalRafKernel {
    fn new(
        context: &'static MetalContext,
        rounds: usize,
        tables: RafTables<Fr>,
    ) -> Result<Self, MetalError> {
        let len = tables.ra_folded.len();
        if len < 2 {
            return Err(MetalError::UnsupportedShape(
                "RAF evaluation needs a table of length >= 2",
            ));
        }
        Ok(Self {
            rounds,
            rounds_bound: 0,
            len,
            ra_folded: RoundTable::new(context, tables.ra_folded)?,
            unmap: RoundTable::new(context, tables.unmap)?,
            partials: Partials::new(context, 2, len / 2)?,
            device: DeviceRound::new(context, KIND),
        })
    }

    fn bind_bookkeeping(&mut self) {
        self.len /= 2;
        self.rounds_bound += 1;
    }

    /// One fused device round via the shared table-pairs kernel at
    /// `num_tables = 1`: one dispatch, one command buffer, one wait.
    fn device_round(
        &self,
        context: &MetalContext,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let num_tgs = num_threadgroups(groups);
        // TablePairsParams: [log_h, num_tables, len, do_bind, num_tgs, r].
        let mut params = vec![
            groups.ilog2(),
            1,
            self.len as u32,
            u32::from(bind.is_some()),
            num_tgs as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let buffers = [
            self.ra_folded.cur().device_buffer(),
            self.unmap.cur().device_buffer(),
            self.ra_folded.nxt().device_buffer(),
            self.unmap.nxt().device_buffer(),
            self.partials.buffer().device_buffer(),
        ];
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::TablePairsRound,
            &params,
            &[
                &buffers[0],
                &buffers[1],
                &buffers[2],
                &buffers[3],
                &buffers[4],
            ],
            groups,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    /// The CPU twin over the same unified-memory tables.
    fn cpu_round(&mut self, bind: Option<Fr>) -> Vec<Fr> {
        if let Some(challenge) = bind {
            let len = self.len;
            self.ra_folded.bind_cpu(len, challenge);
            self.unmap.bind_cpu(len, challenge);
            self.bind_bookkeeping();
        }
        let ra = self.ra_folded.cur_slice(self.len);
        let unmap = self.unmap.cur_slice(self.len);
        let group = |y: usize| -> [Fr; 2] {
            let at_two = |lo: Fr, hi: Fr| hi + hi - lo;
            [
                ra[2 * y] * unmap[2 * y],
                at_two(ra[2 * y], ra[2 * y + 1]) * at_two(unmap[2 * y], unmap[2 * y + 1]),
            ]
        };
        let half = self.len / 2;
        #[cfg(feature = "parallel")]
        {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![Fr::from_u64(0); 2],
                    |mut acc, y| {
                        let evals = group(y);
                        acc[0] += evals[0];
                        acc[1] += evals[1];
                        acc
                    },
                )
                .reduce(|| vec![Fr::from_u64(0); 2], merge_evals)
        }
        #[cfg(not(feature = "parallel"))]
        (0..half).fold(vec![Fr::from_u64(0); 2], |mut acc, y| {
            let evals = group(y);
            acc[0] += evals[0];
            acc[1] += evals[1];
            acc
        })
    }
}

impl ProveRounds<Fr> for MetalRafKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        let device = if groups == 0 {
            None
        } else {
            self.device.gated(self.len)
        };
        let evals = match device {
            Some(context) => match self.device_round(context, bind, groups) {
                Ok(sums) => {
                    if bind.is_some() {
                        self.ra_folded.swap();
                        self.unmap.swap();
                        self.bind_bookkeeping();
                    }
                    sums
                }
                Err(error) => {
                    self.device.failed(&error);
                    self.cpu_round(bind)
                }
            },
            None => self.cpu_round(bind),
        };
        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        let len = self.len;
        self.ra_folded.bind_cpu(len, bind);
        self.unmap.bind_cpu(len, bind);
        self.bind_bookkeeping();
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalRafKernel {
    type Relation = RamRafEvaluation<Fr>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<RamRafEvaluationOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        Ok(RamRafEvaluationOutputClaims {
            ram_ra: self.ra_folded.cur_slice(1)[0],
        })
    }

    /// The bound unmap column against the verifier's `derive_output_term` —
    /// the same drift detector the naive tier runs on its derived table.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<Fr, Self::Relation>,
        output_points: &SumcheckOutputPoints<Fr, Self::Relation>,
        challenges: &jolt_claims::NoChallenges<Fr>,
    ) -> Result<(), SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        let id = JoltDerivedId::from(RamRafEvaluationPublic::UnmapAddress);
        let expected =
            match relation.derive_output_term(&id, input_points, output_points, challenges) {
                Ok(value) => value,
                Err(VerifierError::MissingStageClaimDerived { .. }) => return Ok(()),
                Err(error) => return Err(error.into()),
            };
        let got = self.unmap.cur_slice(1)[0];
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

/// Parity against the optimized kernel (the naive round loop, whose in-round
/// claim check also cross-validates this kernel's recovered `s(1)`), device
/// path forced and probed.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{
        ram_ra_raf_evaluation, RamRafEvaluationDimensions,
    };
    use jolt_claims::protocols::jolt::relations::ram::RamRafEvaluationInputClaims;
    use jolt_claims::NoChallenges;

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::testing::{
        assert_parity, fixture_lowest_address, random_scalars, with_ram_fixture, FixtureShape,
        RamOp,
    };
    use crate::reference::views::cycle_fold;

    #[test]
    fn matches_optimized_on_mixed_traffic() {
        let _lock = gpu_lock();
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        let shape = FixtureShape { log_t: 4, ram_k: 8 };
        let ops = vec![
            RamOp::Write { word: 6, post: 2 },
            RamOp::Read { word: 6 },
            RamOp::None,
            RamOp::Write { word: 3, post: 9 },
            RamOp::Read { word: 3 },
            RamOp::Read { word: 6 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let tau_low = random_scalars(shape.log_t, 83);
            let read_write_dimensions =
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k());
            let relation = RamRafEvaluation::<Fr>::new(
                read_write_dimensions,
                RamRafEvaluationDimensions::try_from(read_write_dimensions).unwrap(),
                shape.log_k(),
                fixture_lowest_address(),
                tau_low.clone(),
            );
            let claims = RamRafEvaluationInputClaims {
                ram_address: Fr::from_u64(0),
            };
            let points = RamRafEvaluationInputClaims::<Vec<Fr>>::default();
            let challenges = NoChallenges::default();

            let mut optimized_session = ProofSession::default();
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut optimized_session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
            let mut session = ProofSession::default();
            let metal_slot = MetalRamRafEvaluation {
                fallback: OptimizedBackend,
            };
            let metal = metal_slot
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            // The independently folded true input claim:
            // `Σ_k unmap(k) · ra_folded(k)`.
            let ra_folded =
                cycle_fold::<Fr>(witness, ram_ra_raf_evaluation(), shape.log_k(), &tau_low)
                    .unwrap();
            let lowest = fixture_lowest_address();
            let input_claim = (0..shape.ram_k as u64)
                .map(|k| Fr::from_u64(8 * k + lowest) * ra_folded[k as usize])
                .sum();

            let rounds_before = device_probe_count();
            assert_parity(
                optimized,
                metal,
                input_claim,
                &ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
                89,
            );
            assert!(
                device_probe_count() > rounds_before,
                "the metal kernel never dispatched on the device"
            );
        });
    }
}
