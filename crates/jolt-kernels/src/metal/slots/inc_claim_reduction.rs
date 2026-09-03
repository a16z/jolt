//! Metal increment claim reduction (stage 6b): device twin of
//! [`OptimizedIncClaimReduction`], byte-identical round polynomials by
//! construction.
//!
//! The fast prepare path fills paired-eq weights on the device while the host
//! materializes the independent increment columns. Its fallback uses the
//! optimized kernel's [`crate::optimized::inc_claim_reduction::build_inc_tables`].
//! Rounds serve the four dense tables from unified-memory ping-pong buffers:
//! ONE `jk_inc_round` dispatch folds with the previous challenge and
//! tree-reduces the summand's `t ∈ {0, 2}` evaluations to per-threadgroup
//! partials. The host assembles the wire polynomial through the same
//! `round_poly_from_skipped_evals` recipe as the optimized tier.

use jolt_field::{Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6b::inc_claim_reduction::{
    IncClaimReduction, IncClaimReductionOutputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, slot_round_params, DeviceRound, Partials, RoundTable};
use crate::metal::buffers::{OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::runtime::{DetachedPass, KernelId, MetalContext};
use crate::metal::{fr_to_u32_limbs, metal_gate, testing, MetalError};
use crate::optimized::inc_claim_reduction::{
    materialize_inc_columns, validate_inc_relation, OptimizedIncClaimReduction,
};
#[cfg(feature = "parallel")]
use crate::optimized::support::merge_evals;
use crate::optimized::support::{eq_table, round_poly_from_skipped_evals};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "inc_claim_reduction";

/// Slot front: device kernel above the [`metal_gate`] threshold, the
/// optimized fallback below it or on any device failure.
pub struct MetalIncClaimReduction {
    pub fallback: OptimizedIncClaimReduction,
}

impl PrepareKernel<Fr, IncClaimReduction<Fr>> for MetalIncClaimReduction {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, IncClaimReduction<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = IncClaimReduction<Fr>>>, KernelError<Fr>>
    {
        if metal_gate(KIND, 1usize << inputs.relation.rounds()) {
            match MetalContext::global() {
                Ok(context) => {
                    let cycles = validate_inc_relation(inputs.relation)?;
                    let weights = DeviceIncWeights::launch(
                        context,
                        inputs.relation.cycle_points(),
                        inputs.challenges.gamma,
                    );
                    match weights {
                        Ok(weights) => {
                            // The independent raw-column walks cover the
                            // device fill's latency before its only wait.
                            let (ram_inc, rd_inc) = materialize_inc_columns(witness, cycles)?;
                            match weights.wait().and_then(|(ram_weights, rd_weights)| {
                                MetalIncKernel::new_prepared(
                                    context,
                                    inputs.relation.rounds(),
                                    ram_inc,
                                    rd_inc,
                                    ram_weights,
                                    rd_weights,
                                )
                            }) {
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

struct DeviceIncWeights {
    // Drops first on an early return, settling the flight before its buffers.
    pass: DetachedPass,
    ram: RoundTable,
    rd: RoundTable,
    factors: OwnedDeviceBuffer<Fr>,
}

impl DeviceIncWeights {
    fn launch(
        context: &'static MetalContext,
        cycle_points: [&[Fr]; 4],
        gamma: Fr,
    ) -> Result<Self, MetalError> {
        let rounds = cycle_points[0].len();
        let high_bits = rounds / 2;
        let low_bits = rounds - high_bits;
        let n = 1usize << rounds;
        let mut offsets = [0u32; 8];
        let mut flat = Vec::new();
        for (point_index, point) in cycle_points.into_iter().enumerate() {
            for (side, factor) in [eq_table(&point[..high_bits]), eq_table(&point[high_bits..])]
                .into_iter()
                .enumerate()
            {
                offsets[2 * point_index + side] = flat.len() as u32;
                flat.extend(factor);
            }
        }
        let factors = context.own_page_aligned(PageAlignedVec::from_slice(&flat))?;
        let ram = RoundTable::new_device_filled(context, n)?;
        let rd = RoundTable::new_device_filled(context, n)?;
        let gamma2 = gamma * gamma;
        let gamma3 = gamma2 * gamma;
        let mut params = vec![n as u32, low_bits as u32, (n >> high_bits) as u32 - 1];
        params.extend_from_slice(&offsets);
        params.extend_from_slice(&fr_to_u32_limbs(gamma));
        params.extend_from_slice(&fr_to_u32_limbs(gamma2));
        params.extend_from_slice(&fr_to_u32_limbs(gamma3));
        let factor_buffer = factors.device_buffer();
        let ram_buffer = ram.cur().device_buffer();
        let rd_buffer = rd.cur().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::IncPrepare,
            &params,
            &[&factor_buffer, &ram_buffer, &rd_buffer],
            n,
        );
        // SAFETY: all three backings move into `Self` and remain host-
        // untouched until `wait` consumes the flight.
        let pass = unsafe { pass.commit().detach() };
        drop((factor_buffer, ram_buffer, rd_buffer));
        Ok(Self {
            pass,
            ram,
            rd,
            factors,
        })
    }

    fn wait(self) -> Result<(RoundTable, RoundTable), MetalError> {
        let Self {
            pass,
            ram,
            rd,
            factors,
        } = self;
        pass.wait()?;
        drop(factors);
        Ok((ram, rd))
    }
}

struct MetalIncKernel {
    /// Declared first so a flight's wait-on-drop runs before the dispatched
    /// tables free.
    in_flight: Option<IncFlight>,
    rounds: usize,
    rounds_bound: usize,
    /// Current logical table length (all four tables stay same-length).
    len: usize,
    ram_inc: RoundTable,
    rd_inc: RoundTable,
    ram_weights: RoundTable,
    rd_weights: RoundTable,
    partials: Partials,
    device: DeviceRound,
}

/// One two-phase round in flight (committed, not yet waited).
struct IncFlight {
    pass: DetachedPass,
    num_tgs: usize,
    /// The round's fold, applied by the launched kernel; advances the
    /// ping-pong on collect success, re-applied host-side on failure.
    bind: Option<Fr>,
}

impl MetalIncKernel {
    fn new_prepared(
        context: &'static MetalContext,
        rounds: usize,
        ram_inc: Vec<Fr>,
        rd_inc: Vec<Fr>,
        ram_weights: RoundTable,
        rd_weights: RoundTable,
    ) -> Result<Self, MetalError> {
        let len = ram_inc.len();
        Ok(Self {
            in_flight: None,
            rounds,
            rounds_bound: 0,
            len,
            ram_inc: RoundTable::new(context, ram_inc)?,
            rd_inc: RoundTable::new(context, rd_inc)?,
            ram_weights,
            rd_weights,
            partials: Partials::new(context, 2, len / 2)?,
            device: DeviceRound::new(context, KIND),
        })
    }

    fn bind_bookkeeping(&mut self) {
        self.len /= 2;
        self.rounds_bound += 1;
    }

    /// Encode + commit the fused round without blocking; the caller decides
    /// whether to wait in place (synchronous tier) or park the flight.
    /// Reads only `cur` tables (writes `nxt` + partials), so a failure
    /// leaves the round re-runnable on the CPU.
    fn commit_round(
        &self,
        context: &MetalContext,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<DetachedPass, MetalError> {
        let num_tgs = num_threadgroups(groups);
        let params = slot_round_params(groups, bind, num_tgs);
        let buffers = [
            self.ram_inc.cur().device_buffer(),
            self.rd_inc.cur().device_buffer(),
            self.ram_weights.cur().device_buffer(),
            self.rd_weights.cur().device_buffer(),
            self.ram_inc.nxt().device_buffer(),
            self.rd_inc.nxt().device_buffer(),
            self.ram_weights.nxt().device_buffer(),
            self.rd_weights.nxt().device_buffer(),
            self.partials.buffer().device_buffer(),
        ];
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::IncRound,
            &params,
            &[
                &buffers[0],
                &buffers[1],
                &buffers[2],
                &buffers[3],
                &buffers[4],
                &buffers[5],
                &buffers[6],
                &buffers[7],
                &buffers[8],
            ],
            groups,
        );
        // SAFETY: every dispatched buffer is kernel-owned (`in_flight` is
        // declared first, so a flight's wait-on-drop precedes their frees)
        // and next host-touched after the wait; params are copied at encode.
        Ok(unsafe { pass.commit().detach() })
    }

    /// The fused device round: one dispatch, one command buffer, one wait.
    fn device_round(
        &self,
        context: &MetalContext,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let num_tgs = num_threadgroups(groups);
        self.commit_round(context, bind, groups)?.wait()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    /// Post-fold ping-pong advance shared by the synchronous and collect
    /// paths.
    fn advance(&mut self) {
        self.ram_inc.swap();
        self.rd_inc.swap();
        self.ram_weights.swap();
        self.rd_weights.swap();
        self.bind_bookkeeping();
    }

    /// The CPU twin (tail rounds and post-failure recovery): same fold, same
    /// summand, over the same unified-memory tables.
    fn cpu_round(&mut self, bind: Option<Fr>) -> Vec<Fr> {
        if let Some(challenge) = bind {
            let len = self.len;
            self.ram_inc.bind_cpu(len, challenge);
            self.rd_inc.bind_cpu(len, challenge);
            self.ram_weights.bind_cpu(len, challenge);
            self.rd_weights.bind_cpu(len, challenge);
            self.bind_bookkeeping();
        }
        let ram = self.ram_inc.cur_slice(self.len);
        let rd = self.rd_inc.cur_slice(self.len);
        let a = self.ram_weights.cur_slice(self.len);
        let b = self.rd_weights.cur_slice(self.len);
        let group = |y: usize| -> [Fr; 2] {
            let at_two = |lo: Fr, hi: Fr| hi + hi - lo;
            [
                a[2 * y] * ram[2 * y] + b[2 * y] * rd[2 * y],
                at_two(a[2 * y], a[2 * y + 1]) * at_two(ram[2 * y], ram[2 * y + 1])
                    + at_two(b[2 * y], b[2 * y + 1]) * at_two(rd[2 * y], rd[2 * y + 1]),
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

impl ProveRounds<Fr> for MetalIncKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        // Post-bind pair count: what the fused kernel binds AND evaluates.
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        // The fused kernel cannot express a fold whose output is a single
        // element (groups == 0), so that degenerate shape stays host-side.
        let device = if groups == 0 {
            None
        } else {
            self.device.gated(self.len)
        };
        let evals = match device {
            Some(context) => match self.device_round(context, bind, groups) {
                Ok(sums) => {
                    if bind.is_some() {
                        self.advance();
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

    fn begin_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        _previous_claim: Fr,
    ) -> Result<bool, SumcheckError<Fr>> {
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        // Declines are stateless: `collect_round` falls through to the
        // synchronous round, which re-runs the same gate.
        if groups == 0 {
            return Ok(false);
        }
        let Some(context) = self.device.gated(self.len) else {
            return Ok(false);
        };
        match self.commit_round(context, bind, groups) {
            Ok(pass) => {
                self.in_flight = Some(IncFlight {
                    pass,
                    num_tgs: num_threadgroups(groups),
                    bind,
                });
                Ok(true)
            }
            Err(error) => {
                // Nothing committed; the synchronous retry (now latched off)
                // recomputes on the CPU from the intact `cur` tables.
                self.device.failed(&error);
                Ok(false)
            }
        }
    }

    fn collect_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(flight) = self.in_flight.take() {
            match flight.pass.wait() {
                Ok(()) => {
                    testing::note_device_round();
                    let sums = self.partials.sums(flight.num_tgs);
                    if flight.bind.is_some() {
                        self.advance();
                    }
                    return Ok(round_poly_from_skipped_evals(&sums, previous_claim));
                }
                Err(error) => {
                    // The kernel writes only `nxt` and the partials — the
                    // synchronous fallback below re-runs the SAME round
                    // (fold included) from the intact `cur` tables.
                    self.device.failed(&error);
                }
            }
        }
        self.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        // A single fold to one element per table: always host-side.
        let len = self.len;
        self.ram_inc.bind_cpu(len, bind);
        self.rd_inc.bind_cpu(len, bind);
        self.ram_weights.bind_cpu(len, bind);
        self.rd_weights.bind_cpu(len, bind);
        self.bind_bookkeeping();
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalIncKernel {
    type Relation = IncClaimReduction<Fr>;

    fn output_claims(
        &mut self,
        _inputs: &jolt_verifier::stages::relations::SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<IncClaimReductionOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        Ok(IncClaimReductionOutputClaims {
            ram_inc: self.ram_inc.cur_slice(1)[0],
            rd_inc: self.rd_inc.cur_slice(1)[0],
        })
    }
}

/// Lockstep parity against the optimized kernel (identical round polynomial
/// bytes, identical output claims) with the device path FORCED and probed —
/// a silent CPU fallback fails the test, not just the parity.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::Ring;
    use jolt_verifier::stages::stage6b::inc_claim_reduction::{
        IncClaimReductionChallenges, IncClaimReductionInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};

    fn force_device_gate() {
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
    }

    #[test]
    fn inc_claim_reduction_matches_optimized() {
        let _lock = gpu_lock();
        force_device_gate();
        with_sample_backend(|backend| {
            let log_t = JoltWitnessOracle::<Fr>::shape(
                backend,
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            )
            .unwrap()
            .rows()
            .ilog2() as usize;

            let relation = IncClaimReduction::new(
                TraceDimensions::new(log_t),
                synthetic_point(log_t, 3),
                synthetic_point(log_t, 5),
                synthetic_point(log_t, 7),
                synthetic_point(log_t, 11),
            );
            let challenges = IncClaimReductionChallenges {
                gamma: Fr::from_u64(29),
            };
            let claims = IncClaimReductionInputClaims::<Fr>::default();
            let input_points = IncClaimReductionInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            // The reference kernel exists only to recover the honest input
            // claim — it is the one tier whose round check exposes it.
            let mut reference =
                <crate::ReferenceBackend as PrepareKernel<Fr, IncClaimReduction<Fr>>>::prepare(
                    &crate::ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedIncClaimReduction
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let metal_slot = MetalIncClaimReduction {
                fallback: OptimizedIncClaimReduction,
            };
            let mut metal = metal_slot
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let rounds_before = device_probe_count();
            let claim = probe_input_claim(reference.as_mut());
            let sumcheck_challenges = synthetic_point(log_t, 401);
            run_lockstep(
                optimized.as_mut(),
                metal.as_mut(),
                claim,
                &sumcheck_challenges,
            );
            assert_eq!(
                optimized.output_claims(&claims).unwrap(),
                metal.output_claims(&claims).unwrap()
            );
            assert!(
                device_probe_count() > rounds_before,
                "the metal kernel never dispatched on the device"
            );
        });
    }
}
