//! Metal increment claim reduction (stage 6b): device twin of
//! [`OptimizedIncClaimReduction`], byte-identical round polynomials by
//! construction.
//!
//! Table construction (paired-eq fusion `A·RamInc + B·RdInc`) is shared with
//! the optimized kernel ([`build_inc_tables`]); this slot serves the four
//! dense tables from unified-memory ping-pong buffers and runs each round as
//! ONE `jk_inc_round` dispatch — fold all four tables with the previous
//! challenge and tree-reduce the summand's `t ∈ {0, 2}` evaluations to
//! per-threadgroup partials — in one command buffer with one wait. The host
//! sums the partials and assembles the wire polynomial through the same
//! `round_poly_from_skipped_evals` recipe as the optimized tier.

use jolt_field::{Fr, FromPrimitiveInt};
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
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::inc_claim_reduction::{
    build_inc_tables, IncTables, OptimizedIncClaimReduction,
};
#[cfg(feature = "parallel")]
use crate::optimized::support::merge_evals;
use crate::optimized::support::round_poly_from_skipped_evals;
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
                    // Structural errors propagate — the fallback would fail
                    // identically; only device failures fall back.
                    let tables = build_inc_tables(witness, &inputs)?;
                    match MetalIncKernel::new(context, tables) {
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

struct MetalIncKernel {
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

impl MetalIncKernel {
    fn new(context: &'static MetalContext, tables: IncTables<Fr>) -> Result<Self, MetalError> {
        let len = tables.ram_inc.len();
        Ok(Self {
            rounds: tables.rounds,
            rounds_bound: 0,
            len,
            ram_inc: RoundTable::new(context, tables.ram_inc)?,
            rd_inc: RoundTable::new(context, tables.rd_inc)?,
            ram_weights: RoundTable::new(context, tables.ram_weights)?,
            rd_weights: RoundTable::new(context, tables.rd_weights)?,
            partials: Partials::new(context, 2, len / 2)?,
            device: DeviceRound::new(context, KIND),
        })
    }

    fn bind_bookkeeping(&mut self) {
        self.len /= 2;
        self.rounds_bound += 1;
    }

    /// The fused device round: one dispatch, one command buffer, one wait.
    /// Reads only `cur` tables (writes `nxt` + partials), so a failure
    /// leaves the round re-runnable on the CPU.
    fn device_round(
        &self,
        context: &MetalContext,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
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
        pass.run()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
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
                        self.ram_inc.swap();
                        self.rd_inc.swap();
                        self.ram_weights.swap();
                        self.rd_weights.swap();
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
    use std::time::Instant;

    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::FromPrimitiveInt;
    use jolt_verifier::stages::stage6b::inc_claim_reduction::{
        IncClaimReductionChallenges, IncClaimReductionInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::testing::{copied_buffer_count, device_probe_count, gpu_lock, seeded_frs};
    use crate::optimized::harness::{probe_input_claim, run_lockstep, synthetic_point};

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

    /// Synthetic 2^16-scale parity + timing sanity: a device kernel against
    /// an identical CPU-forced kernel, byte-equal rounds, zero buffer
    /// copies (2 MiB tables are no-copy eligible), and a wall-clock print to
    /// catch pathological slowness (NOT a benchmark).
    #[test]
    #[expect(clippy::print_stdout, reason = "timing sanity readout")]
    fn inc_claim_reduction_device_parity_at_2e16() {
        let _lock = gpu_lock();
        force_device_gate();
        let context = MetalContext::global().unwrap();
        let len = 1usize << 16;
        let tables = || IncTables {
            rounds: 16,
            ram_inc: seeded_frs(0xA1, len),
            rd_inc: seeded_frs(0xA2, len),
            ram_weights: seeded_frs(0xA3, len),
            rd_weights: seeded_frs(0xA4, len),
        };
        // The honest input claim: the summand's full hypercube sum.
        let reference = tables();
        let mut claim = Fr::from_u64(0);
        for j in 0..len {
            claim += reference.ram_weights[j] * reference.ram_inc[j]
                + reference.rd_weights[j] * reference.rd_inc[j];
        }
        assert_ne!(claim, Fr::from_u64(0), "degenerate fixture");

        let copies_before = copied_buffer_count();
        let mut device_kernel = MetalIncKernel::new(context, tables()).unwrap();
        assert_eq!(
            copied_buffer_count(),
            copies_before,
            "2^16 tables must wrap no-copy"
        );
        let mut cpu_kernel = MetalIncKernel::new(context, tables()).unwrap();
        cpu_kernel.device = DeviceRound::disabled(KIND);

        let rounds_before = device_probe_count();
        let challenges = synthetic_point(16, 777);
        let mut device_wall = std::time::Duration::ZERO;
        let mut cpu_wall = std::time::Duration::ZERO;
        for round in 0..16usize {
            let bind = round.checked_sub(1).map(|previous| challenges[previous]);
            let start = Instant::now();
            let device_poly = device_kernel.prove_round(bind, round, claim).unwrap();
            device_wall += start.elapsed();
            let start = Instant::now();
            let cpu_poly = cpu_kernel.prove_round(bind, round, claim).unwrap();
            cpu_wall += start.elapsed();
            assert_eq!(
                device_poly.coefficients(),
                cpu_poly.coefficients(),
                "round {round} diverged"
            );
            claim = device_poly.evaluate(challenges[round]);
        }
        device_kernel.finish_rounds(challenges[15]).unwrap();
        cpu_kernel.finish_rounds(challenges[15]).unwrap();
        let claims = IncClaimReductionInputClaims::<Fr>::default();
        assert_eq!(
            device_kernel.output_claims(&claims).unwrap(),
            cpu_kernel.output_claims(&claims).unwrap()
        );
        assert_eq!(
            device_probe_count() - rounds_before,
            16,
            "every round must have dispatched on the device"
        );
        println!("inc 2^16, 16 rounds: device {device_wall:?}, cpu-in-kernel {cpu_wall:?}");
    }
}
