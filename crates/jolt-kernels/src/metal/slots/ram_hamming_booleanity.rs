//! Metal RAM Hamming-weight booleanity (stage 6b): device twin of
//! [`OptimizedRamHammingBooleanity`], byte-identical round polynomials by
//! construction.
//!
//! The [`GruenSplitEqPolynomial`] stays HOST-side (its tables are `~√T` and
//! its per-round bind is O(1) scalar work); only the dense Hamming table `H`
//! lives in device ping-pong buffers. Each round is one `jk_hamming_round`
//! dispatch: fold `H` with the previous challenge and accumulate the Gruen
//! inner sums — `Σ eq(y)·(h₀²−h₀)` and `Σ eq(y)·(h₁−h₀)²` with
//! `eq(y) = e_out[y >> log_in]·e_in[y & mask]` — as per-threadgroup
//! partials. The device's flat `e_out·e_in·q` products regroup the CPU's
//! `e_out·(Σ e_in·q)` nesting; distributivity is exact over Fr, so the sums
//! (and therefore [`GruenSplitEqPolynomial::gruen_poly_deg_3`]'s output,
//! assembled host-side) are byte-identical. The current `e_in`/`e_out`
//! levels are wrapped per round — no-copy when the slices are page-aligned
//! and ≥ 32 KiB (true at production `T`), tiny counted copies below that.

use jolt_claims::protocols::jolt::{JoltDerivedId, RamHammingBooleanityPublic};
use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::{
    RamHammingBooleanity, RamHammingBooleanityOutputClaims,
};
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use super::{num_threadgroups, own_eq, wrap_eq, DeviceRound, Partials, RoundTable};
use crate::metal::buffers::{DeviceBuffer, OwnedDeviceBuffer};
use crate::metal::field::fr_to_u32_limbs;
use crate::metal::runtime::{DetachedPass, KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::ram_hamming_booleanity::{
    build_hamming_booleanity_inputs, OptimizedRamHammingBooleanity,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "ram_hamming_booleanity";

/// Slot front: device kernel above the [`metal_gate`] threshold, the
/// optimized fallback below it or on any device failure.
pub struct MetalRamHammingBooleanity {
    pub fallback: OptimizedRamHammingBooleanity,
}

impl PrepareKernel<Fr, RamHammingBooleanity<Fr>> for MetalRamHammingBooleanity {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, RamHammingBooleanity<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = RamHammingBooleanity<Fr>>>, KernelError<Fr>>
    {
        if metal_gate(KIND, 1usize << inputs.relation.rounds()) {
            match MetalContext::global() {
                Ok(context) => {
                    // Structural errors propagate — the fallback would fail
                    // identically; only device failures fall back.
                    let built = build_hamming_booleanity_inputs(witness, &inputs)?;
                    match MetalHammingBooleanityKernel::new(context, built) {
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

struct MetalHammingBooleanityKernel {
    /// Declared first so a flight's wait-on-drop runs before the dispatched
    /// tables free.
    in_flight: Option<HammingFlight>,
    rounds: usize,
    rounds_bound: usize,
    /// Current logical length of the Hamming table.
    len: usize,
    eq: GruenSplitEqPolynomial<Fr>,
    hamming: RoundTable,
    partials: Partials,
    device: DeviceRound,
}

/// One two-phase round in flight (committed, not yet waited). `pass` first:
/// its wait-on-drop must precede the eq uploads freeing.
struct HammingFlight {
    pass: DetachedPass,
    num_tgs: usize,
    /// The round's fold, applied by the launched kernel; advances the
    /// ping-pong on collect success, re-applied host-side on failure.
    bind: Option<Fr>,
    /// The round's gruen levels, copied into flight-owned buffers (see
    /// [`own_eq`]).
    _eq: (OwnedDeviceBuffer<Fr>, OwnedDeviceBuffer<Fr>),
}

impl MetalHammingBooleanityKernel {
    fn new(
        context: &'static MetalContext,
        built: crate::optimized::ram_hamming_booleanity::HammingBooleanityInputs<Fr>,
    ) -> Result<Self, MetalError> {
        let len = built.hamming.len();
        Ok(Self {
            in_flight: None,
            rounds: built.rounds,
            rounds_bound: 0,
            len,
            eq: GruenSplitEqPolynomial::new(&built.eq_point, BindingOrder::LowToHigh),
            hamming: RoundTable::new(context, built.hamming)?,
            partials: Partials::new(context, 2, len / 2)?,
            device: DeviceRound::new(context, KIND),
        })
    }

    fn bind_bookkeeping(&mut self) {
        self.len /= 2;
        self.rounds_bound += 1;
    }

    /// Encode + commit the fused round without blocking: fold `H` (optional)
    /// and accumulate the two Gruen sums against the caller's eq buffers
    /// (wait-in-scope borrows on the synchronous path, flight-owned copies
    /// on the launch path). The eq levels are the CURRENT (post-`eq.bind`)
    /// ones — the caller binds eq first.
    fn commit_round(
        &self,
        context: &MetalContext,
        bind: Option<Fr>,
        groups: usize,
        e_in_log2: u32,
        eq: (&DeviceBuffer<'_>, &DeviceBuffer<'_>),
    ) -> Result<DetachedPass, MetalError> {
        let num_tgs = num_threadgroups(groups);
        // HammingParams: [groups, do_bind, num_tgs, log_in, r].
        let mut params = vec![
            groups as u32,
            u32::from(bind.is_some()),
            num_tgs as u32,
            e_in_log2,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let h_cur = self.hamming.cur().device_buffer();
        let h_nxt = self.hamming.nxt().device_buffer();
        let partials = self.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::HammingRound,
            &params,
            &[&h_cur, &h_nxt, eq.0, eq.1, &partials],
            groups,
        );
        // SAFETY: the ping-pong and partials are kernel-owned (`in_flight`
        // first ⇒ a flight's wait-on-drop precedes their frees) and next
        // host-touched after the wait; the eq buffers are the caller's per
        // the signature contract; copied uploads are Metal-owned.
        Ok(unsafe { pass.commit().detach() })
    }

    /// The fused device round: one dispatch, one command buffer, one wait.
    fn device_round(
        &self,
        context: &MetalContext,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let e_in = self.eq.e_in_current();
        let e_out = self.eq.e_out_current();
        let num_tgs = num_threadgroups(groups);
        let (e_in_buffer, e_out_buffer) = wrap_eq(context, e_in, e_out)?;
        self.commit_round(
            context,
            bind,
            groups,
            e_in.len().trailing_zeros(),
            (&e_in_buffer, &e_out_buffer),
        )?
        .wait()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    /// The CPU twin over the same unified-memory table, mirroring the
    /// optimized kernel's fold (which multiplies `e_out` into the inner sums
    /// per block — equal to the device's flat products by distributivity).
    fn cpu_round(&mut self, bind: Option<Fr>) -> Vec<Fr> {
        if let Some(challenge) = bind {
            let len = self.len;
            self.hamming.bind_cpu(len, challenge);
            self.bind_bookkeeping();
        }
        let hamming = self.hamming.cur_slice(self.len);
        let accumulators = self.eq.par_fold_out_in(
            || [Fr::from_u64(0); 2],
            |accumulator, row, _x_in, e_in| {
                let h_0 = hamming[2 * row];
                let h_1 = hamming[2 * row + 1];
                let delta = h_1 - h_0;
                accumulator[0] += e_in * (h_0 * h_0 - h_0);
                accumulator[1] += e_in * (delta * delta);
            },
            |_x_out, e_out, inner| [e_out * inner[0], e_out * inner[1]],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        );
        accumulators.to_vec()
    }

    /// The round after the eq bind (the caller owns it): device tier when
    /// gated in, CPU twin otherwise, gruen assembly host-side.
    fn round_core(
        &mut self,
        bind: Option<Fr>,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        // Post-bind H pair count; the eq levels must tile it exactly.
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        let eq_tiles = self.eq.e_out_current_len() * self.eq.e_in_current_len() == groups;
        let device = if groups == 0 || !eq_tiles {
            None
        } else {
            self.device.gated(self.len)
        };
        let sums = match device {
            Some(context) => match self.device_round(context, bind, groups) {
                Ok(sums) => {
                    if bind.is_some() {
                        self.hamming.swap();
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
        Ok(self.eq.gruen_poly_deg_3(sums[0], sums[1], previous_claim))
    }
}

impl ProveRounds<Fr> for MetalHammingBooleanityKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        // The eq factor binds host-side FIRST (O(1) scalar work) — exactly
        // once per challenge, so a device failure below cannot re-bind it.
        if let Some(challenge) = bind {
            self.eq.bind(challenge);
        }
        self.round_core(bind, previous_claim)
    }

    fn begin_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        _previous_claim: Fr,
    ) -> Result<bool, SumcheckError<Fr>> {
        // `begin_round` owns the eq bind (`collect_round` never re-binds).
        if let Some(challenge) = bind {
            self.eq.bind(challenge);
        }
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        let eq_tiles = self.eq.e_out_current_len() * self.eq.e_in_current_len() == groups;
        if groups == 0 || !eq_tiles {
            return Ok(false);
        }
        let Some(context) = self.device.gated(self.len) else {
            return Ok(false);
        };
        let launch = || -> Result<HammingFlight, MetalError> {
            let eq = own_eq(context, self.eq.e_in_current(), self.eq.e_out_current())?;
            let pass = self.commit_round(
                context,
                bind,
                groups,
                self.eq.e_in_current().len().trailing_zeros(),
                (&eq.0.device_buffer(), &eq.1.device_buffer()),
            )?;
            Ok(HammingFlight {
                pass,
                num_tgs: num_threadgroups(groups),
                bind,
                _eq: eq,
            })
        };
        match launch() {
            Ok(flight) => {
                self.in_flight = Some(flight);
                Ok(true)
            }
            Err(error) => {
                // Nothing committed; the collect fallback recomputes on the
                // CPU from the intact `cur` table.
                self.device.failed(&error);
                Ok(false)
            }
        }
    }

    fn collect_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(flight) = self.in_flight.take() {
            match flight.pass.wait() {
                Ok(()) => {
                    testing::note_device_round();
                    let sums = self.partials.sums(flight.num_tgs);
                    if flight.bind.is_some() {
                        self.hamming.swap();
                        self.bind_bookkeeping();
                    }
                    return Ok(self.eq.gruen_poly_deg_3(sums[0], sums[1], previous_claim));
                }
                Err(error) => {
                    // `cur` is intact (the kernel writes `nxt` + partials);
                    // the fallback below re-runs the SAME round host-side.
                    self.device.failed(&error);
                }
            }
        }
        // `begin_round` already bound eq.
        self.round_core(bind, previous_claim)
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.eq.bind(bind);
        let len = self.len;
        self.hamming.bind_cpu(len, bind);
        self.bind_bookkeeping();
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalHammingBooleanityKernel {
    type Relation = RamHammingBooleanity<Fr>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<RamHammingBooleanityOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        Ok(RamHammingBooleanityOutputClaims {
            ram_hamming_weight: self.hamming.cur_slice(1)[0],
        })
    }

    /// The split-eq scalar against the verifier's `derive_output_term` — the
    /// same drift detector the optimized kernel runs.
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
        let id = JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle);
        let expected =
            match relation.derive_output_term(&id, input_points, output_points, challenges) {
                Ok(value) => value,
                Err(VerifierError::MissingStageClaimDerived { .. }) => return Ok(()),
                Err(error) => return Err(error.into()),
            };
        let got = self.eq.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

/// Lockstep parity against the optimized kernel with the device path forced
/// and probed. The input claim is exactly zero (booleanity), so the drive is
/// hand-rolled like the optimized tier's own parity test.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::booleanity::testing::{test_challenge, with_booleanity_backend};
    use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanityInputClaims;

    fn parity(log_t: usize) {
        with_booleanity_backend(log_t, 4, |backend, _| {
            let stage1_cycle_binding: Vec<Fr> = (0..log_t as u64)
                .map(|index| Fr::from_u64(600 + 41 * index))
                .collect();
            let relation =
                RamHammingBooleanity::new(TraceDimensions::new(log_t), stage1_cycle_binding);
            let claims = RamHammingBooleanityInputClaims::default();
            let points = RamHammingBooleanityInputClaims::default();
            let challenges = jolt_claims::NoChallenges::default();

            let mut optimized = OptimizedRamHammingBooleanity
                .prepare(
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let metal_slot = MetalRamHammingBooleanity {
                fallback: OptimizedRamHammingBooleanity,
            };
            let mut metal = metal_slot
                .prepare(
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let rounds_before = device_probe_count();
            // The Hamming indicator is boolean, so the input claim is zero.
            let mut claim = Fr::from_u64(0);
            let mut bind = None;
            let mut drawn = Vec::new();
            for round in 0..optimized.num_rounds() {
                let expected = optimized.prove_round(bind, round, claim).unwrap();
                let actual = metal.prove_round(bind, round, claim).unwrap();
                assert_eq!(expected, actual, "round {round} polynomial mismatch");
                let challenge = test_challenge(round);
                claim = expected.evaluate(challenge);
                drawn.push(challenge);
                bind = Some(challenge);
            }
            if let Some(last) = drawn.last() {
                optimized.finish_rounds(*last).unwrap();
                metal.finish_rounds(*last).unwrap();
            }

            assert_eq!(
                optimized.output_claims(&claims).unwrap(),
                metal.output_claims(&claims).unwrap()
            );
            let output_points = relation.derive_opening_points(&drawn, &points).unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            metal
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            assert!(
                device_probe_count() > rounds_before,
                "the metal kernel never dispatched on the device"
            );
        });
    }

    #[test]
    fn matches_optimized() {
        let _lock = gpu_lock();
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        parity(3);
    }

    #[test]
    fn matches_optimized_single_round() {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        parity(1);
    }
}
