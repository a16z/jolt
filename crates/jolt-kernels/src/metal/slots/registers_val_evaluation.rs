//! The stage-5 registers value-evaluation slot (W10): fused bind+eval device
//! rounds over the `Inc·Wa·LT` cubic.
//!
//! Table strategy per factor:
//!
//! - **Inc** (`T`-sized): the host oracle vector, no-copy wrapped as the
//!   ping-pong `cur` side, folded `cur → nxt` on device every bind.
//! - **Wa**: served in the kernel straight from the per-cycle rd indices and
//!   the address eq table (the optimized tier's lazy form — the `K × T` grid
//!   never exists) until the first bind densifies it into a device-owned
//!   `T/2` table; a `RoundTable` afterwards.
//! - **LT**: never a big table. The host keeps the optimized tier's
//!   [`SplitLt`] (three ~√T tables), binds its lo table before every device
//!   round (µs of work), and the kernel evaluates
//!   `lt(j) = lt_hi[j >> log_lo] + eq_hi[j >> log_lo]·lt_lo[j & mask]`
//!   in place. The dense transition (lo variables exhausted) happens below
//!   the device gate at production sizes; the kernel still serves a dense LT
//!   for the gate-free test path.
//!
//! Rounds launch through [`ProveRounds::begin_round`] as a detached command
//! buffer, so the batch engine overlaps the dispatch with its synchronous
//! members' CPU work and collects the message afterwards. Byte parity: the
//! kernel accumulates the same `t ∈ {0, 2, 3}` sample sums the optimized
//! tier does — Fr arithmetic is exact on both sides, so any summation
//! regrouping yields identical field values, and the wire polynomial is
//! assembled through the same `from_evals_and_hint` recipe.
//!
//! Kill switch: `JOLT_METAL_REGVAL=0` keeps this slot on the optimized tier
//! (the per-slot `JOLT_METAL_MIN_TERMS_REGISTERS_VAL_EVALUATION` gate
//! override also applies).

use jolt_field::{Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::registers_val_evaluation::{
    RegistersValEvaluation, RegistersValEvaluationOutputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, DeviceRound, Partials, RoundTable};
use crate::metal::buffers::OwnedDeviceBuffer;
use crate::metal::error::MetalError;
use crate::metal::field::fr_to_u32_limbs;
use crate::metal::runtime::{DetachedPass, KernelId, MetalContext};
use crate::metal::{metal_gate, testing};
use crate::optimized::registers_val_evaluation::{
    IncSource, ValEvaluationKernel, ValEvaluationParts, WaState,
};
use crate::optimized::support::SplitLt;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "registers_val_evaluation";

fn missing_state() -> SumcheckError<Fr> {
    SumcheckError::MissingEvaluationSource {
        kind: "Metal registers val-evaluation state",
    }
}

/// Slot front: device rounds above the [`metal_gate`] threshold, the
/// optimized kernel otherwise or on any device failure.
pub struct MetalRegistersValEvaluation;

impl PrepareKernel<Fr, RegistersValEvaluation<Fr>> for MetalRegistersValEvaluation {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, RegistersValEvaluation<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = RegistersValEvaluation<Fr>>>, KernelError<Fr>>
    {
        let parts = ValEvaluationParts::collect(session, witness, &inputs)?;
        let cycles = 1usize << parts.log_t;
        let killed = std::env::var_os("JOLT_METAL_REGVAL").is_some_and(|value| value == "0");
        // Gate on the first dispatch's work items (round-0 pairs).
        if killed || !metal_gate(KIND, cycles / 2) {
            return Ok(Box::new(ValEvaluationKernel::from_parts(parts)));
        }
        let context = match MetalContext::global() {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                return Ok(Box::new(ValEvaluationKernel::from_parts(parts)));
            }
        };
        match MetalValEvaluationKernel::build(context, parts) {
            Ok(kernel) => Ok(Box::new(kernel)),
            Err((mut parts, error)) => {
                tracing::warn!(slot = KIND, %error, "device build failed; using optimized fallback");
                if parts.inc.is_empty() {
                    // The inc table was consumed by a buffer wrap that then
                    // failed — re-collect it (cold path).
                    parts.inc = IncSource::collect(witness, cycles)?;
                }
                Ok(Box::new(ValEvaluationKernel::from_parts(parts)))
            }
        }
    }
}

/// rd hot indices packed four-per-word for the device (`0xFF` = no write).
fn pack_rd(rd: &[Option<u8>]) -> Vec<u32> {
    let pack = |chunk: &[Option<u8>]| {
        let mut word = 0u32;
        for (k, reg) in chunk.iter().enumerate() {
            word |= reg.map_or(0xFF_u32, u32::from) << (k * 8);
        }
        word
    };
    #[cfg(feature = "parallel")]
    let mut words: Vec<u32> = rd.par_chunks(4).map(pack).collect();
    #[cfg(not(feature = "parallel"))]
    let mut words: Vec<u32> = rd.chunks(4).map(pack).collect();
    if words.is_empty() {
        words.push(0);
    }
    words
}

type SmallBuffers = (
    OwnedDeviceBuffer<u32>,
    OwnedDeviceBuffer<Fr>,
    OwnedDeviceBuffer<Fr>,
    OwnedDeviceBuffer<Fr>,
    RoundTable,
    Partials,
);

/// One fused round in flight (committed, not yet waited).
struct ValFlight {
    pass: DetachedPass,
    num_tgs: usize,
    /// The round folded a challenge: advance the ping-pong on collect.
    bound: bool,
    /// This round's fold densified Wa into its `cur` buffer (no swap).
    wa_was_indices: bool,
    /// LT after this round's host-side fold; applied on a successful collect
    /// so a failed round leaves `lt` pre-bind for the host recompute.
    lt_next: SplitLt<Fr>,
    /// Flight-owned upload the committed pass reads.
    _lt_upload: OwnedDeviceBuffer<Fr>,
}

/// Device tables; dropped whole on hand-off so the ping-pong memory frees
/// before the host tail runs.
struct DeviceValState {
    /// Declared first: an in-flight pass settles (DetachedPass waits on
    /// drop) before the buffers it reads/writes free.
    in_flight: Option<ValFlight>,
    inc: RoundTable,
    wa: RoundTable,
    wa_dense: bool,
    rd_words: OwnedDeviceBuffer<u32>,
    eq_address: OwnedDeviceBuffer<Fr>,
    lt_hi: OwnedDeviceBuffer<Fr>,
    eq_hi: OwnedDeviceBuffer<Fr>,
    partials: Partials,
}

struct MetalValEvaluationKernel {
    rounds: usize,
    rounds_bound: usize,
    /// Live inc/wa logical length; halves on every bound round.
    len: usize,
    device: DeviceRound,
    state: Option<DeviceValState>,
    /// Host-owned LT, bound (via a flight clone) alongside every device
    /// round; handed to the host kernel whole on fallback.
    lt: SplitLt<Fr>,
    /// Host copies for a pre-densification fallback.
    rd_host: Option<Vec<Option<u8>>>,
    eq_address_host: Option<Vec<Fr>>,
    host: Option<ValEvaluationKernel<Fr>>,
}

impl MetalValEvaluationKernel {
    fn build(
        context: &'static MetalContext,
        mut parts: ValEvaluationParts<Fr>,
    ) -> Result<Self, (ValEvaluationParts<Fr>, MetalError)> {
        let cycles = 1usize << parts.log_t;
        let zero = Fr::from_u64(0);
        let small = |parts: &ValEvaluationParts<Fr>| -> Result<SmallBuffers, MetalError> {
            let rd_words = context.own_vec(pack_rd(&parts.rd))?;
            let eq_address = context.own_vec(parts.eq_address.clone())?;
            let (lt_hi, eq_hi) = match &parts.lt {
                SplitLt::Split { lt_hi, eq_hi, .. } => (
                    context.own_vec(lt_hi.clone())?,
                    context.own_vec(eq_hi.clone())?,
                ),
                // Tiny traces only (below the gate in production): the round
                // serves LT densely, the hi tables are never read.
                SplitLt::Dense(_) => (context.own_vec(vec![zero])?, context.own_vec(vec![zero])?),
            };
            testing::note_copied_buffers(u64::from(rd_words.was_copied()));
            for buffer in [&eq_address, &lt_hi, &eq_hi] {
                testing::note_copied_buffers(u64::from(buffer.was_copied()));
            }
            let wa = RoundTable::new_device_filled(context, cycles / 2)?;
            let partials = Partials::new(context, 3, cycles / 2)?;
            Ok((rd_words, eq_address, lt_hi, eq_hi, wa, partials))
        };
        let (rd_words, eq_address, lt_hi, eq_hi, wa, partials) = match small(&parts) {
            Ok(buffers) => buffers,
            Err(error) => return Err((parts, error)),
        };
        // The big wrap last: on failure the (consumed) inc table is
        // re-collected by the caller. A deferred source materializes here —
        // the device path uploads at prepare.
        let inc_vec = match parts.inc.take_table() {
            Ok(table) => table,
            Err(_) => {
                return Err((
                    parts,
                    MetalError::UnsupportedShape("increment table failed to materialize"),
                ))
            }
        };
        let inc = match RoundTable::new(context, inc_vec) {
            Ok(table) => table,
            Err(error) => return Err((parts, error)),
        };
        testing::note_copied_buffers(u64::from(inc.cur().was_copied()));
        Ok(Self {
            rounds: parts.log_t,
            rounds_bound: 0,
            len: cycles,
            device: DeviceRound::new(context, KIND),
            state: Some(DeviceValState {
                in_flight: None,
                inc,
                wa,
                wa_dense: false,
                rd_words,
                eq_address,
                lt_hi,
                eq_hi,
                partials,
            }),
            lt: parts.lt,
            rd_host: Some(parts.rd),
            eq_address_host: Some(parts.eq_address),
            host: None,
        })
    }

    /// Encode + commit one fused round without blocking. The LT fold is
    /// applied to a clone carried by the flight — `self.lt` stays pre-bind
    /// until the collect succeeds, so a failed round recomputes host-side
    /// from intact state.
    fn launch(&mut self, bind: Option<Fr>, groups: usize) -> Result<(), MetalError> {
        let context = self
            .device
            .gated(groups)
            .ok_or(MetalError::UnsupportedShape("device round below the gate"))?;
        let state = self
            .state
            .as_mut()
            .ok_or(MetalError::UnsupportedShape("device round without state"))?;
        let mut lt_next = self.lt.clone();
        if let Some(challenge) = bind {
            lt_next.bind(challenge);
        }
        let (lt_dense, log_lo, lo_table): (u32, u32, Vec<Fr>) = match &lt_next {
            SplitLt::Split { lt_lo, .. } => (0, lt_lo.len().trailing_zeros(), lt_lo.clone()),
            SplitLt::Dense(table) => (1, 0, table.clone()),
        };
        let lt_upload = context.own_vec(lo_table)?;
        testing::note_copied_buffers(u64::from(lt_upload.was_copied()));

        let num_tgs = num_threadgroups(groups);
        let mut params = vec![
            groups as u32,
            u32::from(bind.is_some()),
            num_tgs as u32,
            u32::from(state.wa_dense),
            lt_dense,
            log_lo,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));

        let inc_cur = state.inc.cur().device_buffer();
        let inc_nxt = state.inc.nxt().device_buffer();
        let wa_cur = state.wa.cur().device_buffer();
        // The first fold densifies Wa INTO `cur` (sized `T/2`, exactly the
        // dense length); the kernel reads `wa_cur` only in dense mode, so
        // binding the same buffer at both slots is hazard-free.
        let wa_out = if state.wa_dense {
            state.wa.nxt()
        } else {
            state.wa.cur()
        };
        let wa_nxt = wa_out.device_buffer();
        let rd = state.rd_words.device_buffer();
        let eq_address = state.eq_address.device_buffer();
        let lt_lo = lt_upload.device_buffer();
        let lt_hi = state.lt_hi.device_buffer();
        let eq_hi = state.eq_hi.device_buffer();
        let partials = state.partials.buffer().device_buffer();

        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::RegistersValRound,
            &params,
            &[
                &inc_cur,
                &inc_nxt,
                &wa_cur,
                &wa_nxt,
                &rd,
                &eq_address,
                &lt_lo,
                &lt_hi,
                &eq_hi,
                &partials,
            ],
            groups,
        );
        // SAFETY: every bound buffer is either state-owned (the ping-pong
        // pairs, rd, eq/LT hi tables, partials — next touched after the
        // wait; `DeviceValState`'s field order settles an in-flight pass
        // before they drop) or flight-owned (the LT lo upload).
        let pass = unsafe { pass.commit().detach() };
        state.in_flight = Some(ValFlight {
            pass,
            num_tgs,
            bound: bind.is_some(),
            wa_was_indices: !state.wa_dense,
            lt_next,
            _lt_upload: lt_upload,
        });
        Ok(())
    }

    /// Copy the live tables out of unified memory, drop the device state
    /// (freeing the ping-pong allocations), and resume on the optimized
    /// kernel.
    fn hand_off_to_host(&mut self) -> Result<(), SumcheckError<Fr>> {
        let state = self.state.take().ok_or_else(missing_state)?;
        debug_assert!(state.in_flight.is_none());
        let inc = state.inc.cur_slice(self.len).to_vec();
        let wa = if state.wa_dense {
            WaState::Dense(state.wa.cur_slice(self.len).to_vec())
        } else {
            WaState::Indices {
                rd: self.rd_host.take().ok_or_else(missing_state)?,
                eq_address: self.eq_address_host.take().ok_or_else(missing_state)?,
            }
        };
        let lt = std::mem::replace(&mut self.lt, SplitLt::Dense(Vec::new()));
        self.host = Some(ValEvaluationKernel::from_bound_state(
            self.rounds,
            inc,
            wa,
            lt,
            self.rounds_bound,
        ));
        Ok(())
    }
}

impl ProveRounds<Fr> for MetalValEvaluationKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn begin_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        _previous_claim: Fr,
    ) -> Result<bool, SumcheckError<Fr>> {
        if self.host.is_some() {
            return Ok(false);
        }
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        if groups == 0 || self.device.gated(groups).is_none() {
            self.hand_off_to_host()?;
            return Ok(false);
        }
        match self.launch(bind, groups) {
            Ok(()) => Ok(true),
            Err(error) => {
                // Nothing committed: cur tables and the pre-bind LT are
                // intact for the host recompute of the SAME round.
                self.device.failed(&error);
                self.hand_off_to_host()?;
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
        let flight = self.state.as_mut().and_then(|state| state.in_flight.take());
        if let Some(flight) = flight {
            match flight.pass.wait() {
                Ok(()) => {
                    testing::note_device_round();
                    let state = self.state.as_mut().ok_or_else(missing_state)?;
                    let sums = state.partials.sums(flight.num_tgs);
                    self.lt = flight.lt_next;
                    if flight.bound {
                        state.inc.swap();
                        if flight.wa_was_indices {
                            state.wa_dense = true;
                            self.rd_host = None;
                            self.eq_address_host = None;
                        } else {
                            state.wa.swap();
                        }
                        self.len /= 2;
                        self.rounds_bound += 1;
                    }
                    let evals = [sums[0], sums[1], sums[2]];
                    return Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals));
                }
                Err(error) => {
                    // The fused kernel writes only the bind targets and the
                    // partials — cur and the pre-bind LT are intact.
                    self.device.failed(&error);
                    self.hand_off_to_host()?;
                }
            }
        }
        self.host
            .as_mut()
            .ok_or_else(missing_state)?
            .prove_round(bind, round, previous_claim)
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let _ = self.begin_round(bind, round, previous_claim)?;
        self.collect_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        if self.host.is_none() {
            self.hand_off_to_host()?;
        }
        self.host
            .as_mut()
            .ok_or_else(missing_state)?
            .finish_rounds(bind)
    }
}

impl SumcheckKernel<Fr> for MetalValEvaluationKernel {
    type Relation = RegistersValEvaluation<Fr>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        let remaining = self.rounds.saturating_sub(self.rounds_bound);
        self.host
            .as_mut()
            .ok_or(SumcheckKernelError::NotFullyBound { remaining })?
            .output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<Fr, Self::Relation>,
        output_points: &SumcheckOutputPoints<Fr, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<Fr, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<Fr>> {
        self.host
            .as_ref()
            .ok_or(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds.saturating_sub(self.rounds_bound),
            })?
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        TraceDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::stage5::registers_val_evaluation::{
        RegistersValEvaluation, RegistersValEvaluationInputClaims,
    };
    use jolt_witness::{collect_bundles, JoltWitnessOracle};

    use super::MetalRegistersValEvaluation;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::registers_read_write::test_support::{
        assert_kernel_parity_with_session, assert_nontrivial, challenge_sequence,
        structured_fixture, TraceFixture,
    };
    use crate::optimized::registers_read_write::{RegisterCycleRow, SharedRdIndices};
    use crate::ProofSession;

    fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64, parked: bool) {
        let _lock = gpu_lock();
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::remove_var("JOLT_METAL_REGVAL");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        fixture.with_plane(log_t, |backend| {
            let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(log_t));
            let point = challenge_sequence(REGISTER_ADDRESS_BITS + log_t, seed ^ 0x3C3C);
            let grid = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RegistersVal),
            )
            .unwrap();
            let input_claim = Polynomial::new(grid).evaluate(&point);
            assert_nontrivial(input_claim);

            let mut session = ProofSession::default();
            if parked {
                let rows: Vec<RegisterCycleRow> = collect_bundles(backend, 1 << log_t).unwrap();
                session.park(SharedRdIndices(
                    rows.iter().map(|row| row.rd.map(|(k, ..)| k)).collect(),
                ));
            }

            let claims = RegistersValEvaluationInputClaims {
                registers_val: input_claim,
            };
            let points = RegistersValEvaluationInputClaims {
                registers_val: point,
            };
            let round_challenges = challenge_sequence(log_t, seed);
            let before = device_probe_count();
            assert_kernel_parity_with_session(
                &mut session,
                &MetalRegistersValEvaluation,
                backend,
                &relation,
                &claims,
                &points,
                &NoChallenges::default(),
                input_claim,
                &round_challenges,
            );
            assert!(
                device_probe_count() > before,
                "no device round dispatched — the slot fell back silently"
            );
        });
    }

    /// Crosses the Wa densification, several dense device rounds, AND the
    /// split→dense LT transition on device.
    #[test]
    fn parity_structured_deep() {
        run_parity(structured_fixture(64), 6, 101, false);
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fixture(8), 3, 53, false);
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fixture(16), 4, 59, false);
    }

    #[test]
    fn parity_with_parked_indices() {
        run_parity(structured_fixture(8), 3, 71, true);
    }

    /// The kill switch keeps the whole sumcheck on the optimized tier.
    #[test]
    fn kill_switch_stays_on_host() {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        std::env::set_var("JOLT_METAL_REGVAL", "0");
        structured_fixture(8).with_plane(3, |backend| {
            let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(3));
            let point = challenge_sequence(REGISTER_ADDRESS_BITS + 3, 0x7777 ^ 0x3C3C);
            let grid = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RegistersVal),
            )
            .unwrap();
            let input_claim = Polynomial::new(grid).evaluate(&point);
            let claims = RegistersValEvaluationInputClaims {
                registers_val: input_claim,
            };
            let points = RegistersValEvaluationInputClaims {
                registers_val: point,
            };
            let round_challenges = challenge_sequence(3, 0x7777);
            let before = device_probe_count();
            assert_kernel_parity_with_session(
                &mut ProofSession::default(),
                &MetalRegistersValEvaluation,
                backend,
                &relation,
                &claims,
                &points,
                &NoChallenges::default(),
                input_claim,
                &round_challenges,
            );
            assert_eq!(
                device_probe_count(),
                before,
                "kill switch leaked a dispatch"
            );
        });
        std::env::remove_var("JOLT_METAL_REGVAL");
    }
}
