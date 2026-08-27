//! Metal stage-6b bytecode read+RAF cycle member.
//!
//! The device owns the `O(T)` combined cycle table from prepare onward.
//! Early rounds gather the sparse RA factors from packed mapped-PC rows;
//! the third bind adopts all factors into a flat dense ping-pong. A failed
//! or below-gate round publishes the live combined table through the shared
//! recovery cell before the optimized kernel resumes on the CPU.

use std::sync::{Arc, Mutex};

use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, Polynomial};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::JoltWitnessPlane;

use super::{num_threadgroups, own_uninit_frs, st6b_detach_enabled, DeviceRound, Partials};
use crate::metal::buffers::{OwnedDeviceBuffer, PageAlignedVec};
#[cfg(test)]
use crate::metal::field::FR_U32_LIMBS;
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{DetachedPass, KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::bytecode_read_raf::{
    prepare_bytecode_read_raf_cycle, BytecodeCycleDevice, BytecodeCycleDeviceInputs, PcRow, PcRows,
};
use crate::optimized::lazy_ra::LazyRaDevice;
use crate::optimized::support::eq_table;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const KIND: &str = "bytecode_read_raf_cycle";
const MAX_FACTORS: usize = 16;

const _: () = assert!(size_of::<PcRow>() == 2 * size_of::<u32>());

pub struct MetalBytecodeReadRafCycle;

impl PrepareKernel<Fr, BytecodeReadRafCycle<Fr>> for MetalBytecodeReadRafCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, BytecodeReadRafCycle<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = BytecodeReadRafCycle<Fr>>>, KernelError<Fr>>
    {
        let mut kernel = prepare_bytecode_read_raf_cycle(session, witness, inputs, build_driver)?;
        if st6b_detach_enabled() {
            // Round 0 is bind-free, so its lanes are fully determined at
            // prepare end: launch now and the device works under the batch's
            // remaining serial prepares (the engine's round-0 `begin_round`
            // adopts the flight idempotently).
            let _ = kernel.begin_round(None, 0, Fr::from_u64(0));
        }
        Ok(kernel)
    }
}

fn build_driver(inputs: BytecodeCycleDeviceInputs<'_, Fr>) -> Option<BytecodeCycleDevice<Fr>> {
    // With W3A's lean-memory regime, two 2^27 A/Bs reduced st6b
    // 19.131→16.937 s and 18.476→15.826 s; CB timestamps showed no
    // regression in the other members' device execution.
    if !metal_gate(KIND, inputs.rows.len()) {
        return None;
    }
    let context = match MetalContext::global() {
        Ok(context) => context,
        Err(error) => {
            tracing::warn!(slot = KIND, %error, "no device context; staying on the CPU");
            return None;
        }
    };
    let recovery = Arc::new(Mutex::new(None));
    match BytecodeDriver::build(context, &inputs, Arc::clone(&recovery)) {
        Ok(driver) => Some(BytecodeCycleDevice {
            driver: Box::new(driver),
            combined_recovery: recovery,
            launch: st6b_detach_enabled(),
        }),
        Err(error) => {
            tracing::warn!(slot = KIND, %error, "device prepare failed; staying on the CPU");
            None
        }
    }
}

struct BytecodeDriver {
    /// Declared first so a flight's wait-on-drop runs before any dispatched
    /// buffer's backing (`shifts`, `cur`, `nxt`, `partials`) frees.
    in_flight: Option<BytecodeFlight>,
    device: DeviceRound,
    rows: Arc<PcRows>,
    shifts: OwnedDeviceBuffer<u32>,
    partials: Partials,
    cur: OwnedDeviceBuffer<Fr>,
    nxt: OwnedDeviceBuffer<Fr>,
    len: usize,
    factors: usize,
    num_ra: usize,
    k_entries: usize,
    mask: u32,
    pending_lazy_bind: Option<Fr>,
    recovery: Arc<Mutex<Option<Vec<Fr>>>>,
}

/// One two-phase round in flight (committed, not yet waited). `pass` first:
/// its wait-on-drop must precede the upload backing (`_flat`) freeing.
struct BytecodeFlight {
    pass: DetachedPass,
    num_tgs: usize,
    /// The round's pending fold; advances the ping-pong on collect success,
    /// feeds the combined recovery on failure.
    bind: Option<Fr>,
    /// The lazy flight's flattened branch-table upload.
    _flat: Vec<Fr>,
}

// SAFETY: the driver is accessed only through `&mut dyn LazyRaDevice`; its
// shared-storage Metal resources are thread-safe and every pass is joined.
unsafe impl Send for BytecodeDriver {}
// SAFETY: no shared-reference operation mutates or dispatches driver state.
unsafe impl Sync for BytecodeDriver {}

fn output_buffer(
    context: &'static MetalContext,
    len: usize,
) -> Result<OwnedDeviceBuffer<Fr>, MetalError> {
    if let Some(buffer) = own_uninit_frs(context, len)? {
        return Ok(buffer);
    }
    context.own_page_aligned(PageAlignedVec::from_elem(Fr::from_u64(0), len))
}

impl BytecodeDriver {
    fn build(
        context: &'static MetalContext,
        inputs: &BytecodeCycleDeviceInputs<'_, Fr>,
        recovery: Arc<Mutex<Option<Vec<Fr>>>>,
    ) -> Result<Self, MetalError> {
        let cycles = inputs.rows.len();
        let num_ra = inputs.selector_shifts.len();
        let factors = num_ra + 1;
        if factors > MAX_FACTORS || inputs.degree != factors {
            return Err(MetalError::UnsupportedShape(
                "bytecode read RAF factor count exceeds the shader capacity",
            ));
        }

        let log_t = cycles.trailing_zeros() as usize;
        let lo_bits = log_t / 2;
        let hi_bits = log_t - lo_bits;
        let in_len = 1usize << lo_bits;
        let out_len = 1usize << hi_bits;
        let mut e_hi = Vec::with_capacity(5 * out_len);
        let mut e_lo = Vec::with_capacity(5 * in_len);
        for point in inputs.stage_cycle_points {
            e_hi.extend(eq_table(&point[..hi_bits]));
            e_lo.extend(eq_table(&point[hi_bits..]));
        }

        let e_hi = context.own_vec(e_hi)?;
        let e_lo = context.own_vec(e_lo)?;
        let weights = context.own_vec(inputs.stage_weights.to_vec())?;
        let shifts = context.own_vec(inputs.selector_shifts.clone())?;
        testing::note_copied_buffers(
            u64::from(e_hi.was_copied())
                + u64::from(e_lo.was_copied())
                + u64::from(weights.was_copied())
                + u64::from(shifts.was_copied()),
        );

        let cur = output_buffer(context, cycles)?;
        let nxt = output_buffer(context, cycles / 2)?;
        let rows = context.wrap_slice(inputs.rows.as_slice())?;
        testing::note_copied_buffers(u64::from(rows.was_copied()));
        let mut params = vec![cycles as u32, lo_bits as u32, in_len as u32, out_len as u32];
        params.extend_from_slice(&fr_to_u32_limbs(inputs.entry_term));
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::BytecodeInit,
            &params,
            &[
                &e_hi.device_buffer(),
                &e_lo.device_buffer(),
                &weights.device_buffer(),
                &cur.device_buffer(),
            ],
            cycles,
        );
        pass.run()?;
        testing::note_device_round();

        Ok(Self {
            in_flight: None,
            device: DeviceRound::new(context, KIND),
            rows: Arc::clone(inputs.rows),
            shifts,
            partials: Partials::new(context, inputs.degree, cycles / 2)?,
            cur,
            nxt,
            len: cycles,
            factors: 1,
            num_ra,
            k_entries: 1usize << inputs.committed_chunk_bits,
            mask: ((1u64 << inputs.committed_chunk_bits) - 1) as u32,
            pending_lazy_bind: None,
            recovery,
        })
    }

    fn publish_recovery(&mut self, bind: Option<Fr>) {
        let mut combined = Polynomial::new(self.cur.as_slice()[..self.len].to_vec());
        if let Some(challenge) = bind {
            combined.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        let mut recovery = self
            .recovery
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *recovery = Some(combined.evals().to_vec());
        self.pending_lazy_bind = None;
    }

    fn flatten_tables(tables: &[Vec<Fr>], stride: usize) -> Vec<Fr> {
        let mut flat = Vec::with_capacity(tables.len() * stride);
        for table in tables {
            debug_assert_eq!(table.len(), stride);
            flat.extend_from_slice(table);
        }
        flat
    }

    /// Encode + commit the lazy round without blocking; the caller decides
    /// whether to wait in place (synchronous tier) or park the flight.
    fn commit_lazy(
        &self,
        context: &'static MetalContext,
        tables: &[Vec<Fr>],
        width: usize,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<(DetachedPass, Vec<Fr>), MetalError> {
        let rows = context.wrap_slice(self.rows.as_slice())?;
        let flat = Self::flatten_tables(tables, width * self.k_entries);
        let tables = context.wrap_slice(fr_as_u32s(&flat))?;
        testing::note_copied_buffers(u64::from(rows.was_copied()) + u64::from(tables.was_copied()));
        let num_tgs = num_threadgroups(groups);
        let mut params = vec![
            groups as u32,
            u32::from(bind.is_some()),
            num_tgs as u32,
            width as u32,
            self.num_ra as u32,
            self.k_entries as u32,
            self.mask,
            self.len as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::BytecodeLazyRound,
            &params,
            &[
                &rows,
                &self.shifts.device_buffer(),
                &tables,
                &self.cur.device_buffer(),
                &self.nxt.device_buffer(),
                &self.partials.buffer().device_buffer(),
            ],
            groups,
        );
        // SAFETY: every dispatched backing outlives the flight and stays
        // host-untouched until its wait — `flat` rides along in the return
        // (the sync caller holds it across the wait, the launch caller parks
        // it in the in-flight state); `rows` is Arc'd on the driver;
        // `shifts`/`cur`/`nxt`/`partials` are driver-owned and next touched
        // after the wait; copied uploads are Metal-owned (retained by the
        // command buffer).
        Ok((unsafe { pass.commit().detach() }, flat))
    }

    fn dispatch_lazy(
        &self,
        context: &'static MetalContext,
        tables: &[Vec<Fr>],
        width: usize,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let num_tgs = num_threadgroups(groups);
        let (pass, _flat) = self.commit_lazy(context, tables, width, bind, groups)?;
        pass.wait()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    fn dispatch_adopt(
        &self,
        context: &'static MetalContext,
        tables: &[Vec<Fr>],
        bind: Fr,
    ) -> Result<(OwnedDeviceBuffer<Fr>, OwnedDeviceBuffer<Fr>), MetalError> {
        let new_len = self.rows.len() / 8;
        let factors = self.num_ra + 1;
        let rows = context.wrap_slice(self.rows.as_slice())?;
        let flat = Self::flatten_tables(tables, 8 * self.k_entries);
        let tables = context.wrap_slice(fr_as_u32s(&flat))?;
        testing::note_copied_buffers(u64::from(rows.was_copied()) + u64::from(tables.was_copied()));
        let cur = output_buffer(context, factors * new_len)?;
        let nxt = output_buffer(context, factors * (new_len / 2))?;
        let mut params = vec![
            new_len as u32,
            self.num_ra as u32,
            self.k_entries as u32,
            self.mask,
            self.len as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind));
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::BytecodeAdopt,
            &params,
            &[
                &rows,
                &self.shifts.device_buffer(),
                &tables,
                &self.cur.device_buffer(),
                &cur.device_buffer(),
            ],
            factors * new_len,
        );
        pass.run()?;
        testing::note_device_round();
        Ok((cur, nxt))
    }

    /// Encode + commit the fused dense round without blocking; waiting as in
    /// [`commit_lazy`](Self::commit_lazy).
    fn commit_dense(
        &self,
        context: &'static MetalContext,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<DetachedPass, MetalError> {
        let num_tgs = num_threadgroups(groups);
        let mut params = vec![
            groups as u32,
            u32::from(bind.is_some()),
            num_tgs as u32,
            self.factors as u32,
            self.len as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::BytecodeDenseRound,
            &params,
            &[
                &self.cur.device_buffer(),
                &self.nxt.device_buffer(),
                &self.partials.buffer().device_buffer(),
            ],
            groups,
        );
        // SAFETY: as in `commit_lazy` — the ping-pong and the partials are
        // driver-owned and next touched after the wait.
        Ok(unsafe { pass.commit().detach() })
    }

    fn dispatch_dense(
        &self,
        context: &'static MetalContext,
        bind: Option<Fr>,
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let num_tgs = num_threadgroups(groups);
        self.commit_dense(context, bind, groups)?.wait()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    /// Post-wait bookkeeping shared by the synchronous and collect paths: a
    /// bound round advances the ping-pong.
    fn advance(&mut self, bound: bool) {
        if bound {
            std::mem::swap(&mut self.cur, &mut self.nxt);
            self.len /= 2;
        }
        self.pending_lazy_bind = None;
    }
}

impl LazyRaDevice<Fr> for BytecodeDriver {
    fn bind_lazy(&mut self, challenge: Fr) {
        debug_assert!(self.pending_lazy_bind.is_none());
        self.pending_lazy_bind = Some(challenge);
    }

    fn lazy_lanes(
        &mut self,
        tables: &[Vec<Fr>],
        width: usize,
        _e_in: &[Fr],
        _e_out: &[Fr],
    ) -> Option<Vec<Fr>> {
        let bind = self.pending_lazy_bind;
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        let Some(context) = self.device.gated(self.len) else {
            self.publish_recovery(bind);
            return None;
        };
        match self.dispatch_lazy(context, tables, width, bind, groups) {
            Ok(lanes) => {
                self.advance(bind.is_some());
                Some(lanes)
            }
            Err(error) => {
                self.device.failed(&error);
                self.publish_recovery(bind);
                None
            }
        }
    }

    fn launch_lazy(
        &mut self,
        tables: &[Vec<Fr>],
        width: usize,
        _e_in: &[Fr],
        _e_out: &[Fr],
    ) -> bool {
        let bind = self.pending_lazy_bind;
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        // A decline here is stateless: the synchronous retry re-runs the
        // same gate and publishes the recovery itself.
        let Some(context) = self.device.gated(self.len) else {
            return false;
        };
        match self.commit_lazy(context, tables, width, bind, groups) {
            Ok((pass, flat)) => {
                self.in_flight = Some(BytecodeFlight {
                    pass,
                    num_tgs: num_threadgroups(groups),
                    bind,
                    _flat: flat,
                });
                true
            }
            Err(error) => {
                // Nothing committed — lazy rounds only read `cur`, so the
                // synchronous retry (now latched off) recovers host-side.
                self.device.failed(&error);
                false
            }
        }
    }

    fn adopt_dense(&mut self, tables: &[Vec<Fr>]) -> bool {
        let Some(bind) = self.pending_lazy_bind else {
            self.publish_recovery(None);
            return false;
        };
        let Some(context) = self.device.gated(self.len) else {
            self.publish_recovery(Some(bind));
            return false;
        };
        match self.dispatch_adopt(context, tables, bind) {
            Ok((cur, nxt)) => {
                self.cur = cur;
                self.nxt = nxt;
                self.len = self.rows.len() / 8;
                self.factors = self.num_ra + 1;
                self.pending_lazy_bind = None;
                true
            }
            Err(error) => {
                self.device.failed(&error);
                self.publish_recovery(Some(bind));
                false
            }
        }
    }

    fn dense_round(&mut self, bind: Option<Fr>, _e_in: &[Fr], _e_out: &[Fr]) -> Option<Vec<Fr>> {
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        let Some(context) = self.device.gated(self.len) else {
            self.publish_recovery(bind);
            return None;
        };
        match self.dispatch_dense(context, bind, groups) {
            Ok(lanes) => {
                self.advance(bind.is_some());
                Some(lanes)
            }
            Err(error) => {
                self.device.failed(&error);
                self.publish_recovery(bind);
                None
            }
        }
    }

    fn launch_dense(&mut self, bind: Option<Fr>, _e_in: &[Fr], _e_out: &[Fr]) -> bool {
        let groups = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        // A dense decline normalizes upstream (`ensure_host` reclaims the RA
        // factors), so the combined factor must publish here — exactly the
        // synchronous decline contract.
        let Some(context) = self.device.gated(self.len) else {
            self.publish_recovery(bind);
            return false;
        };
        match self.commit_dense(context, bind, groups) {
            Ok(pass) => {
                self.in_flight = Some(BytecodeFlight {
                    pass,
                    num_tgs: num_threadgroups(groups),
                    bind,
                    _flat: Vec::new(),
                });
                true
            }
            Err(error) => {
                self.device.failed(&error);
                self.publish_recovery(bind);
                false
            }
        }
    }

    fn collect_lanes(&mut self) -> Option<Vec<Fr>> {
        let flight = self.in_flight.take()?;
        match flight.pass.wait() {
            Ok(()) => {
                testing::note_device_round();
                let lanes = self.partials.sums(flight.num_tgs);
                self.advance(flight.bind.is_some());
                Some(lanes)
            }
            Err(error) => {
                // The kernel writes only `nxt` and the partials — `cur` is
                // intact, so the recovery publishes the pre-fold combined
                // with the flight's bind applied host-side.
                self.device.failed(&error);
                self.publish_recovery(flight.bind);
                None
            }
        }
    }

    fn take_dense(&mut self) -> Vec<Vec<Fr>> {
        let flat = self.cur.as_slice();
        (1..self.factors)
            .map(|factor| flat[factor * self.len..(factor + 1) * self.len].to_vec())
            .collect()
    }
}

#[cfg(test)]
fn flat_word_offset(factor: usize, len: usize, element: usize) -> u64 {
    ((factor as u64) * (len as u64) + element as u64) * FR_U32_LIMBS as u64
}
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, BytecodeReadRafTableFoldInputs,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::{JoltWitnessOracle, ProgramSource};

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::parity::{
        probe_input_claim, probe_one_hot_family, run_lockstep, synthetic_point,
    };
    use crate::ReferenceBackend;

    fn force_device_gate() {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
    }

    #[test]
    fn bytecode_flat_offset_matches_device_at_2e27_shape() {
        let _lock = gpu_lock();
        let context = MetalContext::global().unwrap();
        let out = context.alloc_u32s(2).unwrap();
        let cases = [
            (0usize, 1usize << 27, 0usize),
            (3, 1usize << 27, (1usize << 27) - 1),
            (4, 1usize << 27, 0),
            (7, 1usize << 27, 17),
        ];
        for (factor, len, element) in cases {
            context
                .run_once(
                    KernelId::BytecodeOffsetProbe,
                    &[factor as u32, len as u32, element as u32],
                    &[&out],
                    1,
                )
                .unwrap();
            let mut got = [0u32; 2];
            out.copy_to_u32s(&mut got);
            let got = u64::from(got[0]) | (u64::from(got[1]) << 32);
            assert_eq!(got, flat_word_offset(factor, len, element));
        }
        assert_eq!(flat_word_offset(4, 1 << 27, 0), 1u64 << 32);
    }

    #[test]
    fn bytecode_read_raf_cycle_matches_optimized() {
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
            let program = backend.program_preprocessing();
            let log_k = program.bytecode.bytecode.len().ilog2() as usize;
            let (bytecode_d, chunk_bits) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::BytecodeRa, log_t);
            let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, bytecode_d);
            let r_address = synthetic_point(log_k, 17);
            let stage_cycle_points = std::array::from_fn(|s| synthetic_point(log_t, 31 + s as u64));
            let register_read_write_point = synthetic_point(REGISTER_ADDRESS_BITS + log_t, 101);
            let register_val_evaluation_point = synthetic_point(REGISTER_ADDRESS_BITS + log_t, 103);
            let address_challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: Fr::from_u64(3),
                stage1_gamma: Fr::from_u64(5),
                stage2_gamma: Fr::from_u64(7),
                stage3_gamma: Fr::from_u64(11),
                stage4_gamma: Fr::from_u64(13),
                stage5_gamma: Fr::from_u64(17),
            };
            let stage_gammas = address_challenges.stage_gamma_powers();
            let relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                dimensions,
                r_address,
                stage_cycle_points,
                entry_bytecode_index: 0,
                committed_chunk_bits: chunk_bits,
                table_fold: Some(BytecodeReadRafTableFoldInputs {
                    bytecode: &program.bytecode.bytecode,
                    register_read_write_point: &register_read_write_point[..REGISTER_ADDRESS_BITS],
                    register_val_evaluation_point: &register_val_evaluation_point
                        [..REGISTER_ADDRESS_BITS],
                    stage_gammas: std::array::from_fn(|s| stage_gammas[s].as_slice()),
                }),
            })
            .unwrap();
            let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: Fr::from_u64(19),
            };
            let claims = BytecodeReadRafInputClaims::<Fr>::default();
            let points = BytecodeReadRafInputClaims::<Vec<Fr>>::default();
            let mut session = ProofSession::default();
            let mut reference = ReferenceBackend
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = crate::optimized::bytecode_read_raf::OptimizedBytecodeReadRafCycle
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let before = device_probe_count();
            let mut metal = MetalBytecodeReadRafCycle
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let claim = probe_input_claim(reference.as_mut());
            run_lockstep(
                optimized.as_mut(),
                metal.as_mut(),
                claim,
                &synthetic_point(log_t, 211),
            );
            assert_eq!(
                optimized.output_claims(&claims).unwrap(),
                metal.output_claims(&claims).unwrap()
            );
            assert_eq!(
                device_probe_count() - before,
                log_t as u64 + 1,
                "combined init and every round must dispatch"
            );
        });
    }
}
