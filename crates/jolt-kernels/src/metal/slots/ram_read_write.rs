//! Metal RAM read/write-checking cycle rounds with an early host handoff.

use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use super::{num_threadgroups, Partials, RoundTable};
use crate::metal::buffers::{OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{DetachedPass, KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::mmap_vec::MmapVec;
use crate::optimized::ram_read_write::RamReadWriteKernel;
use crate::optimized::ram_trace::{RamAccessColumns, NO_ACCESS};
use crate::optimized::rw_matrix::{CycleMajorEntry, CycleMajorMatrix};
use crate::optimized::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "ram_read_write";
const HANDOFF_ROWS: usize = 4096;

fn fused_rounds_enabled() -> bool {
    std::env::var("JOLT_RAMRW_FUSED").is_ok_and(|value| matches!(value.trim(), "1" | "on" | "ON"))
}

fn gpu_prepare_enabled() -> bool {
    std::env::var("JOLT_RAMRW_GPU_PREPARE")
        .is_ok_and(|value| matches!(value.trim(), "1" | "on" | "ON"))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RawRamRwEntry {
    val: Fr,
    ra: Fr,
    prev_val: u64,
    next_val: u64,
    col: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<RawRamRwEntry>() == 88);

#[derive(Clone, Copy)]
struct RamRwBuildInputs<'a> {
    addresses: &'a [u64],
    pre_values: &'a [u64],
    post_values: &'a [u64],
}

impl<'a> From<&'a RamAccessColumns> for RamRwBuildInputs<'a> {
    fn from(columns: &'a RamAccessColumns) -> Self {
        Self {
            addresses: columns.addresses.as_slice(),
            pre_values: columns.pre_values.as_slice(),
            post_values: columns.post_values.as_slice(),
        }
    }
}

fn ram_rw_offsets(inputs: RamRwBuildInputs<'_>) -> (Vec<u32>, usize) {
    let mut offsets = Vec::with_capacity(inputs.addresses.len() + 1);
    offsets.push(0);
    let mut count = 0_u32;
    for &address in inputs.addresses {
        count += u32::from(address != NO_ACCESS);
        offsets.push(count);
    }
    (offsets, count as usize)
}

fn ram_rw_entries_serial(inputs: RamRwBuildInputs<'_>) -> (Vec<RawRamRwEntry>, Vec<u32>) {
    let mut entries = Vec::new();
    let mut offsets = Vec::with_capacity(inputs.addresses.len() + 1);
    offsets.push(0);
    for cycle in 0..inputs.addresses.len() {
        let address = inputs.addresses[cycle];
        if address != NO_ACCESS {
            let pre_value = inputs.pre_values[cycle];
            entries.push(RawRamRwEntry {
                val: Fr::from_u64(pre_value),
                ra: Fr::from_u64(1),
                prev_val: pre_value,
                next_val: inputs.post_values[cycle],
                col: address as u32,
                pad: 0,
            });
        }
        offsets.push(entries.len() as u32);
    }
    (entries, offsets)
}

struct PendingRamRwEntries {
    // The flight must settle before its output buffers drop.
    pass: DetachedPass,
    entries: OwnedDeviceBuffer<RawRamRwEntry>,
    entry_count: usize,
    offsets: OwnedDeviceBuffer<u32>,
}

impl PendingRamRwEntries {
    fn launch(
        context: &'static MetalContext,
        inputs: RamRwBuildInputs<'_>,
    ) -> Result<Self, MetalError> {
        let cycles = inputs.addresses.len();
        let (offsets, entry_count) = ram_rw_offsets(inputs);
        if entry_count == 0 {
            return Err(MetalError::UnsupportedShape(
                "RAM read/write CSR has no entries",
            ));
        }
        let offsets = context.own_vec(offsets)?;
        let entries = context.own_mmap(MmapVec::zeroed(entry_count))?;
        let addresses = context.wrap_slice(inputs.addresses)?;
        let pre_values = context.wrap_slice(inputs.pre_values)?;
        let post_values = context.wrap_slice(inputs.post_values)?;
        let offset_buffer = offsets.device_buffer();
        let entry_buffer = entries.device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::RamRwBuild,
            &[cycles as u32],
            &[
                &addresses,
                &pre_values,
                &post_values,
                &offset_buffer,
                &entry_buffer,
            ],
            cycles,
        );
        // SAFETY: the columns outlive the wait in `prepare`; the output
        // buffers move into this flight and are host-untouched until wait.
        let pass = unsafe { pass.commit().detach() };
        Ok(Self {
            pass,
            entries,
            entry_count,
            offsets,
        })
    }

    fn wait(
        self,
        context: &'static MetalContext,
        inc: Vec<Fr>,
    ) -> Result<DeviceRamRwState, MetalError> {
        let (entries, entry_count, offsets) = self.wait_buffers()?;
        DeviceRamRwState::from_buffers(context, entries, entry_count, offsets, inc)
    }

    fn wait_buffers(
        self,
    ) -> Result<
        (
            OwnedDeviceBuffer<RawRamRwEntry>,
            usize,
            OwnedDeviceBuffer<u32>,
        ),
        MetalError,
    > {
        let Self {
            pass,
            entries,
            entry_count,
            offsets,
        } = self;
        pass.wait()?;
        Ok((entries, entry_count, offsets))
    }
}

struct PingPong<T: Copy> {
    cur: OwnedDeviceBuffer<T>,
    nxt: OwnedDeviceBuffer<T>,
}

impl<T: Copy> PingPong<T> {
    fn swap(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.nxt);
    }
}

struct DeviceRamRwState {
    context: &'static MetalContext,
    entries: PingPong<RawRamRwEntry>,
    entry_count: usize,
    offsets: PingPong<u32>,
    counts: OwnedDeviceBuffer<u32>,
    inc: RoundTable,
    partials: Partials,
    rows: usize,
    counts_valid: bool,
}

impl DeviceRamRwState {
    fn new(
        context: &'static MetalContext,
        entries: Vec<RawRamRwEntry>,
        offsets: Vec<u32>,
        inc: Vec<Fr>,
    ) -> Result<Self, MetalError> {
        let entry_count = entries.len();
        let entries = context.own_vec(entries)?;
        let offsets = context.own_vec(offsets)?;
        Self::from_buffers(context, entries, entry_count, offsets, inc)
    }

    fn from_buffers(
        context: &'static MetalContext,
        entries: OwnedDeviceBuffer<RawRamRwEntry>,
        entry_count: usize,
        offsets: OwnedDeviceBuffer<u32>,
        inc: Vec<Fr>,
    ) -> Result<Self, MetalError> {
        let rows = inc.len();
        Ok(Self {
            context,
            entries: PingPong {
                cur: entries,
                nxt: context.own_page_aligned(PageAlignedVec::from_elem(
                    RawRamRwEntry::default(),
                    entry_count,
                ))?,
            },
            entry_count,
            offsets: PingPong {
                cur: offsets,
                nxt: context.own_page_aligned(PageAlignedVec::from_elem(0_u32, rows + 1))?,
            },
            counts: context
                .own_page_aligned(PageAlignedVec::from_elem(0_u32, (rows / 2).max(1)))?,
            inc: RoundTable::new(context, inc)?,
            partials: Partials::new(context, 2, (rows / 2).max(1))?,
            rows,
            counts_valid: false,
        })
    }

    fn scan_bind_offsets(&mut self) -> Result<usize, MetalError> {
        if !self.counts_valid {
            return Err(MetalError::Execution(
                "RAM read/write bind without a preceding message".to_string(),
            ));
        }
        let pairs = self.rows / 2;
        let mut total = 0_u32;
        let counts = &self.counts.as_slice()[..pairs];
        let offsets = &mut self.offsets.nxt.as_mut_slice()[..=pairs];
        offsets[0] = 0;
        for (index, &count) in counts.iter().enumerate() {
            total = total.checked_add(count).ok_or_else(|| {
                MetalError::Execution("RAM read/write entry count overflow".to_string())
            })?;
            offsets[index + 1] = total;
        }
        let new_count = total as usize;
        if new_count > self.entries.nxt.len() {
            return Err(MetalError::Execution(
                "RAM read/write bind exceeded sparse entry capacity".to_string(),
            ));
        }
        Ok(new_count)
    }

    fn message(
        &mut self,
        gruen: &GruenSplitEqPolynomial<Fr>,
        gamma: Fr,
    ) -> Result<[Fr; 2], MetalError> {
        let pairs = self.rows / 2;
        let num_tgs = num_threadgroups(pairs);
        let e_in = gruen.e_in_current();
        let e_out = gruen.e_out_current();
        let e_in_buffer = self.context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = self.context.wrap_slice(fr_as_u32s(e_out))?;
        let entry_buffer = self.entries.cur.device_buffer();
        let offset_buffer = self.offsets.cur.device_buffer();
        let inc_buffer = self.inc.cur().device_buffer();
        let partial_buffer = self.partials.buffer().device_buffer();
        let count_buffer = self.counts.device_buffer();
        let mut params = vec![
            pairs as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            e_in.len() as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(gamma));
        let mut pass = self.context.begin_pass()?;
        pass.dispatch(
            KernelId::RamRwMessage,
            &params,
            &[
                &entry_buffer,
                &offset_buffer,
                &inc_buffer,
                &e_out_buffer,
                &e_in_buffer,
                &partial_buffer,
                &count_buffer,
            ],
            pairs,
        );
        pass.run()?;
        testing::note_device_round();
        self.counts_valid = true;
        let sums = self.partials.sums(num_tgs);
        Ok([sums[0], sums[1]])
    }

    fn bind(&mut self, challenge: Fr) -> Result<(), MetalError> {
        let pairs = self.rows / 2;
        let new_count = self.scan_bind_offsets()?;
        {
            let entry_buffer = self.entries.cur.device_buffer();
            let offset_buffer = self.offsets.cur.device_buffer();
            let out_offset_buffer = self.offsets.nxt.device_buffer();
            let out_entry_buffer = self.entries.nxt.device_buffer();
            let inc_buffer = self.inc.cur().device_buffer();
            let out_inc_buffer = self.inc.nxt().device_buffer();
            let mut params = vec![pairs as u32];
            params.extend_from_slice(&fr_to_u32_limbs(challenge));
            let mut pass = self.context.begin_pass()?;
            pass.dispatch(
                KernelId::RamRwBind,
                &params,
                &[
                    &entry_buffer,
                    &offset_buffer,
                    &out_offset_buffer,
                    &out_entry_buffer,
                    &inc_buffer,
                    &out_inc_buffer,
                ],
                pairs,
            );
            pass.run()?;
        }
        testing::note_device_round();
        self.entries.swap();
        self.offsets.swap();
        self.inc.swap();
        self.entry_count = new_count;
        self.rows = pairs;
        self.counts_valid = false;
        Ok(())
    }

    fn bind_and_message(
        &mut self,
        challenge: Fr,
        gruen: &GruenSplitEqPolynomial<Fr>,
        gamma: Fr,
    ) -> Result<[Fr; 2], MetalError> {
        let new_count = self.scan_bind_offsets()?;
        let rows = self.rows / 2;
        let pairs = rows / 2;
        let num_tgs = num_threadgroups(pairs);
        let e_in = gruen.e_in_current();
        let e_out = gruen.e_out_current();
        let e_in_buffer = self.context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = self.context.wrap_slice(fr_as_u32s(e_out))?;
        let entry_buffer = self.entries.cur.device_buffer();
        let offset_buffer = self.offsets.cur.device_buffer();
        let out_offset_buffer = self.offsets.nxt.device_buffer();
        let out_entry_buffer = self.entries.nxt.device_buffer();
        let inc_buffer = self.inc.cur().device_buffer();
        let out_inc_buffer = self.inc.nxt().device_buffer();
        let partial_buffer = self.partials.buffer().device_buffer();
        let count_buffer = self.counts.device_buffer();
        let mut bind_params = vec![rows as u32];
        bind_params.extend_from_slice(&fr_to_u32_limbs(challenge));
        let mut message_params = vec![
            pairs as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            e_in.len() as u32,
        ];
        message_params.extend_from_slice(&fr_to_u32_limbs(gamma));
        let mut pass = self.context.begin_pass()?;
        pass.dispatch(
            KernelId::RamRwBind,
            &bind_params,
            &[
                &entry_buffer,
                &offset_buffer,
                &out_offset_buffer,
                &out_entry_buffer,
                &inc_buffer,
                &out_inc_buffer,
            ],
            rows,
        );
        pass.buffer_barrier();
        pass.dispatch(
            KernelId::RamRwMessage,
            &message_params,
            &[
                &out_entry_buffer,
                &out_offset_buffer,
                &out_inc_buffer,
                &e_out_buffer,
                &e_in_buffer,
                &partial_buffer,
                &count_buffer,
            ],
            pairs,
        );
        pass.run()?;
        testing::note_device_round();
        self.entries.swap();
        self.offsets.swap();
        self.inc.swap();
        self.entry_count = new_count;
        self.rows = rows;
        self.counts_valid = true;
        let sums = self.partials.sums(num_tgs);
        Ok([sums[0], sums[1]])
    }

    fn into_cycle_state(self) -> Result<(CycleMajorMatrix<Fr>, Polynomial<Fr>), ()> {
        let offsets = &self.offsets.cur.as_slice()[..=self.rows];
        if offsets[self.rows] as usize != self.entry_count {
            return Err(());
        }
        let raw_entries = &self.entries.cur.as_slice()[..self.entry_count];
        let mut entries = Vec::with_capacity(self.entry_count);
        for row in 0..self.rows {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let Some(row_entries) = raw_entries.get(start..end) else {
                return Err(());
            };
            entries.extend(row_entries.iter().map(|raw| CycleMajorEntry {
                row,
                col: raw.col as usize,
                prev_val: raw.prev_val,
                next_val: raw.next_val,
                val: raw.val,
                ra: raw.ra,
            }));
        }
        let inc = Polynomial::new(self.inc.cur_slice(self.rows).to_vec());
        Ok((CycleMajorMatrix { entries }, inc))
    }
}

pub struct MetalRamReadWriteChecking {
    fallback: OptimizedBackend,
    handoff_rows: usize,
}

impl MetalRamReadWriteChecking {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedBackend,
            handoff_rows: HANDOFF_ROWS,
        }
    }

    #[cfg(test)]
    fn with_handoff_rows(handoff_rows: usize) -> Self {
        Self {
            fallback: OptimizedBackend,
            handoff_rows,
        }
    }
}

impl Default for MetalRamReadWriteChecking {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepareKernel<Fr, RamReadWriteChecking<Fr>> for MetalRamReadWriteChecking {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, RamReadWriteChecking<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = RamReadWriteChecking<Fr>>>, KernelError<Fr>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let log_t = dimensions.log_t();
        let log_k = relation.ram_log_k();
        let tau_low = relation.product_tau_low();
        if dimensions.phase1_num_rounds() != log_t {
            return Err(KernelError::Unsupported {
                reason: "Metal RAM read-write checking supports only the default read-write config",
            });
        }
        if log_t == 0 || dimensions.log_k() != log_k || tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write checking geometry is inconsistent",
            });
        }
        let cycles = 1usize << log_t;
        if !metal_gate(KIND, cycles) || log_t >= 32 {
            return self.fallback.prepare(session, witness, inputs);
        }

        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses(1usize << log_k)?;
        let gpu_prepare = gpu_prepare_enabled();
        let (context, pending_entries, serial_entries) = if gpu_prepare {
            let context = match MetalContext::global() {
                Ok(context) => context,
                Err(error) => {
                    tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                    return self.fallback.prepare(session, witness, inputs);
                }
            };
            let pending = match PendingRamRwEntries::launch(
                context,
                RamRwBuildInputs::from(columns.as_ref()),
            ) {
                Ok(pending) => pending,
                Err(error) => {
                    tracing::warn!(slot = KIND, %error, "device CSR build failed; using optimized fallback");
                    return self.fallback.prepare(session, witness, inputs);
                }
            };
            (Some(context), Some(pending), None)
        } else {
            let serial = ram_rw_entries_serial(RamRwBuildInputs::from(columns.as_ref()));
            if serial.0.is_empty() {
                return self.fallback.prepare(session, witness, inputs);
            }
            (None, None, Some(serial))
        };
        let inc = columns.inc_column::<Fr>();
        let val_final = witness.oracle_table(JoltPolynomialId::Virtual(
            JoltVirtualPolynomial::RamValFinal,
        ))?;
        if inc.len() != cycles || val_final.len() != 1usize << log_k {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write witness tables disagree with the relation geometry",
            });
        }
        let val_init = Polynomial::new(columns.reconstruct_val_init(val_final));
        let context = match context.map_or_else(MetalContext::global, Ok) {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        let device = match (pending_entries, serial_entries) {
            (Some(pending), None) => pending.wait(context, inc),
            (None, Some((entries, offsets))) => {
                DeviceRamRwState::new(context, entries, offsets, inc)
            }
            _ => unreachable!("RAM read/write prepare selects exactly one CSR builder"),
        };
        let device = match device {
            Ok(device) => device,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device preparation failed; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        Ok(Box::new(MetalRamRwKernel {
            device: Some(device),
            gruen: Some(GruenSplitEqPolynomial::new(
                tau_low,
                BindingOrder::LowToHigh,
            )),
            host: None,
            val_init: Some(val_init),
            gamma: inputs.challenges.gamma,
            log_t,
            log_k,
            handoff_rows: self.handoff_rows.max(1),
            fused_rounds: fused_rounds_enabled(),
            rounds_bound: 0,
        }))
    }
}

fn missing_device_state() -> SumcheckError<Fr> {
    SumcheckError::MissingEvaluationSource {
        kind: "Metal RAM read/write device state",
    }
}

struct MetalRamRwKernel {
    device: Option<DeviceRamRwState>,
    gruen: Option<GruenSplitEqPolynomial<Fr>>,
    host: Option<RamReadWriteKernel<Fr>>,
    val_init: Option<Polynomial<Fr>>,
    gamma: Fr,
    log_t: usize,
    log_k: usize,
    handoff_rows: usize,
    fused_rounds: bool,
    rounds_bound: usize,
}

impl MetalRamRwKernel {
    fn transition(&mut self) -> Result<(), SumcheckError<Fr>> {
        let state = self.device.take().ok_or_else(missing_device_state)?;
        let (matrix, inc) = state
            .into_cycle_state()
            .map_err(|()| missing_device_state())?;
        let gruen = self.gruen.take().ok_or_else(missing_device_state)?;
        let val_init = self.val_init.take().ok_or_else(missing_device_state)?;
        self.host = Some(RamReadWriteKernel::from_cycle_state(
            matrix,
            gruen,
            inc,
            val_init,
            self.gamma,
            self.log_t,
            self.log_k,
            self.rounds_bound,
        )?);
        Ok(())
    }

    fn maybe_transition(&mut self) -> Result<(), SumcheckError<Fr>> {
        if self
            .device
            .as_ref()
            .is_some_and(|state| state.rows <= self.handoff_rows)
        {
            self.transition()?;
        }
        Ok(())
    }

    fn host_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        self.host
            .as_mut()
            .ok_or_else(missing_device_state)?
            .prove_round(bind, round, previous_claim)
    }
}

impl ProveRounds<Fr> for MetalRamRwKernel {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if self.host.is_some() {
            return self.host_round(bind, round, previous_claim);
        }
        if let Some(challenge) = bind {
            let can_fuse = self.fused_rounds
                && self
                    .device
                    .as_ref()
                    .is_some_and(|state| state.rows / 2 > self.handoff_rows && state.rows >= 4);
            if can_fuse {
                let mut next_gruen = self
                    .gruen
                    .as_ref()
                    .ok_or_else(missing_device_state)?
                    .clone();
                next_gruen.bind(challenge);
                let result = self
                    .device
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .bind_and_message(challenge, &next_gruen, self.gamma);
                let [q_0, q_inf] = match result {
                    Ok(values) => values,
                    Err(error) => {
                        tracing::warn!(slot = KIND, %error, "fused device round failed; finishing on CPU");
                        self.transition()?;
                        return self.host_round(Some(challenge), round, previous_claim);
                    }
                };
                self.gruen = Some(next_gruen);
                self.rounds_bound += 1;
                return Ok(self
                    .gruen
                    .as_ref()
                    .ok_or_else(missing_device_state)?
                    .gruen_poly_deg_3(q_0, q_inf, previous_claim));
            }
            let bind_result = self
                .device
                .as_mut()
                .ok_or_else(missing_device_state)?
                .bind(challenge);
            if let Err(error) = bind_result {
                tracing::warn!(slot = KIND, %error, "device bind failed; finishing on CPU");
                self.transition()?;
                return self.host_round(Some(challenge), round, previous_claim);
            }
            self.gruen
                .as_mut()
                .ok_or_else(missing_device_state)?
                .bind(challenge);
            self.rounds_bound += 1;
        }
        self.maybe_transition()?;
        if self.host.is_some() {
            return self.host_round(None, round, previous_claim);
        }
        let result = self
            .device
            .as_mut()
            .ok_or_else(missing_device_state)?
            .message(
                self.gruen.as_ref().ok_or_else(missing_device_state)?,
                self.gamma,
            );
        match result {
            Ok([q_0, q_inf]) => Ok(self
                .gruen
                .as_ref()
                .ok_or_else(missing_device_state)?
                .gruen_poly_deg_3(q_0, q_inf, previous_claim)),
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device message failed; finishing on CPU");
                self.transition()?;
                self.host_round(None, round, previous_claim)
            }
        }
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        if let Some(host) = &mut self.host {
            return host.finish_rounds(bind);
        }
        let bind_result = self
            .device
            .as_mut()
            .ok_or_else(missing_device_state)?
            .bind(bind);
        if let Err(error) = bind_result {
            tracing::warn!(slot = KIND, %error, "final device bind failed; finishing on CPU");
            self.transition()?;
            return self
                .host
                .as_mut()
                .ok_or_else(missing_device_state)?
                .finish_rounds(bind);
        }
        self.gruen
            .as_mut()
            .ok_or_else(missing_device_state)?
            .bind(bind);
        self.rounds_bound += 1;
        self.transition()
    }
}

impl SumcheckKernel<Fr> for MetalRamRwKernel {
    type Relation = RamReadWriteChecking<Fr>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<Fr, Self::Relation>, SumcheckKernelError<Fr>> {
        let remaining = self.num_rounds().saturating_sub(self.rounds_bound);
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
                remaining: self.num_rounds().saturating_sub(self.rounds_bound),
            })?
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

#[cfg(feature = "bench-utils")]
pub mod bench {
    #![expect(clippy::expect_used, reason = "benchmark fixtures must fail loudly")]

    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use jolt_sumcheck::ProveRounds;

    use super::*;

    const HANDOFF_LOG_ROWS: usize = 12;

    #[derive(Clone, Copy)]
    pub struct BenchConfig {
        pub log_t: usize,
        pub log_k: usize,
        pub seed: u64,
    }

    impl BenchConfig {
        pub fn production(log_t: usize) -> Self {
            assert!(log_t > HANDOFF_LOG_ROWS);
            Self {
                log_t,
                log_k: 16,
                seed: 0x5743_5241_4D52_5752 ^ log_t as u64,
            }
        }
    }

    struct SyntheticColumns {
        addresses: MmapVec<u64>,
        pre_values: MmapVec<u64>,
        post_values: MmapVec<u64>,
    }

    impl SyntheticColumns {
        fn inputs(&self) -> RamRwBuildInputs<'_> {
            RamRwBuildInputs {
                addresses: self.addresses.as_slice(),
                pre_values: self.pre_values.as_slice(),
                post_values: self.post_values.as_slice(),
            }
        }
    }

    pub struct PrepareFixture {
        columns: SyntheticColumns,
    }

    enum PreparedCsr {
        Host(Vec<RawRamRwEntry>, Vec<u32>),
        Device(
            OwnedDeviceBuffer<RawRamRwEntry>,
            usize,
            OwnedDeviceBuffer<u32>,
        ),
    }

    pub struct PrepareTiming {
        pub total: Duration,
        pub command_buffers: u64,
        pub kernel_dispatches: u64,
        pub entry_bytes: usize,
    }

    pub struct RoundFixture {
        config: BenchConfig,
        entries: Vec<RawRamRwEntry>,
        offsets: Vec<u32>,
        inc: Vec<Fr>,
        r_cycle: Vec<Fr>,
        gamma: Fr,
    }

    pub struct PreparedRounds {
        kernel: MetalRamRwKernel,
        fused: bool,
        rounds: usize,
    }

    pub struct RoundTiming {
        pub total: Duration,
        pub command_buffers: u64,
        pub kernel_dispatches: u64,
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn challenge(round: usize) -> Fr {
        Fr::from_u64(0xA076_1D64_78BD_642F ^ (round as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB))
    }

    fn synthetic_columns(config: BenchConfig) -> SyntheticColumns {
        let cycles = 1usize << config.log_t;
        let ram_k = 1usize << config.log_k;
        let mut columns = SyntheticColumns {
            addresses: MmapVec::filled(cycles, NO_ACCESS),
            pre_values: MmapVec::zeroed(cycles),
            post_values: MmapVec::zeroed(cycles),
        };
        let mut state = config.seed;
        for cycle in 0..cycles {
            if cycle.is_multiple_of(8) {
                let pre = splitmix(&mut state);
                let post = if cycle.is_multiple_of(16) {
                    splitmix(&mut state)
                } else {
                    pre
                };
                columns.addresses[cycle] = splitmix(&mut state) % ram_k as u64;
                columns.pre_values[cycle] = pre;
                columns.post_values[cycle] = post;
            }
        }
        columns
    }

    fn synthetic_inc(inputs: RamRwBuildInputs<'_>) -> Vec<Fr> {
        inputs
            .post_values
            .iter()
            .zip(inputs.pre_values)
            .map(|(&post, &pre)| Fr::from_i128(i128::from(post) - i128::from(pre)))
            .collect()
    }

    impl PrepareFixture {
        pub fn synthetic(config: BenchConfig) -> Self {
            Self {
                columns: synthetic_columns(config),
            }
        }

        pub fn cycles(&self) -> usize {
            self.columns.addresses.len()
        }

        pub fn run(&self, gpu: bool) -> PrepareTiming {
            let context = MetalContext::global().expect("Metal context");
            let dispatches_before = testing::device_dispatch_count();
            let started = Instant::now();
            let prepared = if gpu {
                let (entries, entry_count, offsets) =
                    PendingRamRwEntries::launch(context, self.columns.inputs())
                        .and_then(PendingRamRwEntries::wait_buffers)
                        .expect("GPU RAM read/write CSR build");
                PreparedCsr::Device(entries, entry_count, offsets)
            } else {
                let (entries, offsets) = ram_rw_entries_serial(self.columns.inputs());
                PreparedCsr::Host(entries, offsets)
            };
            let total = started.elapsed();
            let (entry_count, offset_count) = match &prepared {
                PreparedCsr::Host(entries, offsets) => (entries.len(), offsets.len()),
                PreparedCsr::Device(entries, entry_count, offsets) => {
                    assert_eq!(entries.len(), *entry_count);
                    (*entry_count, offsets.len())
                }
            };
            let _ = black_box(&prepared);
            assert_eq!(offset_count, self.cycles() + 1);
            let kernel_dispatches = testing::device_dispatch_count() - dispatches_before;
            assert_eq!(kernel_dispatches, u64::from(gpu));
            PrepareTiming {
                total,
                command_buffers: u64::from(gpu),
                kernel_dispatches,
                entry_bytes: entry_count * std::mem::size_of::<RawRamRwEntry>(),
            }
        }
    }

    impl RoundFixture {
        pub fn synthetic(config: BenchConfig) -> Self {
            let columns = synthetic_columns(config);
            let inputs = columns.inputs();
            let (entries, offsets) = ram_rw_entries_serial(inputs);
            let inc = synthetic_inc(inputs);
            let mut state = config.seed ^ 0x4551_4359_434C_4553;
            let r_cycle = (0..config.log_t)
                .map(|_| Fr::from_u64(splitmix(&mut state)))
                .collect();
            Self {
                config,
                entries,
                offsets,
                inc,
                r_cycle,
                gamma: Fr::from_u64(0x5EED_1234_5678_9ABC),
            }
        }

        pub fn cycles(&self) -> usize {
            1usize << self.config.log_t
        }

        fn device_kernel(&self, fused: bool) -> MetalRamRwKernel {
            let context = MetalContext::global().expect("Metal context");
            let device = DeviceRamRwState::new(
                context,
                self.entries.clone(),
                self.offsets.clone(),
                self.inc.clone(),
            )
            .expect("RAM read/write round fixture");
            MetalRamRwKernel {
                device: Some(device),
                gruen: Some(GruenSplitEqPolynomial::new(
                    &self.r_cycle,
                    BindingOrder::LowToHigh,
                )),
                host: None,
                val_init: Some(Polynomial::new(vec![
                    Fr::from_u64(0);
                    1usize << self.config.log_k
                ])),
                gamma: self.gamma,
                log_t: self.config.log_t,
                log_k: self.config.log_k,
                handoff_rows: HANDOFF_ROWS,
                fused_rounds: fused,
                rounds_bound: 0,
            }
        }

        pub fn prepare(&self, fused: bool) -> PreparedRounds {
            PreparedRounds {
                kernel: self.device_kernel(fused),
                fused,
                rounds: self.config.log_t - HANDOFF_LOG_ROWS,
            }
        }

        pub fn run(&self, fused: bool) -> RoundTiming {
            self.prepare(fused).run()
        }
    }

    impl PreparedRounds {
        pub fn run(&mut self) -> RoundTiming {
            let command_buffers_before = testing::device_probe_count();
            let dispatches_before = testing::device_dispatch_count();
            let started = Instant::now();
            let _ = drive_prefix(&mut self.kernel, self.rounds);
            let total = started.elapsed();
            let command_buffers = testing::device_probe_count() - command_buffers_before;
            let kernel_dispatches = testing::device_dispatch_count() - dispatches_before;
            let expected_dispatches = 2 * self.rounds - 1;
            let expected_command_buffers = if self.fused {
                self.rounds
            } else {
                expected_dispatches
            };
            assert_eq!(command_buffers, expected_command_buffers as u64);
            assert_eq!(kernel_dispatches, expected_dispatches as u64);
            RoundTiming {
                total,
                command_buffers,
                kernel_dispatches,
            }
        }
    }

    fn drive_prefix(kernel: &mut dyn ProveRounds<Fr>, rounds: usize) -> Vec<Vec<Fr>> {
        let mut claim = Fr::from_u64(0xBEEF);
        let mut messages = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let poly = kernel
                .prove_round(bind, round, claim)
                .expect("RAM read/write prefix round");
            claim = poly.evaluate(challenge(round));
            messages.push(poly.coefficients().to_vec());
        }
        messages
    }

    pub fn assert_small_scale_prepare_parity() {
        let fixture = PrepareFixture::synthetic(BenchConfig {
            log_t: 12,
            log_k: 8,
            seed: 17,
        });
        let (expected_entries, expected_offsets) = ram_rw_entries_serial(fixture.columns.inputs());
        let context = MetalContext::global().expect("Metal context");
        let (entries, entry_count, offsets) =
            PendingRamRwEntries::launch(context, fixture.columns.inputs())
                .and_then(PendingRamRwEntries::wait_buffers)
                .expect("GPU RAM read/write CSR oracle");
        assert_eq!(entry_count, expected_entries.len());
        assert_eq!(entries.as_slice(), expected_entries);
        assert_eq!(offsets.as_slice(), expected_offsets);
    }

    pub fn assert_small_scale_round_parity() {
        let fixture = RoundFixture::synthetic(BenchConfig {
            log_t: 14,
            log_k: 8,
            seed: 23,
        });
        let mut host_entries = Vec::with_capacity(fixture.entries.len());
        for row in 0..fixture.cycles() {
            for raw in
                &fixture.entries[fixture.offsets[row] as usize..fixture.offsets[row + 1] as usize]
            {
                host_entries.push(CycleMajorEntry {
                    row,
                    col: raw.col as usize,
                    prev_val: raw.prev_val,
                    next_val: raw.next_val,
                    val: raw.val,
                    ra: raw.ra,
                });
            }
        }
        let mut host = RamReadWriteKernel::from_cycle_state(
            CycleMajorMatrix {
                entries: host_entries,
            },
            GruenSplitEqPolynomial::new(&fixture.r_cycle, BindingOrder::LowToHigh),
            Polynomial::new(fixture.inc.clone()),
            Polynomial::new(vec![Fr::from_u64(0); 1usize << fixture.config.log_k]),
            fixture.gamma,
            fixture.config.log_t,
            fixture.config.log_k,
            0,
        )
        .expect("CPU RAM read/write fixture");
        let rounds = fixture.config.log_t - HANDOFF_LOG_ROWS;
        let expected = drive_prefix(&mut host, rounds);
        for fused in [false, true] {
            assert_eq!(
                drive_prefix(&mut fixture.device_kernel(fused), rounds),
                expected,
                "RAM read/write prefix wire parity failed (fused={fused})"
            );
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{ram_inc, ram_ra, ram_val};
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::stage2::ram_read_write_checking::{
        RamReadWriteChallenges, RamReadWriteInputClaims,
    };

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };

    fn dense_input_claim(
        witness: &dyn JoltWitnessPlane<Fr>,
        tau_low: &[Fr],
        gamma: Fr,
        ram_k: usize,
    ) -> Fr {
        let cycles = 1usize << tau_low.len();
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let ra: Vec<Fr> = witness.oracle_table(ram_ra().polynomial_id()).unwrap();
        let val: Vec<Fr> = witness.oracle_table(ram_val().polynomial_id()).unwrap();
        let inc: Vec<Fr> = witness.oracle_table(ram_inc().polynomial_id()).unwrap();
        let mut claim = Fr::from_u64(0);
        for k in 0..ram_k {
            for j in 0..cycles {
                let index = (k << tau_low.len()) | j;
                claim += eq[j] * ra[index] * (val[index] + gamma * (val[index] + inc[j]));
            }
        }
        claim
    }

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>, handoff_rows: usize) {
        with_ram_fixture(shape, ops, |witness| {
            let tau_low = random_scalars(shape.log_t, 17);
            let gamma = random_scalars(1, 23)[0];
            let relation = RamReadWriteChecking::<Fr>::new(
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k()),
                shape.log_k(),
                tau_low.clone(),
            );
            let claims = RamReadWriteInputClaims {
                ram_read_value: Fr::from_u64(0),
                ram_write_value: Fr::from_u64(0),
            };
            let points = RamReadWriteInputClaims::<Vec<Fr>>::default();
            let challenges = RamReadWriteChallenges { gamma };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            let before = device_probe_count();
            let metal = PrepareKernel::<Fr, _>::prepare(
                &MetalRamReadWriteChecking::with_handoff_rows(handoff_rows),
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            assert_parity(
                optimized,
                metal,
                dense_input_claim(witness, &tau_low, gamma, shape.ram_k),
                &inputs(),
                71,
            );
            assert!(device_probe_count() > before, "device path did not engage");
        });
    }

    fn force_device() {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_RAM_READ_WRITE", "0");
        std::env::set_var("JOLT_RAMRW_GPU_PREPARE", "1");
        std::env::set_var("JOLT_RAMRW_FUSED", "1");
    }

    #[test]
    fn ram_rw_matches_optimized_mid_cycle_handoff() {
        let _lock = gpu_lock();
        force_device();
        run_parity(
            FixtureShape {
                log_t: 5,
                ram_k: 16,
            },
            vec![
                RamOp::Write { word: 3, post: 5 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 3, post: 9 },
                RamOp::Read { word: 7 },
                RamOp::None,
                RamOp::Write { word: 4, post: 2 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 7, post: 6 },
            ],
            4,
        );
    }

    #[test]
    fn ram_rw_matches_optimized_device_cycle_boundary() {
        let _lock = gpu_lock();
        force_device();
        run_parity(
            FixtureShape { log_t: 4, ram_k: 8 },
            vec![
                RamOp::None,
                RamOp::Write { word: 5, post: 11 },
                RamOp::None,
                RamOp::Read { word: 5 },
                RamOp::None,
                RamOp::Write { word: 5, post: 3 },
            ],
            1,
        );
    }

    #[cfg(feature = "bench-utils")]
    #[test]
    fn ram_rw_bench_oracles_and_dispatch_schedule() {
        let _lock = gpu_lock();
        super::bench::assert_small_scale_prepare_parity();
        super::bench::assert_small_scale_round_parity();
        let fixture = super::bench::RoundFixture::synthetic(super::bench::BenchConfig {
            log_t: 14,
            log_k: 8,
            seed: 29,
        });
        let legacy = fixture.run(false);
        let fused = fixture.run(true);
        assert_eq!((legacy.command_buffers, legacy.kernel_dispatches), (3, 3));
        assert_eq!((fused.command_buffers, fused.kernel_dispatches), (2, 3));
    }
}
