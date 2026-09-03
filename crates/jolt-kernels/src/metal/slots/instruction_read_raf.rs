//! Metal instruction read+RAF checking (stage 5): device phase scans behind
//! the optimized kernel's [`PhaseScanner`] seam.
//!
//! This slot deviates from the sibling `MetalX { fallback }` shape on purpose:
//! stage 5's device-worthy work is not the rounds (256-sized chunk tables)
//! but the three `O(T)` passes at each of the 16 phase boundaries —
//! condensation, the fused RAF scan, the per-table suffix scan. The
//! optimized kernel therefore stays the one and only kernel, and the device
//! substitutes just those passes through a scanner installed at prepare:
//! CPU fallback is the scanner being absent (below [`metal_gate`]), it
//! declining a phase (ineligible buffers), or dying mid-proof (dispatch
//! failure ⇒ [`ScanOutcome::Corrupt`] ⇒ the kernel rebuilds `u_evals` from
//! its intact inputs and finishes every remaining phase on the CPU).
//!
//! Per phase the scanner runs ONE command buffer with five dispatches (the
//! fused condense+RAF scan, its two-level reduce, the suffix scan and its
//! reduce) over zero-copy wraps of the shared rows / `u_evals` / flat buckets.
//! Bucket sums come back as exact field elements: cells fed by plain `u`
//! additions are Montgomery-form; cells fed by `u·scalar` products
//! accumulate in RAW space on the device and get one value-space `×R`
//! multiplication here, landing byte-identically on the CPU accumulator's
//! reduction (field sums are grouping-independent). The shared assembly
//! downstream never knows which tier scanned.
//!
//! At the address→cycle boundary the scanner ADOPTS the cycle tables
//! ([`PhaseScanner::adopt_cycle`]): materialization writes all `1 + ra_count`
//! tables into one flat device-owned ping-pong pair, and each cycle round is
//! one fused dispatch — fold the pending challenge out of place and
//! accumulate the product-grid lanes `[q(1), …, q(F−1), q(∞)]` weighted by
//! the host-bound Gruen eq levels — that the kernel assembles through its
//! own `gruen_poly_from_evals` recipe. Below the gate (tail rounds) or on a
//! failure the scanner steps aside and the kernel reclaims the live tables
//! (`take_cycle_tables`, a small copy at the shrunken length) and finishes
//! on the CPU, exactly as the other slots do.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::{Fr, Ring};
use jolt_lookup_tables::tables::suffixes::NUM_SUFFIXES;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::JoltWitnessPlane;

use super::{num_threadgroups, own_eq, own_uninit_frs, uninit_frs, DeviceRound, Partials};
use crate::metal::buffers::{DeviceBuffer, OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{ComputePass, DetachedPass, KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::instruction_read_raf::{
    shared_instruction_rows, CycleInitRequest, CycleTables, InstructionCycleRow, InstructionRows,
    OptimizedInstructionReadRafKernel, PhaseScanRequest, PhaseScanSums, PhaseScanner, RafSums,
    ScanOutcome, ScannerInputs,
};
use crate::reference::views::eq_table;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const KIND: &str = "instruction_read_raf";

/// Per-table suffix capacity baked into the shaders (`JK_IRR_SUF_CELLS`);
/// every lookup table today uses at most 8 suffixes.
const MAX_SUFFIXES: usize = 8;
/// Suffix ids the shader's `jk_suffix_mle` switch covers. A grown `Suffixes`
/// enum must be ported there before this bumps: an uncovered id evaluates to
/// 0 on the device, which is an invalid proof rather than a dispatch failure.
const _: () = assert!(NUM_SUFFIXES == 63);
const CHUNK_SIZE: usize = 256;
const RAF_CELLS: usize = 6 * CHUNK_SIZE;
const SUF_CELLS: usize = MAX_SUFFIXES * CHUNK_SIZE;
/// Simdgroup width the schedules assume (the kernels re-check at runtime).
const SIMD_WIDTH: usize = 32;
/// Flat cycle-table capacity baked into the round shader
/// (`JK_IRR_MAX_FACTORS`): `1 + ra_count` must fit.
const MAX_FACTORS: usize = 16;
/// Scan parallelism: the serial fr_mont_mul occupancy curve (8.3 Gmul/s at
/// 16k threads vs 11.6 saturated) plus the sorted-flush kernel's sweep put
/// the knee at 4096 simdgroups; the reduce tax this adds is paid by the
/// two-level RAF reduce below.
const TARGET_SIMDGROUPS: usize = 4096;
/// First-level fan-in of the RAF reduce: partials collapse to this many
/// rows (threads = chunks x cells), then one final row.
const RAF_REDUCE_CHUNKS: usize = 32;
/// Suffix gathers need more in-flight groups to hide indexed row/weight loads;
/// 2048 wins scan+reduce while 4096 overlaps it at twice the partial memory.
const TARGET_SUFFIX_SIMDGROUPS: usize = 2048;
const MIN_ROWS_PER_SIMDGROUP: usize = 1024;
/// Phase index (of 16) at which the cycle ping-pong is allocated and handed
/// to the driver for wiring: late enough that the added residency window
/// stays short, early enough that ~32 GiB wires under the remaining phase
/// scans.
const PREWIRE_PHASE: usize = 12;

// The device decodes rows as 12 u32 words (offsets pinned by the repr(C)
// asserts next to the struct) and 128-bit lookup indices.
const _: () = assert!(size_of::<InstructionCycleRow>() == 48);
const _: () = assert!(RISCV_XLEN == 64);

/// Slot front: the optimized kernel with the device phase scanner installed
/// when the gate and device cooperate — the optimized twin IS the fallback
/// (a `None` scanner), so table construction and every round are shared.
pub struct MetalInstructionReadRaf;

impl PrepareKernel<Fr, InstructionReadRaf<Fr>> for MetalInstructionReadRaf {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, InstructionReadRaf<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = InstructionReadRaf<Fr>>>, KernelError<Fr>>
    {
        let dimensions = inputs.relation.dimensions();
        // Reclaims the rows the trace record's walk co-produced (parked at
        // stage 1, carried back by every consumer); collects fresh only when
        // no record was built.
        let rows = shared_instruction_rows(session, witness, 1 << dimensions.log_t())?;
        Ok(Box::new(
            OptimizedInstructionReadRafKernel::new_with_scanner(
                dimensions,
                &inputs.points.lookup_output,
                rows,
                inputs.challenges.gamma,
                device_scanner,
            )?,
        ))
    }
}

/// The scanner factory handed to the optimized kernel: `None` (pure CPU)
/// below the gate, without a device, or when schedule/buffer construction
/// fails — a device problem must never fail a proof.
fn device_scanner(inputs: ScannerInputs<'_>) -> Option<Box<dyn PhaseScanner<Fr>>> {
    // Diagnostic: dump the real trace rows (24-byte records: lookup index,
    // table_index_plus_one, raf_flag, pad) for the `irr_roof` bench's
    // real-rows cells (`IRR_ROOF_ROWS`); free when unset.
    if let Some(path) = std::env::var_os("JOLT_IRR_DUMP_ROWS") {
        let mut bytes = Vec::with_capacity(inputs.rows.len() * 24);
        for row in inputs.rows.iter() {
            bytes.extend_from_slice(&row.lookup_index.to_le_bytes());
            bytes.push(row.table_index().map_or(0, |index| index as u8 + 1));
            bytes.push(u8::from(row.raf_flag));
            bytes.extend_from_slice(&[0u8; 6]);
        }
        if let Err(error) = std::fs::write(&path, bytes) {
            tracing::warn!(slot = KIND, %error, "row dump failed");
        }
    }
    if !metal_gate(KIND, inputs.rows.len()) {
        return None;
    }
    let context = match MetalContext::global() {
        Ok(context) => context,
        Err(error) => {
            tracing::warn!(slot = KIND, %error, "no device context; scanning on the CPU");
            return None;
        }
    };
    match DeviceIrrScanner::build(context, &inputs) {
        Ok(scanner) => Some(Box::new(scanner)),
        Err(error) => {
            tracing::warn!(slot = KIND, %error, "device scanner build failed; scanning on the CPU");
            None
        }
    }
}

/// One present table's host-side metadata for unpacking device sums.
struct SlotMeta {
    suffix_count: usize,
    /// Cell space per suffix: `true` = Montgomery adds (0/1-valued MLE),
    /// `false` = raw-space products needing the `×R` fix-up.
    is_01: [bool; MAX_SUFFIXES],
}

/// One two-phase cycle round in flight (committed, not yet waited).
struct IrrInFlight {
    pass: DetachedPass,
    num_tgs: usize,
    /// The fused kernel folded a challenge: advance the ping-pong on
    /// collect.
    bound: bool,
    /// The round's gruen levels, copied into flight-owned buffers (see
    /// [`own_eq`](super::own_eq)).
    _eq: (OwnedDeviceBuffer<Fr>, OwnedDeviceBuffer<Fr>),
}

/// A prepare-time phase-0 launch: one detached command buffer that fills
/// `u_evals = eq ⊗` and runs the phase-0 scan while the stage's remaining
/// prepares execute on the host. The eq half tables are flight-owned (the
/// CB reads them); `u_evals`' backing belongs to the optimized kernel,
/// untouched until collect per the seam contract.
struct Phase0Flight {
    pass: DetachedPass,
    _eq: (OwnedDeviceBuffer<Fr>, OwnedDeviceBuffer<Fr>),
}

/// The scanner's `Drop` settles `phase0` and `in_flight` before retiring
/// the buffers they reference (declaration order is second-line defense).
struct DeviceIrrScanner {
    phase0: Option<Phase0Flight>,
    in_flight: Option<IrrInFlight>,
    device: DeviceRound,
    rows: Arc<InstructionRows>,
    /// Address-phase scan state; dropped when the cycle tables are adopted
    /// (the phases are over) so its buffers free early.
    address: Option<AddressState>,
    /// Adopted cycle tables (the address→cycle handoff onward).
    cycle: Option<CycleRoundsState>,
    /// Cycle ping-pong allocated (and driver-wired) ahead of the handoff —
    /// see [`try_prewire`](Self::try_prewire).
    prewired: Option<PrewiredCycle>,
    /// `1 + ra_count` from prepare-time dimensions (what adoption will ask
    /// for), so the prewire can size the ping-pong before the handoff.
    factors_hint: usize,
    /// Completed `scan_phase` calls — the prewire trigger clock.
    scans_seen: usize,
    /// Value-space `R = 2^256 mod p`: multiplying a raw-space cell by this
    /// re-expresses it in Montgomery form (the exact element the CPU
    /// accumulator reduces to).
    r_mont: Fr,
}

/// The cycle ping-pong, allocated during the address phases and wired by a
/// detached one-thread command buffer: Metal wires (faults + pins) every
/// referenced no-copy buffer when a command buffer is SCHEDULED, and fresh
/// `MAP_ANON` pages wire at ~50 GB/s — ~0.45 s of the measured @2^27
/// `cycle_init`/round-0 blocked waits. Committing the wire CB while the
/// phase-scan CBs still run moves that driver work off the handoff's
/// critical path. Field order: `wire` precedes the buffers so a pending
/// wire settles (DetachedPass waits on drop) before its buffers free.
struct PrewiredCycle {
    wire: Option<DetachedPass>,
    cur: OwnedDeviceBuffer<Fr>,
    nxt: OwnedDeviceBuffer<Fr>,
}

/// A round left in flight must complete (DetachedPass waits on drop) BEFORE
/// the cycle ping-pong it reads/writes frees; declaration order (`in_flight`
/// precedes `cycle`) already guarantees it — this impl documents the
/// invariant explicitly.
impl Drop for DeviceIrrScanner {
    fn drop(&mut self) {
        drop(self.phase0.take());
        drop(self.in_flight.take());
    }
}

/// Phase-scan schedule and working buffers (static across the 16 phases).
struct AddressState {
    bucket_flat: Arc<Vec<u32>>,
    slots: Vec<SlotMeta>,
    // Static schedule (built once; buckets never change across phases).
    sg_slot: OwnedDeviceBuffer<u32>,
    sg_range: OwnedDeviceBuffer<u32>,
    suffix_meta: OwnedDeviceBuffer<u32>,
    // Two-level RAF reduce: 4096 partials rows fan into `raf_chunks` rows,
    // then one (a single 1536-thread pass over 4096 rows would out-cost the
    // scan win it protects).
    raf_group_l1: OwnedDeviceBuffer<u32>,
    raf_group_l2: OwnedDeviceBuffer<u32>,
    suffix_group: OwnedDeviceBuffer<u32>,
    // Working buffers, sized once and reused every phase.
    partials: OwnedDeviceBuffer<Fr>,
    raf_partials2: OwnedDeviceBuffer<Fr>,
    v_prev: OwnedDeviceBuffer<Fr>,
    raf_out: OwnedDeviceBuffer<Fr>,
    suffix_out: OwnedDeviceBuffer<Fr>,
    /// Row count the schedule was built for.
    n: usize,
    raf_chunks: usize,
    num_sgs_raf: usize,
    rows_per_sg_raf: usize,
    num_sgs_suffix: usize,
}

/// Adopted cycle tables: `combined_val` then the `ra` products concatenated
/// in ONE flat ping-pong pair (fixed kernel arity at any `ra_count`), compact
/// at stride `len` in `cur` and `len / 2` in `nxt`.
struct CycleRoundsState {
    cur: OwnedDeviceBuffer<Fr>,
    nxt: OwnedDeviceBuffer<Fr>,
    partials: Partials,
    factors: usize,
    /// Current per-table logical length (= `cur`'s stride).
    len: usize,
}

fn div_ceil_pos(n: usize, d: usize) -> usize {
    n.div_ceil(d).max(1)
}

impl DeviceIrrScanner {
    fn build(
        context: &'static MetalContext,
        inputs: &ScannerInputs<'_>,
    ) -> Result<Self, MetalError> {
        let rows = Arc::clone(inputs.rows);
        let bucket_flat = Arc::clone(inputs.bucket_flat);
        let n = rows.len();

        let rows_per_sg_raf = div_ceil_pos(n, TARGET_SIMDGROUPS).max(MIN_ROWS_PER_SIMDGROUP);
        let num_sgs_raf = div_ceil_pos(n, rows_per_sg_raf);

        // Suffix schedule: simdgroup count per table proportional to its
        // bucket, each simdgroup a contiguous index range of one table.
        let rows_per_sg_suffix =
            div_ceil_pos(bucket_flat.len(), TARGET_SUFFIX_SIMDGROUPS).max(MIN_ROWS_PER_SIMDGROUP);
        let mut sg_slot = Vec::new();
        let mut sg_range = Vec::new();
        let mut suffix_group = Vec::new();
        let mut suffix_meta = Vec::with_capacity(inputs.present.len() * (MAX_SUFFIXES + 1));
        let mut slots = Vec::with_capacity(inputs.present.len());
        for (slot, present) in inputs.present.iter().enumerate() {
            let suffixes = present.table.suffixes();
            if suffixes.len() > MAX_SUFFIXES {
                return Err(MetalError::UnsupportedShape(
                    "lookup table exceeds the shader's per-table suffix capacity",
                ));
            }
            let mut meta = SlotMeta {
                suffix_count: suffixes.len(),
                is_01: [false; MAX_SUFFIXES],
            };
            suffix_meta.push(suffixes.len() as u32);
            for s in 0..MAX_SUFFIXES {
                let packed = suffixes.get(s).map_or(0, |suffix| {
                    let is01 = suffix.is_01_valued();
                    meta.is_01[s] = is01;
                    u32::from(*suffix as u8) | (u32::from(is01) << 8)
                });
                suffix_meta.push(packed);
            }
            slots.push(meta);

            let sg_begin = sg_slot.len();
            let mut start = present.range.start;
            while start < present.range.end {
                let end = (start + rows_per_sg_suffix).min(present.range.end);
                sg_slot.push(slot as u32);
                sg_range.push(start as u32);
                sg_range.push(end as u32);
                start = end;
            }
            suffix_group.push(sg_begin as u32);
            suffix_group.push(sg_slot.len() as u32);
        }
        let num_sgs_suffix = sg_slot.len();

        let partial_cells = (num_sgs_raf * RAF_CELLS).max(num_sgs_suffix * SUF_CELLS);
        let own_u32s = |values: Vec<u32>| -> Result<OwnedDeviceBuffer<u32>, MetalError> {
            // Schedule arrays are tiny; the padded copy is one-time and
            // counted like every other copied wrap.
            let buffer = context.own_vec(if values.is_empty() { vec![0] } else { values })?;
            testing::note_copied_buffers(u64::from(buffer.was_copied()));
            Ok(buffer)
        };
        let raf_chunks = num_sgs_raf.min(RAF_REDUCE_CHUNKS);
        let sgs_per_chunk = div_ceil_pos(num_sgs_raf, raf_chunks);
        let raf_group_l1: Vec<u32> = (0..raf_chunks)
            .flat_map(|chunk| {
                let begin = chunk * sgs_per_chunk;
                [
                    begin as u32,
                    (begin + sgs_per_chunk).min(num_sgs_raf) as u32,
                ]
            })
            .collect();
        let zero = Fr::from_u64(0);
        Ok(Self {
            phase0: None,
            in_flight: None,
            device: DeviceRound::new(context, KIND),
            rows,
            address: Some(AddressState {
                bucket_flat,
                slots,
                n,
                sg_slot: own_u32s(sg_slot)?,
                sg_range: own_u32s(sg_range)?,
                suffix_meta: own_u32s(suffix_meta)?,
                raf_group_l1: own_u32s(raf_group_l1)?,
                raf_group_l2: own_u32s(vec![0, raf_chunks as u32])?,
                suffix_group: own_u32s(suffix_group)?,
                partials: context
                    .own_page_aligned(PageAlignedVec::from_elem(zero, partial_cells))?,
                raf_partials2: context
                    .own_page_aligned(PageAlignedVec::from_elem(zero, raf_chunks * RAF_CELLS))?,
                v_prev: context.own_page_aligned(PageAlignedVec::from_elem(zero, CHUNK_SIZE))?,
                raf_out: context.own_page_aligned(PageAlignedVec::from_elem(zero, RAF_CELLS))?,
                suffix_out: context.own_page_aligned(PageAlignedVec::from_elem(
                    zero,
                    (inputs.present.len() * SUF_CELLS).max(1),
                ))?,
                num_sgs_raf,
                rows_per_sg_raf,
                num_sgs_suffix,
                raf_chunks,
            }),
            cycle: None,
            prewired: None,
            factors_hint: 1 + inputs.ra_count,
            scans_seen: 0,
            r_mont: Fr::from_u64(1).mul_pow_2(128).mul_pow_2(128),
        })
    }
}

impl AddressState {
    /// Encode the phase's dispatches: the fused condense+RAF scan, its
    /// two-level reduce, then (when any table is present) the suffix scan
    /// and its reduce. `condense` is the previous phase's suffix width when
    /// `u_evals` is to be condensed in place.
    fn encode_scan<'b>(
        &'b self,
        pass: &mut ComputePass<'_, 'b>,
        rows: &DeviceBuffer<'b>,
        u_evals: &DeviceBuffer<'b>,
        bucket: &DeviceBuffer<'b>,
        suffix_len: usize,
        condense: Option<usize>,
    ) {
        let scan_params = [
            self.n as u32,
            self.rows_per_sg_raf as u32,
            self.num_sgs_raf as u32,
            suffix_len as u32,
            condense.unwrap_or(0) as u32,
            u32::from(condense.is_some()),
            u32::from(CANONICAL_INSTRUCTION_ADDRESS),
            suffix_len.saturating_sub(RISCV_XLEN) as u32,
        ];
        let raf_l1_cells = self.raf_chunks * RAF_CELLS;
        let raf_reduce_l1_params = [raf_l1_cells as u32, RAF_CELLS as u32, RAF_CELLS as u32];
        let raf_reduce_l2_params = [RAF_CELLS as u32, RAF_CELLS as u32, RAF_CELLS as u32];
        let suffix_params = [self.num_sgs_suffix as u32, suffix_len as u32];
        let suffix_cells = self.slots.len() * SUF_CELLS;
        let suffix_reduce_params = [suffix_cells as u32, SUF_CELLS as u32, SUF_CELLS as u32];

        let partials = self.partials.device_buffer();
        let raf_partials2 = self.raf_partials2.device_buffer();
        let v_prev = self.v_prev.device_buffer();
        let raf_out = self.raf_out.device_buffer();
        let suffix_out = self.suffix_out.device_buffer();
        let sg_slot = self.sg_slot.device_buffer();
        let sg_range = self.sg_range.device_buffer();
        let suffix_meta = self.suffix_meta.device_buffer();
        let raf_group_l1 = self.raf_group_l1.device_buffer();
        let raf_group_l2 = self.raf_group_l2.device_buffer();
        let suffix_group = self.suffix_group.device_buffer();

        pass.dispatch(
            KernelId::IrrPhaseScan,
            &scan_params,
            &[rows, u_evals, &v_prev, &partials],
            self.num_sgs_raf * SIMD_WIDTH,
        );
        pass.dispatch(
            KernelId::IrrReduce,
            &raf_reduce_l1_params,
            &[&partials, &raf_group_l1, &raf_partials2],
            raf_l1_cells,
        );
        pass.dispatch(
            KernelId::IrrReduce,
            &raf_reduce_l2_params,
            &[&raf_partials2, &raf_group_l2, &raf_out],
            RAF_CELLS,
        );
        if !self.slots.is_empty() {
            pass.dispatch(
                KernelId::IrrSuffixScan,
                &suffix_params,
                &[
                    rows,
                    u_evals,
                    bucket,
                    &sg_slot,
                    &sg_range,
                    &suffix_meta,
                    &partials,
                ],
                self.num_sgs_suffix * SIMD_WIDTH,
            );
            pass.dispatch(
                KernelId::IrrReduce,
                &suffix_reduce_params,
                &[&partials, &suffix_group, &suffix_out],
                suffix_cells,
            );
        }
    }

    /// Encode and run the phase's command buffer. `Ok(true)` means every
    /// dispatch ran; `Ok(false)` means a buffer wrap was ineligible and
    /// NOTHING ran (safe to decline).
    fn dispatch_phase(
        &mut self,
        context: &'static MetalContext,
        rows: &[InstructionCycleRow],
        request: &mut PhaseScanRequest<'_, Fr>,
    ) -> Result<bool, MetalError> {
        let condense = request.condense.map(|(v_prev, shift)| {
            debug_assert_eq!(v_prev.len(), CHUNK_SIZE);
            self.v_prev.as_mut_slice().copy_from_slice(v_prev);
            shift
        });

        let Some(rows_buffer) = context.wrap_slice_nocopy(rows) else {
            return Ok(false);
        };
        let Some(u_evals_buffer) = context.wrap_slice_mut_nocopy(request.u_evals) else {
            return Ok(false);
        };
        // Read-only: the copying fallback is correct, just counted.
        let bucket_buffer = context.wrap_slice(self.bucket_flat.as_slice())?;
        testing::note_copied_buffers(u64::from(bucket_buffer.was_copied()));

        let mut pass = context.begin_pass()?;
        self.encode_scan(
            &mut pass,
            &rows_buffer,
            &u_evals_buffer,
            &bucket_buffer,
            request.suffix_len,
            condense,
        );
        tracing::info_span!("IrrScanner::phase_run").in_scope(|| pass.run())?;
        testing::note_device_round();
        Ok(true)
    }

    /// Encode phase 0 — the eq outer-product fill plus the standard five
    /// scan/reduce dispatches — into ONE detached command buffer.
    /// `Ok(None)` = a wrap was ineligible and nothing ran (safe to fall
    /// back); `Err` = encode/commit failed with nothing committed.
    fn launch_phase0(
        &self,
        context: &'static MetalContext,
        rows: &[InstructionCycleRow],
        r_reduction: &[Fr],
        suffix_len: usize,
        u_evals: &mut [Fr],
    ) -> Result<Option<Phase0Flight>, MetalError> {
        let Some(rows_buffer) = context.wrap_slice_nocopy(rows) else {
            return Ok(None);
        };
        let Some(u_evals_buffer) = context.wrap_slice_mut_nocopy(u_evals) else {
            return Ok(None);
        };
        let bucket_buffer = context.wrap_slice(self.bucket_flat.as_slice())?;
        testing::note_copied_buffers(u64::from(bucket_buffer.was_copied()));

        // eq(r, ·) = eq(r_hi, ·) ⊗ eq(r_lo, ·), big-endian: the first
        // `log_t - lo_bits` coordinates pair the index's high bits. Exact by
        // distributivity, so the device fill is byte-identical to the host
        // `eq_table` it replaces.
        let log_t = r_reduction.len();
        let lo_bits = log_t / 2;
        let eq_hi = context.own_page_aligned(PageAlignedVec::from_slice(&eq_table(
            &r_reduction[..log_t - lo_bits],
        )))?;
        let eq_lo = context.own_page_aligned(PageAlignedVec::from_slice(&eq_table(
            &r_reduction[log_t - lo_bits..],
        )))?;
        let eq_params = [rows.len() as u32, lo_bits as u32];
        let eq_hi_buffer = eq_hi.device_buffer();
        let eq_lo_buffer = eq_lo.device_buffer();

        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::IrrEqOuter,
            &eq_params,
            &[&u_evals_buffer, &eq_hi_buffer, &eq_lo_buffer],
            rows.len(),
        );
        self.encode_scan(
            &mut pass,
            &rows_buffer,
            &u_evals_buffer,
            &bucket_buffer,
            suffix_len,
            None,
        );
        // SAFETY: rows/bucket are Arc'd on the scanner, the schedule and
        // working buffers are `self`-owned, the eq halves ride the returned
        // flight, and `u_evals`' backing is the optimized kernel's vec —
        // untouched until `collect_phase0` waits (the seam contract).
        let pass = unsafe { pass.commit().detach() };
        Ok(Some(Phase0Flight {
            pass,
            _eq: (eq_hi, eq_lo),
        }))
    }

    /// Unpack the device sums, applying the raw-space `×R` fix-up to every
    /// scalar-product cell (RAF left/right/identity; non-0/1 suffixes).
    fn collect_sums(&self, r_mont: Fr) -> PhaseScanSums<Fr> {
        let raf = self.raf_out.as_slice();
        let column = |q: usize, fix: bool| -> Vec<Fr> {
            raf[q * CHUNK_SIZE..(q + 1) * CHUNK_SIZE]
                .iter()
                .map(|&value| if fix { value * r_mont } else { value })
                .collect()
        };
        let suffix_cells = self.suffix_out.as_slice();
        let suffix = self
            .slots
            .iter()
            .enumerate()
            .map(|(slot, meta)| {
                let mut flat = Vec::with_capacity(meta.suffix_count * CHUNK_SIZE);
                for s in 0..meta.suffix_count {
                    let base = slot * SUF_CELLS + s * CHUNK_SIZE;
                    let fix = !meta.is_01[s];
                    flat.extend(suffix_cells[base..base + CHUNK_SIZE].iter().map(|&value| {
                        if fix {
                            value * r_mont
                        } else {
                            value
                        }
                    }));
                }
                flat
            })
            .collect();
        PhaseScanSums {
            raf: RafSums {
                shift_half: column(0, false),
                left: column(1, true),
                right: column(2, true),
                shift_full: column(3, false),
                identity: column(4, true),
                upper_all_ones: column(5, false),
            },
            suffix,
        }
    }
}

impl PhaseScanner<Fr> for DeviceIrrScanner {
    fn launch_phase0(&mut self, r_reduction: &[Fr], suffix_len: usize) -> Option<Vec<Fr>> {
        let context = self.device.gated(self.rows.len())?;
        let address = self.address.as_ref()?;
        if self.rows.len() != 1usize << r_reduction.len() {
            return None;
        }
        let mut u_evals = uninit_frs(self.rows.len());
        let launch = tracing::info_span!("IrrScanner::phase0_launch").in_scope(|| {
            address.launch_phase0(context, &self.rows, r_reduction, suffix_len, &mut u_evals)
        });
        match launch {
            Ok(Some(flight)) => {
                self.phase0 = Some(flight);
                self.scans_seen += 1;
                // The CB is committed — the device round is in flight.
                testing::note_device_round();
                Some(u_evals)
            }
            Ok(None) => None,
            Err(error) => {
                // Nothing committed: the CPU path is safe, but the device
                // just failed an encode — stand down for the whole proof.
                self.device.failed(&error);
                None
            }
        }
    }

    fn collect_phase0(&mut self) -> ScanOutcome<Fr> {
        let Some(flight) = self.phase0.take() else {
            return ScanOutcome::Corrupt;
        };
        match tracing::info_span!("IrrScanner::phase0_wait").in_scope(|| flight.pass.wait()) {
            Ok(()) => match self.address.as_ref() {
                Some(address) => ScanOutcome::Scanned(address.collect_sums(self.r_mont)),
                None => ScanOutcome::Corrupt,
            },
            Err(error) => {
                self.device.failed(&error);
                ScanOutcome::Corrupt
            }
        }
    }

    fn scan_phase(&mut self, mut request: PhaseScanRequest<'_, Fr>) -> ScanOutcome<Fr> {
        let Some(context) = self.device.gated(self.rows.len()) else {
            return ScanOutcome::Declined;
        };
        self.scans_seen += 1;
        if self.scans_seen == PREWIRE_PHASE {
            self.try_prewire(context);
        }
        let Some(address) = self.address.as_mut() else {
            return ScanOutcome::Declined;
        };
        match address.dispatch_phase(context, &self.rows, &mut request) {
            Ok(true) => ScanOutcome::Scanned(address.collect_sums(self.r_mont)),
            Ok(false) => ScanOutcome::Declined,
            Err(error) => {
                // The command buffer may have partially executed —
                // condensation writes u_evals in place, so the kernel must
                // rebuild it before falling back.
                self.device.failed(&error);
                ScanOutcome::Corrupt
            }
        }
    }

    fn materialize_cycle(&mut self, request: CycleInitRequest<'_, Fr>) -> Option<CycleTables<Fr>> {
        let context = self.device.gated(self.rows.len())?;
        match self.dispatch_cycle_init(context, &request) {
            Ok(tables) => tables,
            Err(error) => {
                // Materialization is pure: the partially written tables are
                // dropped and the CPU rebuilds from intact inputs.
                self.device.failed(&error);
                None
            }
        }
    }

    fn adopt_cycle(&mut self, request: &CycleInitRequest<'_, Fr>) -> bool {
        let Some(context) = self.device.gated(self.rows.len()) else {
            return false;
        };
        let factors = 1 + request.ra_count;
        if factors > MAX_FACTORS {
            return false;
        }
        match self.build_cycle_state(context, request, factors) {
            Ok(Some(cycle)) => {
                self.cycle = Some(cycle);
                // The phases are over — free their schedule and partials.
                self.address = None;
                true
            }
            // An ineligible wrap: nothing dispatched, CPU path unharmed.
            Ok(None) => false,
            Err(error) => {
                // Materialization is pure (writes only the dropped buffers).
                self.device.failed(&error);
                false
            }
        }
    }

    fn cycle_round(&mut self, bind: Option<Fr>, e_in: &[Fr], e_out: &[Fr]) -> Option<Vec<Fr>> {
        let len = self.cycle.as_ref()?.len;
        let context = self.device.gated(len)?;
        // Post-bind pair count; the eq levels must tile it exactly.
        let groups = if bind.is_some() { len / 4 } else { len / 2 };
        if groups == 0 || e_in.len() * e_out.len() != groups || !e_in.len().is_power_of_two() {
            return None;
        }
        match self.dispatch_cycle_round(context, bind, e_in, e_out, groups) {
            Ok(lanes) => {
                if bind.is_some() {
                    let cycle = self.cycle.as_mut()?;
                    std::mem::swap(&mut cycle.cur, &mut cycle.nxt);
                    cycle.len /= 2;
                }
                Some(lanes)
            }
            Err(error) => {
                // The fused kernel writes only `nxt` and the partials —
                // `cur` (the pre-bind tables) is intact for the handoff.
                self.device.failed(&error);
                None
            }
        }
    }

    fn take_cycle_tables(&mut self) -> Option<CycleTables<Fr>> {
        let _span = tracing::info_span!("IrrScanner::take_cycle_tables").entered();
        let cycle = self.cycle.take()?;
        let len = cycle.len;
        let flat = cycle.cur.as_slice();
        let table = |f: usize| flat[f * len..(f + 1) * len].to_vec();
        let tables = CycleTables {
            combined_val: table(0),
            ra: (1..cycle.factors).map(table).collect(),
        };
        // The live values are host-owned now; the pair is dead weight and
        // frees here (malloc's large cache recycles the pages into the
        // next stage's allocations).
        drop((cycle.cur, cycle.nxt));
        Some(tables)
    }

    fn launch_cycle_round(&mut self, bind: Option<Fr>, e_in: &[Fr], e_out: &[Fr]) -> bool {
        // The same admission as the synchronous `cycle_round`.
        let Some(len) = self.cycle.as_ref().map(|cycle| cycle.len) else {
            return false;
        };
        let Some(context) = self.device.gated(len) else {
            return false;
        };
        let groups = if bind.is_some() { len / 4 } else { len / 2 };
        if groups == 0 || e_in.len() * e_out.len() != groups || !e_in.len().is_power_of_two() {
            return false;
        }
        let launch = |scanner: &Self| -> Result<IrrInFlight, MetalError> {
            let eq = own_eq(context, e_in, e_out)?;
            let pass = scanner.commit_cycle_round(
                context,
                bind,
                e_in.len().trailing_zeros(),
                (&eq.0.device_buffer(), &eq.1.device_buffer()),
                groups,
            )?;
            Ok(IrrInFlight {
                pass,
                num_tgs: num_threadgroups(groups),
                bound: bind.is_some(),
                _eq: eq,
            })
        };
        match tracing::info_span!("IrrScanner::cycle_launch").in_scope(|| launch(self)) {
            Ok(flight) => {
                self.in_flight = Some(flight);
                true
            }
            Err(error) => {
                // Nothing committed — `cur` intact for the host handoff.
                self.device.failed(&error);
                false
            }
        }
    }

    fn collect_cycle_round(&mut self) -> Option<Vec<Fr>> {
        let flight = self.in_flight.take()?;
        match tracing::info_span!("IrrScanner::cycle_wait").in_scope(|| flight.pass.wait()) {
            Ok(()) => {
                testing::note_device_round();
                let cycle = self.cycle.as_mut()?;
                let lanes = cycle.partials.sums(flight.num_tgs);
                if flight.bound {
                    std::mem::swap(&mut cycle.cur, &mut cycle.nxt);
                    cycle.len /= 2;
                }
                Some(lanes)
            }
            Err(error) => {
                // The fused kernel writes only `nxt` and the partials —
                // `cur` (the pre-bind tables) is intact for the handoff.
                self.device.failed(&error);
                None
            }
        }
    }
}

impl DeviceIrrScanner {
    /// Fill `combined_val` and each `ra` product into host-owned buffers
    /// through mutable zero-copy wraps: one command buffer, one dispatch per
    /// output table (the fixed shader signature takes one output). The
    /// buffers are never CPU-touched before the device writes them, so the
    /// virtual allocations fault in under GPU writes — no host page-zeroing.
    fn dispatch_cycle_init(
        &self,
        context: &'static MetalContext,
        request: &CycleInitRequest<'_, Fr>,
    ) -> Result<Option<CycleTables<Fr>>, MetalError> {
        let n = self.rows.len();
        let Some(rows_buffer) = context.wrap_slice_nocopy(self.rows.as_slice()) else {
            return Ok(None);
        };
        let mut v_flat: Vec<Fr> = Vec::with_capacity(request.v_tables.len() * CHUNK_SIZE);
        for table in request.v_tables {
            debug_assert_eq!(table.len(), CHUNK_SIZE);
            v_flat.extend_from_slice(table);
        }
        let v_tables_buffer = context.wrap_slice(&v_flat)?;
        let table_values_buffer = context.wrap_slice(request.table_values)?;
        testing::note_copied_buffers(
            u64::from(v_tables_buffer.was_copied()) + u64::from(table_values_buffer.was_copied()),
        );

        let mut combined_val = uninit_frs(n);
        let mut ra: Vec<Vec<Fr>> = (0..request.ra_count).map(|_| uninit_frs(n)).collect();
        let mut outputs = Vec::with_capacity(1 + request.ra_count);
        let Some(combined_buffer) = context.wrap_slice_mut_nocopy(combined_val.as_mut_slice())
        else {
            return Ok(None);
        };
        outputs.push(combined_buffer);
        for table in &mut ra {
            let Some(buffer) = context.wrap_slice_mut_nocopy(table.as_mut_slice()) else {
                return Ok(None);
            };
            outputs.push(buffer);
        }

        let mut pass = context.begin_pass()?;
        for (position, output) in outputs.iter().enumerate() {
            pass.dispatch(
                KernelId::IrrCycleInit,
                &cycle_init_params(n, position, request),
                &[&rows_buffer, &v_tables_buffer, &table_values_buffer, output],
                n,
            );
        }
        tracing::info_span!("IrrScanner::cycle_init_run").in_scope(|| pass.run())?;
        testing::note_device_round();
        Ok(Some(CycleTables { combined_val, ra }))
    }

    /// Allocate the adoption-shaped ping-pong ahead of the handoff and hand
    /// it to the driver for wiring while the remaining phase scans execute:
    /// one detached single-thread `FrBind` referencing both buffers —
    /// scheduling a command buffer wires every referenced no-copy buffer,
    /// and fresh `MAP_ANON` pages wire at ~50 GB/s, which is the measured
    /// blocked-wait overhang on the init and round-0 CBs. The probe's one
    /// garbage write (`nxt[0]`, from `cur`'s kernel-zeroed pages) lands in
    /// memory the init/fold dispatches fully overwrite before any read.
    /// Best-effort: any failure leaves the allocate-at-adoption path.
    fn try_prewire(&mut self, context: &'static MetalContext) {
        if self.prewired.is_some() || self.factors_hint > MAX_FACTORS {
            return;
        }
        let factors = self.factors_hint;
        let n = self.rows.len();
        let prewire = move || -> Result<Option<PrewiredCycle>, MetalError> {
            let Some(cur) = own_uninit_frs(context, factors * n)? else {
                return Ok(None);
            };
            let Some(nxt) = own_uninit_frs(context, factors * (n / 2))? else {
                return Ok(None);
            };
            let mut params = vec![1u32];
            params.extend_from_slice(&fr_to_u32_limbs(Fr::from_u64(0)));
            let cur_buffer = cur.device_buffer();
            let nxt_buffer = nxt.device_buffer();
            // Side queue: wiring ~32 GiB stalls the committing queue's
            // in-order schedule pipeline — on the shared queue it would
            // steal the same time back from the remaining phase-scan CBs.
            let mut pass = context.begin_pass_side()?;
            pass.dispatch(KernelId::FrBind, &params, &[&cur_buffer, &nxt_buffer], 1);
            drop((cur_buffer, nxt_buffer));
            // SAFETY: the pair is owned by the returned struct, whose field
            // order settles a pending wire (DetachedPass waits on drop)
            // before the buffers free; adoption waits the flight before
            // dispatching into the pair.
            let wire = unsafe { pass.commit().detach() };
            Ok(Some(PrewiredCycle {
                wire: Some(wire),
                cur,
                nxt,
            }))
        };
        match prewire() {
            Ok(prewired) => self.prewired = prewired,
            Err(error) => {
                tracing::warn!(
                    slot = KIND,
                    %error,
                    "cycle prewire dispatch failed; adoption will allocate"
                );
            }
        }
    }

    /// The adopting twin of [`dispatch_cycle_init`](Self::dispatch_cycle_init):
    /// materialize all `factors` tables into ONE flat scanner-owned buffer
    /// (stride `n`) — a single fused dispatch reading each row once — and
    /// set up the round ping-pong. `Ok(None)` = ineligible buffers, nothing
    /// dispatched.
    fn build_cycle_state(
        &mut self,
        context: &'static MetalContext,
        request: &CycleInitRequest<'_, Fr>,
        factors: usize,
    ) -> Result<Option<CycleRoundsState>, MetalError> {
        let n = self.rows.len();
        let Some(rows_buffer) = context.wrap_slice_nocopy(self.rows.as_slice()) else {
            return Ok(None);
        };
        let mut v_flat: Vec<Fr> = Vec::with_capacity(request.v_tables.len() * CHUNK_SIZE);
        for table in request.v_tables {
            debug_assert_eq!(table.len(), CHUNK_SIZE);
            v_flat.extend_from_slice(table);
        }
        let v_tables_buffer = context.wrap_slice(&v_flat)?;
        let table_values_buffer = context.wrap_slice(request.table_values)?;
        testing::note_copied_buffers(
            u64::from(v_tables_buffer.was_copied()) + u64::from(table_values_buffer.was_copied()),
        );

        // Uninitialized fills: the device writes every element of `cur`
        // below, and `nxt` is read (as the swapped-in `cur`) only up to the
        // compact region a fold fully wrote. A copying wrap would read the
        // uninitialized memory, so eligibility is pre-checked and a miss
        // declines adoption before anything runs. The prewired pair (same
        // shape, already wired) is preferred; a wire failure only costs the
        // prewire — the fresh-allocation path is unaffected.
        let mut pair = None;
        if let Some(mut prewired) = self.prewired.take() {
            if prewired.cur.as_slice().len() == factors * n
                && prewired.nxt.as_slice().len() == factors * (n / 2)
            {
                match prewired.wire.take().map(DetachedPass::wait).transpose() {
                    Ok(_) => pair = Some((prewired.cur, prewired.nxt)),
                    Err(error) => {
                        tracing::warn!(slot = KIND, %error, "cycle prewire failed; reallocating");
                    }
                }
            }
        }
        let (cur, nxt) = if let Some(pair) = pair {
            pair
        } else {
            let Some(cur) = own_uninit_frs(context, factors * n)? else {
                return Ok(None);
            };
            let Some(nxt) = own_uninit_frs(context, factors * (n / 2))? else {
                return Ok(None);
            };
            (cur, nxt)
        };

        let cur_buffer = cur.device_buffer();
        // IrrCycleInitFusedParams: [n, ra_count, phases_per_ra,
        // address_bits, raf_interleaved, raf_identity].
        let mut params = vec![
            n as u32,
            request.ra_count as u32,
            request.phases_per_ra as u32,
            request.address_bits as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(request.raf_interleaved));
        params.extend_from_slice(&fr_to_u32_limbs(request.raf_identity));
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::IrrCycleInitFused,
            &params,
            &[
                &rows_buffer,
                &v_tables_buffer,
                &table_values_buffer,
                &cur_buffer,
            ],
            n,
        );
        tracing::info_span!("IrrScanner::cycle_init_run").in_scope(|| pass.run())?;
        testing::note_device_round();
        drop(cur_buffer);

        let partials = Partials::new(context, factors, n / 2)?;
        Ok(Some(CycleRoundsState {
            cur,
            nxt,
            partials,
            factors,
            len: n,
        }))
    }

    /// Encode + commit one fused cycle round (fold-when-binding +
    /// product-grid lanes, one dispatch) without blocking. The caller
    /// supplies the eq buffers — wait-in-scope borrows on the synchronous
    /// path, flight-owned copies on the launch path — and decides whether
    /// to wait in place or park the flight.
    fn commit_cycle_round(
        &self,
        context: &'static MetalContext,
        bind: Option<Fr>,
        e_in_log2: u32,
        eq: (&DeviceBuffer<'_>, &DeviceBuffer<'_>),
        groups: usize,
    ) -> Result<DetachedPass, MetalError> {
        let cycle = self.cycle.as_ref().ok_or(MetalError::UnsupportedShape(
            "cycle round dispatched before adoption",
        ))?;
        let num_tgs = num_threadgroups(groups);
        // IrrCycleRoundParams: [groups, do_bind, num_tgs, log_in,
        // num_tables, len, r].
        let mut params = vec![
            groups as u32,
            u32::from(bind.is_some()),
            num_tgs as u32,
            e_in_log2,
            cycle.factors as u32,
            cycle.len as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let cur = cycle.cur.device_buffer();
        let nxt = cycle.nxt.device_buffer();
        let partials = cycle.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::IrrCycleRound,
            &params,
            &[&cur, &nxt, eq.0, eq.1, &partials],
            groups,
        );
        // SAFETY: the ping-pong and partials are scanner-owned and next
        // touched after the wait (the scanner's Drop settles an in-flight
        // pass before retiring them); the eq buffers are the caller's per
        // the signature contract; copied uploads are Metal-owned (retained
        // by the command buffer).
        Ok(unsafe { pass.commit().detach() })
    }

    /// One fused cycle round: commit + wait + read. The eq levels are the
    /// CURRENT (post-`gruen.bind`) ones — the kernel binds gruen host-side
    /// first.
    fn dispatch_cycle_round(
        &self,
        context: &'static MetalContext,
        bind: Option<Fr>,
        e_in: &[Fr],
        e_out: &[Fr],
        groups: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let e_in_buffer = context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = context.wrap_slice(fr_as_u32s(e_out))?;
        testing::note_copied_buffers(
            u64::from(e_in_buffer.was_copied()) + u64::from(e_out_buffer.was_copied()),
        );
        self.commit_cycle_round(
            context,
            bind,
            e_in.len().trailing_zeros(),
            (&e_in_buffer, &e_out_buffer),
            groups,
        )?
        .wait()?;
        testing::note_device_round();
        let num_tgs = num_threadgroups(groups);
        let cycle = self.cycle.as_ref().ok_or(MetalError::UnsupportedShape(
            "cycle round dispatched before adoption",
        ))?;
        Ok(cycle.partials.sums(num_tgs))
    }
}

/// `IrrCycleInitParams` for output table `position` (0 = combined_val,
/// `1 + i` = `ra_i`).
fn cycle_init_params(n: usize, position: usize, request: &CycleInitRequest<'_, Fr>) -> Vec<u32> {
    let (phase_begin, phase_count) = match position.checked_sub(1) {
        None => (0, 0),
        Some(i) => (i * request.phases_per_ra, request.phases_per_ra),
    };
    let mut params = vec![
        n as u32,
        phase_begin as u32,
        phase_count as u32,
        request.address_bits as u32,
    ];
    params.extend_from_slice(&fr_to_u32_limbs(request.raf_interleaved));
    params.extend_from_slice(&fr_to_u32_limbs(request.raf_identity));
    params
}

#[cfg(feature = "bench-utils")]
pub mod bench {
    use std::time::Duration;

    use jolt_field::{Fr, Ring};
    use jolt_lookup_tables::LookupTableKind;

    use super::*;
    use crate::metal::runtime::VariantPipeline;

    pub struct IrrPhaseScanFixture {
        rows: PageAlignedVec<InstructionCycleRow>,
        u_evals: PageAlignedVec<Fr>,
        v_prev: PageAlignedVec<Fr>,
        partials: PageAlignedVec<Fr>,
        out: PageAlignedVec<Fr>,
        group: PageAlignedVec<u32>,
        params: Vec<u32>,
        reduce_params: Vec<u32>,
        threads: usize,
    }

    pub struct IrrPhaseScanBuffers<'a> {
        context: &'static MetalContext,
        rows: DeviceBuffer<'a>,
        u_evals: DeviceBuffer<'a>,
        v_prev: DeviceBuffer<'a>,
        partials: DeviceBuffer<'a>,
        out: DeviceBuffer<'a>,
        group: DeviceBuffer<'a>,
        params: &'a [u32],
        reduce_params: &'a [u32],
        threads: usize,
    }

    pub struct IrrSuffixScanFixture {
        rows: PageAlignedVec<InstructionCycleRow>,
        u_evals: PageAlignedVec<Fr>,
        bucket_flat: PageAlignedVec<u32>,
        sg_slot: PageAlignedVec<u32>,
        sg_range: PageAlignedVec<u32>,
        suffix_meta: PageAlignedVec<u32>,
        partials: PageAlignedVec<Fr>,
        out: PageAlignedVec<Fr>,
        group: PageAlignedVec<u32>,
        params: Vec<u32>,
        reduce_params: Vec<u32>,
        threads: usize,
    }

    pub struct IrrSuffixScanBuffers<'a> {
        context: &'static MetalContext,
        rows: DeviceBuffer<'a>,
        u_evals: DeviceBuffer<'a>,
        bucket_flat: DeviceBuffer<'a>,
        sg_slot: DeviceBuffer<'a>,
        sg_range: DeviceBuffer<'a>,
        suffix_meta: DeviceBuffer<'a>,
        partials: DeviceBuffer<'a>,
        group: DeviceBuffer<'a>,
        out: DeviceBuffer<'a>,
        params: &'a [u32],
        reduce_params: &'a [u32],
        threads: usize,
        out_len: usize,
    }

    impl IrrPhaseScanFixture {
        pub fn production_geometry(log_t: usize) -> Result<Self, MetalError> {
            let n = 1usize << log_t;
            let mut state = 0x4952_5250_4841_5345u64 ^ log_t as u64;
            let rows = PageAlignedVec::from_fn(n, |j| {
                let lo = splitmix(&mut state);
                let hi = splitmix(&mut state);
                InstructionCycleRow::new(
                    (u128::from(hi) << 64) | u128::from(lo),
                    Some(j % 5),
                    j % 3 == 0,
                    None,
                    None,
                )
            });
            Ok(Self::with_rows(rows, state))
        }

        /// Real-trace rows: 24-byte records — lookup index (16 LE bytes),
        /// `table_index_plus_one`, `raf_flag`, 6 pad — as written by the
        /// `JOLT_IRR_DUMP_ROWS` probe. Truncates to the largest power of two.
        #[expect(clippy::expect_used, reason = "bench fixture loader")]
        pub fn from_rows_file(path: &std::path::Path) -> Result<Self, MetalError> {
            let bytes = std::fs::read(path)
                .map_err(|_| MetalError::UnsupportedShape("rows file unreadable"))?;
            let records = bytes.len() / 24;
            if records < 2 {
                return Err(MetalError::UnsupportedShape("rows file too small"));
            }
            let n = 1usize << records.ilog2();
            let rows = PageAlignedVec::from_fn(n, |j| {
                let record = &bytes[j * 24..(j + 1) * 24];
                let lookup_index = u128::from_le_bytes(record[..16].try_into().expect("16B"));
                let table_index = (record[16] as usize).checked_sub(1);
                InstructionCycleRow::new(lookup_index, table_index, record[17] != 0, None, None)
            });
            Ok(Self::with_rows(rows, 0x4952_5245_414C_5253))
        }

        fn with_rows(rows: PageAlignedVec<InstructionCycleRow>, seed: u64) -> Self {
            let n = rows.len();
            let rows_per_sg = div_ceil_pos(n, TARGET_SIMDGROUPS).max(MIN_ROWS_PER_SIMDGROUP);
            let num_sgs = div_ceil_pos(n, rows_per_sg);
            let mut state = seed;
            let u_evals = PageAlignedVec::from_fn(n, |_| Fr::from_u64(splitmix(&mut state)));
            let v_prev =
                PageAlignedVec::from_fn(CHUNK_SIZE, |_| Fr::from_u64(splitmix(&mut state)));
            let zero = Fr::from_u64(0);
            Self {
                rows,
                u_evals,
                v_prev,
                partials: PageAlignedVec::from_elem(zero, num_sgs * RAF_CELLS),
                out: PageAlignedVec::from_elem(zero, RAF_CELLS),
                group: PageAlignedVec::from_slice(&[0, num_sgs as u32]),
                params: vec![
                    n as u32,
                    rows_per_sg as u32,
                    num_sgs as u32,
                    64,
                    0,
                    0,
                    u32::from(CANONICAL_INSTRUCTION_ADDRESS),
                    0,
                ],
                reduce_params: vec![RAF_CELLS as u32, RAF_CELLS as u32, RAF_CELLS as u32],
                threads: num_sgs * SIMD_WIDTH,
            }
        }

        pub fn buffers(&mut self) -> Result<IrrPhaseScanBuffers<'_>, MetalError> {
            let context = MetalContext::global()?;
            Ok(IrrPhaseScanBuffers {
                context,
                rows: self.rows.device_buffer(context)?,
                u_evals: self.u_evals.device_buffer(context)?,
                v_prev: self.v_prev.device_buffer(context)?,
                partials: self.partials.device_buffer_mut(context)?,
                out: self.out.device_buffer_mut(context)?,
                group: self.group.device_buffer(context)?,
                params: &self.params,
                reduce_params: &self.reduce_params,
                threads: self.threads,
            })
        }

        /// Production phase-`p` dispatch shape: `suffix_len = 128 − 8(p+1)`,
        /// condensation from phase 1 on (`prev_shift = suffix_len + 8`), and
        /// the wide-suffix params exactly as `AddressState::dispatch_phase`
        /// derives them.
        pub fn set_phase_shape(&mut self, suffix_len: usize, condense: bool) {
            self.params[3] = suffix_len as u32;
            self.params[4] = if condense { (suffix_len + 8) as u32 } else { 0 };
            self.params[5] = u32::from(condense);
            self.params[7] = suffix_len.saturating_sub(RISCV_XLEN) as u32;
        }

        /// Rebuild `u_evals` (condensing dispatches scale it in place; reset
        /// between cells so magnitudes stay comparable).
        pub fn reset_u_evals(&mut self) {
            let mut state = 0x5545_5641_4C52_5354u64;
            for u in self.u_evals.iter_mut() {
                *u = Fr::from_u64(splitmix(&mut state));
            }
        }

        /// Re-derive the scan schedule for `target_simdgroups` (attribution
        /// knob; production is `TARGET_SIMDGROUPS`). Field sums regroup
        /// exactly, so any simdgroup count is value-identical.
        pub fn set_simdgroups(&mut self, target_simdgroups: usize) {
            let n = self.rows.len();
            let rows_per_sg = div_ceil_pos(n, target_simdgroups).max(MIN_ROWS_PER_SIMDGROUP);
            let num_sgs = div_ceil_pos(n, rows_per_sg);
            self.params[1] = rows_per_sg as u32;
            self.params[2] = num_sgs as u32;
            self.partials = PageAlignedVec::from_elem(Fr::from_u64(0), num_sgs * RAF_CELLS);
            self.group = PageAlignedVec::from_slice(&[0, num_sgs as u32]);
            self.threads = num_sgs * SIMD_WIDTH;
        }

        /// Compile a probe kernel against the full production library (same
        /// buffer ABI as `jk_irr_phase_scan`), for [`IrrPhaseScanBuffers::
        /// run_timed_probe`].
        pub fn compile_probe(
            &self,
            extra_source: &str,
            entry: &str,
        ) -> Result<VariantPipeline, MetalError> {
            MetalContext::global()?.compile_variant(extra_source, entry)
        }
    }

    impl IrrPhaseScanBuffers<'_> {
        /// One production-shaped command buffer — scan + its reduce —
        /// returning the CB's GPU execution window.
        pub fn run_timed(&self) -> Result<Duration, MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch(
                KernelId::IrrPhaseScan,
                self.params,
                &self.scan_buffers(),
                self.threads,
            );
            pass.dispatch(
                KernelId::IrrReduce,
                self.reduce_params,
                &[&self.partials, &self.group, &self.out],
                RAF_CELLS,
            );
            pass.commit().wait_timed()
        }

        /// Scan dispatch alone (no reduce) — separates the reduce's share of
        /// the production CB window.
        pub fn run_timed_scan_only(&self) -> Result<Duration, MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch(
                KernelId::IrrPhaseScan,
                self.params,
                &self.scan_buffers(),
                self.threads,
            );
            pass.commit().wait_timed()
        }

        /// Production scan (no reduce) at an explicit threadgroup width.
        pub fn run_timed_width(&self, width: usize) -> Result<Duration, MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch_width(
                KernelId::IrrPhaseScan,
                self.params,
                &self.scan_buffers(),
                self.threads,
                width,
            );
            pass.commit().wait_timed()
        }

        /// One probe dispatch (no reduce) with the production buffer ABI.
        pub fn run_timed_probe(
            &self,
            probe: &VariantPipeline,
            width: usize,
        ) -> Result<Duration, MetalError> {
            self.run_timed_probe_threads(probe, self.params, self.threads, width)
        }

        /// One probe dispatch at an explicit thread count (occupancy-curve
        /// probes that ignore the fixture schedule).
        pub fn run_timed_probe_threads(
            &self,
            probe: &VariantPipeline,
            params: &[u32],
            threads: usize,
            width: usize,
        ) -> Result<Duration, MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch_variant(probe, params, &self.scan_buffers(), threads, width);
            pass.commit().wait_timed()
        }

        /// The scan's argument table (`jk_irr_phase_scan` buffers 0–3).
        fn scan_buffers(&self) -> [&DeviceBuffer<'_>; 4] {
            [&self.rows, &self.u_evals, &self.v_prev, &self.partials]
        }
    }

    impl IrrSuffixScanFixture {
        pub fn production_geometry(log_t: usize) -> Result<Self, MetalError> {
            let n = 1usize << log_t;
            let all_tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
            let table_indices = [0, 3, 7, 11, LookupTableKind::<RISCV_XLEN>::COUNT - 1];
            let tables: Vec<_> = table_indices
                .into_iter()
                .map(|index| all_tables[index])
                .collect();
            let mut buckets = vec![Vec::new(); tables.len()];
            let mut state = 0x4952_5253_5546_4649u64 ^ log_t as u64;
            let rows = PageAlignedVec::from_fn(n, |j| {
                let lo = splitmix(&mut state);
                let hi = splitmix(&mut state);
                let slot = j % tables.len();
                let table_index = (j % 7 != 3).then_some(tables[slot].index());
                if table_index.is_some() {
                    buckets[slot].push(j as u32);
                }
                InstructionCycleRow::new(
                    (u128::from(hi) << 64) | u128::from(lo),
                    table_index,
                    j % 3 == 0,
                    None,
                    None,
                )
            });
            let u_evals = PageAlignedVec::from_fn(n, |_| Fr::from_u64(splitmix(&mut state)));
            Ok(Self::from_parts(rows, u_evals, &tables, buckets))
        }

        /// Real-trace rows (the `JOLT_IRR_DUMP_ROWS` format): production
        /// buckets grouped by the dumped table indices, `suffix_len = 64`
        /// (the fixture's fixed key position — phase-7 shape).
        #[expect(clippy::expect_used, reason = "bench fixture loader")]
        pub fn from_rows_file(path: &std::path::Path) -> Result<Self, MetalError> {
            let bytes = std::fs::read(path)
                .map_err(|_| MetalError::UnsupportedShape("rows file unreadable"))?;
            let records = bytes.len() / 24;
            if records < 2 {
                return Err(MetalError::UnsupportedShape("rows file too small"));
            }
            let n = 1usize << records.ilog2();
            let all_tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
            let mut buckets = vec![Vec::new(); all_tables.len()];
            let rows = PageAlignedVec::from_fn(n, |j| {
                let record = &bytes[j * 24..(j + 1) * 24];
                let lookup_index = u128::from_le_bytes(record[..16].try_into().expect("16B"));
                let table_index = (record[16] as usize).checked_sub(1);
                if let Some(index) = table_index {
                    buckets[index].push(j as u32);
                }
                InstructionCycleRow::new(lookup_index, table_index, record[17] != 0, None, None)
            });
            let mut tables = Vec::new();
            let mut present_buckets = Vec::new();
            for (index, bucket) in buckets.into_iter().enumerate() {
                if !bucket.is_empty() {
                    tables.push(all_tables[index]);
                    present_buckets.push(bucket);
                }
            }
            let mut state = 0x5355_4652_4541_4C53u64;
            let u_evals = PageAlignedVec::from_fn(n, |_| Fr::from_u64(splitmix(&mut state)));
            Ok(Self::from_parts(rows, u_evals, &tables, present_buckets))
        }

        /// The production suffix schedule over `buckets` (one per table, in
        /// `tables` order): `TARGET_SUFFIX_SIMDGROUPS` simdgroups shared in
        /// proportion to bucket size, each a contiguous range of one table.
        fn from_parts(
            rows: PageAlignedVec<InstructionCycleRow>,
            u_evals: PageAlignedVec<Fr>,
            tables: &[LookupTableKind<RISCV_XLEN>],
            buckets: Vec<Vec<u32>>,
        ) -> Self {
            let bucket_len: usize = buckets.iter().map(Vec::len).sum();
            let rows_per_sg =
                div_ceil_pos(bucket_len, TARGET_SUFFIX_SIMDGROUPS).max(MIN_ROWS_PER_SIMDGROUP);
            let mut bucket_flat = Vec::with_capacity(bucket_len);
            let mut sg_slot = Vec::new();
            let mut sg_range = Vec::new();
            let mut group = Vec::with_capacity(2 * tables.len());
            for (slot, bucket) in buckets.into_iter().enumerate() {
                let range_start = bucket_flat.len();
                bucket_flat.extend_from_slice(&bucket);
                let range_end = bucket_flat.len();
                group.push(sg_slot.len() as u32);
                let mut start = range_start;
                while start < range_end {
                    let end = (start + rows_per_sg).min(range_end);
                    sg_slot.push(slot as u32);
                    sg_range.extend_from_slice(&[start as u32, end as u32]);
                    start = end;
                }
                group.push(sg_slot.len() as u32);
            }
            let mut suffix_meta = Vec::with_capacity(tables.len() * (MAX_SUFFIXES + 1));
            for table in tables {
                let suffixes = table.suffixes();
                suffix_meta.push(suffixes.len() as u32);
                for index in 0..MAX_SUFFIXES {
                    suffix_meta.push(suffixes.get(index).map_or(0, |suffix| {
                        u32::from(*suffix as u8) | (u32::from(suffix.is_01_valued()) << 8)
                    }));
                }
            }
            let num_sgs = sg_slot.len();
            let total_cells = tables.len() * SUF_CELLS;
            let zero = Fr::from_u64(0);
            Self {
                rows,
                u_evals,
                bucket_flat: PageAlignedVec::from_slice(&bucket_flat),
                sg_slot: PageAlignedVec::from_slice(&sg_slot),
                sg_range: PageAlignedVec::from_slice(&sg_range),
                suffix_meta: PageAlignedVec::from_slice(&suffix_meta),
                partials: PageAlignedVec::from_elem(zero, num_sgs * SUF_CELLS),
                out: PageAlignedVec::from_elem(zero, total_cells),
                group: PageAlignedVec::from_slice(&group),
                params: vec![num_sgs as u32, 64],
                reduce_params: vec![total_cells as u32, SUF_CELLS as u32, SUF_CELLS as u32],
                threads: num_sgs * SIMD_WIDTH,
            }
        }

        pub fn buffers(&mut self) -> Result<IrrSuffixScanBuffers<'_>, MetalError> {
            let context = MetalContext::global()?;
            let out_len = self.out.len();
            Ok(IrrSuffixScanBuffers {
                context,
                rows: self.rows.device_buffer(context)?,
                u_evals: self.u_evals.device_buffer(context)?,
                bucket_flat: self.bucket_flat.device_buffer(context)?,
                sg_slot: self.sg_slot.device_buffer(context)?,
                sg_range: self.sg_range.device_buffer(context)?,
                suffix_meta: self.suffix_meta.device_buffer(context)?,
                partials: self.partials.device_buffer_mut(context)?,
                group: self.group.device_buffer(context)?,
                out: self.out.device_buffer_mut(context)?,
                params: &self.params,
                reduce_params: &self.reduce_params,
                threads: self.threads,
                out_len,
            })
        }
    }

    impl IrrSuffixScanBuffers<'_> {
        /// One production-shaped command buffer (suffix scan + its reduce)
        /// returning the CB's GPU execution window.
        pub fn run_timed(&self) -> Result<Duration, MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch(
                KernelId::IrrSuffixScan,
                self.params,
                &self.scan_buffers(),
                self.threads,
            );
            pass.dispatch(
                KernelId::IrrReduce,
                self.reduce_params,
                &[&self.partials, &self.group, &self.out],
                self.out_len,
            );
            pass.commit().wait_timed()
        }

        /// Suffix scan alone (no reduce) — the reduce's share of the CB.
        pub fn run_timed_scan_only(&self) -> Result<Duration, MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch(
                KernelId::IrrSuffixScan,
                self.params,
                &self.scan_buffers(),
                self.threads,
            );
            pass.commit().wait_timed()
        }

        /// One probe dispatch (no reduce) with the production buffer ABI.
        pub fn run_timed_probe(
            &self,
            probe: &VariantPipeline,
            width: usize,
        ) -> Result<Duration, MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch_variant(
                probe,
                self.params,
                &self.scan_buffers(),
                self.threads,
                width,
            );
            pass.commit().wait_timed()
        }

        /// The scan's argument table (`jk_irr_suffix_scan` buffers 0–6).
        fn scan_buffers(&self) -> [&DeviceBuffer<'_>; 7] {
            [
                &self.rows,
                &self.u_evals,
                &self.bucket_flat,
                &self.sg_slot,
                &self.sg_range,
                &self.suffix_meta,
                &self.partials,
            ]
        }
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Device-vs-CPU parity: the suffix MLE library case by case, then the full
/// kernel round loop with the scanner forced on.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::num::NonZeroUsize;

    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_lookup_tables::tables::suffixes::Suffixes;
    use jolt_lookup_tables::{LookupBits, LookupTableKind};
    use jolt_sumcheck::ProveRounds;

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Every variant in declaration order; pins the `as u8` discriminants
    /// the device switch mirrors (a reordered enum fails the count or the
    /// per-id comparison below).
    const ALL_SUFFIXES: [Suffixes; 63] = [
        Suffixes::One,
        Suffixes::And,
        Suffixes::AndNot,
        Suffixes::Xor,
        Suffixes::Or,
        Suffixes::RightOperand,
        Suffixes::RightOperandW,
        Suffixes::UpperWord,
        Suffixes::LowerWord,
        Suffixes::LowerHalfWord,
        Suffixes::LessThan,
        Suffixes::GreaterThan,
        Suffixes::Eq,
        Suffixes::LeftOperandIsZero,
        Suffixes::RightOperandIsZero,
        Suffixes::Lsb,
        Suffixes::DivByZero,
        Suffixes::Pow2,
        Suffixes::Pow2W,
        Suffixes::Rev8W,
        Suffixes::RightShiftPadding,
        Suffixes::RightShift,
        Suffixes::RightShiftHelper,
        Suffixes::SignExtension,
        Suffixes::LeftShift,
        Suffixes::TwoLsb,
        Suffixes::SignExtensionUpperHalf,
        Suffixes::SignExtensionRightOperand,
        Suffixes::RightShiftW,
        Suffixes::RightShiftWHelper,
        Suffixes::LeftShiftWHelper,
        Suffixes::LeftShiftW,
        Suffixes::OverflowBitsZero,
        Suffixes::XorRot16,
        Suffixes::XorRot24,
        Suffixes::XorRot32,
        Suffixes::XorRot63,
        Suffixes::XorRotW16,
        Suffixes::XorRotW12,
        Suffixes::XorRotW8,
        Suffixes::XorRotW7,
        Suffixes::Pow2OffsetW,
        Suffixes::Pext,
        Suffixes::PextHelper,
        Suffixes::WindowSign,
        Suffixes::WindowSignPow2,
        Suffixes::XorRotW22,
        Suffixes::XorRotW19,
        Suffixes::XorRotW6,
        Suffixes::SignExtensionW,
        Suffixes::X31Y0,
        Suffixes::Pow2OffsetB,
        Suffixes::Pow2OffsetH,
        Suffixes::AlignAddr,
        Suffixes::ShiftDataB,
        Suffixes::ShiftDataH,
        Suffixes::ShiftDataW,
        Suffixes::OffsetScaleB,
        Suffixes::OffsetScaleH,
        Suffixes::OffsetScaleW,
        Suffixes::XorRotL1Pairs,
        Suffixes::TopYBit,
        Suffixes::BottomXBit,
    ];

    #[test]
    fn suffix_mle_matches_rust_exhaustively() {
        let _lock = gpu_lock();
        assert_eq!(ALL_SUFFIXES.len(), NUM_SUFFIXES);
        let context = MetalContext::global().unwrap();
        // The catch_unwind below skips domain-undefined cases; keep their
        // panics out of the test output.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        // Cases: every variant × every phase suffix width × edge and random
        // bit patterns (the device sees only masked values, as production
        // does — LookupBits masks on construction).
        let mut cases: Vec<u32> = Vec::new();
        let mut expected: Vec<u64> = Vec::new();
        let mut state = 0x0057_A6E5_u64;
        for (id, suffix) in ALL_SUFFIXES.iter().enumerate() {
            assert_eq!(*suffix as u8 as usize, id, "enum order drifted");
            for len in (0..=120usize).step_by(8) {
                let mask = if len == 0 { 0 } else { (1u128 << len) - 1 };
                let mut patterns = vec![
                    0u128,
                    mask,
                    0x5555_5555_5555_5555_5555_5555_5555_5555 & mask,
                ];
                for _ in 0..48 {
                    patterns.push(
                        (((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128)
                            & mask,
                    );
                }
                // Low-entropy patterns exercise the all-ones/zero guards.
                for _ in 0..16 {
                    patterns.push((splitmix(&mut state) % 97) as u128 & mask);
                }
                for bits in patterns {
                    // Debug builds panic on shift overflow for inputs no
                    // real trace produces (e.g. LeftShiftWHelper with ≥ 32
                    // leading ones); those cases are domain-undefined, so
                    // only defined-behavior cases are compared.
                    let Ok(want) =
                        std::panic::catch_unwind(|| suffix.suffix_mle(LookupBits::new(bits, len)))
                    else {
                        continue;
                    };
                    cases.extend_from_slice(&[
                        bits as u32,
                        (bits >> 32) as u32,
                        (bits >> 64) as u32,
                        (bits >> 96) as u32,
                        id as u32,
                        len as u32,
                    ]);
                    expected.push(want);
                }
            }
        }

        std::panic::set_hook(default_hook);
        let n = expected.len();
        // Sanity: skipping must stay the rare exception.
        assert!(n > 40_000, "too many domain-undefined cases skipped: {n}");
        let cases_buffer = context.wrap_slice(&cases).unwrap();
        let out = context.alloc_u32s(2 * n).unwrap();
        context
            .run_once(
                KernelId::SuffixProbe,
                &[n as u32],
                &[&cases_buffer, &out],
                n,
            )
            .unwrap();
        let mut got = vec![0u32; 2 * n];
        out.copy_to_u32s(&mut got);
        for (case, &want) in expected.iter().enumerate() {
            let device = u64::from(got[2 * case]) | (u64::from(got[2 * case + 1]) << 32);
            assert_eq!(
                device,
                want,
                "suffix {:?} len {} bits {:#x}",
                ALL_SUFFIXES[(cases[case * 6 + 4]) as usize],
                cases[case * 6 + 5],
                u128::from(cases[case * 6])
                    | (u128::from(cases[case * 6 + 1]) << 32)
                    | (u128::from(cases[case * 6 + 2]) << 64)
                    | (u128::from(cases[case * 6 + 3]) << 96),
            );
        }
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0x9E37_79B9_7F4A_7C15 ^ (round as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0x11)
    }

    /// Synthetic rows; `skewed` keeps operands tiny so early-phase chunks
    /// collapse to zero (uniform tiles, long equal-key runs), while `!skewed`
    /// uses full random indices (every tile flushes through the sorted scan).
    fn fixture_rows(log_t: usize, seed: u64, skewed: bool) -> Vec<InstructionCycleRow> {
        let count = LookupTableKind::<RISCV_XLEN>::COUNT;
        let tables = [0, 3 % count, 7 % count, 11 % count, count - 1];
        let mut state = seed;
        (0..1usize << log_t)
            .map(|j| {
                let lookup_index = match j {
                    0 => 0u128,
                    1 => u128::MAX,
                    2 => ((u64::MAX as u128) << 64) | splitmix(&mut state) as u128,
                    _ if skewed => (splitmix(&mut state) & 0xFFFF) as u128,
                    _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
                };
                let table_index = if j % 7 == 3 {
                    None
                } else {
                    Some(tables[j % tables.len()])
                };
                InstructionCycleRow::new(lookup_index, table_index, j % 3 == 0, None, None)
            })
            .collect()
    }

    /// Full round-loop parity, device scanner vs pure CPU, byte-equal round
    /// polynomials and output claims. `min_terms` sets the gate;
    /// `device_cycle_rounds` is the number of cycle messages it admits
    /// (below-gate rounds hand off mid-sumcheck and finish on the CPU) —
    /// the exact probe count proves where the device actually ran.
    fn assert_scanner_parity(
        log_t: usize,
        seed: u64,
        skewed: bool,
        min_terms: usize,
        device_cycle_rounds: usize,
        two_phase: bool,
    ) {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", min_terms.to_string());
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        let rows = Arc::new(InstructionRows::new(
            fixture_rows(log_t, seed, skewed).into_iter().collect(),
        ));
        let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(1000 + 37 * i as u64)).collect();
        let gamma = fr(0xACE1_57EF);

        let mut cpu = OptimizedInstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            Arc::clone(&rows),
            gamma,
        )
        .unwrap();
        let probes_before = device_probe_count();
        let mut device = OptimizedInstructionReadRafKernel::new_with_scanner(
            dimensions,
            &r_reduction,
            rows,
            gamma,
            device_scanner,
        )
        .unwrap();
        assert!(
            device_probe_count() > probes_before,
            "phase 0 never dispatched on the device"
        );

        let rounds = cpu.num_rounds();
        // Parity is claim-agnostic: both kernels consume the same claim
        // stream whatever its starting value.
        let mut claim = fr(0x5EED);
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let cpu_poly = cpu.prove_round(bind, round, claim).unwrap();
            // Two-phase mode mirrors the engine: begin (a cycle round with a
            // device driver launches), then collect.
            let device_poly = if two_phase {
                let _launched = device.begin_round(bind, round, claim).unwrap();
                device.collect_round(bind, round, claim).unwrap()
            } else {
                device.prove_round(bind, round, claim).unwrap()
            };
            assert_eq!(
                cpu_poly.coefficients(),
                device_poly.coefficients(),
                "round {round} polynomial mismatch (log_t={log_t}, skewed={skewed})"
            );
            claim = cpu_poly.evaluate(challenge(round));
        }
        cpu.finish_rounds(challenge(rounds - 1)).unwrap();
        device.finish_rounds(challenge(rounds - 1)).unwrap();
        // 16 phase command buffers + the adopting materialization + one per
        // admitted cycle round — a silent fallback anywhere shows up as a
        // missing probe.
        assert_eq!(
            device_probe_count() - probes_before,
            16 + 1 + device_cycle_rounds as u64,
            "device dispatch count drifted (phases + adoption + cycle rounds)"
        );

        let inputs =
            jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims {
                lookup_output: fr(0),
                left_lookup_operand: fr(0),
                right_lookup_operand: fr(0),
            };
        let cpu_outputs = cpu.output_claims(&inputs).unwrap();
        let device_outputs = device.output_claims(&inputs).unwrap();
        assert_eq!(
            cpu_outputs.lookup_table_flags,
            device_outputs.lookup_table_flags
        );
        assert_eq!(cpu_outputs.instruction_ra, device_outputs.instruction_ra);
        assert_eq!(
            cpu_outputs.instruction_raf_flag,
            device_outputs.instruction_raf_flag
        );
    }

    #[test]
    fn scanner_parity_random_indices() {
        let _lock = gpu_lock();
        assert_scanner_parity(13, 12345, false, 0, 13, false);
    }

    /// Full-device run through the engine's two-phase path: cycle rounds
    /// launch in `begin_round` and collect after — byte-equal to the CPU
    /// twin, same dispatch count.
    #[test]
    fn scanner_parity_random_indices_two_phase() {
        let _lock = gpu_lock();
        assert_scanner_parity(13, 12345, false, 0, 13, true);
    }

    #[test]
    fn scanner_parity_skewed_indices() {
        let _lock = gpu_lock();
        assert_scanner_parity(13, 67890, true, 0, 13, false);
    }

    /// The gate declines mid-sumcheck: the device runs the phases, adopts,
    /// and folds while the pre-bind length clears 2^11 (messages at 8192,
    /// 8192→4096, 4096→2048, 2048→1024), then hands the live tables (with
    /// the next challenge pending) back to the CPU tail.
    #[test]
    fn scanner_parity_cycle_handoff() {
        let _lock = gpu_lock();
        assert_scanner_parity(13, 424_242, false, 2048, 4, false);
    }

    /// Two-phase drive across the mid-sumcheck gate handoff: launches
    /// decline below the gate and `collect_round` recomputes through the
    /// reclaimed host tables.
    #[test]
    fn scanner_parity_cycle_handoff_two_phase() {
        let _lock = gpu_lock();
        assert_scanner_parity(13, 424_242, false, 2048, 4, true);
    }
}
