//! Metal instruction read+RAF checking (stage 5): device phase scans behind
//! the optimized kernel's [`PhaseScanner`] seam.
//!
//! This slot deviates from the W2 `MetalX { fallback }` shape on purpose:
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
//! Per phase the scanner runs ONE command buffer with four dispatches
//! (fused condense+RAF scan, its reduce, the suffix scan, its reduce) over
//! zero-copy wraps of the shared rows / `u_evals` / flat bucket arrays.
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
//! on the CPU, exactly as the W2 slots do.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::{Fr, FromPrimitiveInt, MulPow2};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::JoltWitnessPlane;

use super::{num_threadgroups, own_eq, own_uninit_frs, uninit_frs, DeviceRound, Partials};
use crate::metal::buffers::{DeviceBuffer, OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{DetachedPass, KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::instruction_read_raf::{
    shared_instruction_rows, CycleInitRequest, CycleTables, InstructionCycleRow, InstructionRows,
    OptimizedInstructionReadRafKernel, PhaseScanRequest, PhaseScanSums, PhaseScanner, RafSums,
    ScanOutcome, ScannerInputs,
};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const KIND: &str = "instruction_read_raf";

/// Per-table suffix capacity baked into the shaders (`JK_IRR_SUF_CELLS`);
/// every lookup table today uses at most 8 suffixes.
const MAX_SUFFIXES: usize = 8;
const CHUNK_SIZE: usize = 256;
const RAF_CELLS: usize = 6 * CHUNK_SIZE;
const SUF_CELLS: usize = MAX_SUFFIXES * CHUNK_SIZE;
/// Simdgroup width the schedules assume (the kernels re-check at runtime).
const SIMD_WIDTH: usize = 32;
/// Flat cycle-table capacity baked into the round shader
/// (`JK_IRR_MAX_FACTORS`): `1 + ra_count` must fit.
const MAX_FACTORS: usize = 16;
/// Scan parallelism: enough simdgroups to fill the target GPU, big enough
/// row runs to amortize the per-simdgroup bucket rows.
const TARGET_SIMDGROUPS: usize = 512;
/// Suffix gathers need more in-flight groups to hide indexed row/weight loads;
/// 2048 wins scan+reduce while 4096 overlaps it at twice the partial memory.
const TARGET_SUFFIX_SIMDGROUPS: usize = 2048;
const MIN_ROWS_PER_SIMDGROUP: usize = 1024;

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

/// The scanner's `Drop` settles `in_flight` before retiring the cycle
/// ping-pong (see below); `in_flight` also precedes `cycle` in declaration
/// order as second-line defense.
struct DeviceIrrScanner {
    in_flight: Option<IrrInFlight>,
    device: DeviceRound,
    rows: Arc<InstructionRows>,
    /// Address-phase scan state; dropped when the cycle tables are adopted
    /// (the phases are over) so its buffers free early.
    address: Option<AddressState>,
    /// Adopted cycle tables (the address→cycle handoff onward).
    cycle: Option<CycleRoundsState>,
    /// Value-space `R = 2^256 mod p`: multiplying a raw-space cell by this
    /// re-expresses it in Montgomery form (the exact element the CPU
    /// accumulator reduces to).
    r_mont: Fr,
}

/// A round left in flight must complete (DetachedPass waits on drop) BEFORE
/// the cycle ping-pong it reads/writes frees; declaration order (`in_flight`
/// precedes `cycle`) already guarantees it — this impl documents the
/// invariant explicitly.
impl Drop for DeviceIrrScanner {
    fn drop(&mut self) {
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
    raf_group: OwnedDeviceBuffer<u32>,
    suffix_group: OwnedDeviceBuffer<u32>,
    // Working buffers, sized once and reused every phase.
    partials: OwnedDeviceBuffer<Fr>,
    v_prev: OwnedDeviceBuffer<Fr>,
    raf_out: OwnedDeviceBuffer<Fr>,
    suffix_out: OwnedDeviceBuffer<Fr>,
    num_sgs_raf: usize,
    rows_per_sg_raf: usize,
    phase_scan_kernel: KernelId,
    suffix_scan_kernel: KernelId,
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
        let suffix_legacy =
            std::env::var_os("JOLT_IRR_SUFFIX_SCAN_LEGACY").is_some_and(|value| value != "0");
        let suffix_simdgroups = if suffix_legacy {
            TARGET_SIMDGROUPS
        } else {
            TARGET_SUFFIX_SIMDGROUPS
        };
        let rows_per_sg_suffix =
            div_ceil_pos(bucket_flat.len(), suffix_simdgroups).max(MIN_ROWS_PER_SIMDGROUP);
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
        let zero = Fr::from_u64(0);
        Ok(Self {
            in_flight: None,
            device: DeviceRound::new(context, KIND),
            rows,
            address: Some(AddressState {
                bucket_flat,
                slots,
                sg_slot: own_u32s(sg_slot)?,
                sg_range: own_u32s(sg_range)?,
                suffix_meta: own_u32s(suffix_meta)?,
                raf_group: own_u32s(vec![0, num_sgs_raf as u32])?,
                suffix_group: own_u32s(suffix_group)?,
                partials: context
                    .own_page_aligned(PageAlignedVec::from_elem(zero, partial_cells))?,
                v_prev: context.own_page_aligned(PageAlignedVec::from_elem(zero, CHUNK_SIZE))?,
                raf_out: context.own_page_aligned(PageAlignedVec::from_elem(zero, RAF_CELLS))?,
                suffix_out: context.own_page_aligned(PageAlignedVec::from_elem(
                    zero,
                    (inputs.present.len() * SUF_CELLS).max(1),
                ))?,
                num_sgs_raf,
                rows_per_sg_raf,
                phase_scan_kernel: if std::env::var_os("JOLT_IRR_PHASE_SCAN_LEGACY")
                    .is_some_and(|value| value != "0")
                {
                    KernelId::IrrPhaseScanLegacy
                } else {
                    KernelId::IrrPhaseScan
                },
                suffix_scan_kernel: if suffix_legacy {
                    KernelId::IrrSuffixScanLegacy
                } else {
                    KernelId::IrrSuffixScan
                },
                num_sgs_suffix,
            }),
            cycle: None,
            r_mont: Fr::from_u64(1).mul_pow_2(128).mul_pow_2(128),
        })
    }
}

impl AddressState {
    /// Encode and run the phase's command buffer. `Ok(true)` means every
    /// dispatch ran; `Ok(false)` means a buffer wrap was ineligible and
    /// NOTHING ran (safe to decline).
    fn dispatch_phase(
        &mut self,
        context: &'static MetalContext,
        rows: &[InstructionCycleRow],
        request: &mut PhaseScanRequest<'_, Fr>,
    ) -> Result<bool, MetalError> {
        let suffix_len = request.suffix_len as u32;
        let (do_condense, prev_shift) = match request.condense {
            Some((v_prev, shift)) => {
                debug_assert_eq!(v_prev.len(), CHUNK_SIZE);
                self.v_prev.as_mut_slice().copy_from_slice(v_prev);
                (1u32, shift as u32)
            }
            None => (0u32, 0u32),
        };

        let Some(rows_buffer) = context.wrap_slice_nocopy(rows) else {
            return Ok(false);
        };
        let Some(u_evals_buffer) = context.wrap_slice_mut_nocopy(request.u_evals) else {
            return Ok(false);
        };
        // Read-only: the copying fallback is correct, just counted.
        let bucket_buffer = context.wrap_slice(self.bucket_flat.as_slice())?;
        testing::note_copied_buffers(u64::from(bucket_buffer.was_copied()));

        let n = rows.len();
        let scan_params: Vec<u32> = vec![
            n as u32,
            self.rows_per_sg_raf as u32,
            self.num_sgs_raf as u32,
            suffix_len,
            prev_shift,
            do_condense,
            u32::from(CANONICAL_INSTRUCTION_ADDRESS),
            request.suffix_len.saturating_sub(RISCV_XLEN) as u32,
        ];
        let raf_reduce_params: Vec<u32> =
            vec![RAF_CELLS as u32, RAF_CELLS as u32, RAF_CELLS as u32];
        let suffix_params: Vec<u32> = vec![self.num_sgs_suffix as u32, suffix_len];
        let suffix_cells = self.slots.len() * SUF_CELLS;
        let suffix_reduce_params: Vec<u32> =
            vec![suffix_cells as u32, SUF_CELLS as u32, SUF_CELLS as u32];

        let partials = self.partials.device_buffer();
        let v_prev = self.v_prev.device_buffer();
        let raf_out = self.raf_out.device_buffer();
        let suffix_out = self.suffix_out.device_buffer();
        let sg_slot = self.sg_slot.device_buffer();
        let sg_range = self.sg_range.device_buffer();
        let suffix_meta = self.suffix_meta.device_buffer();
        let raf_group = self.raf_group.device_buffer();
        let suffix_group = self.suffix_group.device_buffer();
        let raf_threads = self.num_sgs_raf * SIMD_WIDTH;

        // Attribution-only split: GPU timestamps are per command buffer, so
        // separate buffers name the phase and suffix owners without changing
        // either kernel's inputs or outputs.
        if std::env::var_os("JOLT_IRR_SPLIT_TRACE").is_some_and(|value| value != "0") {
            let mut pass = context.begin_pass()?;
            pass.dispatch(
                self.phase_scan_kernel,
                &scan_params,
                &[&rows_buffer, &u_evals_buffer, &v_prev, &partials],
                raf_threads,
            );
            pass.dispatch(
                KernelId::IrrReduce,
                &raf_reduce_params,
                &[&partials, &raf_group, &raf_out],
                RAF_CELLS,
            );
            pass.run()?;

            let mut pass = context.begin_pass()?;
            if !self.slots.is_empty() {
                pass.dispatch(
                    self.suffix_scan_kernel,
                    &suffix_params,
                    &[
                        &rows_buffer,
                        &u_evals_buffer,
                        &bucket_buffer,
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
            pass.run()?;
        } else {
            let mut pass = context.begin_pass()?;
            pass.dispatch(
                self.phase_scan_kernel,
                &scan_params,
                &[&rows_buffer, &u_evals_buffer, &v_prev, &partials],
                raf_threads,
            );
            pass.dispatch(
                KernelId::IrrReduce,
                &raf_reduce_params,
                &[&partials, &raf_group, &raf_out],
                RAF_CELLS,
            );
            if !self.slots.is_empty() {
                pass.dispatch(
                    self.suffix_scan_kernel,
                    &suffix_params,
                    &[
                        &rows_buffer,
                        &u_evals_buffer,
                        &bucket_buffer,
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
            pass.run()?;
        }
        testing::note_device_round();
        Ok(true)
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
    fn scan_phase(&mut self, mut request: PhaseScanRequest<'_, Fr>) -> ScanOutcome<Fr> {
        let Some(context) = self.device.gated(self.rows.len()) else {
            return ScanOutcome::Declined;
        };
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
        // next stage's allocations — the W1D ablation measured a placement
        // arena over this pair perf-neutral at every scale).
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
        match launch(self) {
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
        match flight.pass.wait() {
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
                &cycle_init_params(n, position, 0, request),
                &[&rows_buffer, &v_tables_buffer, &table_values_buffer, output],
                n,
            );
        }
        pass.run()?;
        testing::note_device_round();
        Ok(Some(CycleTables { combined_val, ra }))
    }

    /// The adopting twin of [`dispatch_cycle_init`](Self::dispatch_cycle_init):
    /// materialize all `factors` tables into ONE flat scanner-owned buffer
    /// (stride `n`) and set up the round ping-pong. `Ok(None)` = ineligible
    /// buffers, nothing dispatched.
    fn build_cycle_state(
        &self,
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
        // declines adoption before anything runs.
        let Some(cur) = own_uninit_frs(context, factors * n)? else {
            return Ok(None);
        };
        let Some(nxt) = own_uninit_frs(context, factors * (n / 2))? else {
            return Ok(None);
        };

        let cur_buffer = cur.device_buffer();
        let mut pass = context.begin_pass()?;
        for position in 0..factors {
            pass.dispatch(
                KernelId::IrrCycleInit,
                &cycle_init_params(n, position, position * n, request),
                &[
                    &rows_buffer,
                    &v_tables_buffer,
                    &table_values_buffer,
                    &cur_buffer,
                ],
                n,
            );
        }
        pass.run()?;
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
/// `1 + i` = `ra_i`) written at element offset `out_base`.
fn cycle_init_params(
    n: usize,
    position: usize,
    out_base: usize,
    request: &CycleInitRequest<'_, Fr>,
) -> Vec<u32> {
    let (phase_begin, phase_count) = match position.checked_sub(1) {
        None => (0, 0),
        Some(i) => (i * request.phases_per_ra, request.phases_per_ra),
    };
    let mut params = vec![
        n as u32,
        phase_begin as u32,
        phase_count as u32,
        request.address_bits as u32,
        out_base as u32,
    ];
    params.extend_from_slice(&fr_to_u32_limbs(request.raf_interleaved));
    params.extend_from_slice(&fr_to_u32_limbs(request.raf_identity));
    params
}

#[cfg(feature = "bench-utils")]
pub mod bench {
    use jolt_field::{Fr, FromPrimitiveInt, MulPow2};
    use jolt_lookup_tables::{LookupBits, LookupTableKind};
    use rayon::prelude::*;

    use super::*;

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
        params: &'a [u32],
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
        tables: Vec<LookupTableKind<RISCV_XLEN>>,
        ranges: Vec<std::ops::Range<usize>>,
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
            let rows_per_sg = div_ceil_pos(n, TARGET_SIMDGROUPS).max(MIN_ROWS_PER_SIMDGROUP);
            let num_sgs = div_ceil_pos(n, rows_per_sg);
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
            let u_evals = PageAlignedVec::from_fn(n, |_| Fr::from_u64(splitmix(&mut state)));
            let v_prev =
                PageAlignedVec::from_fn(CHUNK_SIZE, |_| Fr::from_u64(splitmix(&mut state)));
            let zero = Fr::from_u64(0);
            Ok(Self {
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
            })
        }

        pub fn assert_oracle(&mut self) -> Result<(), MetalError> {
            let context = MetalContext::global()?;
            {
                let rows = self.rows.device_buffer(context)?;
                let u_evals = self.u_evals.device_buffer(context)?;
                let v_prev = self.v_prev.device_buffer(context)?;
                let partials = self.partials.device_buffer_mut(context)?;
                let out = self.out.device_buffer_mut(context)?;
                let group = self.group.device_buffer(context)?;
                let mut pass = context.begin_pass()?;
                pass.dispatch(
                    KernelId::IrrPhaseScan,
                    &self.params,
                    &[&rows, &u_evals, &v_prev, &partials],
                    self.threads,
                );
                pass.dispatch(
                    KernelId::IrrReduce,
                    &self.reduce_params,
                    &[&partials, &group, &out],
                    RAF_CELLS,
                );
                pass.run()?;
            }

            let expected = self
                .rows
                .par_iter()
                .zip(self.u_evals.par_iter())
                .fold(
                    || vec![Fr::from_u64(0); RAF_CELLS],
                    |mut sums, (row, &u)| {
                        let suffix = row.lookup_index & u64::MAX as u128;
                        let chunk = ((row.lookup_index >> 64) as usize) & (CHUNK_SIZE - 1);
                        if row.raf_flag {
                            sums[3 * CHUNK_SIZE + chunk] += u;
                            sums[4 * CHUNK_SIZE + chunk] += u * Fr::from_u128(suffix);
                            if CANONICAL_INSTRUCTION_ADDRESS {
                                sums[5 * CHUNK_SIZE + chunk] += u;
                            }
                        } else {
                            let (left, right) = LookupBits::new(suffix, 64).uninterleave();
                            sums[chunk] += u;
                            sums[CHUNK_SIZE + chunk] += u * Fr::from_u64(u64::from(left));
                            sums[2 * CHUNK_SIZE + chunk] += u * Fr::from_u64(u64::from(right));
                        }
                        sums
                    },
                )
                .reduce(
                    || vec![Fr::from_u64(0); RAF_CELLS],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                );
            let r_mont = Fr::from_u64(1).mul_pow_2(128).mul_pow_2(128);
            for (index, (&got, &expected)) in self.out.iter().zip(&expected).enumerate() {
                let got = if matches!(index / CHUNK_SIZE, 1 | 2 | 4) {
                    got * r_mont
                } else {
                    got
                };
                assert_eq!(got, expected, "phase-scan cell {index}");
            }

            {
                let rows = self.rows.device_buffer(context)?;
                let u_evals = self.u_evals.device_buffer(context)?;
                let v_prev = self.v_prev.device_buffer(context)?;
                let partials = self.partials.device_buffer_mut(context)?;
                let out = self.out.device_buffer_mut(context)?;
                let group = self.group.device_buffer(context)?;
                let mut pass = context.begin_pass()?;
                pass.dispatch(
                    KernelId::IrrPhaseScanLegacy,
                    &self.params,
                    &[&rows, &u_evals, &v_prev, &partials],
                    self.threads,
                );
                pass.dispatch(
                    KernelId::IrrReduce,
                    &self.reduce_params,
                    &[&partials, &group, &out],
                    RAF_CELLS,
                );
                pass.run()?;
            }
            for (index, (&got, &expected)) in self.out.iter().zip(&expected).enumerate() {
                let got = if matches!(index / CHUNK_SIZE, 1 | 2 | 4) {
                    got * r_mont
                } else {
                    got
                };
                assert_eq!(got, expected, "legacy phase-scan cell {index}");
            }
            Ok(())
        }

        pub fn buffers(&mut self) -> Result<IrrPhaseScanBuffers<'_>, MetalError> {
            let context = MetalContext::global()?;
            Ok(IrrPhaseScanBuffers {
                context,
                rows: self.rows.device_buffer(context)?,
                u_evals: self.u_evals.device_buffer(context)?,
                v_prev: self.v_prev.device_buffer(context)?,
                partials: self.partials.device_buffer_mut(context)?,
                params: &self.params,
                threads: self.threads,
            })
        }
    }

    impl IrrPhaseScanBuffers<'_> {
        pub fn run(&self) -> Result<(), MetalError> {
            self.context.run_once(
                KernelId::IrrPhaseScan,
                self.params,
                &[&self.rows, &self.u_evals, &self.v_prev, &self.partials],
                self.threads,
            )
        }

        pub fn run_legacy(&self) -> Result<(), MetalError> {
            self.context.run_once(
                KernelId::IrrPhaseScanLegacy,
                self.params,
                &[&self.rows, &self.u_evals, &self.v_prev, &self.partials],
                self.threads,
            )
        }
    }

    impl IrrSuffixScanFixture {
        pub fn production_geometry(log_t: usize) -> Result<Self, MetalError> {
            Self::with_simdgroups(log_t, TARGET_SUFFIX_SIMDGROUPS)
        }

        pub fn with_simdgroups(log_t: usize, target_simdgroups: usize) -> Result<Self, MetalError> {
            if target_simdgroups == 0 {
                return Err(MetalError::UnsupportedShape(
                    "suffix-scan fixture requires a nonzero simdgroup target",
                ));
            }
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

            let bucket_len: usize = buckets.iter().map(Vec::len).sum();
            let rows_per_sg =
                div_ceil_pos(bucket_len, target_simdgroups).max(MIN_ROWS_PER_SIMDGROUP);
            let mut bucket_flat = Vec::with_capacity(bucket_len);
            let mut ranges = Vec::with_capacity(tables.len());
            let mut sg_slot = Vec::new();
            let mut sg_range = Vec::new();
            let mut group = Vec::with_capacity(2 * tables.len());
            for (slot, bucket) in buckets.into_iter().enumerate() {
                let range_start = bucket_flat.len();
                bucket_flat.extend_from_slice(&bucket);
                let range_end = bucket_flat.len();
                ranges.push(range_start..range_end);
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
            for table in &tables {
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
            Ok(Self {
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
                tables,
                ranges,
            })
        }

        pub fn assert_oracle(&mut self) -> Result<(), MetalError> {
            let mut expected = vec![Fr::from_u64(0); self.tables.len() * SUF_CELLS];
            for (slot, (table, range)) in self.tables.iter().zip(&self.ranges).enumerate() {
                for &j in &self.bucket_flat[range.clone()] {
                    let row = &self.rows[j as usize];
                    let suffix_bits = LookupBits::new(row.lookup_index & u64::MAX as u128, 64);
                    let chunk = ((row.lookup_index >> 64) as usize) & (CHUNK_SIZE - 1);
                    let u = self.u_evals[j as usize];
                    for (suffix_index, suffix) in table.suffixes().iter().enumerate() {
                        let scalar = suffix.suffix_mle(suffix_bits);
                        expected[slot * SUF_CELLS + suffix_index * CHUNK_SIZE + chunk] +=
                            u * Fr::from_u64(scalar);
                    }
                }
            }

            let context = MetalContext::global()?;
            self.dispatch_and_reduce(context, KernelId::IrrSuffixScan)?;
            self.assert_output(&expected, "suffix-scan");
            self.dispatch_and_reduce(context, KernelId::IrrSuffixScanLegacy)?;
            self.assert_output(&expected, "legacy suffix-scan");
            Ok(())
        }

        fn dispatch_and_reduce(
            &mut self,
            context: &'static MetalContext,
            kernel: KernelId,
        ) -> Result<(), MetalError> {
            let out_len = self.out.len();
            let rows = self.rows.device_buffer(context)?;
            let u_evals = self.u_evals.device_buffer(context)?;
            let bucket_flat = self.bucket_flat.device_buffer(context)?;
            let sg_slot = self.sg_slot.device_buffer(context)?;
            let sg_range = self.sg_range.device_buffer(context)?;
            let suffix_meta = self.suffix_meta.device_buffer(context)?;
            let partials = self.partials.device_buffer_mut(context)?;
            let out = self.out.device_buffer_mut(context)?;
            let group = self.group.device_buffer(context)?;
            let mut pass = context.begin_pass()?;
            pass.dispatch(
                kernel,
                &self.params,
                &[
                    &rows,
                    &u_evals,
                    &bucket_flat,
                    &sg_slot,
                    &sg_range,
                    &suffix_meta,
                    &partials,
                ],
                self.threads,
            );
            pass.dispatch(
                KernelId::IrrReduce,
                &self.reduce_params,
                &[&partials, &group, &out],
                out_len,
            );
            pass.run()
        }

        fn assert_output(&self, expected: &[Fr], label: &str) {
            let r_mont = Fr::from_u64(1).mul_pow_2(128).mul_pow_2(128);
            for (slot, table) in self.tables.iter().enumerate() {
                for (suffix_index, suffix) in table.suffixes().iter().enumerate() {
                    let start = slot * SUF_CELLS + suffix_index * CHUNK_SIZE;
                    let end = start + CHUNK_SIZE;
                    for (offset, (&got, &expected)) in self.out[start..end]
                        .iter()
                        .zip(&expected[start..end])
                        .enumerate()
                    {
                        let index = start + offset;
                        let got = if suffix.is_01_valued() {
                            got
                        } else {
                            got * r_mont
                        };
                        assert_eq!(got, expected, "{label} cell {index}");
                    }
                }
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
        pub fn run(&self) -> Result<(), MetalError> {
            self.dispatch(KernelId::IrrSuffixScan)
        }

        pub fn run_legacy(&self) -> Result<(), MetalError> {
            self.dispatch(KernelId::IrrSuffixScanLegacy)
        }

        pub fn run_with_reduce(&self) -> Result<(), MetalError> {
            let mut pass = self.context.begin_pass()?;
            pass.dispatch(
                KernelId::IrrSuffixScan,
                self.params,
                &[
                    &self.rows,
                    &self.u_evals,
                    &self.bucket_flat,
                    &self.sg_slot,
                    &self.sg_range,
                    &self.suffix_meta,
                    &self.partials,
                ],
                self.threads,
            );
            pass.dispatch(
                KernelId::IrrReduce,
                self.reduce_params,
                &[&self.partials, &self.group, &self.out],
                self.out_len,
            );
            pass.run()
        }

        fn dispatch(&self, kernel: KernelId) -> Result<(), MetalError> {
            self.context.run_once(
                kernel,
                self.params,
                &[
                    &self.rows,
                    &self.u_evals,
                    &self.bucket_flat,
                    &self.sg_slot,
                    &self.sg_range,
                    &self.suffix_meta,
                    &self.partials,
                ],
                self.threads,
            )
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
    use jolt_lookup_tables::tables::suffixes::{Suffixes, NUM_SUFFIXES};
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
    const ALL_SUFFIXES: [Suffixes; 43] = [
        Suffixes::One,
        Suffixes::And,
        Suffixes::AndNot,
        Suffixes::Xor,
        Suffixes::Or,
        Suffixes::RightOperand,
        Suffixes::RightOperandW,
        Suffixes::ChangeDivisor,
        Suffixes::ChangeDivisorW,
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
    /// collapse to zero (the butterfly fast path), while `!skewed` uses full
    /// random indices (the serial emit path).
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
    #[ignore = "large attribution fixture; run explicitly"]
    fn scanner_attribution_scale() {
        let _lock = gpu_lock();
        let log_t = std::env::var("JOLT_IRR_ATTR_LOG_T")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(22);
        assert_scanner_parity(log_t, 12345, false, 0, log_t, true);
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
