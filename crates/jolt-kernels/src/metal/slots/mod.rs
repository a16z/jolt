//! Device sumcheck slots: `Metal*` twins of optimized kernels, installed by
//! [`JoltBackend::metal`](crate::JoltBackend::metal).
//!
//! # The slot pattern (W2 — later waves copy this shape)
//!
//! Each slot is a `MetalX { fallback: OptimizedX }` front whose `prepare`
//! consults [`metal_gate`](super::metal_gate) with the slot's initial
//! work-item count. Below threshold, or on ANY device failure, it silently
//! delegates to the optimized fallback — a device error must never fail a
//! proof. Structural errors (witness access, geometry) are NOT device
//! failures: table construction is shared with the optimized twin and its
//! `KernelError`s propagate identically from either tier.
//!
//! A prepared device kernel owns its round tables as [`RoundTable`]
//! ping-pong pairs of unified-memory [`OwnedDeviceBuffer`]s: the fused round
//! kernel folds the previous challenge OUT OF PLACE (`cur → nxt`, no
//! intra-dispatch hazard) while accumulating the new table's round-poly
//! evaluations as per-threadgroup partials, all in ONE command buffer with
//! ONE synchronization per round (dispatch floors at ~97 µs on the target
//! M4, so per-round sync count is the budget). The HOST finishes the partial
//! sums and assembles the wire polynomial through the exact optimized-tier
//! recipe (`round_poly_from_skipped_evals`, `gruen_poly_deg_3`): field
//! arithmetic is exact on both sides, so any summation regrouping yields
//! byte-identical round polynomials — pinned by the lockstep parity tests in
//! each slot module and by `byte_diff`'s metal arms.
//!
//! Rounds re-consult the gate as tables shrink: once a round's work drops
//! below the slot threshold the kernel finishes on the CPU over the same
//! unified-memory slices (no download — the buffers ARE host memory), which
//! is also the recovery path when a dispatch fails mid-proof
//! ([`DeviceRound::failed`]). `finish_rounds` (a single fold to one element)
//! is always host-side.

mod bytecode_read_raf;
mod hamming_weight_claim_reduction;
mod inc_claim_reduction;
mod instruction_claim_reduction;
mod instruction_input;
mod instruction_read_raf;
mod joint_opening;
mod ra_lazy;
mod ram_hamming_booleanity;
mod ram_raf_evaluation;
mod ram_read_write;
mod registers_read_write;
mod registers_val_evaluation;
mod spartan_outer;
mod spartan_product;

pub use bytecode_read_raf::MetalBytecodeReadRafCycle;
pub use hamming_weight_claim_reduction::MetalHammingWeightClaimReduction;
pub use inc_claim_reduction::MetalIncClaimReduction;
pub use instruction_claim_reduction::MetalInstructionClaimReduction;
pub use instruction_input::MetalInstructionInput;
#[cfg(feature = "bench-utils")]
pub use instruction_read_raf::bench::{IrrPhaseScanFixture, IrrSuffixScanFixture};
pub use instruction_read_raf::MetalInstructionReadRaf;
pub use joint_opening::MetalJointOpening;
pub use ra_lazy::{
    MetalBooleanityCycle, MetalInstructionRaVirtualization, MetalRamRaVirtualization,
};
pub use ram_hamming_booleanity::MetalRamHammingBooleanity;
pub use ram_raf_evaluation::MetalRamRafEvaluation;
pub use ram_read_write::MetalRamReadWriteChecking;
pub use registers_read_write::MetalRegistersReadWriteChecking;
pub use registers_val_evaluation::MetalRegistersValEvaluation;
pub use spartan_outer::{MetalOuterRemainder, MetalOuterUniskip};
pub use spartan_product::{MetalProductRemainder, MetalProductUniskip};

use jolt_field::{Fr, Ring};

use super::buffers::{OwnedDeviceBuffer, PageAlignedVec};
use super::error::MetalError;
use super::field::fr_to_u32_limbs;
use super::runtime::{MetalContext, THREADGROUP_SIZE};
use super::testing;
use crate::mmap_vec::MmapVec;

/// A slot table's ping-pong buffer pair. `cur` holds the live table (logical
/// length tracked by the owning kernel — buffers never shrink); `nxt` is the
/// bind target, swapped in after each fold. Capacities alternate between the
/// initial length and half of it, which every later round's write fits.
pub(super) struct RoundTable {
    cur: OwnedDeviceBuffer<Fr>,
    nxt: OwnedDeviceBuffer<Fr>,
}

impl RoundTable {
    /// Wrap `table` (no-copy when eligible; copies are counted in
    /// [`testing::copied_buffer_count`]) and allocate its half-size bind
    /// target.
    pub(super) fn new(ctx: &'static MetalContext, table: Vec<Fr>) -> Result<Self, MetalError> {
        let nxt_len = table.len() / 2;
        let cur = ctx.own_vec(table)?;
        testing::note_copied_buffers(u64::from(cur.was_copied()));
        // The bind target is written out of place before any read (the
        // ping-pong contract), so it takes the fresh never-zeroed uninit
        // path. Zero-fill remains only for sizes the no-copy wrap can't
        // serve.
        let nxt = match own_uninit_frs(ctx, nxt_len)? {
            Some(buffer) => buffer,
            None => ctx.own_page_aligned(PageAlignedVec::from_elem(Fr::from_u64(0), nxt_len))?,
        };
        Ok(Self { cur, nxt })
    }

    /// Allocate both ping-pong sides for a device fill of `cur` before the
    /// table becomes visible to any round.
    pub(super) fn new_device_filled(
        ctx: &'static MetalContext,
        len: usize,
    ) -> Result<Self, MetalError> {
        let cur = own_uninit_frs(ctx, len)?.ok_or(MetalError::UnsupportedShape(
            "zero-length device-filled round table",
        ))?;
        let nxt = own_uninit_frs(ctx, len / 2)?.ok_or(MetalError::UnsupportedShape(
            "device-filled round table has no bind target",
        ))?;
        Ok(Self { cur, nxt })
    }

    pub(super) fn cur(&self) -> &OwnedDeviceBuffer<Fr> {
        &self.cur
    }

    pub(super) fn nxt(&self) -> &OwnedDeviceBuffer<Fr> {
        &self.nxt
    }

    /// The live table's first `len` evaluations (host view).
    pub(super) fn cur_slice(&self, len: usize) -> &[Fr] {
        &self.cur.as_slice()[..len]
    }

    /// After a device round bound `cur[..len]` into `nxt[..len / 2]`.
    pub(super) fn swap(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.nxt);
    }

    /// Host-side low-to-high fold of `cur[..len]` into the live position:
    /// the CPU twin of the device bind, for below-threshold tail rounds and
    /// post-failure recovery.
    pub(super) fn bind_cpu(&mut self, len: usize, r: Fr) {
        let half = len / 2;
        let source = &self.cur.as_slice()[..len];
        let target = &mut self.nxt.as_mut_slice()[..half];
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            target
                .par_iter_mut()
                .zip(source.par_chunks_exact(2))
                .for_each(|(out, pair)| *out = pair[0] + r * (pair[1] - pair[0]));
        }
        #[cfg(not(feature = "parallel"))]
        for (out, pair) in target.iter_mut().zip(source.chunks_exact(2)) {
            *out = pair[0] + r * (pair[1] - pair[0]);
        }
        self.swap();
    }
}

/// The round kernels' partial-sum sink: `points` per-threadgroup partial
/// rows, sized once for the widest (first) round and reused.
pub(super) struct Partials {
    buffer: OwnedDeviceBuffer<Fr>,
    points: usize,
}

impl Partials {
    pub(super) fn new(
        ctx: &MetalContext,
        points: usize,
        max_threads: usize,
    ) -> Result<Self, MetalError> {
        let max_tgs = num_threadgroups(max_threads);
        Ok(Self {
            buffer: ctx
                .own_page_aligned(PageAlignedVec::from_elem(Fr::from_u64(0), points * max_tgs))?,
            points,
        })
    }

    pub(super) fn buffer(&self) -> &OwnedDeviceBuffer<Fr> {
        &self.buffer
    }

    /// Host finish of the device tree reduction: sum each point's `num_tgs`
    /// partials (order-independent — Fr addition is exact).
    pub(super) fn sums(&self, num_tgs: usize) -> Vec<Fr> {
        let partials = self.buffer.as_slice();
        (0..self.points)
            .map(|point| {
                partials[point * num_tgs..(point + 1) * num_tgs]
                    .iter()
                    .fold(Fr::from_u64(0), |acc, term| acc + *term)
            })
            .collect()
    }
}

pub(super) fn num_threadgroups(threads: usize) -> usize {
    threads.div_ceil(THREADGROUP_SIZE).max(1)
}

/// An uninitialized device-owned field-element buffer, or `None` when the
/// allocation would not wrap no-copy (see the SAFETY-adjacent contract on
/// [`uninit_frs`] — a copy would read the uninitialized memory).
///
/// Always freshly allocated. A retired-buffer placement arena used to live
/// here (park a finished slot's ping-pong pair, carve later adoptions from
/// its pages); the W1D park-vs-free ablation measured it perf-neutral at
/// every scale — malloc's large-entry cache recycles the just-freed pages
/// into the next stage's allocations equally warm — so producers now simply
/// drop their pairs (lane report `.journals/lane-reports/w1d-rootcause.md`).
pub(super) fn own_uninit_frs(
    context: &'static MetalContext,
    len: usize,
) -> Result<Option<OwnedDeviceBuffer<Fr>>, MetalError> {
    if len == 0 {
        return Ok(None);
    }
    // mmap-backed: page-aligned by construction (always no-copy eligible),
    // kernel-zeroed instead of uninit (device fills before any read either
    // way), and — the point — munmapped out of phys_footprint the moment
    // the buffer drops, instead of lingering in libmalloc as the corpse
    // pile that feeds the stage-6b compressor storm (W3A root-cause).
    let buffer = context.own_mmap(MmapVec::zeroed(len))?;
    debug_assert!(!buffer.was_copied());
    Ok(Some(buffer))
}

/// An uninitialized field-element buffer for device fills.
///
/// SAFETY-adjacent contract: callers must guarantee every element the host
/// (or a later device pass) reads was device-written first. `Fr` is plain
/// limb data — no drop glue, no invalid representations.
#[expect(clippy::uninit_vec, reason = "device-filled before any read")]
pub(super) fn uninit_frs(len: usize) -> Vec<Fr> {
    let mut buffer = Vec::with_capacity(len);
    // SAFETY: capacity == len, and per the contract above the contents are
    // fully overwritten by the device before being read.
    unsafe {
        buffer.set_len(len);
    }
    buffer
}

/// Copy the current gruen levels into flight-owned buffers: a parked flight
/// (a committed round awaiting its two-phase collect) must not reference
/// the consumer's eq backings — they drop with the consumer on paths the
/// flight doesn't control — so the launch tier pays a small per-round copy
/// the synchronous tier avoids.
pub(super) fn own_eq(
    context: &'static MetalContext,
    e_in: &[Fr],
    e_out: &[Fr],
) -> Result<(OwnedDeviceBuffer<Fr>, OwnedDeviceBuffer<Fr>), MetalError> {
    let own = |values: &[Fr]| -> Result<OwnedDeviceBuffer<Fr>, MetalError> {
        let buffer = context.own_vec(values.to_vec())?;
        testing::note_copied_buffers(u64::from(buffer.was_copied()));
        Ok(buffer)
    };
    Ok((own(e_in)?, own(e_out)?))
}

/// Stage 6b's synchronous device members (bytecode cycle, increment claim
/// reduction, RAM Hamming booleanity) run two-phase: launch in
/// `begin_round`, wait in `collect_round`, so one queue drain per round
/// replaces three serial blocking waits behind the already-detached
/// Bool/RAV gather command buffers. `JOLT_ST6B_DETACH=0` restores the
/// synchronous rounds (same-binary ablation knob).
pub(super) fn st6b_detach_enabled() -> bool {
    !std::env::var("JOLT_ST6B_DETACH")
        .is_ok_and(|value| matches!(value.trim(), "0" | "off" | "OFF"))
}

/// `[groups, do_bind, num_tgs, r]` — the shared `SlotRoundParams` head of
/// the slot round kernels (`shaders/kernels.metal`).
pub(super) fn slot_round_params(groups: usize, bind: Option<Fr>, num_tgs: usize) -> Vec<u32> {
    let mut params = vec![groups as u32, u32::from(bind.is_some()), num_tgs as u32];
    params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
    params
}

/// One slot's device liveness: the context while healthy, dropped (with one
/// warning) on the first dispatch failure so every later round stays on the
/// CPU. A failed round never corrupts `cur` — the fused kernel only writes
/// `nxt` and the partials — so recovery recomputes the SAME round host-side
/// from intact inputs.
pub(super) struct DeviceRound {
    context: Option<&'static MetalContext>,
    kind: &'static str,
}

impl DeviceRound {
    pub(super) fn new(context: &'static MetalContext, kind: &'static str) -> Self {
        Self {
            context: Some(context),
            kind,
        }
    }

    /// A never-device instance: the parity/timing tests build one kernel in
    /// this state as the CPU reference for its device twin.
    #[cfg(test)]
    pub(super) fn disabled(kind: &'static str) -> Self {
        Self {
            context: None,
            kind,
        }
    }

    /// The context, when the device is still healthy AND `work_items` clears
    /// the slot's gate for this round (tail rounds shrink below it).
    pub(super) fn gated(&self, work_items: usize) -> Option<&'static MetalContext> {
        self.context
            .filter(|_| super::metal_gate(self.kind, work_items))
    }

    /// Record a mid-proof device failure: warn once, fall back forever.
    pub(super) fn failed(&mut self, error: &MetalError) {
        tracing::warn!(
            slot = self.kind,
            %error,
            "device round failed; finishing this sumcheck on the CPU"
        );
        self.context = None;
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;

    /// The uninit path must hand back a no-copy device buffer of the exact
    /// requested length (a copying wrap would read uninitialized memory).
    #[test]
    fn own_uninit_frs_wraps_nocopy() {
        let _lock = testing::gpu_lock();
        let context = MetalContext::global().unwrap();

        let buffer = own_uninit_frs(context, 1 << 15).unwrap().unwrap();
        assert!(!buffer.was_copied());
        assert_eq!(buffer.as_slice().len(), 1 << 15);
    }
}
