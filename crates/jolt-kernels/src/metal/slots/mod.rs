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

mod hamming_weight_claim_reduction;
mod inc_claim_reduction;
mod instruction_read_raf;
mod joint_opening;
mod ra_lazy;
mod ram_hamming_booleanity;
mod ram_raf_evaluation;

pub use hamming_weight_claim_reduction::MetalHammingWeightClaimReduction;
pub use inc_claim_reduction::MetalIncClaimReduction;
pub use instruction_read_raf::MetalInstructionReadRaf;
pub use joint_opening::MetalJointOpening;
pub use ra_lazy::{
    MetalBooleanityCycle, MetalInstructionRaVirtualization, MetalRamRaVirtualization,
};
pub use ram_hamming_booleanity::MetalRamHammingBooleanity;
pub use ram_raf_evaluation::MetalRamRafEvaluation;

use std::sync::Mutex;

use jolt_field::{Fr, FromPrimitiveInt};

use super::buffers::{OwnedDeviceBuffer, PageAlignedVec, MALLOC_LARGE_THRESHOLD, PAGE_SIZE};
use super::error::MetalError;
use super::field::fr_to_u32_limbs;
use super::runtime::{MetalContext, THREADGROUP_SIZE};
use super::testing;

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
    pub(super) fn new(ctx: &MetalContext, table: Vec<Fr>) -> Result<Self, MetalError> {
        let nxt_len = table.len() / 2;
        let cur = ctx.own_vec(table)?;
        testing::note_copied_buffers(u64::from(cur.was_copied()));
        let nxt = ctx.own_page_aligned(PageAlignedVec::from_elem(Fr::from_u64(0), nxt_len))?;
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

/// Retired device buffers awaiting reuse: a slot whose device phase ended
/// parks its big ping-pong pair here ([`retire_frs`]) so a LATER stage's
/// [`own_uninit_frs`] hands the pages back instead of allocating fresh ones
/// (a fresh unified allocation page-zeroes on first touch — the stage-5
/// scanner's flat pair alone is GiB-scale at production traces, and the
/// stage-6b dense adoptions fit inside it). Stale contents satisfy the
/// uninit contract: every consumer device-writes before any read.
///
/// Process-global because the consumers (mid-sumcheck adoptions) have no
/// session access; proof-scoped because the producing slot's `prepare`
/// parks a [`RetiredPoolGuard`] whose drop drains the pool — buffers never
/// outlive the proof that retired them.
struct RetiredPool(Mutex<Vec<OwnedDeviceBuffer<Fr>>>);

// SAFETY: `OwnedDeviceBuffer` is a Metal buffer handle plus its host
// backing memory. `MTLBuffer` is an `MTLResource`, which Metal documents as
// thread-safe (only command encoders are not); the backing memory is plain
// bytes with no thread affinity; the mutex serializes every pool access.
unsafe impl Send for RetiredPool {}
// SAFETY: as for `Send` — all access goes through the mutex.
unsafe impl Sync for RetiredPool {}

static RETIRED: RetiredPool = RetiredPool(Mutex::new(Vec::new()));

/// Park device buffers for reuse by a later stage's [`own_uninit_frs`].
/// Callers (or an earlier slot in the same proof) must have parked a
/// [`RetiredPoolGuard`] in the session so the pool drains at proof end.
#[expect(
    clippy::expect_used,
    reason = "pool mutex cannot be poisoned: no panics inside"
)]
pub(super) fn retire_frs(buffers: impl IntoIterator<Item = OwnedDeviceBuffer<Fr>>) {
    RETIRED
        .0
        .lock()
        .expect("retired-buffer pool poisoned")
        .extend(buffers);
}

/// The smallest retired buffer holding at least `len` elements, if any.
#[expect(
    clippy::expect_used,
    reason = "pool mutex cannot be poisoned: no panics inside"
)]
fn take_retired(len: usize) -> Option<OwnedDeviceBuffer<Fr>> {
    let mut pool = RETIRED.0.lock().expect("retired-buffer pool poisoned");
    let best = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.as_slice().len() >= len)
        .min_by_key(|(_, buffer)| buffer.as_slice().len())
        .map(|(index, _)| index)?;
    Some(pool.swap_remove(best))
}

/// Session-parked drain guard: dropping it (with the proof's session)
/// releases every still-parked retired buffer, so pooled pages never leak
/// across proofs.
pub(super) struct RetiredPoolGuard;

impl Drop for RetiredPoolGuard {
    fn drop(&mut self) {
        if let Ok(mut pool) = RETIRED.0.lock() {
            pool.clear();
        }
    }
}

/// An uninitialized device-owned field-element buffer — reusing a retired
/// buffer when one fits — or `None` when a fresh allocation would not wrap
/// no-copy (see the SAFETY-adjacent contract on [`uninit_frs`] — a copy
/// would read the uninitialized memory).
pub(super) fn own_uninit_frs(
    context: &'static MetalContext,
    len: usize,
) -> Result<Option<OwnedDeviceBuffer<Fr>>, MetalError> {
    if let Some(buffer) = take_retired(len) {
        testing::note_pool_reuse();
        return Ok(Some(buffer));
    }
    let vec = uninit_frs(len);
    let len_bytes = std::mem::size_of_val(vec.as_slice());
    let aligned = (vec.as_ptr() as usize).is_multiple_of(PAGE_SIZE);
    let page_granular = len_bytes.is_multiple_of(PAGE_SIZE) || len_bytes >= MALLOC_LARGE_THRESHOLD;
    if len_bytes == 0 || !aligned || !page_granular {
        return Ok(None);
    }
    let buffer = context.own_vec(vec)?;
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

    /// Retire → smallest-fit reuse → guard drain. Pointer identity proves
    /// the pages came back; the drained pool allocates fresh again.
    #[test]
    fn retired_pool_roundtrip() {
        let _lock = testing::gpu_lock();
        let context = MetalContext::global().unwrap();

        let small = own_uninit_frs(context, 1 << 13).unwrap().unwrap();
        let large = own_uninit_frs(context, 1 << 15).unwrap().unwrap();
        let small_ptr = small.as_slice().as_ptr();
        let large_ptr = large.as_slice().as_ptr();
        retire_frs([large, small]);

        // Smallest fitting buffer wins; an oversized backing is fine.
        let reused = own_uninit_frs(context, 1 << 12).unwrap().unwrap();
        assert_eq!(reused.as_slice().as_ptr(), small_ptr);
        assert!(reused.as_slice().len() >= 1 << 13);
        let reuses_before = testing::pool_reuse_count();
        let reused_large = own_uninit_frs(context, 1 << 14).unwrap().unwrap();
        assert_eq!(reused_large.as_slice().as_ptr(), large_ptr);
        assert_eq!(testing::pool_reuse_count(), reuses_before + 1);

        // Guard drop drains whatever is still parked.
        retire_frs([reused, reused_large]);
        drop(RetiredPoolGuard);
        assert!(take_retired(1).is_none());
    }
}
