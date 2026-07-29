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
mod ram_hamming_booleanity;

pub use hamming_weight_claim_reduction::MetalHammingWeightClaimReduction;
pub use inc_claim_reduction::MetalIncClaimReduction;
pub use ram_hamming_booleanity::MetalRamHammingBooleanity;

use jolt_field::{Fr, FromPrimitiveInt};

use super::buffers::{OwnedDeviceBuffer, PageAlignedVec};
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
