//! The Metal witness-commitment slot: tier-1 one-hot G1 accumulation on the
//! device, pipelined with the CPU work only the CPU can do.
//!
//! The optimized CPU kernel spends its streaming phase on one-hot G1 batch
//! additions (~55% of the @2^23 stage-0 wall) and then a strictly-after
//! finish phase on tier-2 pairings (~40%). This slot restructures the pass
//! into three concurrent lanes so the pairings overlap the group work
//! instead of following it:
//!
//! ```text
//! driver   [extract sc0][extract sc1][extract sc2]…
//! builder  ......[bucket|digit-recode sc0][bucket|digit-recode sc1]…
//! gpu      ............[G1 seg sums sc0]......[G1 seg sums sc1]......
//! tier-2   ..................[Miller absorb sc0]…[final exps]
//! ```
//!
//! - The **driver** (calling thread) streams superchunks through one fused
//!   subchunk-parallel pass: each trace row is extracted once and its
//!   one-hot hot addresses and increment scalars staged with per-subchunk
//!   bucket counts into a recycled [`StagedChunk`] (double-buffered, so
//!   the next superchunk extracts while the previous one builds). Below
//!   the `commit_inc` gate the increment columns also run the identical
//!   [`StreamingCommitment::feed_i128_rows`] calls the optimized kernel
//!   makes.
//! - The **builder lane** turns each staging into a device job: a compact
//!   pass scatters the staged addresses into per `(column, window, k)`
//!   gather segments and signed base-256 digit buckets (per
//!   `(column, window, slot, |digit|)`, scalar and digit signs folded into
//!   the gather index's negation bit), and ships both segment families to
//!   the GPU lane as one job. Job slabs recycle through the GPU lane, so
//!   steady-state superchunks re-fill warm pages.
//! - The **GPU lane** (dedicated thread; command buffers are not `Send`)
//!   runs `jk_g1_seg_sum` over each superchunk — one command buffer, one
//!   dispatch per segment family: every segment thread sums its selected
//!   (possibly negated) SRS bases in Jacobian form. One-hot buckets are the
//!   work the CPU tier does with batch-affine additions; increment buckets
//!   are the Pippenger inner loop of the CPU tier's `msm_i128`.
//! - The **tier-2 lane** decodes finished segments, reduces multi-segment
//!   buckets, batch-normalizes, records the row commitments, and multiplies
//!   the rows' Miller loops into per-column [`Tier2Accumulator`]s via the
//!   tiled prepared-coefficient table kernel (one final exponentiation per
//!   column at the end). Increment buckets it reduces
//!   arithmetically instead: per `(column, window)` a weighted running sum
//!   over digit magnitudes and a base-256 Horner across digit slots yield
//!   the row commitment — the same group value `msm_i128` produces.
//!
//! Row commitments equal the CPU path's at group level by construction
//! (both are sums of the same base subsets), and after batch normalization
//! they are coordinate-identical (Z = 1). The accumulated tier-2 GT is
//! value-identical to `multi_pair_g2_setup` (see [`jolt_dory`]'s `tier2`
//! module), so commitments, hints, and downstream proof bytes all match the
//! optimized kernel — pinned by `metal_commit_matches_optimized`.
//!
//! Increment columns and every non-streaming geometry (address-major
//! orders, widened grids, advice) take the optimized/reference path
//! unchanged. Device errors fall back to the optimized kernel on a fresh
//! stream pass (fail-closed, never mid-proof wrong).

use std::any::TypeId;
use std::sync::mpsc::{sync_channel, SyncSender};

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, Zero};
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_crypto::Bn254G1;
use jolt_dory::{
    one_hot_output_from_rows, DoryCommitment, DoryHint, DoryPartialCommitment, DoryProverSetup,
    DoryScheme, DoryTier2Prep, Tier2Accumulator,
};
use jolt_field::{Field, Fr};
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
use jolt_witness::{JoltWitnessOracle, RowSource, WitnessBundle, WitnessError};
use rayon::prelude::*;

#[cfg(feature = "bench-utils")]
use super::field::FR_U32_LIMBS;
use super::g1::{
    bases_as_u32s, jac_from_device_limbs, JAC_U32S, SEG_INDEX_SIGN_BIT, SEG_SUM_WIDTH,
};
use super::miller::{
    g1_points_as_u32s, miller_table_params, product_of_partials, MillerTile, FQ12_U32S,
};
use super::runtime::{KernelId, MetalContext};
use super::{metal_gate, DeviceBuffer, MetalError, PageAlignedVec};
use crate::commitment::{
    CommitWitness, CommitmentGrid, CommittedColumnsWitness, WitnessCommitment,
};
use crate::reference::commitment::{column_kinds, ColumnKind};
use crate::{KernelError, OptimizedBackend, ProofSession};

#[cfg(feature = "bench-utils")]
mod bench;
mod builder;
#[cfg(test)]
mod tests;
mod tier2;

#[cfg(feature = "bench-utils")]
pub use bench::{G1SegBenchCase, G1SegBenchFixture, G1SegBenchSample};
use builder::{
    build_inc_job, build_one_hot_job, reduce_inc_superchunk, DriverScratch, MetalColumns, SlabPool,
};
use tier2::{absorb_superchunk, drain_miller_lane, MillerBatch, MillerLane};

/// Cycles per superchunk — the optimized kernel's width, so both backends
/// stream the same window sequences.
const SUPERCHUNK_CYCLES: usize = 1 << 17;

/// Job/result queue depth. Two keeps each lane one superchunk ahead of the
/// next without unbounded buffering (a queued job is ~tens of MB of gather
/// indices at production widths).
const PIPELINE_DEPTH: usize = 2;

/// Staging sets circulating between the driver and the builder lane. Two
/// overlaps superchunk N's job build (and its GPU-queue backpressure) with
/// superchunk N+1's extraction, so the two host phases run concurrently
/// instead of serializing on one thread; a set is ~20 MB at 2^27 geometry.
const STAGING_SETS: usize = 2;

/// Per-thread segment cap: buckets larger than this split into several
/// device threads whose partial sums the tier-2 lane re-adds. The kernel is
/// thread-starved at production segment counts (W9: mul-chain ceiling 5.9
/// Gmul/s at ~12k threads vs 11.6 saturated), so more, shorter segments buy
/// device throughput; 128 beat 256 by −7.7% and 64 inverted (in-pipeline
/// commit ABBA @2^25) — the tier-2 lane re-adds 2× the partials, affordable
/// since the W7/W8 host cuts.
const MAX_SEGMENT_LEN: usize = 128;

/// Pairs per device Miller thread. W13 re-sweep at the post-W7 65536-pair
/// flush (cap-32 pipeline, production row striping): ppt 1/2/4/8 →
/// 1.53/1.15/1.03/1.66 µs/pair — 4 amortizes the per-thread squaring
/// ladder to its knee; 8 inverts (8192 threads starve the device).
const MILLER_TABLE_SEG_PAIRS: usize = 4;

/// Pairs per CPU absorb sub-shard. Small enough that rayon can steal a big
/// column off a straggler core mid-superchunk, large enough that the
/// per-shard squaring ladder stays noise (64 Fq12 squarings per shard
/// ≈ +1% of its line-fold work).
const MILLER_CPU_SHARD: usize = 64;

/// CPU share of each superchunk's Miller pairs. Tier2Accumulator is
/// partition-invariant, so ANY split is byte-identical — the share is pure
/// load balance. The tiled table kernel absorbs a pair ~20× cheaper than
/// the CPU path (1.03 vs ~21 µs), and at flagship scale the tier-2 host
/// lane (decode + reduce_inc) is the pipeline's second-binding lane while
/// the device holds slack — so every pair rides the device; the CPU absorb
/// remains the device-failure recovery arm.
const MILLER_CPU_FRACTION: f64 = 0.0;

/// Queued device pairs flush when they reach this count: dispatches batch
/// across superchunks — byte-free by the same partition invariance as the
/// CPU/device split itself. The Miller kernels starve below ~32k-pair
/// dispatches (W7 fly retune −40% CB mass at 65536; W13 table scale
/// curve agrees); diminishing past that, and the stream-end drain batch
/// grows with the threshold.
const MILLER_FLUSH_PAIRS: usize = 65536;

/// Signed base-256 digit slots covering any `i128` magnitude (16 bytes plus
/// a possible recoding carry into a 17th slot).
const INC_SLOTS: usize = 17;

/// Digit magnitudes after signed recoding span `1..=128`.
const INC_MAGNITUDES: usize = 128;

/// Digit buckets per increment `(column, window)` block.
const INC_BUCKETS: usize = INC_SLOTS * INC_MAGNITUDES;

/// Rows per fused-pass subchunk: the parallel grain of extraction, digit
/// counting, and both scatters. A production superchunk fans out into
/// `superchunk / 1024 = 128` tasks even at the widest row geometry (where
/// whole-window blocks collapse to 1-2 per superchunk) — enough that the
/// last scheduling round wastes ≤ a few percent of the burst — while
/// per-subchunk count tables stay cache-resident.
const SUBCHUNK_ROWS: usize = 1024;

/// `hot` staging entry for a cycle with no hot address. Addresses fit u16
/// (the device gate requires `one_hot_k < u16::MAX`), halving the staging
/// array's write+read traffic against u32.
const HOT_NONE: u16 = u16::MAX;

/// Signed base-256 recoding: `value = Σ ±magnitude · 256^slot` over the
/// emitted `(slot, magnitude, negate)` digits, `magnitude ∈ 1..=128`,
/// `slot < INC_SLOTS`. The scalar's sign folds into each digit's `negate`,
/// so bucket sums need only base negation. Any correct decomposition yields
/// the same MSM group value, which is all downstream parity needs.
#[inline]
fn for_each_signed_digit(value: i128, mut digit: impl FnMut(u32, u32, bool)) {
    let negative = value < 0;
    let mut magnitude_bytes = value.unsigned_abs();
    let mut carry = 0u32;
    let mut slot = 0u32;
    while magnitude_bytes != 0 || carry != 0 {
        let d = ((magnitude_bytes & 0xff) as u32) + carry;
        magnitude_bytes >>= 8;
        carry = u32::from(d > 128);
        let (magnitude, digit_negative) = if d > 128 { (256 - d, true) } else { (d, false) };
        if magnitude != 0 {
            digit(slot, magnitude, digit_negative != negative);
        }
        slot += 1;
    }
}

/// Install the Dory-specialized Metal commit slot when the backend is
/// instantiated at `(Fr, DoryScheme)` — the only pairing the device tier
/// implements. Other instantiations keep the optimized slot.
pub(super) fn dory_commit_slot<F, PCS>() -> Option<Box<dyn CommitWitness<F, PCS>>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + StreamingCommitment + 'static,
{
    if TypeId::of::<PCS>() != TypeId::of::<DoryScheme>() {
        return None;
    }
    let boxed: Box<dyn CommitWitness<Fr, DoryScheme>> = Box::new(MetalCommitWitness);
    // SAFETY: the TypeId check proves PCS == DoryScheme, and the
    // `CommitmentScheme<Field = F>` bound then forces F == DoryScheme::Field
    // == Fr, so the source and target trait-object types are the same type;
    // the fat pointer (data + vtable) is reinterpreted, not changed.
    Some(unsafe {
        std::mem::transmute::<Box<dyn CommitWitness<Fr, DoryScheme>>, Box<dyn CommitWitness<F, PCS>>>(
            boxed,
        )
    })
}

/// The Metal commit slot: device tier-1 with pipelined CPU tier-2, falling
/// back to [`OptimizedBackend`] below the gate, off the streaming geometry,
/// or on any device error.
struct MetalCommitWitness;

impl CommitWitness<Fr, DoryScheme> for MetalCommitWitness {
    #[tracing::instrument(
        skip_all,
        name = "commit_witness",
        fields(columns = ids.len(), total_vars = grid.total_vars)
    )]
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        source: &dyn RowSource,
        ids: &[JoltCommittedPolynomial],
        grid: CommitmentGrid,
        setup: &DoryProverSetup,
    ) -> Result<Vec<WitnessCommitment<DoryScheme>>, KernelError<Fr>> {
        let cycles = 1usize << grid.log_t;
        let row_width = grid.num_columns();
        let kinds = column_kinds::<Fr>(ids, grid)?;

        let streaming_geometry =
            grid.order == TracePolynomialOrder::CycleMajor && row_width <= cycles;
        let device = streaming_geometry
            && kinds.iter().any(|kind| kind.is_one_hot())
            // Hot addresses stage as u16 (HOT_NONE reserved).
            && (1usize << grid.log_k_chunk) < usize::from(u16::MAX)
            && metal_gate("commit", cycles)
            && MetalContext::global().is_ok();
        if !device {
            return OptimizedBackend.commit_witness(session, source, ids, grid, setup);
        }

        #[expect(clippy::expect_used, reason = "checked by the device gate above")]
        let ctx = MetalContext::global().expect("gated on a live context");
        // Separately gated so the increment path can be ablated (or raised)
        // without touching the one-hot pipeline.
        let inc_device = metal_gate("commit_inc", cycles);
        match commit_streaming_metal(
            ctx,
            source,
            &kinds,
            grid,
            setup,
            SUPERCHUNK_CYCLES,
            MAX_SEGMENT_LEN,
            MILLER_FLUSH_PAIRS,
            inc_device,
        ) {
            Ok(outputs) => Ok(outputs
                .into_iter()
                .zip(ids)
                .map(|((commitment, hint), &id)| WitnessCommitment {
                    id,
                    commitment,
                    hint,
                })
                .collect()),
            Err(MetalCommitError::Witness(error)) => Err(error.into()),
            Err(MetalCommitError::Device(error)) => {
                tracing::warn!(%error, "Metal commit failed; re-running on the optimized kernel");
                OptimizedBackend.commit_witness(session, source, ids, grid, setup)
            }
        }
    }

    fn commit_advice(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessOracle<Fr>,
        id: JoltCommittedPolynomial,
        grid: CommitmentGrid,
        setup: &DoryProverSetup,
    ) -> Result<WitnessCommitment<DoryScheme>, KernelError<Fr>> {
        // Advice grids are small single-column dense commits — no one-hot
        // tier-1 to offload.
        OptimizedBackend.commit_advice(session, witness, id, grid, setup)
    }
}

/// Device errors are recoverable (the wrapper re-runs the pass on the CPU);
/// witness errors are the caller's.
#[derive(Debug)]
enum MetalCommitError {
    Device(MetalError),
    Witness(WitnessError),
}

impl From<MetalError> for MetalCommitError {
    fn from(error: MetalError) -> Self {
        Self::Device(error)
    }
}

/// One segment's destination: which one-hot column (ordinal among one-hot
/// columns) and which tier-2 row (`k · windows_total + window`).
#[derive(Clone, Copy)]
struct SegOut {
    column: u32,
    row: u32,
}

/// One increment segment's destination: the increment column ordinal, the
/// absolute window (= tier-2 row), and the signed-digit bucket it belongs
/// to. Segments are emitted sorted by all four fields, so split buckets are
/// consecutive and `(column, window)` groups are contiguous runs.
#[derive(Clone, Copy, PartialEq, Eq)]
struct IncSeg {
    column: u32,
    window: u32,
    slot: u8,
    magnitude: u8,
}

/// The increment columns' half of a superchunk job: signed gather indices
/// (bit 31 = negated base) grouped by digit bucket.
struct IncJob {
    indices: PageAlignedVec<u32>,
    seg_bounds: PageAlignedVec<u32>,
    segs: Vec<IncSeg>,
}

/// A superchunk's device job: gather indices grouped by segment, the
/// length-sorted device bounds triples, and each segment's destination
/// (in bucket-walk order — bounds carry the out slot); plus the increment
/// columns' bucket family when they ride the device.
struct GpuJob {
    indices: PageAlignedVec<u32>,
    seg_bounds: PageAlignedVec<u32>,
    segs: Vec<SegOut>,
    inc: Option<IncJob>,
}

/// [`super::g1::seg_bounds_sorted`] into a recycled slab.
fn seg_bounds_into_slab(seg_starts: &[u32], pool: &mut SlabPool) -> PageAlignedVec<u32> {
    let bounds = super::g1::seg_bounds_sorted(seg_starts);
    let mut slab = pool.take(bounds.len().max(1));
    slab[..bounds.len()].copy_from_slice(&bounds);
    slab
}

/// A finished job: decoded destinations plus the raw Jacobian limb results,
/// per segment family.
struct GpuDone {
    segs: Vec<SegOut>,
    jac: Vec<u32>,
    inc: Option<(Vec<IncSeg>, Vec<u32>)>,
}

/// One extracted superchunk in flight to the builder lane: staged hot
/// addresses / bucket counts / increment scalars plus the chunk geometry
/// the builds need. Sets recycle through a free-list channel
/// ([`STAGING_SETS`]).
struct StagedChunk {
    scratch: DriverScratch,
    inc_vals: Vec<Vec<i128>>,
    n: usize,
    window_base: usize,
}

impl StagedChunk {
    fn new(row_width: usize, n_inc: usize) -> Self {
        Self {
            scratch: DriverScratch::new(row_width),
            inc_vals: vec![Vec::new(); n_inc],
            n: 0,
            window_base: 0,
        }
    }
}

/// The streaming metal commit at explicit superchunk width, segment cap, and
/// Miller flush threshold (tests shrink all three to force multi-delivery,
/// multi-segment reduction, and mid-stream Miller flushes). Returns
/// per-column outputs in `kinds` order. `inc_device` routes the increment
/// columns' row MSMs through the device as signed digit buckets; off, they
/// run the CPU `feed_i128_rows` path.
#[expect(
    clippy::expect_used,
    reason = "worker joins fail only on a panicked worker (which must propagate); the \
              output expect is the one-hot/increment kind exhaustiveness"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "internal seam; the trailing four are the geometry/gate knobs the tests shrink"
)]
fn commit_streaming_metal(
    ctx: &'static MetalContext,
    source: &dyn RowSource,
    kinds: &[ColumnKind],
    grid: CommitmentGrid,
    setup: &DoryProverSetup,
    superchunk_cycles: usize,
    max_segment_len: usize,
    miller_flush_pairs: usize,
    inc_device: bool,
) -> Result<Vec<(DoryCommitment, DoryHint)>, MetalCommitError> {
    let cycles = 1usize << grid.log_t;
    let row_width = grid.num_columns();
    let one_hot_k = 1usize << grid.log_k_chunk;
    let windows_total = cycles / row_width;
    let windows_per_sc = (superchunk_cycles / row_width).clamp(1, windows_total);
    let one_hot_rows = windows_total * one_hot_k;

    let one_hot: Vec<(usize, ColumnKind)> = kinds
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, kind)| kind.is_one_hot())
        .collect();
    let n_one_hot = one_hot.len();
    let n_inc = kinds.len() - n_one_hot;

    // Shared pass state, built while nothing else is running: the prepared
    // G2 generators every tier-2 absorb pairs against, and the affine SRS
    // bases the device gathers from (one copy for all columns — the CPU
    // path materializes one per column).
    let (prep, bases) = rayon::join(
        || DoryScheme::prepare_tier2(setup, one_hot_rows.max(windows_total)),
        || {
            tracing::info_span!("MetalCommit::base_affine_cache")
                .in_scope(|| DoryScheme::begin_one_hot_column_major_stream(setup, row_width))
        },
    );
    assert!(
        !bases.par_iter().any(AffineRepr::is_zero),
        "SRS G1 bases must not contain the identity (the device view never \
         reads the infinity flag)"
    );

    let miller_device = metal_gate("miller", n_one_hot * one_hot_rows);

    std::thread::scope(|scope| {
        let (tx_jobs, rx_jobs) = sync_channel::<GpuJob>(PIPELINE_DEPTH);
        let (tx_done, rx_done) = sync_channel::<GpuDone>(PIPELINE_DEPTH);
        let (tx_recycle, rx_recycle) = std::sync::mpsc::channel::<PageAlignedVec<u32>>();
        let (tx_staged, rx_staged) = sync_channel::<StagedChunk>(1);
        let (tx_free, rx_free) = std::sync::mpsc::channel::<StagedChunk>();
        for _ in 0..STAGING_SETS {
            let _ = tx_free.send(StagedChunk::new(row_width, n_inc));
        }

        let bases_ref = &bases;
        let gpu = scope.spawn(move || -> Result<(), MetalError> {
            // Command buffers and MTLBuffers are not Send: every device
            // object lives and dies on this thread.
            let bases_buf = ctx.wrap_slice(bases_as_u32s(bases_ref))?;
            loop {
                let Ok(job) =
                    tracing::info_span!("MetalCommit::gpu_recv_wait").in_scope(|| rx_jobs.recv())
                else {
                    break;
                };
                let n_segs = job.segs.len();
                let wrap_span = tracing::info_span!("MetalCommit::gpu_wrap").entered();
                let indices_buf = job.indices.device_buffer(ctx)?;
                let bounds_buf = job.seg_bounds.device_buffer(ctx)?;
                let out_buf = ctx.alloc_u32s(n_segs * JAC_U32S)?;
                let mut pass = ctx.begin_pass()?;
                pass.dispatch_width(
                    KernelId::G1SegSum,
                    &[u32::try_from(n_segs).map_err(|_| {
                        MetalError::Execution("segment count overflows u32".to_owned())
                    })?],
                    &[&bases_buf, &indices_buf, &bounds_buf, &out_buf],
                    n_segs,
                    SEG_SUM_WIDTH,
                );
                // The increment family joins the same command buffer: a
                // second dispatch over disjoint buffers, one wait for both.
                let inc_bufs = match &job.inc {
                    Some(inc) => {
                        let n_inc_segs = inc.segs.len();
                        let inc_indices_buf = inc.indices.device_buffer(ctx)?;
                        let inc_bounds_buf = inc.seg_bounds.device_buffer(ctx)?;
                        let inc_out_buf = ctx.alloc_u32s(n_inc_segs * JAC_U32S)?;
                        pass.dispatch_width(
                            KernelId::G1SegSum,
                            &[u32::try_from(n_inc_segs).map_err(|_| {
                                MetalError::Execution(
                                    "increment segment count overflows u32".to_owned(),
                                )
                            })?],
                            &[&bases_buf, &inc_indices_buf, &inc_bounds_buf, &inc_out_buf],
                            n_inc_segs,
                            SEG_SUM_WIDTH,
                        );
                        Some((n_inc_segs, inc_indices_buf, inc_bounds_buf, inc_out_buf))
                    }
                    None => None,
                };
                drop(wrap_span);
                tracing::info_span!("MetalCommit::gpu_run").in_scope(|| pass.run())?;
                let readback_span = tracing::info_span!("MetalCommit::gpu_readback").entered();
                let mut jac = vec![0u32; n_segs * JAC_U32S];
                out_buf.copy_to_u32s(&mut jac);
                let inc_jac = inc_bufs.as_ref().map(|(n_inc_segs, _, _, inc_out_buf)| {
                    let mut inc_jac = vec![0u32; n_inc_segs * JAC_U32S];
                    inc_out_buf.copy_to_u32s(&mut inc_jac);
                    inc_jac
                });
                drop(readback_span);
                // Release the borrows on the job's buffers before recycling
                // the slabs back to the driver.
                drop(inc_bufs);
                drop(indices_buf);
                drop(bounds_buf);
                let GpuJob {
                    indices,
                    seg_bounds,
                    segs,
                    inc,
                } = job;
                let _ = tx_recycle.send(indices);
                let _ = tx_recycle.send(seg_bounds);
                let inc_done = inc.and_then(|inc| {
                    let _ = tx_recycle.send(inc.indices);
                    let _ = tx_recycle.send(inc.seg_bounds);
                    inc_jac.map(|inc_jac| (inc.segs, inc_jac))
                });
                if tx_done
                    .send(GpuDone {
                        segs,
                        jac,
                        inc: inc_done,
                    })
                    .is_err()
                {
                    // The tier-2 lane never closes first unless the scope is
                    // unwinding; stop quietly either way.
                    return Ok(());
                }
            }
            Ok(())
        });

        let prep_ref = &prep;
        type Tier2Out = (Vec<Tier2Accumulator>, Vec<Vec<Bn254G1>>, Vec<Vec<Bn254G1>>);
        let tier2 = scope.spawn(move || -> Tier2Out {
            let mut accumulators: Vec<Tier2Accumulator> =
                (0..n_one_hot).map(|_| Tier2Accumulator::new()).collect();
            let mut rows: Vec<Vec<Bn254G1>> = (0..n_one_hot)
                .map(|_| vec![Default::default(); one_hot_rows])
                .collect();
            // The Miller lane's device objects live and die on this thread
            // (MTLBuffers are not Send).
            let mut miller_lane = miller_device.then(|| MillerLane {
                flush_pairs: miller_flush_pairs,
                failed: false,
                queue: MillerBatch::new(),
                in_flight: None,
                tiles: Vec::new(),
            });
            // Increment row commitments by absolute window; windows whose
            // scalars are all zero receive no segments and stay identity —
            // exactly the CPU path's zero-MSM rows.
            let mut inc_rows: Vec<Vec<Bn254G1>> = (0..n_inc)
                .map(|_| vec![Default::default(); windows_total])
                .collect();
            loop {
                let Ok(done) =
                    tracing::info_span!("MetalCommit::tier2_recv_wait").in_scope(|| rx_done.recv())
                else {
                    break;
                };
                absorb_superchunk(
                    ctx,
                    prep_ref,
                    &done,
                    &mut accumulators,
                    &mut rows,
                    miller_lane.as_mut(),
                );
                if let Some((inc_segs, inc_jac)) = &done.inc {
                    let _span = tracing::info_span!("MetalCommit::tier2_reduce_inc").entered();
                    reduce_inc_superchunk(inc_segs, inc_jac, &mut inc_rows);
                }
            }
            if let Some(lane) = miller_lane.as_mut() {
                let _span = tracing::info_span!("MetalCommit::miller_drain").entered();
                drain_miller_lane(ctx, prep_ref, lane, &mut accumulators);
            }
            (accumulators, rows, inc_rows)
        });

        // The builder lane: turns staged superchunks into device jobs —
        // layout prefixes, segment walks, both scatters, and the GPU-queue
        // send (with its backpressure) — off the driver thread, which keeps
        // extracting the next superchunk meanwhile. Staging sets recycle
        // through `tx_free`, job slabs through the GPU lane's `tx_recycle`.
        // Returns whether a job send failed (device lane death).
        let has_inc = inc_device && n_inc > 0;
        let builder = scope.spawn(move || -> bool {
            let mut pool = SlabPool::with_returns(rx_recycle);
            let mut tx_jobs = Some(tx_jobs);
            let mut send_failed = false;
            loop {
                let Ok(mut staged) = tracing::info_span!("MetalCommit::builder_recv_wait")
                    .in_scope(|| rx_staged.recv())
                else {
                    break;
                };
                if let Some(tx) = &tx_jobs {
                    let mut job = {
                        let _span = tracing::info_span!("MetalCommit::build_gpu_job").entered();
                        build_one_hot_job(
                            &mut staged.scratch,
                            &mut pool,
                            staged.n,
                            n_one_hot,
                            one_hot_k,
                            row_width,
                            staged.window_base,
                            windows_total,
                            max_segment_len,
                        )
                    };
                    if has_inc {
                        let _span = tracing::info_span!("MetalCommit::build_inc_job").entered();
                        job.inc = build_inc_job(
                            &staged.inc_vals,
                            row_width,
                            staged.window_base,
                            max_segment_len,
                            &mut staged.scratch,
                            &mut pool,
                        );
                    }
                    if tracing::info_span!("MetalCommit::send_wait")
                        .in_scope(|| tx.send(job).is_err())
                    {
                        // Device lane died; its join surfaces the error after
                        // the stream completes.
                        send_failed = true;
                        tx_jobs = None;
                    }
                }
                if tx_free.send(staged).is_err() {
                    // Driver gone (stream error unwinding); stop quietly.
                    break;
                }
            }
            send_failed
        });

        let increments: Vec<(usize, ColumnKind, DoryPartialCommitment)> = kinds
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, kind)| !kind.is_one_hot())
            .map(|(index, kind)| (index, kind, DoryScheme::begin(setup)))
            .collect();
        let inc_kinds: Vec<ColumnKind> = increments.iter().map(|(_, kind, _)| *kind).collect();
        let mut consumer = MetalColumns {
            tx: Some(tx_staged),
            free: rx_free,
            one_hot: &one_hot,
            increments,
            inc_kinds,
            inc_device,
            row_width,
            one_hot_k,
            windows_fed: 0,
            setup,
        };
        let stream_result =
            tracing::info_span!("stream_witnesses", cycles = cycles).in_scope(|| {
                source.visit_chunks(
                    0..cycles,
                    row_width * windows_per_sc,
                    &mut |rows, next_after, env| consumer.consume_rows(rows, next_after, env),
                )
            });
        // Close the staged queue so the builder drains and exits (dropping
        // the job queue); the GPU lane then drains and exits, and the tier-2
        // lane last.
        let drain_span = tracing::info_span!("MetalCommit::lane_drain").entered();
        consumer.tx = None;
        let send_failed = builder.join().expect("builder lane panicked");
        let gpu_result = gpu.join().expect("GPU lane panicked");
        let (accumulators, rows, inc_rows) = tier2.join().expect("tier-2 lane panicked");
        drop(drain_span);

        stream_result.map_err(MetalCommitError::Witness)?;
        gpu_result?;
        if send_failed {
            // Unreachable when gpu_result was Ok, but keep the pass honest.
            return Err(MetalCommitError::Device(MetalError::Execution(
                "GPU lane closed mid-pass".to_owned(),
            )));
        }

        // Assemble outputs in `kinds` order: one-hot columns from the
        // accumulated rows, increment columns through the shared-prep finish
        // (the optimized kernel's own call).
        let _finish_span = tracing::info_span!("MetalCommit::finish_columns").entered();
        let mut outputs: Vec<Option<(DoryCommitment, DoryHint)>> = vec![None; kinds.len()];
        for ((ids_index, _), (acc, column_rows)) in
            one_hot.iter().zip(accumulators.into_iter().zip(rows))
        {
            outputs[*ids_index] = Some(one_hot_output_from_rows(setup, column_rows, acc));
        }
        for ((ids_index, _, mut partial), rows) in consumer.increments.into_iter().zip(inc_rows) {
            if inc_device {
                // The CPU partial never fed; adopt the device-reduced rows.
                partial.row_commitments = rows;
            }
            outputs[ids_index] = Some(DoryScheme::finish_with_hint_prepared(
                partial, setup, prep_ref,
            ));
        }
        Ok(outputs
            .into_iter()
            .map(|output| output.expect("every column is one-hot or increment"))
            .collect())
    })
}
