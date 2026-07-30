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
//! driver   [extract sc0|bucket|digit-recode][extract sc1|bucket|digit-recode]…
//! gpu      ......[G1 seg sums sc0]......[G1 seg sums sc1]......
//! tier-2   ............[Miller absorb sc0]...[Miller absorb sc1]…[final exps]
//! ```
//!
//! - The **driver** (calling thread) streams superchunks, derives each
//!   one-hot column's hot addresses, bucket-sorts them into per
//!   `(column, window, k)` gather segments, recodes the increment columns'
//!   i128 row scalars into signed base-256 digit buckets (per
//!   `(column, window, slot, |digit|)`, scalar and digit signs folded into
//!   the gather index's negation bit), and ships both segment families to
//!   the GPU lane as one job. Below the `commit_inc` gate the increment
//!   columns instead run the identical
//!   [`StreamingCommitment::feed_i128_rows`] calls the optimized kernel
//!   makes.
//! - The **GPU lane** (dedicated thread; command buffers are not `Send`)
//!   runs `jk_g1_seg_sum` over each superchunk — one command buffer, one
//!   dispatch per segment family: every segment thread sums its selected
//!   (possibly negated) SRS bases in Jacobian form. One-hot buckets are the
//!   work the CPU tier does with batch-affine additions; increment buckets
//!   are the Pippenger inner loop of the CPU tier's `msm_i128`.
//! - The **tier-2 lane** decodes finished segments, reduces multi-segment
//!   buckets, batch-normalizes, records the row commitments, and multiplies
//!   the rows' Miller loops into per-column [`Tier2Accumulator`]s (one final
//!   exponentiation per column at the end). Increment buckets it reduces
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
use jolt_witness::{stream_witnesses, JoltWitnessOracle, RowSource, StreamConsumer, WitnessError};
use rayon::prelude::*;

use super::g1::{bases_as_u32s, jac_from_device_limbs, JAC_U32S, SEG_INDEX_SIGN_BIT};
use super::miller::{
    flatten_prepared_coeffs, g1_points_as_u32s, miller_table_params, product_of_partials, FQ12_U32S,
};
use super::runtime::{KernelId, MetalContext};
use super::{metal_gate, DeviceBuffer, MetalError, PageAlignedVec};
use crate::commitment::{
    CommitWitness, CommitmentGrid, CommittedColumnsWitness, WitnessCommitment,
};
use crate::reference::commitment::{column_kinds, ColumnKind};
use crate::{KernelError, OptimizedBackend, ProofSession};

/// Cycles per superchunk — the optimized kernel's width, so both backends
/// stream the same window sequences.
const SUPERCHUNK_CYCLES: usize = 1 << 17;

/// Job/result queue depth. Two keeps each lane one superchunk ahead of the
/// next without unbounded buffering (a queued job is ~tens of MB of gather
/// indices at production widths).
const PIPELINE_DEPTH: usize = 2;

/// Per-thread segment cap: buckets larger than this split into several
/// device threads whose partial sums the tier-2 lane re-adds. Sized so an
/// average production bucket (`row_width / one_hot_k` = 512–1024 entries)
/// spans 2–4 threads — enough threads per superchunk (~20k) to fill the
/// device, enough serial adds per thread to amortize scheduling.
const MAX_SEGMENT_LEN: usize = 256;

/// Pairs per device Miller thread. The W6 microbench is occupancy-bound at
/// production pair counts — per-pair rate keeps improving down to 2 pairs
/// per thread (6.6 µs/pair vs 12.2 at 8) despite the extra per-thread
/// squaring ladders.
const MILLER_SEG_PAIRS: usize = 2;

/// Pairs per CPU absorb sub-shard. Small enough that rayon can steal a big
/// column off a straggler core mid-superchunk, large enough that the
/// per-shard squaring ladder stays noise (64 Fq12 squarings per shard
/// ≈ +1% of its line-fold work).
const MILLER_CPU_SHARD: usize = 64;

/// Default CPU share of each superchunk's Miller pairs (see
/// [`miller_cpu_fraction`]). The CPU cores idle once extraction is fed
/// while the device carries tier-1 + Miller; giving them a pair share
/// balances the lanes. Tier2Accumulator is partition-invariant, so ANY
/// split is byte-identical — the knob is pure load balance. In-pipeline
/// sweep (sha2-chain @2^22, stage-0 wall): 0 → 3.25 s, 0.3 → 3.15,
/// **0.5 → 2.82**, 0.65 → 3.34, 0.8 → 4.02, gate-closed → 4.78.
const MILLER_CPU_FRACTION_DEFAULT: f64 = 0.5;

/// `JOLT_METAL_MILLER_CPU_FRACTION` — CPU share of Miller pairs in
/// `[0, 1]`; `0` = all-device, `1` = all-CPU (the ablation arm). Read once
/// per commit pass.
fn miller_cpu_fraction() -> f64 {
    std::env::var("JOLT_METAL_MILLER_CPU_FRACTION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .map_or(MILLER_CPU_FRACTION_DEFAULT, |fraction| {
            fraction.clamp(0.0, 1.0)
        })
}

/// Signed base-256 digit slots covering any `i128` magnitude (16 bytes plus
/// a possible recoding carry into a 17th slot).
const INC_SLOTS: usize = 17;

/// Digit magnitudes after signed recoding span `1..=128`.
const INC_MAGNITUDES: usize = 128;

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
    seg_starts: PageAlignedVec<u32>,
    segs: Vec<IncSeg>,
}

/// A superchunk's device job: gather indices grouped by segment, the
/// segment prefix, and each segment's destination; plus the increment
/// columns' bucket family when they ride the device.
struct GpuJob {
    indices: PageAlignedVec<u32>,
    seg_starts: PageAlignedVec<u32>,
    segs: Vec<SegOut>,
    inc: Option<IncJob>,
}

/// A finished job: decoded destinations plus the raw Jacobian limb results,
/// per segment family.
struct GpuDone {
    segs: Vec<SegOut>,
    jac: Vec<u32>,
    inc: Option<(Vec<IncSeg>, Vec<u32>)>,
}

/// The streaming metal commit at explicit superchunk width and segment cap
/// (tests shrink both to force multi-delivery and multi-segment reduction).
/// Returns per-column outputs in `kinds` order. `inc_device` routes the
/// increment columns' row MSMs through the device as signed digit buckets;
/// off, they run the CPU `feed_i128_rows` path.
#[expect(
    clippy::expect_used,
    reason = "worker joins fail only on a panicked worker (which must propagate); the \
              output expect is the one-hot/increment kind exhaustiveness"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "internal seam; the trailing three are the test/ablation knobs"
)]
fn commit_streaming_metal(
    ctx: &'static MetalContext,
    source: &dyn RowSource,
    kinds: &[ColumnKind],
    grid: CommitmentGrid,
    setup: &DoryProverSetup,
    superchunk_cycles: usize,
    max_segment_len: usize,
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
        || DoryScheme::begin_one_hot_column_major_stream(setup, row_width),
    );
    assert!(
        !bases.par_iter().any(AffineRepr::is_zero),
        "SRS G1 bases must not contain the identity (the device view never \
         reads the infinity flag)"
    );

    // The device Miller lane's shared coefficient table: the SAME line
    // coefficients `prep` holds, flattened once per pass into the
    // step-major layout `jk_miller_table` streams (~16.7 KB per row; all
    // one-hot columns share it). Gated on the pass's total pair-eval count.
    let miller_cpu_share = miller_cpu_fraction();
    let miller_table: Option<Vec<u32>> =
        (metal_gate("miller", n_one_hot * one_hot_rows) && miller_cpu_share < 1.0).then(|| {
            let refs: Vec<&_> = prep.prepared().iter().collect();
            let mut table = Vec::new();
            flatten_prepared_coeffs(&refs, &mut table);
            table
        });

    std::thread::scope(|scope| {
        let (tx_jobs, rx_jobs) = sync_channel::<GpuJob>(PIPELINE_DEPTH);
        let (tx_done, rx_done) = sync_channel::<GpuDone>(PIPELINE_DEPTH);

        let bases_ref = &bases;
        let gpu = scope.spawn(move || -> Result<(), MetalError> {
            // Command buffers and MTLBuffers are not Send: every device
            // object lives and dies on this thread.
            let bases_buf = ctx.wrap_slice(bases_as_u32s(bases_ref))?;
            while let Ok(job) = rx_jobs.recv() {
                let n_segs = job.segs.len();
                let indices_buf = job.indices.device_buffer(ctx)?;
                let starts_buf = job.seg_starts.device_buffer(ctx)?;
                let out_buf = ctx.alloc_u32s(n_segs * JAC_U32S)?;
                let mut pass = ctx.begin_pass()?;
                pass.dispatch(
                    KernelId::G1SegSum,
                    &[u32::try_from(n_segs).map_err(|_| {
                        MetalError::Execution("segment count overflows u32".to_owned())
                    })?],
                    &[&bases_buf, &indices_buf, &starts_buf, &out_buf],
                    n_segs,
                );
                // The increment family joins the same command buffer: a
                // second dispatch over disjoint buffers, one wait for both.
                let inc_bufs = match &job.inc {
                    Some(inc) => {
                        let n_inc_segs = inc.segs.len();
                        let inc_indices_buf = inc.indices.device_buffer(ctx)?;
                        let inc_starts_buf = inc.seg_starts.device_buffer(ctx)?;
                        let inc_out_buf = ctx.alloc_u32s(n_inc_segs * JAC_U32S)?;
                        pass.dispatch(
                            KernelId::G1SegSum,
                            &[u32::try_from(n_inc_segs).map_err(|_| {
                                MetalError::Execution(
                                    "increment segment count overflows u32".to_owned(),
                                )
                            })?],
                            &[&bases_buf, &inc_indices_buf, &inc_starts_buf, &inc_out_buf],
                            n_inc_segs,
                        );
                        Some((n_inc_segs, inc_indices_buf, inc_starts_buf, inc_out_buf))
                    }
                    None => None,
                };
                pass.run()?;
                let mut jac = vec![0u32; n_segs * JAC_U32S];
                out_buf.copy_to_u32s(&mut jac);
                let inc_jac = inc_bufs.as_ref().map(|(n_inc_segs, _, _, inc_out_buf)| {
                    let mut inc_jac = vec![0u32; n_inc_segs * JAC_U32S];
                    inc_out_buf.copy_to_u32s(&mut inc_jac);
                    inc_jac
                });
                // Release the borrows on job.inc's buffers before moving its
                // segment table out.
                drop(inc_bufs);
                let inc_done = match (job.inc, inc_jac) {
                    (Some(inc), Some(inc_jac)) => Some((inc.segs, inc_jac)),
                    _ => None,
                };
                if tx_done
                    .send(GpuDone {
                        segs: job.segs,
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
        let miller_table_ref = miller_table.as_deref();
        type Tier2Out = (Vec<Tier2Accumulator>, Vec<Vec<Bn254G1>>, Vec<Vec<Bn254G1>>);
        let tier2 = scope.spawn(move || -> Tier2Out {
            let mut accumulators: Vec<Tier2Accumulator> =
                (0..n_one_hot).map(|_| Tier2Accumulator::new()).collect();
            let mut rows: Vec<Vec<Bn254G1>> = (0..n_one_hot)
                .map(|_| vec![Default::default(); one_hot_rows])
                .collect();
            // The Miller lane's device objects live and die on this thread
            // (MTLBuffers are not Send); a declined wrap only costs the
            // device share — the pass continues all-CPU.
            let mut miller_lane =
                miller_table_ref.and_then(|coeffs| match ctx.wrap_slice(coeffs) {
                    Ok(buffer) => Some(MillerLane {
                        coeffs: buffer,
                        n_rows: prep_ref.prepared().len(),
                        cpu_share: miller_cpu_share,
                        failed: false,
                        queue: MillerBatch::new(),
                    }),
                    Err(error) => {
                        tracing::warn!(%error, "miller table wrap failed; tier-2 stays on CPU");
                        None
                    }
                });
            // Increment row commitments by absolute window; windows whose
            // scalars are all zero receive no segments and stay identity —
            // exactly the CPU path's zero-MSM rows.
            let mut inc_rows: Vec<Vec<Bn254G1>> = (0..n_inc)
                .map(|_| vec![Default::default(); windows_total])
                .collect();
            while let Ok(done) = rx_done.recv() {
                absorb_superchunk(
                    ctx,
                    prep_ref,
                    &done,
                    &mut accumulators,
                    &mut rows,
                    miller_lane.as_mut(),
                );
                if let Some((inc_segs, inc_jac)) = &done.inc {
                    reduce_inc_superchunk(inc_segs, inc_jac, &mut inc_rows);
                }
            }
            if let Some(lane) = miller_lane.as_mut() {
                drain_miller_lane(ctx, prep_ref, lane, &mut accumulators);
            }
            (accumulators, rows, inc_rows)
        });

        let mut consumers = (MetalColumns {
            tx: Some(tx_jobs),
            send_failed: false,
            one_hot: &one_hot,
            increments: kinds
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, kind)| !kind.is_one_hot())
                .map(|(index, kind)| (index, kind, DoryScheme::begin(setup)))
                .collect(),
            inc_device,
            row_width,
            one_hot_k,
            windows_fed: 0,
            windows_total,
            max_segment_len,
            setup,
        },);
        let stream_result = stream_witnesses(
            source,
            0..cycles,
            row_width * windows_per_sc,
            &mut consumers,
        );
        let mut consumer = consumers.0;
        // Close the job queue so the GPU lane drains and exits; the tier-2
        // lane then drains the done queue and exits.
        consumer.tx = None;
        let gpu_result = gpu.join().expect("GPU lane panicked");
        let (accumulators, rows, inc_rows) = tier2.join().expect("tier-2 lane panicked");

        stream_result.map_err(MetalCommitError::Witness)?;
        gpu_result?;
        if consumer.send_failed {
            // Unreachable when gpu_result was Ok, but keep the pass honest.
            return Err(MetalCommitError::Device(MetalError::Execution(
                "GPU lane closed mid-pass".to_owned(),
            )));
        }

        // Assemble outputs in `kinds` order: one-hot columns from the
        // accumulated rows, increment columns through the shared-prep finish
        // (the optimized kernel's own call).
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

/// Queued device pairs flush when they reach this count: one superchunk's
/// share sits well under the device's ~4k-thread saturation knee at the
/// deeper geometries (2^23: ~2.5 k pairs/superchunk at the default split),
/// so dispatches batch across superchunks — byte-free by the same
/// partition invariance as the split itself. Override:
/// `JOLT_METAL_MILLER_FLUSH_PAIRS` (tests force mid-stream flushes with
/// it).
const MILLER_FLUSH_PAIRS_DEFAULT: usize = 8192;

fn miller_flush_pairs() -> usize {
    std::env::var("JOLT_METAL_MILLER_FLUSH_PAIRS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(MILLER_FLUSH_PAIRS_DEFAULT)
}

/// The tier-2 Miller device lane's pass-scoped state: the wrapped shared
/// coefficient table, the CPU pair share, and the cross-superchunk device
/// queue. `failed` latches on the first device error — the rest of the
/// pass absorbs on the CPU (the recovered batch included),
/// byte-identically.
struct MillerLane<'b> {
    coeffs: DeviceBuffer<'b>,
    n_rows: usize,
    cpu_share: f64,
    failed: bool,
    queue: MillerBatch,
}

/// One device dispatch's worth of queued pairs: per-thread segments never
/// straddle a column, and each fold entry maps a thread range back to its
/// column ordinal.
struct MillerBatch {
    points: Vec<G1Affine>,
    row_indices: Vec<u32>,
    seg_starts: Vec<u32>,
    folds: Vec<(usize, usize, usize)>,
}

impl MillerBatch {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            row_indices: Vec::new(),
            seg_starts: vec![0],
            folds: Vec::new(),
        }
    }
}

impl MillerLane<'_> {
    fn take_queue(&mut self) -> MillerBatch {
        std::mem::replace(&mut self.queue, MillerBatch::new())
    }
}

/// Fold a dispatched batch's partials into the per-column accumulators, or
/// recover the whole batch on the CPU (latching the lane off) when the
/// dispatch failed.
fn settle_miller_batch(
    prep: &DoryTier2Prep,
    lane: &mut MillerLane<'_>,
    accumulators: &mut [Tier2Accumulator],
    batch: &MillerBatch,
    dispatched: Result<Vec<u32>, MetalError>,
) {
    match dispatched {
        Ok(partials) => {
            let products: Vec<_> = batch
                .folds
                .par_iter()
                .map(|&(column, start, end)| {
                    (
                        column,
                        product_of_partials(&partials[start * FQ12_U32S..end * FQ12_U32S]),
                    )
                })
                .collect();
            for (column, product) in products {
                accumulators[column].merge_miller(product);
            }
        }
        Err(error) => {
            tracing::warn!(
                %error,
                "device miller dispatch failed; absorbing this batch and later \
                 superchunks on the CPU"
            );
            lane.failed = true;
            for &(column, thread_start, thread_end) in &batch.folds {
                let start = batch.seg_starts[thread_start] as usize;
                let end = batch.seg_starts[thread_end] as usize;
                accumulators[column].absorb(
                    prep,
                    &batch.points[start..end],
                    &batch.row_indices[start..end],
                );
            }
        }
    }
}

/// One column's decoded pair set, split for the hybrid absorb:
/// `[..cpu_len]` goes to the CPU `absorb`, the tail to the device kernel.
/// Any split point yields the same accumulator value (partition
/// invariance), so the share is purely a load-balance knob.
struct PendingPairs {
    points: Vec<G1Affine>,
    row_indices: Vec<u32>,
    cpu_len: usize,
}

/// Decode one finished superchunk: reduce multi-segment buckets, batch
/// normalize, record rows — parallel across columns — then absorb the
/// Miller mass: the CPU share through [`Tier2Accumulator::absorb`], the
/// device share as ONE `jk_miller_table` dispatch whose per-thread
/// segments never straddle columns, its execution overlapped with the CPU
/// share, per-column partial products folded in afterward.
fn absorb_superchunk(
    ctx: &MetalContext,
    prep: &DoryTier2Prep,
    done: &GpuDone,
    accumulators: &mut [Tier2Accumulator],
    rows: &mut [Vec<Bn254G1>],
    lane: Option<&mut MillerLane<'_>>,
) {
    // Segments arrive column-major (the driver emits buckets column by
    // column), so each column owns one contiguous segment range.
    let mut column_ranges: Vec<std::ops::Range<usize>> = vec![0..0; accumulators.len()];
    let mut cursor = 0usize;
    while cursor < done.segs.len() {
        let column = done.segs[cursor].column as usize;
        let start = cursor;
        while cursor < done.segs.len() && done.segs[cursor].column as usize == column {
            cursor += 1;
        }
        column_ranges[column] = start..cursor;
    }

    let (lane, cpu_share) = match lane {
        Some(lane) if !lane.failed => {
            let share = lane.cpu_share;
            (Some(lane), share)
        }
        _ => (None, 1.0),
    };

    // Decode phase, parallel across columns (each column's row vector is
    // disjoint): reduce consecutive same-row segments (split buckets) into
    // one point per row, drop identities (empty buckets never emit
    // segments, but a partial sum can still cancel to zero), batch
    // normalize, record rows.
    let pending: Vec<Option<PendingPairs>> =
        rows.par_iter_mut()
            .zip(column_ranges)
            .map(|(column_rows, range)| {
                if range.is_empty() {
                    return None;
                }
                let mut reduced: Vec<(u32, G1Projective)> = Vec::with_capacity(range.len());
                for (seg, jac) in done.segs[range.clone()].iter().zip(
                    done.jac[range.start * JAC_U32S..range.end * JAC_U32S].chunks_exact(JAC_U32S),
                ) {
                    let point = super::g1::jac_from_device_limbs(jac);
                    match reduced.last_mut() {
                        Some((row, sum)) if *row == seg.row => *sum += point,
                        _ => reduced.push((seg.row, point)),
                    }
                }
                reduced.retain(|(_, point)| !point.is_zero());
                if reduced.is_empty() {
                    return None;
                }
                let points: Vec<G1Projective> = reduced.iter().map(|(_, point)| *point).collect();
                let affine = G1Projective::normalize_batch(&points);
                let row_indices: Vec<u32> = reduced.iter().map(|(row, _)| *row).collect();
                for (&row, point) in row_indices.iter().zip(&affine) {
                    column_rows[row as usize] = Bn254G1::from(G1Projective::from(*point));
                }
                #[expect(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "pair counts are tiny; the split point is a load-balance knob"
                )]
                let cpu_len = ((affine.len() as f64) * cpu_share).round() as usize;
                Some(PendingPairs {
                    points: affine,
                    row_indices,
                    cpu_len: cpu_len.min(reduced.len()),
                })
            })
            .collect();

    let cpu_absorb = |accumulators: &mut [Tier2Accumulator]| {
        // (column, sub-shard) tasks instead of one task per column: the
        // fresh absorb loop is serial within a call, so a big column on a
        // straggler core would stretch the closure past the dispatch
        // window. Shards absorb independently and merge per column —
        // partition-invariant, so the value is unchanged.
        let tasks: Vec<(usize, &PendingPairs, std::ops::Range<usize>)> = pending
            .iter()
            .enumerate()
            .filter_map(|(column, pending_column)| pending_column.as_ref().map(|p| (column, p)))
            .flat_map(|(column, p)| {
                (0..p.cpu_len)
                    .step_by(MILLER_CPU_SHARD)
                    .map(move |at| (column, p, at..(at + MILLER_CPU_SHARD).min(p.cpu_len)))
            })
            .collect();
        let partials: Vec<(usize, Tier2Accumulator)> = tasks
            .into_par_iter()
            .map(|(column, p, range)| {
                let mut shard = Tier2Accumulator::new();
                shard.absorb(prep, &p.points[range.clone()], &p.row_indices[range]);
                (column, shard)
            })
            .collect();
        for (column, shard) in partials {
            accumulators[column].merge(shard);
        }
    };

    // Queue the device share — every column's tail slice, per-thread
    // segments kept inside their column — and flush once enough pairs have
    // accumulated to occupy the device, overlapping the dispatch with this
    // superchunk's CPU share.
    match lane {
        Some(lane) => {
            for (column, pending_column) in pending.iter().enumerate() {
                let Some(p) = pending_column else { continue };
                if p.cpu_len == p.points.len() {
                    continue;
                }
                let queue = &mut lane.queue;
                let thread_start = queue.seg_starts.len() - 1;
                queue.points.extend_from_slice(&p.points[p.cpu_len..]);
                queue
                    .row_indices
                    .extend_from_slice(&p.row_indices[p.cpu_len..]);
                let mut at = queue.seg_starts[thread_start] as usize;
                while at < queue.points.len() {
                    at = (at + MILLER_SEG_PAIRS).min(queue.points.len());
                    queue.seg_starts.push(at as u32);
                }
                queue
                    .folds
                    .push((column, thread_start, queue.seg_starts.len() - 1));
            }
            if lane.queue.points.len() >= miller_flush_pairs() {
                let batch = lane.take_queue();
                let dispatched = run_miller_dispatch(
                    ctx,
                    &lane.coeffs,
                    lane.n_rows,
                    &batch.points,
                    &batch.row_indices,
                    &batch.seg_starts,
                    || cpu_absorb(accumulators),
                );
                settle_miller_batch(prep, lane, accumulators, &batch, dispatched);
            } else {
                cpu_absorb(accumulators);
            }
        }
        None => cpu_absorb(accumulators),
    }
}

/// Flush whatever the lane still queues (stream end, or a lane that never
/// reached the flush threshold).
fn drain_miller_lane(
    ctx: &MetalContext,
    prep: &DoryTier2Prep,
    lane: &mut MillerLane<'_>,
    accumulators: &mut [Tier2Accumulator],
) {
    if lane.failed || lane.queue.points.is_empty() {
        return;
    }
    let batch = lane.take_queue();
    let dispatched = run_miller_dispatch(
        ctx,
        &lane.coeffs,
        lane.n_rows,
        &batch.points,
        &batch.row_indices,
        &batch.seg_starts,
        || {},
    );
    settle_miller_batch(prep, lane, accumulators, &batch, dispatched);
}

/// Encode, commit, and collect one `jk_miller_table` dispatch, running
/// `overlap` (the CPU pair share) exactly once on every path — after the
/// commit when the device accepted the work (so both proceed
/// concurrently), before returning when setup failed (so no work is lost).
fn run_miller_dispatch(
    ctx: &MetalContext,
    coeffs: &DeviceBuffer<'_>,
    n_rows: usize,
    points: &[G1Affine],
    row_indices: &[u32],
    seg_starts: &[u32],
    overlap: impl FnOnce(),
) -> Result<Vec<u32>, MetalError> {
    let n_threads = seg_starts.len() - 1;
    let setup = (|| {
        Ok((
            ctx.wrap_slice(g1_points_as_u32s(points))?,
            ctx.wrap_slice(row_indices)?,
            ctx.wrap_slice(seg_starts)?,
            ctx.alloc_u32s(n_threads * FQ12_U32S)?,
            miller_table_params(n_threads, n_rows)?,
        ))
    })();
    let (ps_buf, idx_buf, seg_buf, out_buf, params) = match setup {
        Ok(buffers) => buffers,
        Err(error) => {
            overlap();
            return Err(error);
        }
    };
    let pending = match ctx.begin_pass() {
        Ok(mut pass) => {
            pass.dispatch(
                KernelId::MillerTable,
                &params,
                &[&ps_buf, &idx_buf, &seg_buf, coeffs, &out_buf],
                n_threads,
            );
            pass.commit()
        }
        Err(error) => {
            overlap();
            return Err(error);
        }
    };
    super::testing::note_miller_dispatch();
    overlap();
    pending.wait()?;
    let mut partials = vec![0u32; n_threads * FQ12_U32S];
    out_buf.copy_to_u32s(&mut partials);
    Ok(partials)
}

/// The driver-side stream consumer: buckets one-hot hot addresses into
/// device jobs and advances the increment columns on the CPU.
struct MetalColumns<'a> {
    tx: Option<SyncSender<GpuJob>>,
    send_failed: bool,
    one_hot: &'a [(usize, ColumnKind)],
    increments: Vec<(usize, ColumnKind, DoryPartialCommitment)>,
    inc_device: bool,
    row_width: usize,
    one_hot_k: usize,
    windows_fed: usize,
    windows_total: usize,
    max_segment_len: usize,
    setup: &'a DoryProverSetup,
}

impl StreamConsumer for MetalColumns<'_> {
    type Witness = CommittedColumnsWitness;

    fn consume(&mut self, chunk: &[CommittedColumnsWitness]) {
        debug_assert!(
            chunk.len().is_multiple_of(self.row_width),
            "superchunk must be whole windows"
        );
        let n_windows = chunk.len() / self.row_width;
        let window_base = self.windows_fed;
        self.windows_fed += n_windows;

        if let Some(tx) = &self.tx {
            if !self.send_failed {
                let mut job = build_gpu_job(
                    chunk,
                    self.one_hot,
                    self.row_width,
                    self.one_hot_k,
                    window_base,
                    self.windows_total,
                    self.max_segment_len,
                );
                if self.inc_device && !self.increments.is_empty() {
                    let increments: Vec<Vec<i128>> = self
                        .increments
                        .iter()
                        .map(|(_, kind, _)| chunk.iter().map(|row| kind.increment(row)).collect())
                        .collect();
                    job.inc = build_inc_job(
                        &increments,
                        self.row_width,
                        window_base,
                        self.max_segment_len,
                    );
                }
                if tx.send(job).is_err() {
                    // Device lane died; its join surfaces the error after
                    // the stream completes.
                    self.send_failed = true;
                }
            }
        }

        if !self.inc_device {
            for (_, kind, partial) in &mut self.increments {
                let increments: Vec<i128> = chunk.iter().map(|row| kind.increment(row)).collect();
                DoryScheme::feed_i128_rows(partial, &increments, self.row_width, self.setup);
            }
        }
    }
}

/// Bucket a superchunk's one-hot hot addresses into gather segments:
/// count-sort per `(column, window)` into a flat index array (bucket-major,
/// columns outermost), then split buckets at the segment cap.
fn build_gpu_job(
    chunk: &[CommittedColumnsWitness],
    one_hot: &[(usize, ColumnKind)],
    row_width: usize,
    one_hot_k: usize,
    window_base: usize,
    windows_total: usize,
    max_segment_len: usize,
) -> GpuJob {
    let n_windows = chunk.len() / row_width;
    let n_cw = one_hot.len() * n_windows;
    let buckets_per_cw = one_hot_k;

    // Hot addresses per column — the same per-column derivation the CPU
    // kernel feeds `process_one_hot_chunks`.
    let hot: Vec<Vec<Option<usize>>> = one_hot
        .par_iter()
        .map(|(_, kind)| chunk.iter().map(|row| kind.hot_address(row)).collect())
        .collect();

    // Bucket counts, (column, window) blocks of `one_hot_k` each.
    let mut counts = vec![0u32; n_cw * buckets_per_cw];
    counts
        .par_chunks_mut(buckets_per_cw)
        .enumerate()
        .for_each(|(cw, counts)| {
            let (column, window) = (cw / n_windows, cw % n_windows);
            for hot_row in hot[column][window * row_width..(window + 1) * row_width]
                .iter()
                .flatten()
            {
                counts[*hot_row] += 1;
            }
        });

    // Exclusive prefix over all buckets → gather layout offsets.
    let mut bucket_starts = vec![0u32; counts.len() + 1];
    for (i, &count) in counts.iter().enumerate() {
        bucket_starts[i + 1] = bucket_starts[i] + count;
    }
    let total_hot = bucket_starts[counts.len()] as usize;

    // Scatter window-local base positions into their buckets. Each
    // (column, window) block owns a disjoint `one_hot_k` bucket range, so
    // per-block cursors need no synchronization.
    let mut indices = PageAlignedVec::from_elem(0u32, total_hot.max(1));
    {
        let indices = &mut indices[..];
        // Split the flat array at block boundaries so blocks fill in
        // parallel without synchronization.
        let mut blocks: Vec<&mut [u32]> = Vec::with_capacity(n_cw);
        let mut rest = indices;
        for cw in 0..n_cw {
            let start = bucket_starts[cw * buckets_per_cw] as usize;
            let end = bucket_starts[(cw + 1) * buckets_per_cw] as usize;
            let (block, tail) = rest.split_at_mut(end - start);
            blocks.push(block);
            rest = tail;
        }
        blocks.par_iter_mut().enumerate().for_each(|(cw, block)| {
            let (column, window) = (cw / n_windows, cw % n_windows);
            let base = bucket_starts[cw * buckets_per_cw];
            let mut cursors: Vec<u32> = (0..buckets_per_cw)
                .map(|k| bucket_starts[cw * buckets_per_cw + k] - base)
                .collect();
            for (position, hot_row) in hot[column][window * row_width..(window + 1) * row_width]
                .iter()
                .enumerate()
            {
                if let Some(hot_row) = hot_row {
                    block[cursors[*hot_row] as usize] = position as u32;
                    cursors[*hot_row] += 1;
                }
            }
        });
    }

    // Segment table: split buckets at the cap; empty buckets emit nothing
    // (their rows stay identity, exactly the CPU path's empty-index rows).
    let mut seg_starts_host = vec![0u32];
    let mut segs = Vec::new();
    for cw in 0..n_cw {
        let (column, window) = (cw / n_windows, cw % n_windows);
        for k in 0..buckets_per_cw {
            let start = bucket_starts[cw * buckets_per_cw + k] as usize;
            let end = bucket_starts[cw * buckets_per_cw + k + 1] as usize;
            let mut cursor = start;
            while cursor < end {
                let seg_end = (cursor + max_segment_len).min(end);
                seg_starts_host.push(seg_end as u32);
                segs.push(SegOut {
                    column: column as u32,
                    row: (k * windows_total + window_base + window) as u32,
                });
                cursor = seg_end;
            }
        }
    }

    GpuJob {
        indices,
        seg_starts: PageAlignedVec::from_slice(&seg_starts_host),
        segs,
        inc: None,
    }
}

/// Bucket a superchunk's increment scalars into signed-digit gather
/// segments: per `(column, window)` block, count-sort each scalar's signed
/// base-256 digits into `(slot, magnitude)` buckets whose entries are
/// window-local base positions (bit 31 = negated base), then split buckets
/// at the segment cap. `None` when every scalar is zero (no device work;
/// the rows stay identity).
///
/// `increments` holds one whole-superchunk scalar vector per increment
/// column; `window_base` makes the emitted windows absolute.
fn build_inc_job(
    increments: &[Vec<i128>],
    row_width: usize,
    window_base: usize,
    max_segment_len: usize,
) -> Option<IncJob> {
    assert!(
        row_width < SEG_INDEX_SIGN_BIT as usize,
        "row width must leave the gather sign bit free"
    );
    let n_windows = increments
        .first()
        .map_or(0, |column| column.len() / row_width);
    let n_blocks = increments.len() * n_windows;
    if n_blocks == 0 {
        return None;
    }
    let buckets_per_block = INC_SLOTS * INC_MAGNITUDES;

    // Bucket counts, (column, window) blocks of `buckets_per_block` each.
    let mut counts = vec![0u32; n_blocks * buckets_per_block];
    counts
        .par_chunks_mut(buckets_per_block)
        .enumerate()
        .for_each(|(block, counts)| {
            let (column, window) = (block / n_windows, block % n_windows);
            for &value in &increments[column][window * row_width..(window + 1) * row_width] {
                for_each_signed_digit(value, |slot, magnitude, _| {
                    counts[slot as usize * INC_MAGNITUDES + (magnitude - 1) as usize] += 1;
                });
            }
        });

    // Exclusive prefix over all buckets → gather layout offsets.
    let mut bucket_starts = vec![0u32; counts.len() + 1];
    for (i, &count) in counts.iter().enumerate() {
        bucket_starts[i + 1] = bucket_starts[i] + count;
    }
    let total_digits = bucket_starts[counts.len()] as usize;
    if total_digits == 0 {
        return None;
    }

    // Scatter window-local base positions (plus sign) into their buckets;
    // blocks own disjoint bucket ranges, so they fill in parallel.
    let mut indices = PageAlignedVec::from_elem(0u32, total_digits);
    {
        let indices = &mut indices[..];
        let mut blocks: Vec<&mut [u32]> = Vec::with_capacity(n_blocks);
        let mut rest = indices;
        for block in 0..n_blocks {
            let start = bucket_starts[block * buckets_per_block] as usize;
            let end = bucket_starts[(block + 1) * buckets_per_block] as usize;
            let (head, tail) = rest.split_at_mut(end - start);
            blocks.push(head);
            rest = tail;
        }
        blocks.par_iter_mut().enumerate().for_each(|(block, out)| {
            let (column, window) = (block / n_windows, block % n_windows);
            let base = bucket_starts[block * buckets_per_block];
            let mut cursors: Vec<u32> = (0..buckets_per_block)
                .map(|bucket| bucket_starts[block * buckets_per_block + bucket] - base)
                .collect();
            for (position, &value) in increments[column]
                [window * row_width..(window + 1) * row_width]
                .iter()
                .enumerate()
            {
                for_each_signed_digit(value, |slot, magnitude, negate| {
                    let bucket = slot as usize * INC_MAGNITUDES + (magnitude - 1) as usize;
                    let sign = if negate { SEG_INDEX_SIGN_BIT } else { 0 };
                    out[cursors[bucket] as usize] = position as u32 | sign;
                    cursors[bucket] += 1;
                });
            }
        });
    }

    // Segment table: split buckets at the cap; empty buckets emit nothing.
    let mut seg_starts = vec![0u32];
    let mut segs = Vec::new();
    for block in 0..n_blocks {
        let (column, window) = (block / n_windows, block % n_windows);
        for slot in 0..INC_SLOTS {
            for magnitude in 1..=INC_MAGNITUDES {
                let bucket = block * buckets_per_block + slot * INC_MAGNITUDES + magnitude - 1;
                let end = bucket_starts[bucket + 1] as usize;
                let mut cursor = bucket_starts[bucket] as usize;
                while cursor < end {
                    let seg_end = (cursor + max_segment_len).min(end);
                    seg_starts.push(seg_end as u32);
                    segs.push(IncSeg {
                        column: column as u32,
                        window: (window_base + window) as u32,
                        slot: slot as u8,
                        magnitude: magnitude as u8,
                    });
                    cursor = seg_end;
                }
            }
        }
    }

    Some(IncJob {
        indices,
        seg_starts: PageAlignedVec::from_slice(&seg_starts),
        segs,
    })
}

/// Decode one superchunk's increment segment sums into row commitments.
/// Per `(column, window)` group (a contiguous run of segments sorted by
/// ascending `(slot, magnitude)`): merge split buckets, form each slot's
/// weighted total `T_slot = Σ_m m · S_m` by a descending running sum, and
/// combine slots by base-256 Horner — `row = Σ_slot 256^slot · T_slot`,
/// the exact MSM group value of the window's scalars.
fn reduce_inc_superchunk(segs: &[IncSeg], jac: &[u32], inc_rows: &mut [Vec<Bn254G1>]) {
    let mut groups: Vec<(usize, usize, std::ops::Range<usize>)> = Vec::new();
    let mut cursor = 0usize;
    while cursor < segs.len() {
        let (column, window) = (segs[cursor].column as usize, segs[cursor].window as usize);
        let start = cursor;
        while cursor < segs.len()
            && segs[cursor].column as usize == column
            && segs[cursor].window as usize == window
        {
            cursor += 1;
        }
        groups.push((column, window, start..cursor));
    }

    let rows: Vec<(usize, usize, Bn254G1)> = groups
        .into_par_iter()
        .map(|(column, window, range)| {
            let mut row = G1Projective::zero();
            let mut cursor = range.end;
            for slot in (0..INC_SLOTS).rev() {
                // Horner shift for the slots already combined above this one.
                if !row.is_zero() {
                    for _ in 0..8 {
                        row = row.double();
                    }
                }
                let slot_end = cursor;
                while cursor > range.start && segs[cursor - 1].slot as usize == slot {
                    cursor -= 1;
                }
                if cursor == slot_end {
                    continue;
                }
                // Descending running sum: after bucket m folds into `acc`,
                // every later `total += acc` re-adds it — m times in all.
                // Identity-accumulator adds early-out, so the dense loop
                // costs real adds only below the largest live magnitude.
                let slot_segs = &segs[cursor..slot_end];
                let mut acc = G1Projective::zero();
                let mut total = G1Projective::zero();
                let mut next = slot_segs.len();
                for magnitude in (1..=INC_MAGNITUDES as u8).rev() {
                    while next > 0 && slot_segs[next - 1].magnitude == magnitude {
                        let seg = cursor + next - 1;
                        acc += jac_from_device_limbs(&jac[seg * JAC_U32S..(seg + 1) * JAC_U32S]);
                        next -= 1;
                    }
                    if !acc.is_zero() {
                        total += acc;
                    }
                }
                row += total;
            }
            (column, window, Bn254G1::from(row))
        })
        .collect();
    for (column, window, point) in rows {
        inc_rows[column][window] = point;
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests: fail loudly"
)]
mod tests {
    use ark_bn254::G1Affine;
    use ark_ff::{Field, UniformRand};
    use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
    use jolt_witness::RowSource;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::super::g1::g1_seg_sum_dispatch;
    use super::super::testing::gpu_lock;
    use super::*;
    use crate::optimized::testing::{with_ram_fixture, FixtureShape, RamOp};

    /// Interesting signed-recoding inputs: carry chains, magnitude-128
    /// boundaries, and the extreme i128s (`unsigned_abs` of `i128::MIN`).
    const EDGE_VALUES: [i128; 20] = [
        0,
        1,
        -1,
        127,
        128,
        129,
        -127,
        -128,
        -129,
        255,
        256,
        257,
        -255,
        -256,
        65535,
        -65536,
        i64::MAX as i128,
        i64::MIN as i128,
        i128::MAX,
        i128::MIN,
    ];

    /// The recoding is a correct signed-digit decomposition: digits stay in
    /// range and reconstruct the value in the scalar field.
    #[test]
    fn signed_digits_reconstruct() {
        let check = |value: i128| {
            let mut sum = ark_bn254::Fr::from(0u64);
            let mut digits = 0usize;
            for_each_signed_digit(value, |slot, magnitude, negate| {
                assert!((1..=INC_MAGNITUDES as u32).contains(&magnitude), "{value}");
                assert!((slot as usize) < INC_SLOTS, "{value}");
                let term = ark_bn254::Fr::from(u64::from(magnitude))
                    * ark_bn254::Fr::from(256u64).pow([u64::from(slot)]);
                if negate {
                    sum -= term;
                } else {
                    sum += term;
                }
                digits += 1;
            });
            assert_eq!(sum, ark_bn254::Fr::from(value), "value {value}");
            if value == 0 {
                assert_eq!(digits, 0);
            }
        };
        for value in EDGE_VALUES {
            check(value);
        }
        let mut rng = ChaCha20Rng::seed_from_u64(23);
        for _ in 0..500 {
            let value = ((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())) as i128;
            check(value >> (rng.next_u32() % 128));
        }
    }

    /// Full increment path against arkworks: recode + bucket, device segment
    /// sums (split at a tiny cap), weighted reduction — every window's row
    /// equals the direct per-scalar MSM, at a nonzero window base.
    #[test]
    fn inc_rows_match_direct_msm() {
        let _lock = gpu_lock();
        let ctx = MetalContext::global().expect("metal context");
        let mut rng = ChaCha20Rng::seed_from_u64(29);
        let row_width = 8usize;
        let n_windows = 3usize;
        let window_base = 5usize;
        let bases: Vec<G1Affine> = (0..row_width).map(|_| G1Affine::rand(&mut rng)).collect();

        // Column 0: the edge values (padded with a repeat digit-collision
        // value to force bucket splits). Column 1: random full-range i128s.
        let mut columns = vec![Vec::new(), Vec::new()];
        columns[0].extend_from_slice(&EDGE_VALUES[..row_width * n_windows - 4]);
        columns[0].extend([42i128; 4]);
        for _ in 0..row_width * n_windows {
            columns[1]
                .push(((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())) as i128);
        }

        let inc = build_inc_job(&columns, row_width, window_base, 2).expect("nonzero scalars");
        let n_segs = inc.segs.len();
        let bases_buf = ctx.wrap_slice(bases_as_u32s(&bases)).unwrap();
        let indices_buf = inc.indices.device_buffer(ctx).unwrap();
        let starts_buf = inc.seg_starts.device_buffer(ctx).unwrap();
        let out_buf = ctx.alloc_u32s(n_segs * JAC_U32S).unwrap();
        g1_seg_sum_dispatch(ctx, &bases_buf, &indices_buf, &starts_buf, &out_buf, n_segs)
            .expect("dispatch");
        let mut jac = vec![0u32; n_segs * JAC_U32S];
        out_buf.copy_to_u32s(&mut jac);

        let mut inc_rows: Vec<Vec<Bn254G1>> =
            vec![vec![Default::default(); window_base + n_windows]; columns.len()];
        reduce_inc_superchunk(&inc.segs, &jac, &mut inc_rows);

        for (column, values) in columns.iter().enumerate() {
            for window in 0..n_windows {
                let expected = values[window * row_width..(window + 1) * row_width]
                    .iter()
                    .zip(&bases)
                    .fold(G1Projective::zero(), |acc, (&value, base)| {
                        acc + *base * ark_bn254::Fr::from(value)
                    });
                assert_eq!(
                    inc_rows[column][window_base + window],
                    Bn254G1::from(expected),
                    "column {column} window {window}"
                );
            }
            for row in &inc_rows[column][..window_base] {
                assert_eq!(*row, Bn254G1::default(), "untouched row");
            }
        }
    }

    fn assert_same(
        cpu: &[WitnessCommitment<DoryScheme>],
        device: &[(DoryCommitment, DoryHint)],
        label: &str,
    ) {
        assert_eq!(cpu.len(), device.len());
        for (cpu, (commitment, hint)) in cpu.iter().zip(device) {
            assert_eq!(
                &cpu.commitment, commitment,
                "{label}: {:?} commitment diverged",
                cpu.id
            );
            assert_eq!(&cpu.hint, hint, "{label}: {:?} hint diverged", cpu.id);
        }
    }

    /// The device pipeline must reproduce the optimized kernel's commitments
    /// and hints exactly: whole-trace superchunks with production segment
    /// caps, and single-window superchunks with a 1-entry segment cap (every
    /// addition its own device thread — the deepest multi-segment reduction
    /// and multi-delivery sequencing). The Miller gate is forced open, so
    /// every arm also exercises the hybrid tier-2 absorb at the default CPU
    /// share; dedicated arms pin the all-device and all-CPU extremes
    /// (partition invariance makes every split byte-identical).
    #[test]
    fn metal_commit_matches_optimized() {
        let _lock = gpu_lock();
        // nextest runs one process per test, so the env writes cannot race
        // another test. The tiny flush threshold forces mid-stream batch
        // flushes (production only reaches them at deep geometries).
        std::env::set_var("JOLT_METAL_MIN_TERMS_MILLER", "1");
        std::env::set_var("JOLT_METAL_MILLER_FLUSH_PAIRS", "8");
        let shape = FixtureShape {
            log_t: 6,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 2, post: 17 },
            RamOp::Read { word: 2 },
            RamOp::None,
            RamOp::Write { word: 5, post: 3 },
            RamOp::Read { word: 5 },
            RamOp::Write { word: 2, post: 9 },
            RamOp::Read { word: 3 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let ids: Vec<JoltCommittedPolynomial> = witness
                .committed_order()
                .unwrap()
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
            let grid = CommitmentGrid {
                total_vars: 4 + shape.log_t,
                log_t: shape.log_t,
                log_k_chunk: 4,
                order: TracePolynomialOrder::CycleMajor,
            };
            let setup = DoryScheme::setup_prover(grid.total_vars);
            let source: &dyn RowSource = witness;
            let kinds = column_kinds::<Fr>(&ids, grid).unwrap();
            let ctx = MetalContext::global().expect("metal context");

            let optimized = <OptimizedBackend as CommitWitness<Fr, DoryScheme>>::commit_witness(
                &OptimizedBackend,
                &mut ProofSession::default(),
                source,
                &ids,
                grid,
                &setup,
            )
            .unwrap();

            let miller_dispatches = super::super::testing::miller_dispatch_count();
            let whole_trace = commit_streaming_metal(
                ctx,
                source,
                &kinds,
                grid,
                &setup,
                1 << shape.log_t,
                MAX_SEGMENT_LEN,
                true,
            )
            .expect("whole-trace metal commit");
            assert_same(&optimized, &whole_trace, "whole-trace superchunk");
            assert!(
                super::super::testing::miller_dispatch_count() > miller_dispatches,
                "the hybrid absorb never dispatched a device Miller batch"
            );

            let single_window = commit_streaming_metal(
                ctx,
                source,
                &kinds,
                grid,
                &setup,
                grid.num_columns(),
                1,
                true,
            )
            .expect("single-window metal commit");
            assert_same(&optimized, &single_window, "single-window superchunk");

            let inc_on_cpu = commit_streaming_metal(
                ctx,
                source,
                &kinds,
                grid,
                &setup,
                1 << shape.log_t,
                MAX_SEGMENT_LEN,
                false,
            )
            .expect("cpu-increment metal commit");
            assert_same(&optimized, &inc_on_cpu, "cpu-increment fallback");

            // The split extremes: all pairs on device, then all on CPU
            // (which also skips the table build entirely).
            std::env::set_var("JOLT_METAL_MILLER_CPU_FRACTION", "0");
            let all_device = commit_streaming_metal(
                ctx,
                source,
                &kinds,
                grid,
                &setup,
                1 << shape.log_t,
                MAX_SEGMENT_LEN,
                true,
            )
            .expect("all-device miller commit");
            assert_same(&optimized, &all_device, "all-device miller split");

            std::env::set_var("JOLT_METAL_MILLER_CPU_FRACTION", "1");
            let all_cpu = commit_streaming_metal(
                ctx,
                source,
                &kinds,
                grid,
                &setup,
                1 << shape.log_t,
                MAX_SEGMENT_LEN,
                true,
            )
            .expect("all-cpu miller commit");
            assert_same(&optimized, &all_cpu, "all-cpu miller split");
            std::env::remove_var("JOLT_METAL_MILLER_CPU_FRACTION");
        });
    }

    /// The full slot path: with the gate forced open, `MetalCommitWitness`
    /// routes through the device and matches the optimized kernel; advice
    /// stays delegated.
    #[test]
    fn metal_slot_matches_optimized_through_gate() {
        let _lock = gpu_lock();
        // nextest runs one process per test, so the env writes cannot race
        // another test.
        std::env::set_var("JOLT_METAL_MIN_TERMS_COMMIT", "1");
        std::env::set_var("JOLT_METAL_MIN_TERMS_COMMIT_INC", "1");
        let shape = FixtureShape {
            log_t: 6,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::Write { word: 1, post: 5 },
            RamOp::Read { word: 1 },
            RamOp::Write { word: 7, post: 2 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let ids: Vec<JoltCommittedPolynomial> = witness
                .committed_order()
                .unwrap()
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
            let grid = CommitmentGrid {
                total_vars: 4 + shape.log_t,
                log_t: shape.log_t,
                log_k_chunk: 4,
                order: TracePolynomialOrder::CycleMajor,
            };
            let setup = DoryScheme::setup_prover(grid.total_vars);
            let source: &dyn RowSource = witness;

            let optimized = <OptimizedBackend as CommitWitness<Fr, DoryScheme>>::commit_witness(
                &OptimizedBackend,
                &mut ProofSession::default(),
                source,
                &ids,
                grid,
                &setup,
            )
            .unwrap();
            let metal = MetalCommitWitness
                .commit_witness(&mut ProofSession::default(), source, &ids, grid, &setup)
                .unwrap();

            assert_eq!(optimized.len(), metal.len());
            for (cpu, device) in optimized.iter().zip(&metal) {
                assert_eq!(cpu.id, device.id);
                assert_eq!(cpu.commitment, device.commitment, "{:?}", cpu.id);
                assert_eq!(cpu.hint, device.hint, "{:?}", cpu.id);
            }
        });
    }
}
