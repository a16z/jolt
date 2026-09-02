//! The tier-2 Miller lane: batches finished tier-1 rows into device Miller
//! dispatches (tiled prepared-coefficient table by default, fly arm behind
//! [`miller_commit_fly`]), settles them asynchronously, and falls back to
//! CPU absorption per batch on any device failure.

use super::*;

/// Queued device pairs flush when they reach this count: dispatches batch
/// across superchunks — byte-free by the same partition invariance as the
/// CPU/device split itself. Both Miller kernels starve below ~32k-pair
/// dispatches (W7 fly retune −40% CB mass at 65536; W13 table scale
/// curve agrees); diminishing past that, and the stream-end drain batch
/// grows with the threshold. Override: `JOLT_METAL_MILLER_FLUSH_PAIRS`
/// (tests force mid-stream flushes with it).
const MILLER_FLUSH_PAIRS_DEFAULT: usize = 65536;

fn miller_flush_pairs() -> usize {
    std::env::var("JOLT_METAL_MILLER_FLUSH_PAIRS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(MILLER_FLUSH_PAIRS_DEFAULT)
}

pub(super) enum MillerInput {
    Table,
    Fly(Vec<G2Affine>),
}

pub(super) struct MillerLane<'b> {
    /// `Some` = fly arm (the wrapped affine G2 setup rows); `None` = the
    /// tiled prepared-coefficient table (default, [`miller_commit_fly`]).
    pub(super) fly_qs: Option<DeviceBuffer<'b>>,
    pub(super) cpu_share: f64,
    pub(super) failed: bool,
    pub(super) queue: MillerBatch,
    /// The committed-but-unsettled previous dispatch (see
    /// [`InFlightMiller`]); at most one, settled at the next flush point or
    /// at drain.
    pub(super) in_flight: Option<InFlightMiller>,
    /// Recycled tile backing (≤ 2: one building + one in flight).
    pub(super) tiles: Vec<MillerTile>,
}

/// `JOLT_METAL_MILLER_ASYNC=0` restores the synchronous settle (wait + fold
/// immediately after each Miller dispatch, the pre-W7 shape). Default on:
/// the tier-2 lane keeps decoding later superchunks while the device
/// crunches, instead of sleeping in `miller_wait` and backpressuring the
/// whole pipeline through the depth-2 queues.
fn miller_async() -> bool {
    static ASYNC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASYNC.get_or_init(|| std::env::var("JOLT_METAL_MILLER_ASYNC").as_deref() != Ok("0"))
}

/// One committed, un-waited Miller dispatch. `batch` and `tile` own the
/// backing memory of every wrapped input buffer (heap `Vec`s —
/// address-stable across moves); `batch` doubles as the CPU-recovery input
/// on a device error (its row indices stay TABLE-global — only the device
/// sees the tile remap); `out_buf` is device-allocated, read after the wait.
pub(super) struct InFlightMiller {
    batch: MillerBatch,
    tile: Option<MillerTile>,
    out_buf: DeviceBuffer<'static>,
    pending: crate::metal::runtime::DetachedPass,
    n_threads: usize,
}

/// One device dispatch's worth of queued pairs: per-thread segments never
/// straddle a column, and each fold entry maps a thread range back to its
/// column ordinal.
pub(super) struct MillerBatch {
    points: Vec<G1Affine>,
    row_indices: Vec<u32>,
    seg_starts: Vec<u32>,
    folds: Vec<(usize, usize, usize)>,
}

impl MillerBatch {
    pub(super) fn new() -> Self {
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

    const fn pairs_per_thread(&self) -> usize {
        match self.fly_qs {
            Some(_) => 1,
            None => MILLER_TABLE_SEG_PAIRS,
        }
    }

    /// On the table arm, gather `batch`'s rows into a recycled tile (see
    /// [`miller_commit_fly`]); `None` on the fly arm.
    fn build_tile(&mut self, prep: &DoryTier2Prep, batch: &MillerBatch) -> Option<MillerTile> {
        self.fly_qs.is_none().then(|| {
            let mut tile = self.tiles.pop().unwrap_or_default();
            tile.build(prep.prepared(), &batch.row_indices);
            tile
        })
    }
}

/// Wait on a committed Miller dispatch and fold its per-thread partials
/// into the per-column accumulators — or recover the whole batch on the
/// CPU (latching the lane off) when the device surfaced an error. The
/// merge order per column is dispatch order either way, and GT products
/// are exact field ops, so settle timing never changes a byte.
fn settle_in_flight(
    prep: &DoryTier2Prep,
    lane: &mut MillerLane<'_>,
    accumulators: &mut [Tier2Accumulator],
    in_flight: InFlightMiller,
) {
    let InFlightMiller {
        batch,
        tile,
        out_buf,
        pending,
        n_threads,
    } = in_flight;
    let waited = tracing::info_span!("MetalCommit::miller_wait").in_scope(|| pending.wait());
    lane.tiles.extend(tile);
    match waited {
        Ok(()) => {
            let _span = tracing::info_span!("MetalCommit::miller_fold").entered();
            // Shared storage: fold straight out of the device buffer.
            let partials: &[u32] = out_buf.typed_slice(n_threads * FQ12_U32S);
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
            absorb_batch_cpu(prep, accumulators, &batch);
        }
    }
}

/// The device-error recovery arm: absorb a batch's pairs through the CPU
/// path, column by column.
fn absorb_batch_cpu(
    prep: &DoryTier2Prep,
    accumulators: &mut [Tier2Accumulator],
    batch: &MillerBatch,
) {
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
/// device share queued and dispatched as one table or indexed-fly command
/// buffer per flush threshold, whose per-thread segments never straddle
/// columns. Dispatches settle (wait + per-column partial-product fold) one
/// flush later ([`InFlightMiller`]), so the device crunches a batch while
/// this lane decodes the following superchunks.
#[tracing::instrument(skip_all, name = "MetalCommit::tier2_absorb")]
pub(super) fn absorb_superchunk(
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
    let decode_span = tracing::info_span!("MetalCommit::tier2_decode").entered();
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
                    let point = crate::metal::g1::jac_from_device_limbs(jac);
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
    drop(decode_span);

    let cpu_absorb = |accumulators: &mut [Tier2Accumulator]| {
        let _span = tracing::info_span!("MetalCommit::tier2_cpu_absorb").entered();
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
            let pairs_per_thread = lane.pairs_per_thread();
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
                    at = (at + pairs_per_thread).min(queue.points.len());
                    queue.seg_starts.push(at as u32);
                }
                queue
                    .folds
                    .push((column, thread_start, queue.seg_starts.len() - 1));
            }
            if lane.queue.points.len() >= miller_flush_pairs() {
                let batch = lane.take_queue();
                let tile = lane.build_tile(prep, &batch);
                match commit_miller_dispatch(ctx, lane.fly_qs.as_ref(), tile, batch) {
                    Ok(dispatched) => {
                        // The new dispatch crunches while this superchunk's
                        // CPU share absorbs and the PREVIOUS dispatch —
                        // committed a full flush interval ago — settles
                        // with a near-zero residual wait.
                        cpu_absorb(accumulators);
                        if let Some(previous) = lane.in_flight.take() {
                            settle_in_flight(prep, lane, accumulators, previous);
                        }
                        if miller_async() && !lane.failed {
                            lane.in_flight = Some(dispatched);
                        } else {
                            settle_in_flight(prep, lane, accumulators, dispatched);
                        }
                    }
                    Err((batch, error)) => {
                        tracing::warn!(
                            %error,
                            "device miller dispatch failed; absorbing this batch and \
                             later superchunks on the CPU"
                        );
                        lane.failed = true;
                        cpu_absorb(accumulators);
                        if let Some(previous) = lane.in_flight.take() {
                            settle_in_flight(prep, lane, accumulators, previous);
                        }
                        absorb_batch_cpu(prep, accumulators, &batch);
                    }
                }
            } else {
                cpu_absorb(accumulators);
            }
        }
        None => cpu_absorb(accumulators),
    }
}

/// Settle the in-flight dispatch and flush whatever the lane still queues
/// (stream end, or a lane that never reached the flush threshold).
pub(super) fn drain_miller_lane(
    ctx: &MetalContext,
    prep: &DoryTier2Prep,
    lane: &mut MillerLane<'_>,
    accumulators: &mut [Tier2Accumulator],
) {
    if let Some(previous) = lane.in_flight.take() {
        settle_in_flight(prep, lane, accumulators, previous);
    }
    if lane.failed || lane.queue.points.is_empty() {
        return;
    }
    let batch = lane.take_queue();
    let tile = lane.build_tile(prep, &batch);
    match commit_miller_dispatch(ctx, lane.fly_qs.as_ref(), tile, batch) {
        Ok(dispatched) => settle_in_flight(prep, lane, accumulators, dispatched),
        Err((batch, error)) => {
            tracing::warn!(%error, "device miller drain failed; absorbing on the CPU");
            lane.failed = true;
            absorb_batch_cpu(prep, accumulators, &batch);
        }
    }
}

/// Encode and commit one device Miller dispatch without waiting. On any
/// encode/commit failure the batch comes back to the caller for CPU
/// recovery (the pre-built `tile` drops; recovery reads the batch's
/// table-global indices).
fn commit_miller_dispatch(
    ctx: &MetalContext,
    fly_qs: Option<&DeviceBuffer<'_>>,
    tile: Option<MillerTile>,
    batch: MillerBatch,
) -> Result<InFlightMiller, (MillerBatch, MetalError)> {
    let n_threads = batch.seg_starts.len() - 1;
    let committed = (|| {
        let ps_buf = ctx.wrap_slice(g1_points_as_u32s(&batch.points))?;
        let out_buf = ctx.alloc_u32s(n_threads * FQ12_U32S)?;
        let mut pass = ctx.begin_pass()?;
        // Per-arm wrappers may drop before commit — the encoder retains each
        // bound MTLBuffer; only the BACKING memory must outlive the pass.
        match (&tile, fly_qs) {
            (Some(tile), _) => {
                let params = miller_table_params(n_threads, tile.n_rows())?;
                let idx_buf = ctx.wrap_slice(&tile.local_rows)?;
                let seg_buf = ctx.wrap_slice(&batch.seg_starts)?;
                let coeffs_buf = ctx.wrap_slice(&tile.coeffs)?;
                pass.dispatch(
                    KernelId::MillerTable,
                    &params,
                    &[&ps_buf, &idx_buf, &seg_buf, &coeffs_buf, &out_buf],
                    n_threads,
                );
            }
            (None, Some(qs)) => {
                assert_eq!(n_threads, batch.points.len());
                let params = [u32::try_from(n_threads)
                    .map_err(|_| MetalError::Execution("pair count overflows u32".into()))?];
                let idx_buf = ctx.wrap_slice(&batch.row_indices)?;
                pass.dispatch(
                    KernelId::MillerFlyIndexed,
                    &params,
                    &[&ps_buf, &idx_buf, qs, &out_buf],
                    n_threads,
                );
            }
            (None, None) => unreachable!("table dispatch always builds a tile"),
        }
        // SAFETY: the returned `InFlightMiller` owns `batch` and `tile` —
        // the heap backing of every wrapped input buffer, address-stable
        // across moves, neither written nor dropped until
        // `settle_in_flight` waits — and holds `out_buf`, which the host
        // reads only after the wait.
        Ok((out_buf, unsafe { pass.commit().detach() }))
    })();
    match committed {
        Ok((out_buf, pending)) => {
            crate::metal::testing::note_miller_dispatch();
            Ok(InFlightMiller {
                batch,
                tile,
                out_buf,
                pending,
                n_threads,
            })
        }
        Err(error) => Err((batch, error)),
    }
}
