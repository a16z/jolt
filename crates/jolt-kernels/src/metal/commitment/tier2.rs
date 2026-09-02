//! The tier-2 Miller lane: batches finished tier-1 rows into device Miller
//! dispatches (`jk_miller_table`, 2.2× less ALU per pair than the on-the-fly
//! ladder, over per-flush coefficient tiles), settles them asynchronously,
//! and falls back to CPU absorption per batch on any device failure.

use super::*;

pub(super) struct MillerLane {
    /// Queued pairs flush into one device dispatch at this count
    /// ([`MILLER_FLUSH_PAIRS`] in production; tests shrink it to force
    /// mid-stream flushes).
    pub(super) flush_pairs: usize,
    pub(super) failed: bool,
    pub(super) queue: MillerBatch,
    /// The committed-but-unsettled previous dispatch (see
    /// [`InFlightMiller`]); at most one, settled at the next flush point or
    /// at drain, so this lane keeps decoding later superchunks while the
    /// device crunches instead of sleeping in `miller_wait` and
    /// backpressuring the whole pipeline through the depth-2 queues.
    pub(super) in_flight: Option<InFlightMiller>,
    /// Recycled tile backing (≤ 2: one building + one in flight).
    pub(super) tiles: Vec<MillerTile>,
}

/// One committed, un-waited Miller dispatch. `batch` and `tile` own the
/// backing memory of every wrapped input buffer (heap `Vec`s —
/// address-stable across moves); `batch` doubles as the CPU-recovery input
/// on a device error (its row indices stay TABLE-global — only the device
/// sees the tile remap); `out_buf` is device-allocated, read after the wait.
pub(super) struct InFlightMiller {
    batch: MillerBatch,
    tile: MillerTile,
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

impl MillerLane {
    fn take_queue(&mut self) -> MillerBatch {
        std::mem::replace(&mut self.queue, MillerBatch::new())
    }

    /// Gather `batch`'s rows into a recycled tile, so each dispatch
    /// references O(flush) coefficient bytes at every scale — the row-scaled
    /// whole table wired 2.19 GiB at 2^27 and stretched co-running command
    /// buffers by +2.59 s.
    fn build_tile(&mut self, prep: &DoryTier2Prep, batch: &MillerBatch) -> MillerTile {
        let mut tile = self.tiles.pop().unwrap_or_default();
        tile.build(prep.prepared(), &batch.row_indices);
        tile
    }
}

/// Wait on a committed Miller dispatch and fold its per-thread partials
/// into the per-column accumulators — or recover the whole batch on the
/// CPU (latching the lane off) when the device surfaced an error. The
/// merge order per column is dispatch order either way, and GT products
/// are exact field ops, so settle timing never changes a byte.
fn settle_in_flight(
    prep: &DoryTier2Prep,
    lane: &mut MillerLane,
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
    lane.tiles.push(tile);
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
/// device share queued and dispatched as one table command buffer per
/// flush threshold, whose per-thread segments never straddle columns.
/// Dispatches settle (wait + per-column partial-product fold) one flush
/// later ([`InFlightMiller`]), so the device crunches a batch while this
/// lane decodes the following superchunks.
#[tracing::instrument(skip_all, name = "MetalCommit::tier2_absorb")]
pub(super) fn absorb_superchunk(
    ctx: &MetalContext,
    prep: &DoryTier2Prep,
    done: &GpuDone,
    accumulators: &mut [Tier2Accumulator],
    rows: &mut [Vec<Bn254G1>],
    lane: Option<&mut MillerLane>,
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
        Some(lane) if !lane.failed => (Some(lane), MILLER_CPU_FRACTION),
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
                    at = (at + MILLER_TABLE_SEG_PAIRS).min(queue.points.len());
                    queue.seg_starts.push(at as u32);
                }
                queue
                    .folds
                    .push((column, thread_start, queue.seg_starts.len() - 1));
            }
            if lane.queue.points.len() >= lane.flush_pairs {
                let batch = lane.take_queue();
                let tile = lane.build_tile(prep, &batch);
                match commit_miller_dispatch(ctx, tile, batch) {
                    Ok(dispatched) => {
                        // The new dispatch crunches while this superchunk's
                        // CPU share absorbs and the PREVIOUS dispatch —
                        // committed a full flush interval ago — settles
                        // with a near-zero residual wait.
                        cpu_absorb(accumulators);
                        if let Some(previous) = lane.in_flight.take() {
                            settle_in_flight(prep, lane, accumulators, previous);
                        }
                        if lane.failed {
                            settle_in_flight(prep, lane, accumulators, dispatched);
                        } else {
                            lane.in_flight = Some(dispatched);
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
    lane: &mut MillerLane,
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
    match commit_miller_dispatch(ctx, tile, batch) {
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
    tile: MillerTile,
    batch: MillerBatch,
) -> Result<InFlightMiller, (MillerBatch, MetalError)> {
    let n_threads = batch.seg_starts.len() - 1;
    let committed = (|| {
        let ps_buf = ctx.wrap_slice(g1_points_as_u32s(&batch.points))?;
        let out_buf = ctx.alloc_u32s(n_threads * FQ12_U32S)?;
        let mut pass = ctx.begin_pass()?;
        // Wrappers may drop before commit — the encoder retains each bound
        // MTLBuffer; only the BACKING memory must outlive the pass.
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
