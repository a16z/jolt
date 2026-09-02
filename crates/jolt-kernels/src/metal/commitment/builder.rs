//! The driver/builder machinery: fused row extraction into staged one-hot
//! addresses and increment scalars ([`extract_columns`], [`MetalColumns`]),
//! the builder lane's scatter passes that turn a staging into a device job
//! ([`build_one_hot_job`], [`build_inc_job`]), recycled job slabs
//! ([`SlabPool`]), and the arithmetic reduction of increment digit buckets
//! ([`reduce_inc_superchunk`]).

use super::*;

/// Shared raw destination for the order-preserving parallel scatters.
pub(super) struct ScatterPtr<T>(pub(super) *mut T);

// SAFETY: every user writes a set of element indices disjoint across
// workers (layout cursors partition the gather index space; workers own
// disjoint row ranges of the increment columns), so no element aliases.
unsafe impl<T: Send> Send for ScatterPtr<T> {}
// SAFETY: as for `Send` — writes are index-disjoint, and nothing reads
// through the pointer while workers hold it.
unsafe impl<T: Send> Sync for ScatterPtr<T> {}

/// Recycled job slabs: the GPU lane returns each job's gather/segment
/// buffers once its command buffer completes, so steady-state superchunks
/// re-fill warm pages instead of faulting and zeroing fresh allocations
/// every job. Slabs travel at full length; segment tables name only the
/// written prefix, so stale tails are never read.
pub(super) struct SlabPool {
    free: Vec<PageAlignedVec<u32>>,
    returns: Option<std::sync::mpsc::Receiver<PageAlignedVec<u32>>>,
}

impl SlabPool {
    /// A pool with no return path (bench/test builders drop their slabs).
    #[cfg(any(test, feature = "bench-utils"))]
    pub(super) fn detached() -> Self {
        Self {
            free: Vec::new(),
            returns: None,
        }
    }

    pub(super) fn with_returns(returns: std::sync::mpsc::Receiver<PageAlignedVec<u32>>) -> Self {
        Self {
            free: Vec::new(),
            returns: Some(returns),
        }
    }

    /// A slab of at least `len` u32s: the best recycled fit, else a fresh
    /// zero-filled allocation with headroom so per-superchunk count jitter
    /// settles on a stable slab set.
    pub(super) fn take(&mut self, len: usize) -> PageAlignedVec<u32> {
        if let Some(returns) = &self.returns {
            self.free.extend(returns.try_iter());
        }
        let fit = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, slab)| slab.len() >= len)
            .min_by_key(|(_, slab)| slab.len())
            .map(|(index, _)| index);
        match fit {
            Some(index) => self.free.swap_remove(index),
            None => PageAlignedVec::from_elem(0u32, (len + len / 8).next_multiple_of(1024)),
        }
    }
}

/// Reused staging for the fused extract→bucket build. `hot[s]` holds
/// `n_one_hot × sub` addresses (column-major within the subchunk block).
/// `oh_bases` / `inc_bases` hold bucket counts in **bucket-major,
/// subchunk-minor** layout — entry `bucket · subs_per_window + sw` — i.e.
/// exactly the flat gather-array order, so the layout prefix that turns
/// counts into write cursors is one sequential dependent-add sweep (a
/// strided sweep here measured 44 ns/element at 2^25 geometry).
pub(super) struct DriverScratch {
    /// Rows per subchunk; divides the row width (both powers of two).
    pub(super) sub: usize,
    pub(super) hot: Vec<u16>,
    pub(super) oh_bases: Vec<u32>,
    oh_bucket_starts: Vec<u32>,
    inc_bases: Vec<u32>,
    inc_bucket_starts: Vec<u32>,
    starts_host: Vec<u32>,
}

impl DriverScratch {
    pub(super) fn new(row_width: usize) -> Self {
        Self {
            sub: SUBCHUNK_ROWS.min(row_width),
            hot: Vec::new(),
            oh_bases: Vec::new(),
            oh_bucket_starts: Vec::new(),
            inc_bases: Vec::new(),
            inc_bucket_starts: Vec::new(),
            starts_host: Vec::new(),
        }
    }

    /// Size (and zero the count region of) the one-hot staging for an
    /// `n`-cycle chunk.
    pub(super) fn ensure_one_hot(&mut self, n: usize, n_one_hot: usize, one_hot_k: usize) {
        debug_assert!(n.is_multiple_of(self.sub));
        let counts = (n / self.sub) * n_one_hot * one_hot_k;
        if self.hot.len() < n_one_hot * n {
            self.hot.resize(n_one_hot * n, 0);
        }
        if self.oh_bases.len() < counts {
            self.oh_bases.resize(counts, 0);
        }
        self.oh_bases[..counts].fill(0);
    }
}

/// Index arithmetic for the bucket-major/subchunk-minor count-and-cursor
/// matrices: entry `(ci, window, bucket, sw)` lives at
/// `ci · buckets · S + ((window · buckets + bucket) << log2_spw) + sw`,
/// where `S` is the chunk's subchunk count and `spw = S / n_windows`.
#[derive(Clone, Copy)]
pub(super) struct BucketLayout {
    /// Per-column stride: `buckets_per_block · s_count`.
    ci_stride: usize,
    /// `log2(subs_per_window)` — bucket index shifts, never multiplies.
    log2_spw: usize,
    /// Subchunk-window mask: `sw = s & (spw - 1)`.
    spw_mask: usize,
    /// Per-window bucket-group stride in buckets: `buckets_per_block`.
    buckets: usize,
}

impl BucketLayout {
    pub(super) fn new(s_count: usize, subs_per_window: usize, buckets: usize) -> Self {
        debug_assert!(subs_per_window.is_power_of_two());
        Self {
            ci_stride: buckets * s_count,
            log2_spw: subs_per_window.trailing_zeros() as usize,
            spw_mask: subs_per_window - 1,
            buckets,
        }
    }

    /// The subchunk-invariant part of an entry index: window group + sw.
    #[inline]
    pub(super) fn subchunk_base(self, s: usize) -> usize {
        (((s >> self.log2_spw) * self.buckets) << self.log2_spw) + (s & self.spw_mask)
    }

    /// Entry index from a subchunk base and bucket ordinal within the block.
    #[inline]
    fn entry(self, ci: usize, subchunk_base: usize, bucket: usize) -> usize {
        ci * self.ci_stride + (bucket << self.log2_spw) + subchunk_base
    }
}

/// Stage one row's hot addresses: addresses into the subchunk's `hot` block
/// (column-major within the block), bucket counts into the shared
/// bucket-major count matrix.
///
/// # Safety
///
/// `counts` entries touched here are `(·, s)` slots owned exclusively by
/// this row's subchunk; concurrent callers must hold distinct subchunks.
#[expect(
    clippy::too_many_arguments,
    reason = "internal seam shared by the fused pass and the bench fixture"
)]
#[inline]
pub(super) unsafe fn stage_one_hot_row(
    facts: &CommittedColumnsWitness,
    one_hot: &[(usize, ColumnKind)],
    layout: BucketLayout,
    subchunk_base: usize,
    sub: usize,
    local: usize,
    hot: &mut [u16],
    counts: *mut u32,
) {
    for (ci, (_, kind)) in one_hot.iter().enumerate() {
        match kind.hot_address(facts) {
            Some(address) => {
                hot[ci * sub + local] = address as u16;
                // SAFETY: caller contract — this subchunk owns the slot.
                unsafe { *counts.add(layout.entry(ci, subchunk_base, address)) += 1 };
            }
            None => hot[ci * sub + local] = HOT_NONE,
        }
    }
}

/// The fused extraction pass: one subchunk-parallel walk over the trace
/// rows extracts each row's fact bundle exactly once and stages every
/// column family — one-hot addresses plus bucket counts into `scratch`,
/// increment scalars into `inc_vals` (indexed by chunk row).
#[expect(
    clippy::too_many_arguments,
    reason = "internal seam between the stream visitor and the staging state"
)]
pub(super) fn extract_columns(
    rows: &[TraceRow],
    next_after: Option<&TraceRow>,
    env: &WitnessEnv<'_>,
    one_hot: &[(usize, ColumnKind)],
    inc_kinds: &[ColumnKind],
    one_hot_k: usize,
    row_width: usize,
    scratch: &mut DriverScratch,
    inc_vals: &mut [Vec<i128>],
) -> Result<(), WitnessError> {
    let n = rows.len();
    let n_one_hot = one_hot.len();
    let sub = scratch.sub;
    let subs_per_window = row_width / sub;
    scratch.ensure_one_hot(n, n_one_hot, one_hot_k);
    for column in inc_vals.iter_mut() {
        column.resize(n, 0);
    }
    let inc_ptrs: Vec<ScatterPtr<i128>> = inc_vals
        .iter_mut()
        .map(|column| ScatterPtr(column.as_mut_ptr()))
        .collect();
    let inc_ptrs = &inc_ptrs;
    let layout = BucketLayout::new(n / sub, subs_per_window, one_hot_k);
    let counts = ScatterPtr(scratch.oh_bases.as_mut_ptr());
    let counts = &counts;

    let hot_stride = n_one_hot * sub;
    let error = std::sync::Mutex::new(None);
    scratch.hot[..n_one_hot * n]
        .par_chunks_mut(hot_stride)
        .enumerate()
        .for_each(|(s, hot)| {
            let subchunk_base = layout.subchunk_base(s);
            for local in 0..sub {
                let r = s * sub + local;
                let next = rows.get(r + 1).or(next_after);
                let facts = match CommittedColumnsWitness::from_row(&rows[r], next, env) {
                    Ok(facts) => facts,
                    Err(failure) => {
                        if let Ok(mut guard) = error.try_lock() {
                            let _ = guard.get_or_insert(failure);
                        }
                        return;
                    }
                };
                // SAFETY: this worker owns subchunk `s` alone — all `(·, s)`
                // count slots and rows `[s·sub, (s+1)·sub)` are single-writer.
                unsafe {
                    stage_one_hot_row(
                        &facts,
                        one_hot,
                        layout,
                        subchunk_base,
                        sub,
                        local,
                        hot,
                        counts.0,
                    );
                }
                for (ci, kind) in inc_kinds.iter().enumerate() {
                    // SAFETY: row `r` belongs to this worker alone (workers
                    // own disjoint row ranges), and `resize` above covers it.
                    unsafe { *inc_ptrs[ci].0.add(r) = kind.increment(&facts) };
                }
            }
        });
    #[expect(clippy::unwrap_used, reason = "no lock user can panic")]
    if let Some(failure) = error.into_inner().unwrap() {
        return Err(failure);
    }
    Ok(())
}

/// The driver-side stream consumer: extracts each superchunk once into a
/// recycled staging set, advances the below-gate increment columns on the
/// CPU, and ships the staging to the builder lane for job assembly.
pub(super) struct MetalColumns<'a> {
    /// Staged-superchunk queue to the builder lane; `None` after the
    /// builder hangs up (its join surfaces the cause).
    pub(super) tx: Option<SyncSender<StagedChunk>>,
    /// Recycled staging sets coming back from the builder.
    pub(super) free: std::sync::mpsc::Receiver<StagedChunk>,
    pub(super) one_hot: &'a [(usize, ColumnKind)],
    pub(super) increments: Vec<(usize, ColumnKind, DoryPartialCommitment)>,
    pub(super) inc_kinds: Vec<ColumnKind>,
    pub(super) inc_device: bool,
    pub(super) row_width: usize,
    pub(super) one_hot_k: usize,
    pub(super) windows_fed: usize,
    pub(super) setup: &'a DoryProverSetup,
}

impl MetalColumns<'_> {
    pub(super) fn consume_rows(
        &mut self,
        rows: &[TraceRow],
        next_after: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<(), WitnessError> {
        let _span = tracing::info_span!("MetalCommit::consume").entered();
        debug_assert!(
            rows.len().is_multiple_of(self.row_width),
            "superchunk must be whole windows"
        );
        let n_windows = rows.len() / self.row_width;
        let window_base = self.windows_fed;
        self.windows_fed += n_windows;

        // A recycled staging set — or a fresh one when the builder lane is
        // gone (its join surfaces the cause; the stream keeps feeding the
        // CPU-side increment path meanwhile).
        let mut staging = tracing::info_span!("MetalCommit::stage_wait")
            .in_scope(|| self.free.recv())
            .unwrap_or_else(|_| StagedChunk::new(self.row_width, self.inc_kinds.len()));
        {
            let _span = tracing::info_span!("MetalCommit::extract_bucket").entered();
            extract_columns(
                rows,
                next_after,
                env,
                self.one_hot,
                &self.inc_kinds,
                self.one_hot_k,
                self.row_width,
                &mut staging.scratch,
                &mut staging.inc_vals,
            )?;
        }
        staging.n = rows.len();
        staging.window_base = window_base;

        if !self.inc_device {
            for (ci, (_, _, partial)) in self.increments.iter_mut().enumerate() {
                let vals: &[i128] = &staging.inc_vals[ci];
                DoryScheme::feed_i128_rows_with(
                    partial,
                    |index| vals[index],
                    vals.len(),
                    self.row_width,
                    self.setup,
                );
            }
        }
        if let Some(tx) = &self.tx {
            if tx.send(staging).is_err() {
                self.tx = None;
            }
        }
        Ok(())
    }
}

/// Bucket the staged one-hot addresses into gather segments: an in-place
/// layout prefix turns the per-subchunk counts into write cursors over the
/// flat index array (bucket-major, columns outermost — the layout the CPU
/// kernel feeds `process_one_hot_chunks` bucket by bucket), then a
/// subchunk-parallel scatter fills it in cycle order.
#[expect(
    clippy::too_many_arguments,
    reason = "internal seam between the fused pass and the job layout"
)]
pub(super) fn build_one_hot_job(
    scratch: &mut DriverScratch,
    pool: &mut SlabPool,
    n: usize,
    n_one_hot: usize,
    one_hot_k: usize,
    row_width: usize,
    window_base: usize,
    windows_total: usize,
    max_segment_len: usize,
) -> GpuJob {
    let sub = scratch.sub;
    let s_count = n / sub;
    let n_windows = n / row_width;
    let subs_per_window = row_width / sub;
    let n_cw = n_one_hot * n_windows;
    let layout = BucketLayout::new(s_count, subs_per_window, one_hot_k);

    // In-place exclusive prefix: the bucket-major/subchunk-minor count
    // layout equals the flat gather order, so cursors and per-bucket
    // boundaries fall out of one sequential sweep.
    let prefix_span = tracing::info_span!("MetalCommit::oh_prefix").entered();
    scratch.oh_bucket_starts.clear();
    scratch.oh_bucket_starts.push(0);
    let mut running = 0u32;
    for (index, slot) in scratch.oh_bases[..n_one_hot * one_hot_k * s_count]
        .iter_mut()
        .enumerate()
    {
        let count = *slot;
        *slot = running;
        running += count;
        if (index + 1) & layout.spw_mask == 0 {
            scratch.oh_bucket_starts.push(running);
        }
    }
    let total_hot = running as usize;
    drop(prefix_span);

    // Segment table: split buckets at the cap; empty buckets emit nothing
    // (their rows stay identity, exactly the CPU path's empty-index rows).
    let segwalk_span = tracing::info_span!("MetalCommit::oh_segwalk").entered();
    scratch.starts_host.clear();
    scratch.starts_host.push(0);
    let mut segs = Vec::new();
    for cw in 0..n_cw {
        let (column, window) = (cw / n_windows, cw % n_windows);
        for k in 0..one_hot_k {
            let start = scratch.oh_bucket_starts[cw * one_hot_k + k] as usize;
            let end = scratch.oh_bucket_starts[cw * one_hot_k + k + 1] as usize;
            let mut cursor = start;
            while cursor < end {
                let seg_end = (cursor + max_segment_len).min(end);
                scratch.starts_host.push(seg_end as u32);
                segs.push(SegOut {
                    column: column as u32,
                    row: (k * windows_total + window_base + window) as u32,
                });
                cursor = seg_end;
            }
        }
    }

    drop(segwalk_span);
    // Scatter window-local base positions into their buckets: every
    // `(bucket, subchunk)` cursor has one owning subchunk, so subchunks fill
    // the shared array in parallel and every bucket keeps ascending cycle
    // order.
    let scatter_span = tracing::info_span!("MetalCommit::oh_scatter").entered();
    let mut indices = pool.take(total_hot.max(1));
    let target = ScatterPtr(indices.as_mut_ptr());
    let target = &target;
    let cursors = ScatterPtr(scratch.oh_bases.as_mut_ptr());
    let cursors = &cursors;
    scratch.hot[..n_one_hot * n]
        .par_chunks(n_one_hot * sub)
        .enumerate()
        .for_each(|(s, hot)| {
            let subchunk_base = layout.subchunk_base(s);
            let window_local = ((s & layout.spw_mask) * sub) as u32;
            for ci in 0..n_one_hot {
                for local in 0..sub {
                    let address = hot[ci * sub + local];
                    if address != HOT_NONE {
                        // SAFETY: cursor slot `(ci, address, s)` is owned by
                        // this subchunk alone, and the layout prefix
                        // partitions `[0, total_hot)` across cursors, which
                        // `indices` covers.
                        unsafe {
                            let cursor =
                                cursors
                                    .0
                                    .add(layout.entry(ci, subchunk_base, address as usize));
                            *target.0.add(*cursor as usize) = window_local + local as u32;
                            *cursor += 1;
                        }
                    }
                }
            }
        });

    drop(scatter_span);
    let seg_bounds = seg_bounds_into_slab(&scratch.starts_host, pool);
    GpuJob {
        indices,
        seg_bounds,
        segs,
        inc: None,
    }
}

/// Bucket the increment scalars into signed-digit gather segments: a
/// subchunk-parallel count pass over each scalar's signed base-256 digits,
/// an in-place layout prefix per `(column, window, slot, magnitude)`
/// bucket, and a subchunk-parallel scatter of window-local base positions
/// (bit 31 = negated base), split at the segment cap. `None` when every
/// scalar is zero (no device work; the rows stay identity).
///
/// `increments` holds one whole-superchunk scalar vector per increment
/// column; `window_base` makes the emitted windows absolute.
pub(super) fn build_inc_job(
    increments: &[Vec<i128>],
    row_width: usize,
    window_base: usize,
    max_segment_len: usize,
    scratch: &mut DriverScratch,
    pool: &mut SlabPool,
) -> Option<IncJob> {
    assert!(
        row_width < SEG_INDEX_SIGN_BIT as usize,
        "row width must leave the gather sign bit free"
    );
    let n = increments.first().map_or(0, Vec::len);
    let n_windows = n / row_width;
    let n_inc = increments.len();
    let n_blocks = n_inc * n_windows;
    if n_blocks == 0 {
        return None;
    }
    let sub = scratch.sub.min(row_width);
    debug_assert!(n.is_multiple_of(sub) && row_width.is_multiple_of(sub));
    let s_count = n / sub;
    let subs_per_window = row_width / sub;
    let layout = BucketLayout::new(s_count, subs_per_window, INC_BUCKETS);
    let len = n_inc * INC_BUCKETS * s_count;

    // Digit counts in bucket-major/subchunk-minor layout (single-writer per
    // `(·, s)` slot — workers own disjoint subchunks).
    if scratch.inc_bases.len() < len {
        scratch.inc_bases.resize(len, 0);
    }
    scratch.inc_bases[..len].fill(0);
    let counts = ScatterPtr(scratch.inc_bases.as_mut_ptr());
    let counts = &counts;
    (0..s_count).into_par_iter().for_each(|s| {
        let subchunk_base = layout.subchunk_base(s);
        for (ci, column) in increments.iter().enumerate() {
            for &value in &column[s * sub..(s + 1) * sub] {
                for_each_signed_digit(value, |slot, magnitude, _| {
                    let bucket = slot as usize * INC_MAGNITUDES + (magnitude - 1) as usize;
                    // SAFETY: this worker owns subchunk `s` alone.
                    unsafe { *counts.0.add(layout.entry(ci, subchunk_base, bucket)) += 1 };
                });
            }
        }
    });

    // In-place exclusive prefix: the count layout equals the flat gather
    // order (column, window, slot, magnitude, subchunk) — one sequential
    // sweep yields cursors and per-bucket boundaries.
    scratch.inc_bucket_starts.clear();
    scratch.inc_bucket_starts.push(0);
    let mut running = 0u32;
    for (index, slot) in scratch.inc_bases[..len].iter_mut().enumerate() {
        let count = *slot;
        *slot = running;
        running += count;
        if (index + 1) & layout.spw_mask == 0 {
            scratch.inc_bucket_starts.push(running);
        }
    }
    let total_digits = running as usize;
    if total_digits == 0 {
        return None;
    }

    // Segment table: split buckets at the cap; empty buckets emit nothing.
    scratch.starts_host.clear();
    scratch.starts_host.push(0);
    let mut segs = Vec::new();
    for block in 0..n_blocks {
        let (column, window) = (block / n_windows, block % n_windows);
        for slot in 0..INC_SLOTS {
            for magnitude in 1..=INC_MAGNITUDES {
                let bucket = block * INC_BUCKETS + slot * INC_MAGNITUDES + magnitude - 1;
                let end = scratch.inc_bucket_starts[bucket + 1] as usize;
                let mut cursor = scratch.inc_bucket_starts[bucket] as usize;
                while cursor < end {
                    let seg_end = (cursor + max_segment_len).min(end);
                    scratch.starts_host.push(seg_end as u32);
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

    // Scatter window-local base positions (plus sign) into their buckets;
    // every `(bucket, subchunk)` cursor has one owning subchunk, so
    // subchunks fill the shared array in parallel and every bucket keeps
    // ascending cycle order.
    let mut indices = pool.take(total_digits);
    let target = ScatterPtr(indices.as_mut_ptr());
    let target = &target;
    // Fresh pointer: the prefix's `iter_mut` invalidated the count pass's.
    let cursors = ScatterPtr(scratch.inc_bases.as_mut_ptr());
    let cursors = &cursors;
    (0..s_count).into_par_iter().for_each(|s| {
        let subchunk_base = layout.subchunk_base(s);
        let window_local = ((s & layout.spw_mask) * sub) as u32;
        for (ci, column) in increments.iter().enumerate() {
            for (local, &value) in column[s * sub..(s + 1) * sub].iter().enumerate() {
                for_each_signed_digit(value, |slot, magnitude, negate| {
                    let bucket = slot as usize * INC_MAGNITUDES + (magnitude - 1) as usize;
                    let sign = if negate { SEG_INDEX_SIGN_BIT } else { 0 };
                    // SAFETY: cursor slot `(ci, bucket, s)` is owned by this
                    // subchunk alone; the layout prefix partitions
                    // `[0, total_digits)` across cursors, which `indices`
                    // covers.
                    unsafe {
                        let cursor = cursors.0.add(layout.entry(ci, subchunk_base, bucket));
                        *target.0.add(*cursor as usize) = (window_local + local as u32) | sign;
                        *cursor += 1;
                    }
                });
            }
        }
    });

    let seg_bounds = seg_bounds_into_slab(&scratch.starts_host, pool);
    Some(IncJob {
        indices,
        seg_bounds,
        segs,
    })
}

/// Decode one superchunk's increment segment sums into row commitments.
/// Per `(column, window)` group (a contiguous run of segments sorted by
/// ascending `(slot, magnitude)`): merge split buckets, form each slot's
/// weighted total `T_slot = Σ_m m · S_m` by a descending running sum, and
/// combine slots by base-256 Horner — `row = Σ_slot 256^slot · T_slot`,
/// the exact MSM group value of the window's scalars.
pub(super) fn reduce_inc_superchunk(segs: &[IncSeg], jac: &[u32], inc_rows: &mut [Vec<Bn254G1>]) {
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
