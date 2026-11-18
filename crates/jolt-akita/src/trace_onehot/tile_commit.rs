//! The one-hot trace commitment over [`tile_dot`]: a batch of twelve key
//! positions is laid out once for every limb, and one sequential pass over it
//! serves every limb of a tile of columns. [`super::ntt_commit`] computes the
//! same commitment with one pointwise dot per limb and column; `commit` takes
//! this path where the transform kernels are NEON, which is where it was
//! measured.

use std::ops::Range;

use akita_algebra::ntt::butterfly::inverse_ntt;
use akita_algebra::ntt::tile_dot::{self, SLOTS};
use akita_algebra::tables::q128_primes;
use akita_algebra::{CrtNttParamSet, CyclotomicRing, DigitMontLut, MontCoeff};
use akita_error::AkitaError;
use akita_pcs::custom_source::CommitInnerPlan;
use jolt_field::Fp128x8i32;
use rayon::prelude::*;

use super::digit_windows::{flush_digit_accumulators, DigitAccumulator, DigitWindows};
use super::ntt_commit::{
    NttRing, COLUMN_TILE, COMMIT_ONE_HOT_K, COMMIT_RING_DIMENSION, DIGIT_BITS, DIGIT_LIMBS,
    MAX_ACCUMULATED_POSITIONS, MAX_NTT_POSITIONS,
};
use super::one_hot_ntt::OneHotNtt;
use super::source::TraceOneHotColumn;
use super::traversal::{row_is_committed, visit_segment_ring_row_range};
use crate::AkitaField;

/// Key positions one dot pass covers.
const DOT_BATCH: usize = SLOTS;
/// Positions whose digit windows a sparse pass keeps loaded. Half a dot
/// batch: the windows of twelve positions overflow the first-level cache.
const SPARSE_SLOTS: usize = DOT_BATCH / 2;

/// What a batch's work costs, in cycles measured on an M5 Max Super core. A
/// column's batch goes one of two ways: the transform of every position that
/// has rows and one dot pass over the whole batch, or one digit-window
/// addition per committed row. The windows are loaded only when some column
/// uses them, so the choice is made per batch: the columns the windows would
/// serve cheaper take them only if together they save more than loading the
/// windows costs.
///
/// One position of one column in a dot pass (all limbs).
const DOT_COST: usize = 830;
/// One transform, before its rows.
const TRANSFORM_COST: usize = 540;
/// One committed row of a transform.
const TRANSFORM_ROW_COST: usize = 8;
/// One committed row through the digit windows.
const WINDOW_ROW_COST: usize = 262;
/// Loading one position's digit windows.
const WINDOW_LOAD_COST: usize = 4450;

/// A column with committed rows in the batch: its shifts are `lens[slot]`
/// entries per slot, in slot order, from `start` on.
struct BatchColumn {
    column: usize,
    start: usize,
    lens: [usize; DOT_BATCH],
}

impl BatchColumn {
    fn selected(&self) -> usize {
        self.lens.iter().sum()
    }

    /// Cycles the transform and dot take beyond what the digit windows
    /// would, if positive.
    fn window_saving(&self) -> Option<usize> {
        let selected = self.selected();
        let transforms = self.lens.iter().filter(|&&len| len != 0).count();
        let dense =
            DOT_BATCH * DOT_COST + transforms * TRANSFORM_COST + selected * TRANSFORM_ROW_COST;
        dense.checked_sub(selected * WINDOW_ROW_COST)
    }
}
const _: () = assert!(MAX_NTT_POSITIONS.is_multiple_of(DOT_BATCH));

pub(super) struct PreparedTileCommit<'a, const D: usize> {
    params: CrtNttParamSet<i32, 1, D>,
    lut: DigitMontLut<i32, 1>,
    one_hot: Option<OneHotNtt>,
    /// The key's limbs by batch of `DOT_BATCH` positions, each batch in
    /// [`tile_dot`]'s key layout and zero past the last position.
    key: Vec<i32>,
    coefficients: &'a [CyclotomicRing<AkitaField, D>],
}

impl<'a, const D: usize> PreparedTileCommit<'a, D> {
    pub(super) fn new(
        source: &TraceOneHotColumn,
        plan: CommitInnerPlan,
        matrix_row: &'a [CyclotomicRing<AkitaField, D>],
    ) -> Option<Self> {
        if D != COMMIT_RING_DIMENSION
            || source.one_hot_k != COMMIT_ONE_HOT_K
            || plan.n_a != 1
            || plan.num_digits_inner != 1
        {
            return None;
        }
        let active_positions = plan.num_positions_per_block.min(
            source
                .rows
                .nonzero_prefix_len()
                .div_ceil(D / source.one_hot_k),
        );
        if active_positions < 32 {
            return None;
        }
        let params = CrtNttParamSet::new([q128_primes()[0]]);
        let prime = params.primes[0];
        if tile_dot::max_passes(prime) < MAX_NTT_POSITIONS / DOT_BATCH {
            return None;
        }
        // Split A into canonical base-2^16 limbs. Every integer convolution
        // coefficient is bounded by positions * (D/K) * (2^16-1), so one
        // small prime suffices when we reconstruct every 252 positions.
        // The binary input transform is shared by all eight A limbs.
        let coefficient_bound = MAX_NTT_POSITIONS * (D / source.one_hot_k) * 65535;
        if coefficient_bound as u64 >= prime.p as u64 / 2 {
            return None;
        }
        let _span = tracing::info_span!("trace_onehot_commit_prepare_ntt").entered();
        let lut = DigitMontLut::new_with_digit_bound(&params, 2);
        let one_hot = OneHotNtt::new(&params, &lut);
        // One pass over the positions: a coefficient leaves Montgomery form
        // once for its eight digits.
        let fill_batch = |batch: &mut [i32], rings: &[CyclotomicRing<AkitaField, D>]| {
            for (slot, ring) in rings.iter().enumerate() {
                let canonical: [u128; D] =
                    std::array::from_fn(|j| ring.coeffs[j].to_canonical_u128());
                for limb in 0..DIGIT_LIMBS {
                    let digits = std::array::from_fn(|j| {
                        ((canonical[j] >> (limb * DIGIT_BITS)) & 65535) as i128
                    });
                    let mut limb_ring = NttRing::from_centered_coefficients(&digits, &params);
                    if let Some(one_hot) = &one_hot {
                        one_hot.to_lane_order(&mut limb_ring.limbs[0]);
                    }
                    tile_dot::write_key(batch, DIGIT_LIMBS, limb, slot, &limb_ring.limbs[0], prime);
                }
            }
        };
        let batch_len = tile_dot::key_len(DIGIT_LIMBS, D);
        let mut key = vec![0; active_positions.div_ceil(DOT_BATCH) * batch_len];
        key.par_chunks_mut(batch_len)
            .zip(matrix_row[..active_positions].par_chunks(DOT_BATCH))
            .for_each(|(batch, rings)| fill_batch(batch, rings));
        Some(Self {
            params,
            lut,
            one_hot,
            key,
            coefficients: matrix_row,
        })
    }

    fn key_batch(&self, batch: usize) -> &[i32] {
        let batch_len = tile_dot::key_len(DIGIT_LIMBS, D);
        &self.key[batch * batch_len..][..batch_len]
    }

    pub(super) fn commit_part(
        &self,
        source: &TraceOneHotColumn,
        block_ring_start: usize,
        rings: Range<usize>,
    ) -> Result<Vec<CyclotomicRing<AkitaField, D>>, AkitaError> {
        let mut scratch = TileCommitScratch::<D>::new(source);
        let mut result = vec![CyclotomicRing::zero(); source.rows.num_columns()];
        let mut accumulated_positions = 0;
        for start in (rings.start..rings.end).step_by(MAX_NTT_POSITIONS) {
            let end = start.saturating_add(MAX_NTT_POSITIONS).min(rings.end);
            visit_segment_ring_row_range::<D>(source, start, end, |ring, rows, masks| {
                scratch.push(self, ring - block_ring_start, rows, masks);
            })?;
            scratch.accumulate(self);
            scratch.reconstruct(self);
            accumulated_positions += end - start;
            if accumulated_positions == MAX_ACCUMULATED_POSITIONS {
                flush_digit_accumulators(&mut scratch.digit_accumulators, &mut result);
                accumulated_positions = 0;
            }
        }
        if accumulated_positions != 0 {
            flush_digit_accumulators(&mut scratch.digit_accumulators, &mut result);
        }
        Ok(result)
    }
}

struct TileCommitScratch<const D: usize> {
    num_columns: usize,
    rows_per_ring: usize,
    one_hot_k: usize,
    rows: Vec<u8>,
    masks: Vec<u64>,
    populated_rows: [usize; DOT_BATCH],
    buffered: usize,
    batch_start: usize,
    /// One [`tile_dot`] input per column of a tile.
    transformed: Vec<i32>,
    /// One transform's output on its way into `transformed`, and one limb's
    /// residues on their way out of `sums`.
    ring: NttRing<D>,
    /// Every column's [`tile_dot`] sums since the last reconstruction.
    sums: Vec<i64>,
    ntt_used: Vec<bool>,
    digit_accumulators: Vec<DigitAccumulator<D>>,
    windows: Vec<DigitWindows<D>>,
    /// The batch's columns with committed rows and their shifts, back to back.
    columns: Vec<BatchColumn>,
    shifts: Vec<usize>,
    /// Indices into `columns` of the batch's digit-window columns.
    sparse: Vec<usize>,
}

impl<const D: usize> TileCommitScratch<D> {
    fn new(source: &TraceOneHotColumn) -> Self {
        let num_columns = source.rows.num_columns();
        let rows_per_ring = D / source.one_hot_k;
        Self {
            num_columns,
            rows_per_ring,
            one_hot_k: source.one_hot_k,
            rows: vec![0; DOT_BATCH * rows_per_ring * num_columns],
            masks: vec![0; DOT_BATCH * rows_per_ring],
            populated_rows: [0; DOT_BATCH],
            buffered: 0,
            batch_start: 0,
            transformed: vec![0; COLUMN_TILE * tile_dot::input_len(D)],
            ring: NttRing::zero(),
            sums: vec![0; num_columns * tile_dot::sums_len(DIGIT_LIMBS, D)],
            ntt_used: vec![false; num_columns],
            digit_accumulators: vec![[Fp128x8i32([0; 8]); D]; num_columns],
            windows: (0..SPARSE_SLOTS).map(|_| DigitWindows::new()).collect(),
            columns: Vec::with_capacity(num_columns),
            // One slot more than every row of every column: a row's shift is
            // written before it is known to be committed.
            shifts: vec![0; num_columns * DOT_BATCH * rows_per_ring + 1],
            sparse: Vec::with_capacity(num_columns),
        }
    }

    fn push(
        &mut self,
        prepared: &PreparedTileCommit<'_, D>,
        position: usize,
        rows: &[u8],
        masks: &[u64],
    ) {
        if self.buffered == 0 {
            // A batch is one key entry: a range that starts inside one leaves
            // the slots before it empty.
            let lead = position % DOT_BATCH;
            self.batch_start = position - lead;
            self.populated_rows[..lead].fill(0);
            self.buffered = lead;
        }
        let row_start = self.buffered * self.rows_per_ring * self.num_columns;
        self.rows[row_start..row_start + rows.len()].copy_from_slice(rows);
        let mask_start = self.buffered * self.rows_per_ring;
        self.masks[mask_start..mask_start + masks.len()].copy_from_slice(masks);
        self.populated_rows[self.buffered] = masks.len();
        self.buffered += 1;
        if self.buffered == DOT_BATCH {
            self.accumulate(prepared);
        }
    }

    fn accumulate(&mut self, prepared: &PreparedTileCommit<'_, D>) {
        if self.buffered == 0 {
            return;
        }
        self.collect_shifts();
        self.choose_sparse();
        let mut tile_columns = [0usize; COLUMN_TILE];
        let mut tile_len = 0;
        // `sparse` is increasing.
        let mut next_sparse = 0;
        for index in 0..self.columns.len() {
            if self.sparse.get(next_sparse) == Some(&index) {
                next_sparse += 1;
                continue;
            }
            let batch = &self.columns[index];
            let input_len = tile_dot::input_len(D);
            let input = &mut self.transformed[tile_len * input_len..][..input_len];
            let mut offset = batch.start;
            let mut previous: &[usize] = &[];
            for (slot, &len) in batch.lens.iter().enumerate() {
                let slot_shifts = &self.shifts[offset..offset + len];
                offset += len;
                if slot_shifts.is_empty() {
                    tile_dot::clear_input(input, slot, D);
                } else if slot > 0 && slot_shifts == previous {
                    // Equal coefficient positions have the same transform, even
                    // though the dot product uses a different key for each slot.
                    tile_dot::copy_input(input, slot - 1, slot, D);
                } else if let Some(one_hot) = &prepared.one_hot {
                    one_hot.transform_lanes(slot_shifts, tile_dot::input_slot_mut(input, slot, D));
                } else {
                    let mut coefficients = [0i8; D];
                    for &shift in slot_shifts {
                        coefficients[shift] = 1;
                    }
                    self.ring
                        .assign_i8_with_lut(&coefficients, &prepared.params, &prepared.lut);
                    prepared.params.primes[0].reduce_range_in_place(&mut self.ring.limbs[0]);
                    tile_dot::write_input(input, slot, &self.ring.limbs[0]);
                }
                previous = slot_shifts;
            }
            self.ntt_used[batch.column] = true;
            tile_columns[tile_len] = batch.column;
            tile_len += 1;
            if tile_len == COLUMN_TILE {
                self.accumulate_tile(prepared, &tile_columns);
                tile_len = 0;
            }
        }
        self.accumulate_tile(prepared, &tile_columns[..tile_len]);
        self.accumulate_sparse(prepared);
        self.buffered = 0;
    }

    /// Lists the batch's columns that have committed rows, with their shifts.
    fn collect_shifts(&mut self) {
        let num_columns = self.num_columns;
        let rows_per_ring = self.rows_per_ring;
        let one_hot_k = self.one_hot_k;
        // Slices, not the vectors: through `&Vec` every row would reload the
        // pointer and length past the store into `shifts`.
        let rows = self.rows.as_slice();
        let masks = self.masks.as_slice();
        let shifts = self.shifts.as_mut_slice();
        self.columns.clear();
        let mut fill = 0;
        for column in 0..num_columns {
            let start = fill;
            let mut lens = [0; DOT_BATCH];
            for (slot, len) in lens.iter_mut().enumerate().take(self.buffered) {
                let row_start = slot * rows_per_ring;
                let slot_start = fill;
                for row in 0..self.populated_rows[slot] {
                    let hot = rows[(row_start + row) * num_columns + column];
                    shifts[fill] = row * one_hot_k + usize::from(hot);
                    fill += usize::from(row_is_committed(hot, masks[row_start + row], column));
                }
                *len = fill - slot_start;
            }
            if fill != start {
                self.columns.push(BatchColumn {
                    column,
                    start,
                    lens,
                });
            }
        }
    }

    /// Fills `sparse` with the columns the digit windows serve in this batch.
    fn choose_sparse(&mut self) {
        self.sparse.clear();
        let mut saving = 0;
        let mut loaded = [false; DOT_BATCH.div_ceil(SPARSE_SLOTS)];
        for (index, batch) in self.columns.iter().enumerate() {
            let Some(column_saving) = batch.window_saving() else {
                continue;
            };
            saving += column_saving;
            self.sparse.push(index);
            for (group, lens) in loaded.iter_mut().zip(batch.lens.chunks(SPARSE_SLOTS)) {
                *group |= lens.iter().any(|&len| len != 0);
            }
        }
        let loads = loaded.iter().filter(|&&group| group).count() * SPARSE_SLOTS;
        if saving <= loads * WINDOW_LOAD_COST {
            self.sparse.clear();
        }
    }

    /// Adds the batch's sparse columns through the digit windows, a few
    /// positions at a time so that their windows stay cache-resident across
    /// the columns.
    fn accumulate_sparse(&mut self, prepared: &PreparedTileCommit<'_, D>) {
        for first in (0..self.buffered).step_by(SPARSE_SLOTS) {
            let slots = first..(first + SPARSE_SLOTS).min(self.buffered);
            if self.sparse.iter().all(|&index| {
                self.columns[index].lens[slots.clone()]
                    .iter()
                    .all(|&len| len == 0)
            }) {
                continue;
            }
            for (window, slot) in self.windows.iter_mut().zip(slots.clone()) {
                window.load(&prepared.coefficients[self.batch_start + slot]);
            }
            for &index in &self.sparse {
                let batch = &self.columns[index];
                let mut offset = batch.start + batch.lens[..first].iter().sum::<usize>();
                let mut slot_shifts = [&[][..]; SPARSE_SLOTS];
                for (slot_shifts, &len) in slot_shifts.iter_mut().zip(&batch.lens[slots.clone()]) {
                    *slot_shifts = &self.shifts[offset..offset + len];
                    offset += len;
                }
                DigitWindows::accumulate_slots(
                    &self.windows[..slots.len()],
                    &mut self.digit_accumulators[batch.column],
                    &slot_shifts[..slots.len()],
                );
            }
        }
    }

    /// One pass over the batch's key entry serves every limb of every column
    /// in the tile.
    fn accumulate_tile(&mut self, prepared: &PreparedTileCommit<'_, D>, columns: &[usize]) {
        if columns.is_empty() {
            return;
        }
        let per_column = tile_dot::sums_len(DIGIT_LIMBS, D);
        let mut tile: [&mut [i64]; COLUMN_TILE] = Default::default();
        let mut rest = self.sums.as_mut_slice();
        let mut next_column = 0;
        // Tile columns are increasing, so their sums split off in order.
        for (sums, &column) in tile.iter_mut().zip(columns) {
            let (_, tail) =
                std::mem::take(&mut rest).split_at_mut((column - next_column) * per_column);
            let (chosen, tail) = tail.split_at_mut(per_column);
            *sums = chosen;
            rest = tail;
            next_column = column + 1;
        }
        tile_dot::accumulate(
            &mut tile[..columns.len()],
            prepared.key_batch(self.batch_start / DOT_BATCH),
            &self.transformed[..columns.len() * tile_dot::input_len(D)],
            DIGIT_LIMBS,
            D,
            prepared.params.primes[0],
            prepared.params.kernel_plan(),
        );
    }

    fn reconstruct(&mut self, prepared: &PreparedTileCommit<'_, D>) {
        let prime = prepared.params.primes[0];
        let per_column = tile_dot::sums_len(DIGIT_LIMBS, D);
        for column in 0..self.num_columns {
            if !std::mem::take(&mut self.ntt_used[column]) {
                continue;
            }
            let sums = &mut self.sums[column * per_column..][..per_column];
            let residues = &mut self.ring.limbs[0];
            for limb in 0..DIGIT_LIMBS {
                tile_dot::reduce(
                    sums,
                    DIGIT_LIMBS,
                    limb,
                    residues,
                    prime,
                    prepared.params.kernel_plan(),
                );
                if let Some(one_hot) = &prepared.one_hot {
                    one_hot.to_library_order(residues);
                }
                inverse_ntt(
                    residues,
                    prime,
                    &prepared.params.twiddles[0],
                    prepared.params.kernel_plan(),
                );
                for (digits, &value) in self.digit_accumulators[column]
                    .iter_mut()
                    .zip(residues.iter())
                {
                    // Multiplication by raw 1 removes Montgomery scaling. Akita's signed
                    // reduction returns z = (a - t*p) / 2^32 with a, t signed 32-bit words;
                    // `NttPrime::new` asserts p < 2^30, so |a - t*p| < 2^62 fits in i64 and
                    // |z| < (p+1)/2: z is already the centered residue in [-(p-1)/2, (p-1)/2].
                    digits.0[limb] += prime.mul(value, MontCoeff::from_raw(1)).raw();
                }
            }
            sums.fill(0);
        }
    }
}
