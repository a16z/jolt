use std::ops::Range;

use akita_algebra::ntt::butterfly::inverse_ntt;
use akita_algebra::tables::q128_primes;
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut};
use akita_error::AkitaError;
use akita_pcs::custom_source::CommitInnerPlan;
use jolt_field::Fp128x8i32;
use rayon::prelude::*;

use super::digit_windows::{flush_digit_accumulators, DigitAccumulator, DigitWindows};
use super::source::TraceOneHotColumn;
use super::traversal::{row_is_committed, visit_segment_ring_row_range};
use super::MAX_WIDE_ACCUMULATIONS;
use crate::AkitaField;

const DIGIT_LIMBS: usize = 8;
const DIGIT_BITS: usize = 16;
const MAX_NTT_POSITIONS: usize = 252;
const DOT_BATCH: usize = 6;
const COLUMN_TILE: usize = 4;
const COMMIT_RING_DIMENSION: usize = 512;
const COMMIT_ONE_HOT_K: usize = 16;
const MAX_ACCUMULATED_POSITIONS: usize =
    (MAX_WIDE_ACCUMULATIONS / (COMMIT_RING_DIMENSION / COMMIT_ONE_HOT_K) / MAX_NTT_POSITIONS)
        * MAX_NTT_POSITIONS;
const NTT_MIN_SHIFTS_PER_RING: usize = 10;
const _: () = assert!(MAX_NTT_POSITIONS.is_multiple_of(DOT_BATCH));
const _: () = assert!(MAX_ACCUMULATED_POSITIONS >= MAX_NTT_POSITIONS);
type NttRing<const D: usize> = CyclotomicCrtNtt<i32, 1, D>;

pub(super) struct PreparedNttCommit<'a, const D: usize> {
    params: CrtNttParamSet<i32, 1, D>,
    lut: DigitMontLut<i32, 1>,
    matrix: [Vec<NttRing<D>>; DIGIT_LIMBS],
    coefficients: &'a [CyclotomicRing<AkitaField, D>],
}

impl<'a, const D: usize> PreparedNttCommit<'a, D> {
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
        if params.pointwise_dot_batch_size() != DOT_BATCH {
            return None;
        }
        // Split A into canonical base-2^16 limbs. Every integer convolution
        // coefficient is bounded by positions * (D/K) * (2^16-1), so one
        // small prime suffices when we reconstruct every 252 positions.
        // The binary input transform is shared by all eight A limbs.
        let coefficient_bound = MAX_NTT_POSITIONS * (D / source.one_hot_k) * 65535;
        if coefficient_bound as u64 >= params.primes[0].p as u64 / 2 {
            return None;
        }
        let _span = tracing::info_span!("trace_onehot_commit_prepare_ntt").entered();
        let lut = DigitMontLut::new_with_digit_bound(&params, 2);
        let matrix = std::array::from_fn(|limb| {
            matrix_row[..active_positions]
                .par_iter()
                .map(|ring| {
                    let digits = std::array::from_fn(|j| {
                        ((ring.coeffs[j].to_canonical_u128() >> (limb * DIGIT_BITS)) & 65535)
                            as i128
                    });
                    NttRing::from_centered_coefficients(&digits, &params)
                })
                .collect()
        });
        Some(Self {
            params,
            lut,
            matrix,
            coefficients: matrix_row,
        })
    }

    pub(super) fn commit_part(
        &self,
        source: &TraceOneHotColumn,
        block_ring_start: usize,
        rings: Range<usize>,
    ) -> Result<Vec<CyclotomicRing<AkitaField, D>>, AkitaError> {
        let mut scratch = NttCommitScratch::<D>::new(source);
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
        flush_digit_accumulators(&mut scratch.digit_accumulators, &mut result);
        Ok(result)
    }
}

struct NttCommitScratch<const D: usize> {
    num_columns: usize,
    rows_per_ring: usize,
    one_hot_k: usize,
    rows: Vec<u8>,
    masks: Vec<u64>,
    populated_rows: [usize; DOT_BATCH],
    buffered: usize,
    batch_start: usize,
    transformed: Vec<[NttRing<D>; DOT_BATCH]>,
    accumulators: Vec<[NttRing<D>; DIGIT_LIMBS]>,
    ntt_used: Vec<bool>,
    digit_accumulators: Vec<DigitAccumulator<D>>,
    windows: Vec<DigitWindows<D>>,
    shifts: [Vec<usize>; DOT_BATCH],
}

impl<const D: usize> NttCommitScratch<D> {
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
            transformed: (0..COLUMN_TILE)
                .map(|_| std::array::from_fn(|_| NttRing::zero()))
                .collect(),
            accumulators: (0..num_columns)
                .map(|_| std::array::from_fn(|_| NttRing::zero()))
                .collect(),
            ntt_used: vec![false; num_columns],
            digit_accumulators: vec![[Fp128x8i32([0; 8]); D]; num_columns],
            windows: (0..DOT_BATCH).map(|_| DigitWindows::new()).collect(),
            shifts: std::array::from_fn(|_| Vec::with_capacity(rows_per_ring)),
        }
    }

    fn push(
        &mut self,
        prepared: &PreparedNttCommit<'_, D>,
        position: usize,
        rows: &[u8],
        masks: &[u64],
    ) {
        if self.buffered == 0 {
            self.batch_start = position;
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

    fn accumulate(&mut self, prepared: &PreparedNttCommit<'_, D>) {
        if self.buffered == 0 {
            return;
        }
        let mut windows_loaded = false;
        let mut tile_columns = [0usize; COLUMN_TILE];
        let mut tile_len = 0;
        for column in 0..self.num_columns {
            let mut selected = 0;
            for slot in 0..self.buffered {
                let shifts = &mut self.shifts[slot];
                shifts.resize(self.rows_per_ring, 0);
                let row_start = slot * self.rows_per_ring;
                let mut len = 0;
                for row in 0..self.populated_rows[slot] {
                    let hot = self.rows[(row_start + row) * self.num_columns + column];
                    shifts[len] = row * self.one_hot_k + usize::from(hot);
                    len += usize::from(row_is_committed(hot, self.masks[row_start + row], column));
                }
                shifts.truncate(len);
                selected += len;
            }
            if selected == 0 {
                continue;
            }
            if selected < self.buffered * NTT_MIN_SHIFTS_PER_RING {
                if !windows_loaded {
                    for slot in 0..self.buffered {
                        self.windows[slot].load(&prepared.coefficients[self.batch_start + slot]);
                    }
                    windows_loaded = true;
                }
                DigitWindows::accumulate_slots(
                    &self.windows[..self.buffered],
                    &mut self.digit_accumulators[column],
                    &self.shifts[..self.buffered],
                );
                continue;
            }
            for slot in 0..self.buffered {
                if self.shifts[slot].is_empty() {
                    self.transformed[tile_len][slot] = NttRing::zero();
                } else {
                    let mut coefficients = [0i8; D];
                    for &shift in &self.shifts[slot] {
                        coefficients[shift] = 1;
                    }
                    self.transformed[tile_len][slot].assign_i8_with_lut(
                        &coefficients,
                        &prepared.params,
                        &prepared.lut,
                    );
                }
            }
            self.ntt_used[column] = true;
            tile_columns[tile_len] = column;
            tile_len += 1;
            if tile_len == COLUMN_TILE {
                self.accumulate_tile(prepared, &tile_columns);
                tile_len = 0;
            }
        }
        self.accumulate_tile(prepared, &tile_columns[..tile_len]);
        self.buffered = 0;
    }

    /// Reuse one matrix limb across several columns while its six rings
    /// remain cache-resident. The arithmetic and reconstruction bounds are
    /// unchanged; only the order of independent column updates changes.
    fn accumulate_tile(&mut self, prepared: &PreparedNttCommit<'_, D>, columns: &[usize]) {
        if columns.is_empty() {
            return;
        }
        for (limb, matrix) in prepared.matrix.iter().enumerate() {
            let matrix = &matrix[self.batch_start..self.batch_start + self.buffered];
            for (&column, transformed) in columns.iter().zip(&self.transformed) {
                self.accumulators[column][limb].add_assign_pointwise_dot(
                    matrix,
                    &transformed[..self.buffered],
                    &prepared.params,
                );
            }
        }
    }

    fn reconstruct(&mut self, prepared: &PreparedNttCommit<'_, D>) {
        let prime = prepared.params.primes[0];
        for column in 0..self.num_columns {
            if !std::mem::take(&mut self.ntt_used[column]) {
                continue;
            }
            for (limb, accumulator) in self.accumulators[column].iter_mut().enumerate() {
                inverse_ntt(
                    &mut accumulator.limbs[0],
                    prime,
                    &prepared.params.twiddles[0],
                    prepared.params.kernel_plan(),
                );
                for (digits, &value) in self.digit_accumulators[column]
                    .iter_mut()
                    .zip(&accumulator.limbs[0])
                {
                    digits.0[limb] += prime.center(prime.to_canonical(value));
                }
                *accumulator = NttRing::zero();
            }
        }
    }
}
