use akita_algebra::{ring::WideCyclotomicRing, CyclotomicRing};
use akita_error::AkitaError;
use akita_prover::compute::SubringCoefficientPackingPlan;
use akita_prover::RootPolyShape;
use akita_types::FpExtEncoding;
use jolt_field::{CanonicalEncoding, ExtField, PseudoMersenne, Unreduced, Zero};

use super::source::{validate_dimension, TracePackedOneHot};
use super::{K256_ROW_BATCH, NO_SELECTED_ROW, SHARED_SHIFT_MIN_COLUMNS, TASKS_PER_RAYON_WORKER};
use crate::AkitaField;

#[inline(always)]
pub(super) fn row_is_committed(selected_row: u8, committed_zero_mask: u64, column: usize) -> bool {
    selected_row != NO_SELECTED_ROW || committed_zero_mask & (1u64 << column) != 0
}

pub(super) type AkitaWideRing<const D: usize> =
    WideCyclotomicRing<<AkitaField as Unreduced>::Wide, D>;

// Canonical reduction on every add costs more than tracking 2^128 wraps and
// applying 2^128 = MODULUS_OFFSET only when the tile is flushed.
#[derive(Clone)]
pub(super) struct DeferredFp128Ring<const D: usize> {
    pub(super) lo: [u64; D],
    pub(super) hi: [u64; D],
    pub(super) wraps: [i16; D],
}

impl<const D: usize> DeferredFp128Ring<D> {
    pub(super) fn zero() -> Self {
        Self {
            lo: [0; D],
            hi: [0; D],
            wraps: [0; D],
        }
    }

    #[inline(always)]
    pub(super) fn add_coefficient(&mut self, index: usize, value: AkitaField) {
        let [value_lo, value_hi] = value.to_limbs();
        let (lo, carry_lo) = self.lo[index].overflowing_add(value_lo);
        let (hi, carry_hi) = self.hi[index].carrying_add(value_hi, carry_lo);
        self.lo[index] = lo;
        self.hi[index] = hi;
        self.wraps[index] += i16::from(carry_hi);
    }

    #[inline(always)]
    pub(super) fn sub_coefficient(&mut self, index: usize, value: AkitaField) {
        let [value_lo, value_hi] = value.to_limbs();
        let (lo, borrow_lo) = self.lo[index].overflowing_sub(value_lo);
        let (hi, borrow_hi) = self.hi[index].borrowing_sub(value_hi, borrow_lo);
        self.lo[index] = lo;
        self.hi[index] = hi;
        self.wraps[index] -= i16::from(borrow_hi);
    }

    #[inline(always)]
    pub(super) fn shift_accumulate(
        &mut self,
        source: &CyclotomicRing<AkitaField, D>,
        shift: usize,
    ) {
        debug_assert!(shift < D);
        let (lo, hi) = source.coefficients().split_at(D - shift);
        for (index, &value) in lo.iter().enumerate() {
            self.add_coefficient(index + shift, value);
        }
        for (index, &value) in hi.iter().enumerate() {
            self.sub_coefficient(index, value);
        }
    }

    pub(super) fn reduce_and_clear(&mut self) -> CyclotomicRing<AkitaField, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            let lo = std::mem::take(&mut self.lo[index]);
            let hi = std::mem::take(&mut self.hi[index]);
            let wraps = std::mem::take(&mut self.wraps[index]);
            debug_assert!(usize::from(wraps.unsigned_abs()) <= K256_ROW_BATCH);

            let base = AkitaField::from_u128_reduced(u128::from(lo) | (u128::from(hi) << 64));
            let correction = AkitaField::from_u128_reduced(
                u128::from(wraps.unsigned_abs()) * <AkitaField as PseudoMersenne>::OFFSET,
            );
            if wraps >= 0 {
                base + correction
            } else {
                base - correction
            }
        }))
    }
}

/// Groups columns that share four consecutive K=16 row shifts. Adaptive
/// dimensions use one, two, or four of these groups per ring, preserving the
/// useful four-row reuse pattern without dimension-specific implementations.
pub(super) struct K16FourRowShiftGroups {
    group_by_key: Vec<(u64, u8)>,
    group_columns: Vec<u8>,
    group_counts: Vec<u8>,
    group_shifts: Vec<[usize; 4]>,
    partial_columns: Vec<u8>,
    row_start: usize,
    num_groups: u8,
}

impl K16FourRowShiftGroups {
    pub(super) fn new(num_columns: usize, row_start: usize) -> Option<Self> {
        if num_columns >= usize::from(u8::MAX) {
            return None;
        }
        let key_slots = (2 * num_columns).next_power_of_two();
        Some(Self {
            group_by_key: vec![(0, u8::MAX); key_slots],
            group_columns: vec![u8::MAX; num_columns * num_columns],
            group_counts: vec![0; num_columns],
            group_shifts: vec![[0; 4]; num_columns],
            partial_columns: Vec::with_capacity(num_columns),
            row_start,
            num_groups: 0,
        })
    }

    pub(super) fn build(
        &mut self,
        selected_rows: &[u8],
        committed_zero_masks: &[u64],
        num_columns: usize,
    ) -> bool {
        self.group_by_key.fill((0, u8::MAX));
        self.partial_columns.clear();
        self.num_groups = 0;
        if selected_rows.len() != 4 * num_columns || committed_zero_masks.len() != 4 {
            return false;
        }

        for column in 0..num_columns {
            let mut key = 0u64;
            let mut shifts = [0usize; 4];
            let mut complete = true;
            for (row_offset, (row_indices, &committed_zero_mask)) in selected_rows
                .chunks_exact(num_columns)
                .zip(committed_zero_masks)
                .enumerate()
            {
                let hot = row_indices[column];
                if !row_is_committed(hot, committed_zero_mask, column) {
                    complete = false;
                    break;
                }
                key |= u64::from(hot) << (4 * row_offset);
                shifts[row_offset] = 16 * (self.row_start + row_offset) + usize::from(hot);
            }
            if !complete {
                self.partial_columns.push(column as u8);
                continue;
            }
            let slot_mask = self.group_by_key.len() - 1;
            let mut slot = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) as usize & slot_mask;
            let group = loop {
                let (stored_key, stored_group) = self.group_by_key[slot];
                if stored_group != u8::MAX && stored_key == key {
                    break stored_group;
                }
                if stored_group == u8::MAX {
                    let group = self.num_groups;
                    self.num_groups += 1;
                    self.group_by_key[slot] = (key, group);
                    self.group_counts[usize::from(group)] = 0;
                    self.group_shifts[usize::from(group)] = shifts;
                    break group;
                }
                slot = (slot + 1) & slot_mask;
            };
            let group = usize::from(group);
            let count = usize::from(self.group_counts[group]);
            self.group_columns[group * num_columns + count] = column as u8;
            self.group_counts[group] += 1;
        }
        true
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the fused shift kernel keeps its source, destination, rank, and row views explicit"
    )]
    pub(super) fn accumulate<const D: usize>(
        &self,
        src: &AkitaWideRing<D>,
        dst: &mut [AkitaWideRing<D>],
        a: usize,
        n_a: usize,
        selected_rows: &[u8],
        committed_zero_masks: &[u64],
        num_columns: usize,
    ) {
        for group in 0..self.num_groups {
            let group = usize::from(group);
            let count = usize::from(self.group_counts[group]);
            let columns = &self.group_columns[group * num_columns..group * num_columns + count];
            if self.group_counts[group] >= SHARED_SHIFT_MIN_COLUMNS {
                let mut shifted_sum = AkitaWideRing::zero();
                for &shift in &self.group_shifts[group] {
                    src.shift_accumulate_into(&mut shifted_sum, shift);
                }
                for &column in columns {
                    dst[usize::from(column) * n_a + a] += shifted_sum;
                }
            } else {
                for &column in columns {
                    let dst = &mut dst[usize::from(column) * n_a + a];
                    for &shift in &self.group_shifts[group] {
                        src.shift_accumulate_into(dst, shift);
                    }
                }
            }
        }
        for &column in &self.partial_columns {
            let column = usize::from(column);
            for (row_offset, (row_indices, &committed_zero_mask)) in selected_rows
                .chunks_exact(num_columns)
                .zip(committed_zero_masks)
                .enumerate()
            {
                let hot = row_indices[column];
                if row_is_committed(hot, committed_zero_mask, column) {
                    src.shift_accumulate_into(
                        &mut dst[column * n_a + a],
                        (self.row_start + row_offset) * 16 + usize::from(hot),
                    );
                }
            }
        }
    }
}

/// Visits ring elements within one semantic column segment. Each callback
/// receives the segment-relative ring index and `(column, coefficient)` pairs
/// contributed by the same trace rows.
pub(super) fn visit_segment_ring_range<const D: usize>(
    source: &TracePackedOneHot,
    ring_start: usize,
    ring_end: usize,
    mut visit: impl FnMut(usize, &[(usize, usize)]),
) -> Result<(), AkitaError> {
    validate_dimension::<D>(source.one_hot_k)?;
    let segment_rings = source.segment_ring_elems::<D>()?;
    if ring_start > ring_end || ring_end > segment_rings {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot ring range {ring_start}..{ring_end} exceeds segment size {segment_rings}"
        )));
    }
    let k = source.one_hot_k;
    let num_columns = source.rows.num_columns();
    let mut selected_rows = vec![NO_SELECTED_ROW; num_columns];
    if k >= D {
        let rings_per_row = k / D;
        let row_start = ring_start / rings_per_row;
        let row_end = ring_end.div_ceil(rings_per_row);
        let mut buckets = vec![Vec::new(); rings_per_row];
        for row in row_start..row_end.min(source.rows.num_rows()) {
            source.rows.fill_row(row, &mut selected_rows);
            let committed_zero_mask = source.rows.committed_digit_zero_mask(row);
            for bucket in &mut buckets {
                bucket.clear();
            }
            for (column, &hot) in selected_rows.iter().enumerate() {
                if !row_is_committed(hot, committed_zero_mask, column) {
                    continue;
                }
                let hot = usize::from(hot);
                if hot >= k {
                    return Err(AkitaError::InvalidInput(format!(
                        "trace one-hot row {hot} is outside K={k}"
                    )));
                }
                buckets[hot / D].push((column, hot % D));
            }
            for (offset, contributions) in buckets.iter().enumerate() {
                let ring = row * rings_per_row + offset;
                if ring_start <= ring && ring < ring_end {
                    visit(ring, contributions);
                }
            }
        }
    } else {
        let rows_per_ring = D / k;
        let mut contributions = Vec::with_capacity(num_columns * rows_per_ring);
        for ring in ring_start..ring_end {
            contributions.clear();
            for row_offset in 0..rows_per_ring {
                let row = ring * rows_per_ring + row_offset;
                if row >= source.rows.num_rows() {
                    break;
                }
                source.rows.fill_row(row, &mut selected_rows);
                let committed_zero_mask = source.rows.committed_digit_zero_mask(row);
                for (column, &hot) in selected_rows.iter().enumerate() {
                    if !row_is_committed(hot, committed_zero_mask, column) {
                        continue;
                    }
                    let hot = usize::from(hot);
                    if hot >= k {
                        return Err(AkitaError::InvalidInput(format!(
                            "trace one-hot row {hot} is outside K={k}"
                        )));
                    }
                    contributions.push((column, row_offset * k + hot));
                }
            }
            visit(ring, &contributions);
        }
    }
    Ok(())
}

/// Visits K<D ring elements as row indices for the D/K trace rows packed
/// into each ring. This avoids expanding the row buffer into contribution
/// tuples when a kernel can consume the indices directly.
pub(super) fn visit_segment_ring_row_range<const D: usize>(
    source: &TracePackedOneHot,
    ring_start: usize,
    ring_end: usize,
    mut visit: impl FnMut(usize, &[u8], &[u64]),
) -> Result<(), AkitaError> {
    validate_dimension::<D>(source.one_hot_k)?;
    if source.one_hot_k >= D {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot row traversal requires K={} < D={D}",
            source.one_hot_k
        )));
    }
    let segment_rings = source.segment_ring_elems::<D>()?;
    if ring_start > ring_end || ring_end > segment_rings {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot ring range {ring_start}..{ring_end} exceeds segment size {segment_rings}"
        )));
    }
    let rows_per_ring = D / source.one_hot_k;
    let num_columns = source.rows.num_columns();
    if !source.rows.num_rows().is_multiple_of(rows_per_ring) {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot row count {} is not aligned to {rows_per_ring} rows per D={D} ring",
            source.rows.num_rows()
        )));
    }
    let row_index_count = num_columns.checked_mul(rows_per_ring).ok_or_else(|| {
        AkitaError::InvalidInput("trace one-hot row-index buffer size overflow".to_string())
    })?;
    let mut selected_rows = vec![NO_SELECTED_ROW; row_index_count];
    let mut committed_zero_masks = vec![0u64; rows_per_ring];
    for ring in ring_start..ring_end {
        let row_start = ring * rows_per_ring;
        let populated_rows = source
            .rows
            .num_rows()
            .saturating_sub(row_start)
            .min(rows_per_ring);
        let populated_indices = &mut selected_rows[..populated_rows * num_columns];
        let populated_masks = &mut committed_zero_masks[..populated_rows];
        source.rows.fill_rows(row_start, populated_indices);
        source
            .rows
            .fill_committed_digit_zero_masks(row_start, populated_masks);
        for &hot in populated_indices.iter() {
            if hot != NO_SELECTED_ROW && usize::from(hot) >= source.one_hot_k {
                return Err(AkitaError::InvalidInput(format!(
                    "trace one-hot row {hot} is outside K={}",
                    source.one_hot_k
                )));
            }
        }
        visit(ring, populated_indices, populated_masks);
    }
    Ok(())
}

pub(super) fn coefficient_packing_partials_packed<E, const D: usize>(
    source: &TracePackedOneHot,
    plan: SubringCoefficientPackingPlan<'_, E>,
) -> Result<Vec<AkitaField>, AkitaError>
where
    E: ExtField<AkitaField> + FpExtEncoding<AkitaField>,
{
    plan.validate::<D>(source.num_vars)?;
    let point = plan.point;
    let expected = point.num_live_positions();
    let actual = RootPolyShape::<AkitaField, D>::num_ring_elems(source);
    if actual != expected {
        return Err(AkitaError::InvalidSize { expected, actual });
    }

    let geometry = point.geometry();
    if E::DEGREE != geometry.extension_degree() {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing field extension degree mismatch".to_string(),
        ));
    }
    let num_blocks = point.num_live_blocks();
    let subring_dimension = geometry.challenge_subring_dimension();
    let stride = geometry.subring_embedding_stride();
    if point.packing_weights().len() != stride {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing weight width disagrees with its geometry".to_string(),
        ));
    }
    let packed_len = num_blocks.checked_mul(subring_dimension).ok_or_else(|| {
        AkitaError::InvalidInput("coefficient-packing accumulator length overflow".to_string())
    })?;
    let mut packed = vec![E::zero(); packed_len];
    let positions_per_block = point.num_positions_per_block();
    let segment_rings = source.segment_ring_elems::<D>()?;

    visit_segment_ring_range::<D>(source, 0, segment_rings, |ring, contributions| {
        for &(column, coefficient) in contributions {
            let position = column * segment_rings + ring;
            let block = position / positions_per_block;
            let position_in_block = position % positions_per_block;
            let subring = coefficient / stride;
            let low_coefficient = coefficient % stride;
            packed[block * subring_dimension + subring] += point.position_weights()
                [position_in_block]
                * point.packing_weights()[low_coefficient];
        }
    })?;

    let partial_width = geometry.partial_base_field_width();
    let output_len = num_blocks.checked_mul(partial_width).ok_or_else(|| {
        AkitaError::InvalidInput("coefficient-packing output length overflow".to_string())
    })?;
    let mut coordinates = vec![AkitaField::zero(); output_len];
    for (packed_index, coefficient) in packed.into_iter().enumerate() {
        let block = packed_index / subring_dimension;
        let subring = packed_index % subring_dimension;
        let extension_coordinates = coefficient.ext_coords();
        if extension_coordinates.len() != geometry.extension_degree() {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing extension encoding width mismatch".to_string(),
            ));
        }
        for (extension_coordinate, &coordinate) in extension_coordinates.iter().enumerate() {
            let local_index =
                geometry.partial_base_field_coordinate_index(extension_coordinate, subring)?;
            coordinates[block * partial_width + local_index] = coordinate;
        }
    }
    Ok(coordinates)
}

pub(super) fn flush_wide<const D: usize>(
    wide: &mut [AkitaWideRing<D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
) {
    for (wide, reduced) in wide.iter_mut().zip(reduced) {
        *reduced += std::mem::replace(wide, WideCyclotomicRing::zero()).reduce();
    }
}

pub(super) fn flush_deferred_rank<const D: usize>(
    rank_deferred: &mut [DeferredFp128Ring<D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
    n_a: usize,
    a: usize,
) {
    for (column, value) in rank_deferred.iter_mut().enumerate() {
        let index = column * n_a + a;
        reduced[index] += value.reduce_and_clear();
    }
}

#[inline(always)]
pub(super) fn full_row_coefficients<const N: usize>(
    selected_rows: &[u8],
    committed_zero_masks: &[u64],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
) -> Option<[usize; N]> {
    if selected_rows.len() != N * num_columns || committed_zero_masks.len() != N {
        return None;
    }
    let coefficients = std::array::from_fn(|row| {
        row * one_hot_k + usize::from(selected_rows[row * num_columns + column])
    });
    (0..N)
        .all(|row| {
            row_is_committed(
                selected_rows[row * num_columns + column],
                committed_zero_masks[row],
                column,
            )
        })
        .then_some(coefficients)
}

#[inline(always)]
pub(super) fn shift_accumulate_full_rows<const D: usize, const N: usize>(
    src: &AkitaWideRing<D>,
    dst: &mut AkitaWideRing<D>,
    selected_rows: &[u8],
    committed_zero_masks: &[u64],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
) -> bool {
    let Some(coefficients) = full_row_coefficients::<N>(
        selected_rows,
        committed_zero_masks,
        num_columns,
        column,
        one_hot_k,
    ) else {
        return false;
    };
    for coefficient in coefficients {
        src.shift_accumulate_into(dst, coefficient);
    }
    true
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the fixed-row fast path keeps its source, destination, row views, and geometry explicit"
)]
pub(super) fn try_shift_accumulate_full_rows<const D: usize>(
    src: &AkitaWideRing<D>,
    dst: &mut AkitaWideRing<D>,
    selected_rows: &[u8],
    committed_zero_masks: &[u64],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
    rows_per_ring: usize,
) -> bool {
    match rows_per_ring {
        2 => shift_accumulate_full_rows::<D, 2>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        4 => shift_accumulate_full_rows::<D, 4>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        8 => shift_accumulate_full_rows::<D, 8>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        16 => shift_accumulate_full_rows::<D, 16>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        32 => shift_accumulate_full_rows::<D, 32>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        _ => false,
    }
}

pub(super) fn validate_block_geometry(
    segment_rings: usize,
    column_capacity: usize,
    num_positions: usize,
) -> Result<(usize, usize), AkitaError> {
    if num_positions == 0 || !num_positions.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot positions per block {num_positions} must be a nonzero power of two"
        )));
    }
    let total_rings = segment_rings
        .checked_mul(column_capacity)
        .ok_or_else(|| AkitaError::InvalidInput("trace one-hot ring count overflow".to_string()))?;
    Ok((total_rings, total_rings.div_ceil(num_positions)))
}

pub(super) fn trace_block_task_parts<const D: usize>(
    one_hot_k: usize,
    num_positions: usize,
    blocks_per_column: usize,
) -> usize {
    let ring_alignment = (one_hot_k / D).max(1);
    debug_assert_eq!(num_positions % ring_alignment, 0);
    let max_parts = num_positions / ring_alignment;
    let target_tasks = rayon::current_num_threads()
        .saturating_mul(TASKS_PER_RAYON_WORKER)
        .max(1);
    target_tasks.div_ceil(blocks_per_column).clamp(1, max_parts)
}

pub(super) fn trace_block_part_range(
    num_positions: usize,
    ring_alignment: usize,
    part: usize,
    parts: usize,
) -> (usize, usize) {
    debug_assert_eq!(num_positions % ring_alignment, 0);
    let aligned_positions = num_positions / ring_alignment;
    (
        part * aligned_positions / parts * ring_alignment,
        (part + 1) * aligned_positions / parts * ring_alignment,
    )
}
