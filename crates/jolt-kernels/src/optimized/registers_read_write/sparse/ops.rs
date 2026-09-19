use core::{cmp::Ordering, mem::MaybeUninit};

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::{Accumulator, JoltField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    layout::{
        merge_bind, merge_count, merge_soa, split_indexed, split_pair_group, split_soa_pair_group,
        Cell, IndexedMeta, MatrixEntry, SeedEntry, SoaSpareBlock, SparseEntry,
    },
    CoeffLut, CycleState, LutIndex, OneHotCoeff,
};

/// Block boundaries advanced so no `row >> pair_bits` group is split.
fn pair_aligned_bounds<E: Cell>(entries: &[E], pair_bits: u32) -> Vec<usize> {
    const BLOCK_TARGET: usize = 1 << 14;
    let len = entries.len();
    let block_count = len.div_ceil(BLOCK_TARGET).max(1);
    let mut bounds: Vec<usize> = Vec::with_capacity(block_count + 1);
    bounds.push(0);
    for block in 1..block_count {
        let mut index = block * len / block_count;
        while index < len
            && index > 0
            && entries[index].row() >> pair_bits == entries[index - 1].row() >> pair_bits
        {
            index += 1;
        }
        #[expect(clippy::unwrap_used, reason = "bounds starts non-empty")]
        if index > *bounds.last().unwrap() && index < len {
            bounds.push(index);
        }
    }
    bounds.push(len);
    bounds
}

/// Bind and compact entries within pair-aligned blocks.
///
/// Writes stay behind unread groups because merging never grows a group.
/// A group uses scratch until its output fits entirely in the vacated prefix.
pub(super) fn bind_sparse_entries_in_place<E>(
    entries: &mut Vec<E>,
    bind: impl Fn(Option<&E>, Option<&E>) -> E + Sync,
) where
    E: Cell,
{
    let bounds = pair_aligned_bounds(entries, 1);
    let blocks = bounds.len() - 1;

    let bind_block = |scratch: &mut Vec<E>, block: &mut [E]| -> usize {
        let len = block.len();
        let mut write = 0usize;
        let mut group_start = 0usize;
        while group_start < len {
            let pair_row = block[group_start].row() / 2;
            let mut group_end = group_start + 1;
            while group_end < len && block[group_end].row() / 2 == pair_row {
                group_end += 1;
            }
            if write + (group_end - group_start) <= group_start {
                let (out, tail) = block.split_at_mut(group_start);
                let (evens, odds) = split_pair_group(&tail[..group_end - group_start]);
                merge_bind(evens, odds, &bind, |entry| {
                    out[write] = entry;
                    write += 1;
                });
            } else {
                scratch.clear();
                let (evens, odds) = split_pair_group(&block[group_start..group_end]);
                merge_bind(evens, odds, &bind, |entry| scratch.push(entry));
                debug_assert!(write + scratch.len() <= group_end);
                block[write..write + scratch.len()].copy_from_slice(scratch);
                write += scratch.len();
            }
            group_start = group_end;
        }
        write
    };

    let mut block_slices = Vec::with_capacity(blocks);
    let mut rest: &mut [E] = entries;
    for block in 0..blocks {
        let (head, tail) = rest.split_at_mut(bounds[block + 1] - bounds[block]);
        block_slices.push(head);
        rest = tail;
    }
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = block_slices
        .into_par_iter()
        .map_init(
            || Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
            |scratch, block| bind_block(scratch, block),
        )
        .collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = {
        let mut scratch = Vec::with_capacity(2 << REGISTER_ADDRESS_BITS);
        block_slices
            .into_iter()
            .map(|block| bind_block(&mut scratch, block))
            .collect()
    };

    // Each compacted run stays within its original block.
    let mut total = counts[0];
    for block in 1..blocks {
        let src = bounds[block];
        if src != total {
            entries.copy_within(src..src + counts[block], total);
        }
        total += counts[block];
    }
    entries.truncate(total);
}

/// [`bind_sparse_entries_in_place`] for index-parallel SoA columns.
pub(super) fn bind_indexed_in_place_soa<F: JoltField>(
    vals: &mut Vec<F>,
    metas: &mut Vec<IndexedMeta>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r: F,
) {
    debug_assert_eq!(vals.len(), metas.len());
    let bounds = pair_aligned_bounds(metas, 1);
    let blocks = bounds.len() - 1;

    let bind_block = |scratch: &mut (Vec<F>, Vec<IndexedMeta>),
                      vals_block: &mut [F],
                      metas_block: &mut [IndexedMeta]|
     -> usize {
        let len = metas_block.len();
        let mut write = 0usize;
        let mut group_start = 0usize;
        while group_start < len {
            let pair_row = metas_block[group_start].row() / 2;
            let mut group_end = group_start + 1;
            while group_end < len && metas_block[group_end].row() / 2 == pair_row {
                group_end += 1;
            }
            let group_len = group_end - group_start;
            let bind_pair = |even: Option<SparseEntry<F, LutIndex>>,
                             odd: Option<SparseEntry<F, LutIndex>>| {
                split_indexed(SparseEntry::<F, LutIndex>::bind(
                    even.as_ref(),
                    odd.as_ref(),
                    r,
                    ra_lut,
                    wa_lut,
                ))
            };
            if write + group_len <= group_start {
                let (vals_out, vals_tail) = vals_block.split_at_mut(group_start);
                let (metas_out, metas_tail) = metas_block.split_at_mut(group_start);
                let (evens, odds) =
                    split_soa_pair_group(&vals_tail[..group_len], &metas_tail[..group_len]);
                merge_soa(evens, odds, |even, odd| {
                    let (val, meta) = bind_pair(even, odd);
                    vals_out[write] = val;
                    metas_out[write] = meta;
                    write += 1;
                });
            } else {
                scratch.0.clear();
                scratch.1.clear();
                let (evens, odds) = split_soa_pair_group(
                    &vals_block[group_start..group_end],
                    &metas_block[group_start..group_end],
                );
                merge_soa(evens, odds, |even, odd| {
                    let (val, meta) = bind_pair(even, odd);
                    scratch.0.push(val);
                    scratch.1.push(meta);
                });
                debug_assert!(write + scratch.0.len() <= group_end);
                vals_block[write..write + scratch.0.len()].copy_from_slice(&scratch.0);
                metas_block[write..write + scratch.1.len()].copy_from_slice(&scratch.1);
                write += scratch.0.len();
            }
            group_start = group_end;
        }
        write
    };

    let mut block_slices = Vec::with_capacity(blocks);
    let mut vals_rest: &mut [F] = vals;
    let mut metas_rest: &mut [IndexedMeta] = metas;
    for block in 0..blocks {
        let take = bounds[block + 1] - bounds[block];
        let (vals_head, vals_tail) = vals_rest.split_at_mut(take);
        let (metas_head, metas_tail) = metas_rest.split_at_mut(take);
        block_slices.push((vals_head, metas_head));
        vals_rest = vals_tail;
        metas_rest = metas_tail;
    }
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = block_slices
        .into_par_iter()
        .map_init(
            || {
                (
                    Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
                    Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
                )
            },
            |scratch, (vals_block, metas_block)| bind_block(scratch, vals_block, metas_block),
        )
        .collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = {
        let mut scratch = (
            Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
            Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
        );
        block_slices
            .into_iter()
            .map(|(vals_block, metas_block)| bind_block(&mut scratch, vals_block, metas_block))
            .collect()
    };

    // Compact both columns in lockstep.
    let mut total = counts[0];
    for block in 1..blocks {
        let src = bounds[block];
        if src != total {
            vals.copy_within(src..src + counts[block], total);
            metas.copy_within(src..src + counts[block], total);
        }
        total += counts[block];
    }
    vals.truncate(total);
    metas.truncate(total);
}

/// Merge-bind indexed SoA entries directly into field entries.
pub(super) fn bind_indexed_to_direct<F: JoltField>(
    vals: &[F],
    metas: &[IndexedMeta],
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r: F,
) -> Vec<SparseEntry<F, F>> {
    let pair_predicate = |a: &IndexedMeta, b: &IndexedMeta| a.row() / 2 == b.row() / 2;
    let bounds = pair_aligned_bounds(metas, 1);
    let blocks = bounds.len() - 1;

    let count_block = |block: usize| -> usize {
        metas[bounds[block]..bounds[block + 1]]
            .chunk_by(pair_predicate)
            .map(|group| {
                let (evens, odds) = split_pair_group(group);
                merge_count(evens, odds)
            })
            .sum()
    };
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = (0..blocks).into_par_iter().map(count_block).collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = (0..blocks).map(count_block).collect();

    let bound_length: usize = counts.iter().sum();
    let mut bound: Vec<SparseEntry<F, F>> = Vec::with_capacity(bound_length);
    let mut out_slices = Vec::with_capacity(blocks);
    let mut out_rest = bound.spare_capacity_mut();
    for &count in &counts {
        let (out_slice, next_out) = out_rest.split_at_mut(count);
        out_rest = next_out;
        out_slices.push(out_slice);
    }

    let unused = CycleState::<F>::unused_lut();
    let deref = |entry: SparseEntry<F, LutIndex>| SparseEntry::<F, F> {
        val: entry.val,
        prev_val: entry.prev_val,
        next_val: entry.next_val,
        row: entry.row,
        ra: entry.ra.value(ra_lut),
        wa: entry.wa.value(wa_lut),
        col: entry.col,
    };
    let fill_block = |(block, out): (usize, &mut [MaybeUninit<SparseEntry<F, F>>])| {
        let mut written = 0usize;
        let mut cursor = bounds[block];
        for group in metas[bounds[block]..bounds[block + 1]].chunk_by(pair_predicate) {
            let group_vals = &vals[cursor..cursor + group.len()];
            cursor += group.len();
            let (evens, odds) = split_soa_pair_group(group_vals, group);
            merge_soa(evens, odds, |even, odd| {
                out[written] = MaybeUninit::new(SparseEntry::<F, F>::bind(
                    even.map(&deref).as_ref(),
                    odd.map(&deref).as_ref(),
                    r,
                    &unused,
                    &unused,
                ));
                written += 1;
            });
        }
        debug_assert_eq!(written, out.len());
    };
    #[cfg(feature = "parallel")]
    out_slices.into_par_iter().enumerate().for_each(fill_block);
    #[cfg(not(feature = "parallel"))]
    out_slices.into_iter().enumerate().for_each(fill_block);

    // SAFETY: the count pass sized every block's output slice exactly, the
    // slices partition `bound`'s spare capacity up to `bound_length`, and
    // the merge writes each slot exactly once.
    unsafe {
        bound.set_len(bound_length);
    }
    bound
}

/// Accumulate one row pair's `[q(0), q(∞)]` terms.
fn accumulate_pair_group<F, E>(
    evens: &[E],
    odds: &[E],
    inc_evals: [F; 2],
    inner: &mut [F::Accumulator; 2],
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
) where
    F: JoltField,
    E: MatrixEntry<F>,
{
    let mut i = 0;
    let mut j = 0;
    while i < evens.len() && j < odds.len() {
        match evens[i].col().cmp(&odds[j].col()) {
            Ordering::Equal => {
                E::accumulate_pair_evals(
                    Some(&evens[i]),
                    Some(&odds[j]),
                    inc_evals,
                    inner,
                    ra_lut,
                    wa_lut,
                );
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                E::accumulate_pair_evals(Some(&evens[i]), None, inc_evals, inner, ra_lut, wa_lut);
                i += 1;
            }
            Ordering::Greater => {
                E::accumulate_pair_evals(None, Some(&odds[j]), inc_evals, inner, ra_lut, wa_lut);
                j += 1;
            }
        }
    }
    for even in &evens[i..] {
        E::accumulate_pair_evals(Some(even), None, inc_evals, inner, ra_lut, wa_lut);
    }
    for odd in &odds[j..] {
        E::accumulate_pair_evals(None, Some(odd), inc_evals, inner, ra_lut, wa_lut);
    }
}

/// Cycle-round `[q(0), leading coefficient]` over sparse entries.
pub(super) fn sparse_quadratic<F, E>(
    entries: &[E],
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    e_in: &[F],
    e_out: &[F],
    inc_evals_at: impl Fn(usize) -> [F; 2] + Sync,
) -> [F; 2]
where
    F: JoltField,
    E: MatrixEntry<F>,
{
    let e_in_len = e_in.len();
    let in_bits = if e_in_len <= 1 {
        0
    } else {
        e_in_len.trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let group_contribution = |group: &[E]| -> [F; 2] {
        let x_out = (group[0].row() / 2) >> in_bits;
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        for pair_group in group.chunk_by(|a, b| a.row() / 2 == b.row() / 2) {
            let z = pair_group[0].row() / 2;
            let e_in_eval = if e_in_len <= 1 {
                F::one()
            } else {
                e_in[z & mask]
            };
            let inc_evals = inc_evals_at(z);

            let mut inner = [F::Accumulator::default(), F::Accumulator::default()];
            let (evens, odds) = split_pair_group(pair_group);
            accumulate_pair_group(evens, odds, inc_evals, &mut inner, ra_lut, wa_lut);

            acc[0].fmadd(e_in_eval, inner[0].reduce());
            acc[1].fmadd(e_in_eval, inner[1].reduce());
        }
        let e_out_eval = e_out[x_out];
        [e_out_eval * acc[0].reduce(), e_out_eval * acc[1].reduce()]
    };

    let group_predicate = |a: &E, b: &E| (a.row() / 2) >> in_bits == (b.row() / 2) >> in_bits;
    #[cfg(feature = "parallel")]
    {
        entries
            .par_chunk_by(group_predicate)
            .map(group_contribution)
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
    #[cfg(not(feature = "parallel"))]
    {
        entries
            .chunk_by(group_predicate)
            .map(group_contribution)
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
}

/// [`sparse_quadratic`] for indexed SoA columns.
pub(super) fn sparse_quadratic_soa<F: JoltField>(
    vals: &[F],
    metas: &[IndexedMeta],
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    e_in: &[F],
    e_out: &[F],
    inc_evals_at: impl Fn(usize) -> [F; 2] + Sync,
) -> [F; 2] {
    let e_in_len = e_in.len();
    let in_bits = if e_in_len <= 1 {
        0
    } else {
        e_in_len.trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let group_contribution = |group: &[IndexedMeta]| -> [F; 2] {
        // SAFETY: `group` is a sub-slice of `metas`, so the offset is the
        // group's start index; `vals` is index-parallel to `metas`.
        let start = unsafe { group.as_ptr().offset_from(metas.as_ptr()) } as usize;
        let group_vals = &vals[start..start + group.len()];
        let x_out = (group[0].row() / 2) >> in_bits;
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        let mut offset = 0usize;
        while offset < group.len() {
            let z = group[offset].row() / 2;
            let mut end = offset + 1;
            while end < group.len() && group[end].row() / 2 == z {
                end += 1;
            }
            let e_in_eval = if e_in_len <= 1 {
                F::one()
            } else {
                e_in[z & mask]
            };
            let inc_evals = inc_evals_at(z);

            let mut inner = [F::Accumulator::default(), F::Accumulator::default()];
            let (evens, odds) = split_soa_pair_group(&group_vals[offset..end], &group[offset..end]);
            merge_soa(evens, odds, |even, odd| {
                <SparseEntry<F, LutIndex> as MatrixEntry<F>>::accumulate_pair_evals(
                    even.as_ref(),
                    odd.as_ref(),
                    inc_evals,
                    &mut inner,
                    ra_lut,
                    wa_lut,
                );
            });

            acc[0].fmadd(e_in_eval, inner[0].reduce());
            acc[1].fmadd(e_in_eval, inner[1].reduce());
            offset = end;
        }
        let e_out_eval = e_out[x_out];
        [e_out_eval * acc[0].reduce(), e_out_eval * acc[1].reduce()]
    };

    let group_predicate =
        |a: &IndexedMeta, b: &IndexedMeta| (a.row() / 2) >> in_bits == (b.row() / 2) >> in_bits;
    #[cfg(feature = "parallel")]
    {
        metas
            .par_chunk_by(group_predicate)
            .map(group_contribution)
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
    #[cfg(not(feature = "parallel"))]
    {
        metas
            .chunk_by(group_predicate)
            .map(group_contribution)
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
}

/// Per-thread intermediate rows for one 4-row group.
type FusedScratch<F> = (Vec<SparseEntry<F, LutIndex>>, Vec<SparseEntry<F, LutIndex>>);

fn fused_scratch<F: JoltField>() -> FusedScratch<F> {
    (
        Vec::with_capacity(1 << REGISTER_ADDRESS_BITS),
        Vec::with_capacity(1 << REGISTER_ADDRESS_BITS),
    )
}

/// Rebuild one 4-row group's two first-bind intermediates.
#[inline]
fn fused_intermediates<F: JoltField>(
    group: &[SeedEntry],
    seed_ra_lut: &CoeffLut<F>,
    seed_wa_lut: &CoeffLut<F>,
    r1: F,
    scratch: &mut FusedScratch<F>,
) {
    scratch.0.clear();
    scratch.1.clear();
    let half_start = group.partition_point(|entry| entry.row % 4 < 2);
    let (first, second) = group.split_at(half_start);
    let bind = |even: Option<&SeedEntry>, odd: Option<&SeedEntry>| {
        SeedEntry::bind(even, odd, r1, seed_ra_lut, seed_wa_lut)
    };
    let (evens, odds) = split_pair_group(first);
    merge_bind(evens, odds, &bind, |entry| scratch.0.push(entry));
    let (evens, odds) = split_pair_group(second);
    merge_bind(evens, odds, &bind, |entry| scratch.1.push(entry));
}

/// Round-1 quadratic with first-bind rows rebuilt in per-thread scratch.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors sparse_quadratic plus the two table generations"
)]
pub(super) fn sparse_quadratic_fused<F: JoltField>(
    entries: &[SeedEntry],
    seed_ra_lut: &CoeffLut<F>,
    seed_wa_lut: &CoeffLut<F>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r1: F,
    e_in: &[F],
    e_out: &[F],
    inc_evals_at: impl Fn(usize) -> [F; 2] + Sync,
) -> [F; 2] {
    let e_in_len = e_in.len();
    let in_bits = if e_in_len <= 1 {
        0
    } else {
        e_in_len.trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let group_contribution = |scratch: &mut FusedScratch<F>, group: &[SeedEntry]| -> [F; 2] {
        let x_out = (group[0].row() / 4) >> in_bits;
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        for pair_group in group.chunk_by(|a, b| a.row() / 4 == b.row() / 4) {
            let z = pair_group[0].row() / 4;
            let e_in_eval = if e_in_len <= 1 {
                F::one()
            } else {
                e_in[z & mask]
            };
            let inc_evals = inc_evals_at(z);

            fused_intermediates(pair_group, seed_ra_lut, seed_wa_lut, r1, scratch);
            let mut inner = [F::Accumulator::default(), F::Accumulator::default()];
            accumulate_pair_group(
                &scratch.0, &scratch.1, inc_evals, &mut inner, ra_lut, wa_lut,
            );

            acc[0].fmadd(e_in_eval, inner[0].reduce());
            acc[1].fmadd(e_in_eval, inner[1].reduce());
        }
        let e_out_eval = e_out[x_out];
        [e_out_eval * acc[0].reduce(), e_out_eval * acc[1].reduce()]
    };

    let group_predicate =
        |a: &SeedEntry, b: &SeedEntry| (a.row() / 4) >> in_bits == (b.row() / 4) >> in_bits;
    #[cfg(feature = "parallel")]
    {
        entries
            .par_chunk_by(group_predicate)
            .map_init(fused_scratch, |scratch, group| {
                group_contribution(scratch, group)
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = fused_scratch();
        entries
            .chunk_by(group_predicate)
            .map(|group| group_contribution(&mut scratch, group))
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
}

/// Fuse two seed binds into quarter-domain indexed SoA columns.
pub(super) fn bind_seed_entries_fused<F: JoltField>(
    entries: &[SeedEntry],
    seed_ra_lut: &CoeffLut<F>,
    seed_wa_lut: &CoeffLut<F>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r1: F,
    r2: F,
) -> (Vec<F>, Vec<IndexedMeta>) {
    // One bit per register column.
    const _: () = assert!(REGISTER_ADDRESS_BITS <= 7);
    let group_predicate = |a: &SeedEntry, b: &SeedEntry| a.row() / 4 == b.row() / 4;
    let bounds = pair_aligned_bounds(entries, 2);
    let blocks = bounds.len() - 1;

    // Each 4-row group emits one entry per distinct column.
    let count_block = |block: usize| -> usize {
        entries[bounds[block]..bounds[block + 1]]
            .chunk_by(group_predicate)
            .map(|group| {
                group
                    .iter()
                    .fold(0u128, |mask, entry| mask | (1u128 << entry.col))
                    .count_ones() as usize
            })
            .sum()
    };
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = (0..blocks).into_par_iter().map(count_block).collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = (0..blocks).map(count_block).collect();

    let bound_length: usize = counts.iter().sum();
    let mut vals: Vec<F> = Vec::with_capacity(bound_length);
    let mut metas: Vec<IndexedMeta> = Vec::with_capacity(bound_length);
    let mut out_slices = Vec::with_capacity(blocks);
    let mut vals_rest = vals.spare_capacity_mut();
    let mut metas_rest = metas.spare_capacity_mut();
    for &count in &counts {
        let (vals_slice, next_vals) = vals_rest.split_at_mut(count);
        let (metas_slice, next_metas) = metas_rest.split_at_mut(count);
        vals_rest = next_vals;
        metas_rest = next_metas;
        out_slices.push((vals_slice, metas_slice));
    }

    let fill_block =
        |scratch: &mut FusedScratch<F>,
         (block, (vals_out, metas_out)): (usize, SoaSpareBlock<'_, F>)| {
            let mut written = 0usize;
            for group in entries[bounds[block]..bounds[block + 1]].chunk_by(group_predicate) {
                fused_intermediates(group, seed_ra_lut, seed_wa_lut, r1, scratch);
                merge_bind(
                    &scratch.0,
                    &scratch.1,
                    &|even, odd| {
                        split_indexed(SparseEntry::<F, LutIndex>::bind(
                            even, odd, r2, ra_lut, wa_lut,
                        ))
                    },
                    |(val, meta)| {
                        vals_out[written] = MaybeUninit::new(val);
                        metas_out[written] = MaybeUninit::new(meta);
                        written += 1;
                    },
                );
            }
            debug_assert_eq!(written, vals_out.len());
        };
    #[cfg(feature = "parallel")]
    out_slices
        .into_par_iter()
        .enumerate()
        .for_each_init(fused_scratch, |scratch, item| fill_block(scratch, item));
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = fused_scratch();
        out_slices
            .into_iter()
            .enumerate()
            .for_each(|item| fill_block(&mut scratch, item));
    }

    // SAFETY: exact per-block counts partition both spare capacities; the
    // merge initializes every slot once.
    unsafe {
        vals.set_len(bound_length);
        metas.set_len(bound_length);
    }
    (vals, metas)
}
