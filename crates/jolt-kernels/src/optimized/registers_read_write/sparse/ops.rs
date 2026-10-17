use core::{cmp::Ordering, mem::MaybeUninit};

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::{Accumulator, JoltField};
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    layout::{merge_bind, split_pair_group, Cell, MatrixEntry, SeedEntry, SparseEntry},
    CoeffLut, CycleState, LutIndex, Seeds,
};

/// Block boundaries advanced so no `row >> pair_bits` group is split.
pub(crate) fn pair_aligned_bounds<E: Cell>(entries: &[E], pair_bits: u32) -> Vec<usize> {
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
pub(crate) fn bind_sparse_entries_in_place<E>(
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

pub(super) fn accumulate_pair_group<F, E>(
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

type FusedScratch<F> = (Vec<SparseEntry<F, LutIndex>>, Vec<SparseEntry<F, LutIndex>>);

pub(super) fn fused_scratch<F: JoltField>() -> FusedScratch<F> {
    (
        Vec::with_capacity(1 << REGISTER_ADDRESS_BITS),
        Vec::with_capacity(1 << REGISTER_ADDRESS_BITS),
    )
}

#[inline]
pub(super) fn fused_intermediates<F: JoltField>(
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

/// Cycles per second-bind block: a multiple of four, so no 4-row group
/// straddles two blocks.
const BIND_BLOCK_CYCLES: usize = 1 << 12;

/// Fuse two seed binds into quarter-domain field-coefficient entries.
#[expect(
    clippy::too_many_arguments,
    reason = "the seed source and its cycle count, both table generations, and both challenges"
)]
pub(super) fn bind_seed_entries_fused<F: JoltField>(
    seeds: &Seeds,
    cycles: usize,
    seed_ra_lut: &CoeffLut<F>,
    seed_wa_lut: &CoeffLut<F>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r1: F,
    r2: F,
) -> Result<Vec<SparseEntry<F, F>>, WitnessError> {
    const _: () = assert!(REGISTER_ADDRESS_BITS <= 7);
    let block_cycles = BIND_BLOCK_CYCLES.min(cycles);
    let blocks = cycles / block_cycles;
    let block_rows = |block: usize| block * block_cycles..(block + 1) * block_cycles;

    let count_block = |block: usize| -> Result<usize, WitnessError> {
        let mut count = 0usize;
        seeds.for_each_group(block_rows(block), 2, |_, quad| {
            count += quad
                .iter()
                .fold(0u128, |mask, entry| mask | (1u128 << entry.col))
                .count_ones() as usize;
        })?;
        Ok(count)
    };
    #[cfg(feature = "parallel")]
    let counts = (0..blocks)
        .into_par_iter()
        .map(count_block)
        .collect::<Result<Vec<usize>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let counts = (0..blocks)
        .map(count_block)
        .collect::<Result<Vec<usize>, _>>()?;

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
    let to_field = |entry: &SparseEntry<F, LutIndex>| entry.to_field(ra_lut, wa_lut);
    let fill_block = |scratch: &mut FusedScratch<F>,
                      (block, out): (usize, &mut [MaybeUninit<SparseEntry<F, F>>])|
     -> Result<(), WitnessError> {
        let mut written = 0usize;
        seeds.for_each_group(block_rows(block), 2, |_, quad| {
            fused_intermediates(quad, seed_ra_lut, seed_wa_lut, r1, scratch);
            merge_bind(
                &scratch.0,
                &scratch.1,
                &|even, odd| {
                    SparseEntry::<F, F>::bind(
                        even.map(to_field).as_ref(),
                        odd.map(to_field).as_ref(),
                        r2,
                        &unused,
                        &unused,
                    )
                },
                |entry| {
                    out[written] = MaybeUninit::new(entry);
                    written += 1;
                },
            );
        })?;
        assert_eq!(
            written,
            out.len(),
            "second-bind fill diverged from its count"
        );
        Ok(())
    };
    #[cfg(feature = "parallel")]
    out_slices
        .into_par_iter()
        .enumerate()
        .try_for_each_init(fused_scratch, |scratch, item| fill_block(scratch, item))?;
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = fused_scratch();
        out_slices
            .into_iter()
            .enumerate()
            .try_for_each(|item| fill_block(&mut scratch, item))?;
    }

    // SAFETY: exact per-block counts partition the spare capacity, and every
    // fill block asserted it initialized each of its slots.
    unsafe {
        bound.set_len(bound_length);
    }
    Ok(bound)
}
