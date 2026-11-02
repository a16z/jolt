//! Products of categorical selector pairs in the first cycle round.
//!
//! A pair of 4-bit selectors has only 256 possible endpoint products.
//! Four table reads recover its quadratic at 1, 2 and infinity, replacing
//! three field multiplications without expanding any cycle-indexed table.

use super::super::lazy_ra::ChunkIndexSource;
use super::{LazyFoldedRa, OptimizedInstructionRaVirtualizationKernel};
use jolt_field::{Accumulator, JoltField};

struct Quartet<F> {
    products: [Vec<F>; 2],
    tail_index: usize,
    tail_value: F,
}

#[inline(always)]
fn quadratic<F: JoltField>(table: &[F], low: usize, high: usize, radix: usize) -> [F; 3] {
    let mask = radix - 1;
    let ll = table[low];
    let hh = table[high];
    let lh = table[(low & !mask) | (high & mask)];
    let hl = table[(high & !mask) | (low & mask)];
    let leading = hh + ll - lh - hl;
    let next = hh + leading;
    [hh, next + next - ll, leading]
}

#[inline(always)]
fn third<F: JoltField>(samples: [F; 3]) -> F {
    let next = samples[1] + samples[2];
    next + next - samples[0]
}

impl<F: JoltField> OptimizedInstructionRaVirtualizationKernel<F> {
    pub(super) fn categorical_lanes(&self, prefix_pairs: usize) -> Option<Vec<F::Accumulator>> {
        if self.num_committed_per_virtual != 4 {
            return None;
        }
        let LazyFoldedRa::Lazy {
            tables,
            width: 1,
            factors: 1,
            source,
            ..
        } = &self.folded_ra
        else {
            return None;
        };
        let radix = tables.first()?.len();
        if radix > 16 || prefix_pairs < radix * radix {
            return None;
        }
        let bits = radix.trailing_zeros() as usize;
        let pair_bits = 2 * bits;
        let pair_mask = radix * radix - 1;
        let group_mask = (1u128 << (4 * bits)) - 1;
        let tail_index = source.packed_index(source.cycles() - 1);
        let groups: Vec<_> = tables
            .chunks_exact(4)
            .take(self.active_virtuals)
            .enumerate()
            .map(|(g, tables)| {
                let shift = (source.num_columns() - 4 * (g + 1)) * bits;
                Quartet {
                    products: std::array::from_fn(|pair| {
                        tables[2 * pair]
                            .iter()
                            .flat_map(|a| tables[2 * pair + 1].iter().map(|b| *a * *b))
                            .collect()
                    }),
                    tail_index: ((tail_index >> shift) & group_mask) as usize,
                    tail_value: self.folded_ra.tail_values()[4 * g..4 * g + 4]
                        .iter()
                        .copied()
                        .product(),
                }
            })
            .collect();
        let lanes = self.gruen.par_fold_out_in_prefix(
            prefix_pairs,
            || [F::Accumulator::default(); 4],
            |lanes, row, _, weight| {
                let lo = source.packed_index(2 * row);
                let hi = source.packed_index(2 * row + 1);
                let mut row_lanes = [F::Accumulator::default(); 4];
                for (g, group) in groups.iter().enumerate() {
                    let shift = (source.num_columns() - 4 * (g + 1)) * bits;
                    let low = ((lo >> shift) & group_mask) as usize;
                    let high = ((hi >> shift) & group_mask) as usize;
                    if low == high {
                        let value = if low == group.tail_index {
                            group.tail_value
                        } else {
                            group.products[0][low >> pair_bits] * group.products[1][low & pair_mask]
                        };
                        for lane in &mut row_lanes[..3] {
                            lane.add(value);
                        }
                        continue;
                    }
                    let left = quadratic(
                        &group.products[0],
                        low >> pair_bits,
                        high >> pair_bits,
                        radix,
                    );
                    let right =
                        quadratic(&group.products[1], low & pair_mask, high & pair_mask, radix);
                    row_lanes[0].fmadd(left[0], right[0]);
                    row_lanes[1].fmadd(left[1], right[1]);
                    row_lanes[2].fmadd(third(left), third(right));
                    row_lanes[3].fmadd(left[2], right[2]);
                }
                for (lane, row_lane) in lanes.iter_mut().zip(row_lanes) {
                    lane.fmadd(weight, row_lane.reduce());
                }
            },
            |_, weight, lanes| {
                lanes.map(|lane| {
                    let mut result = F::Accumulator::default();
                    result.fmadd(weight, lane.reduce());
                    result
                })
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    a.merge(b);
                }
                a
            },
        );
        Some(lanes.to_vec())
    }
}
