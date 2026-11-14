//! The streaming twin of [`LazyFoldedRa`]'s lazy state: address-folded
//! one-hot selectors bound `LowToHigh` and gathered from blocks of the
//! witness's resident rows instead of by cycle index over a collected column.
//!
//! An unbound selector column is a point mass, one scale-table lookup per
//! cycle, and after `b` binds the bound value at index `j` is
//!
//! ```text
//! value(i, j) = Σ_{branch < 2^b} tables[i][branch][index_i(j·2^b + branch)]
//! ```
//!
//! where branch table `branch` is the base table pre-scaled by that bound-bit
//! pattern's eq weight ([`LazyFoldedRa`] documents the derivation). The bind
//! that reaches [`materialize_depth`] gathers every polynomial dense, and the
//! kernel continues over the default family's dense state. A gather adds the
//! table entries the default family adds, so every value equals its value.

use std::ops::Range;

use jolt_field::{Accumulator, JoltField};
use jolt_poly::GruenSplitEqPolynomial;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::{WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{dense_crossover, task_ranges, VISIT_ROWS};
use crate::optimized::lazy_ra::{double_branches, LazyFoldedRa, DEFAULT_MAX_WIDTH};
use crate::optimized::support::BundleStore;

/// Bytes of each dense polynomial a streaming family materializes
/// ([`materialize_depth`]).
pub(crate) const MATERIALIZE_BYTES: usize = 1 << 26;

/// The default family's materializing bind, its fourth.
const DEFAULT_MATERIALIZE_DEPTH: usize = (2 * DEFAULT_MAX_WIDTH).ilog2() as usize;

/// The bind depth at which a family over `2^rounds` cycles materializes: the
/// first whose polynomials, `2^rounds >> depth` entries each, fit `bytes`; at
/// least the default family's, so no streamed polynomial is longer than the
/// default family's where the domain has that many binds and its rows end in
/// no constant suffix: the default family stops at one, where this family
/// holds every polynomial over the whole domain; and at most the
/// last bind, so the final claims are always dense entries. Sizes are
/// compared in entries against the entries `bytes` holds, which is the same
/// test as bytes against `bytes` and cannot overflow where the byte count of
/// `2^rounds` entries would, for `rounds < usize::BITS`
/// ([`BundleStore::resident`] refuses every larger domain).
fn materialize_depth<F>(rounds: usize, bytes: usize) -> usize {
    dense_crossover(1 << rounds, bytes / size_of::<F>(), rounds)
        .max(DEFAULT_MATERIALIZE_DEPTH)
        .min(rounds)
}

/// Rows per visit callback: whole groups of `unit` consecutive rows,
/// [`VISIT_ROWS`] rows where a group is shorter.
fn callback_rows(unit: usize) -> usize {
    unit * (VISIT_ROWS / unit).max(1)
}

/// The hot scale-table index of each point-mass column, one per selector
/// polynomial, in one cycle's typed witness row.
pub(crate) trait RowIndexSource: Send + Sync {
    type Row: WitnessBundle + Copy + Send + Sync;

    /// Calls `visit(column, index)` once for every column that is hot in
    /// `row`, with that column's scale-table index.
    fn for_each_hot(&self, row: &Self::Row, visit: impl FnMut(usize, usize));
}

/// Selector polynomials bound `LowToHigh` before they materialize.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, S: RowIndexSource + allocative::Allocative")
)]
pub(crate) struct StreamedFoldedRa<F: JoltField, S: RowIndexSource> {
    /// Per polynomial, its base table pre-scaled by each bound-bit pattern's
    /// eq weight, branch-major: `tables[i][branch · K_i + k]` for a
    /// `K_i`-entry base table.
    tables: Vec<Vec<F>>,
    /// Binds so far: every polynomial has `2^depth` branches.
    depth: usize,
    materialize_depth: usize,
    /// log2 of the cycle domain `rows` span.
    rounds: usize,
    source: S,
    rows: BundleStore<S::Row>,
}

impl<F: JoltField, S: RowIndexSource> StreamedFoldedRa<F, S> {
    /// One scale table per selector polynomial, in `source`'s column order,
    /// over the `2^rounds` cycles of `rows`. Each polynomial materializes at
    /// `materialize_bytes` at most ([`materialize_depth`]).
    pub(crate) fn new(
        tables: Vec<Vec<F>>,
        source: S,
        rows: BundleStore<S::Row>,
        rounds: usize,
        materialize_bytes: usize,
    ) -> Self {
        Self {
            tables,
            depth: 0,
            materialize_depth: materialize_depth::<F>(rounds, materialize_bytes),
            rounds,
            source,
            rows,
        }
    }

    /// Every polynomial's bound value at the index whose `2^depth` unbound
    /// cycles are `rows`.
    #[inline]
    fn gather(&self, rows: &[S::Row], values: &mut [F]) {
        values.fill(F::zero());
        for (branch, row) in rows.iter().enumerate() {
            self.source.for_each_hot(row, |column, index| {
                let table = &self.tables[column];
                values[column] += table[branch * (table.len() >> self.depth) + index];
            });
        }
    }

    /// `Σ_y E(y) · pair_terms(lo(y), hi(y))` over the current pair rows `y`,
    /// where `lo(y)` and `hi(y)` hold every polynomial's bound values at
    /// indices `2y` and `2y + 1`, `pair_terms(scratch, lo, hi, terms)` writes
    /// the row's `terms` summands, and `E = e_out ⊗ e_in` are `eq`'s current
    /// tables. The domain splits into [`task_ranges`] of whole pairs; each
    /// task visits its rows once, serially, and keeps one `scratch()`. Every
    /// callback holds whole pairs and a pair is weighted on its own, so no
    /// sum depends on where a callback or a task ends.
    pub(crate) fn sum_pair_terms<T>(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        terms: usize,
        scratch: impl Fn() -> T + Sync,
        pair_terms: impl Fn(&mut T, &[F], &[F], &mut [F]) + Sync,
    ) -> Result<Vec<F>, WitnessError> {
        let e_out = eq.e_out_current();
        let e_in = eq.e_in_current();
        let pair = 2usize << self.depth;
        let zero = || vec![F::Accumulator::default(); terms];
        let task = |range: &Range<usize>| -> Result<Vec<F::Accumulator>, WitnessError> {
            let mut scratch = scratch();
            let mut lo = vec![F::zero(); self.tables.len()];
            let mut hi = vec![F::zero(); self.tables.len()];
            let mut values = vec![F::zero(); terms];
            let mut total = zero();
            self.rows
                .visit(range.clone(), callback_rows(pair), |start, block| {
                    for (y, rows) in (start / pair..).zip(block.chunks_exact(pair)) {
                        let (low, high) = rows.split_at(pair / 2);
                        self.gather(low, &mut lo);
                        self.gather(high, &mut hi);
                        pair_terms(&mut scratch, &lo, &hi, &mut values);
                        let weight = e_out[y / e_in.len()] * e_in[y % e_in.len()];
                        for (total, value) in total.iter_mut().zip(&values) {
                            total.fmadd(weight, *value);
                        }
                    }
                    Ok(())
                })?;
            Ok(total)
        };
        let merge = |mut left: Vec<F::Accumulator>, right: Vec<F::Accumulator>| {
            for (left, right) in left.iter_mut().zip(right) {
                left.merge(right);
            }
            left
        };
        let ranges = task_ranges(e_out.len() * e_in.len() * pair, pair);
        #[cfg(feature = "parallel")]
        let total = ranges
            .par_iter()
            .map(task)
            .try_reduce(zero, |left, right| Ok(merge(left, right)))?;
        #[cfg(not(feature = "parallel"))]
        let total = ranges.iter().try_fold(zero(), |total, range| {
            task(range).map(|part| merge(total, part))
        })?;
        Ok(total.into_iter().map(Accumulator::reduce).collect())
    }

    /// Binds the next cycle variable: every polynomial's branch set doubles.
    pub(crate) fn bind(&mut self, challenge: F) {
        self.tables = double_branches(std::mem::take(&mut self.tables), 1, challenge);
        self.depth += 1;
    }

    /// The default family's row-major dense state, every polynomial gathered
    /// at each of its `2^rounds >> depth` bound indices, once the binds reach
    /// the materializing depth; `None` before. The rows cover the bound
    /// domain, so the state's tail, their last row, never stands in for a row
    /// they lack. The storage is allocated once at its final size and split
    /// at the task boundaries; each task visits its cycles once, serially.
    pub(crate) fn materialized<I>(&self) -> Result<Option<LazyFoldedRa<F, I>>, WitnessError> {
        if self.depth != self.materialize_depth {
            return Ok(None);
        }
        let columns = self.tables.len();
        let width = 1usize << self.depth;
        let len = (1usize << self.rounds) >> self.depth;
        // A product past `usize` overflows the allocation's capacity check
        // instead of wrapping to a shorter vector.
        let mut values: Vec<F> = unsafe_allocate_zero_vec(columns.saturating_mul(len));
        let ranges = task_ranges(1 << self.rounds, width);
        let mut rest = values.as_mut_slice();
        let slices: Vec<&mut [F]> = ranges
            .iter()
            .map(|range| {
                let (head, tail) =
                    std::mem::take(&mut rest).split_at_mut(columns * (range.len() >> self.depth));
                rest = tail;
                head
            })
            .collect();
        let fill = |(range, slice): (&Range<usize>, &mut [F])| {
            let mut bound_rows = slice.chunks_exact_mut(columns);
            self.rows
                .visit(range.clone(), callback_rows(width), |_, block| {
                    for (rows, bound_row) in block.chunks_exact(width).zip(&mut bound_rows) {
                        self.gather(rows, bound_row);
                    }
                    Ok(())
                })
        };
        #[cfg(feature = "parallel")]
        ranges.par_iter().zip(slices).try_for_each(fill)?;
        #[cfg(not(feature = "parallel"))]
        ranges.iter().zip(slices).try_for_each(fill)?;
        let tail = values[values.len() - columns..].to_vec();
        Ok(Some(LazyFoldedRa::Rows { values, tail }))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    #[cfg(feature = "akita")]
    use jolt_akita::AkitaField;
    use jolt_field::{Fr, Ring};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_witness::witnesses::{LookupIndex, RemappedRamAddress};
    #[cfg(feature = "parallel")]
    use rayon::ThreadPoolBuilder;

    use super::*;
    use crate::optimized::instruction_read_raf::LookupIndexChunks;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
    struct TestRow {
        lookup_index: LookupIndex,
        remapped_ram_address: RemappedRamAddress,
    }

    /// Column 0 is the two low bits of the lookup index, always hot; column 1
    /// the three low bits of the RAM address, cold on a cycle without one.
    struct TestSource;

    impl RowIndexSource for TestSource {
        type Row = TestRow;

        fn for_each_hot(&self, row: &TestRow, mut visit: impl FnMut(usize, usize)) {
            visit(0, (row.lookup_index.0 & 3) as usize);
            if let Some(address) = row.remapped_ram_address.0 {
                visit(1, (address & 7) as usize);
            }
        }
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn test_rows(rounds: usize) -> Vec<TestRow> {
        let mut state = 0xA5A5 + rounds as u64;
        (0..1usize << rounds)
            .map(|_| {
                let word = splitmix(&mut state);
                TestRow {
                    lookup_index: LookupIndex(u128::from(word)),
                    remapped_ram_address: RemappedRamAddress(
                        (!(word >> 8).is_multiple_of(3)).then_some(word >> 16),
                    ),
                }
            })
            .collect()
    }

    fn test_tables<F: JoltField>() -> Vec<Vec<F>> {
        vec![
            (0..4).map(|k| F::from_u64(11 + 7 * k)).collect(),
            (0..8).map(|k| F::from_u64(101 + 13 * k)).collect(),
        ]
    }

    /// Binds a family that materializes at `target` binds beside dense
    /// multilinear binds of its columns: before each bind the pair sums of
    /// two summands, one mixing both columns and both halves, equal the
    /// dense columns' sums, and no state materializes; at `target` the dense
    /// state holds exactly the bound columns, with their last row as its
    /// tail.
    fn check_family<F: JoltField>(rounds: usize, target: usize) {
        let rows = test_rows(rounds);
        let tables = test_tables::<F>();
        let mut dense: Vec<Polynomial<F>> = (0..tables.len())
            .map(|column| {
                let values = rows.iter().map(|row| {
                    let mut value = F::zero();
                    TestSource.for_each_hot(row, |hot, index| {
                        if hot == column {
                            value = tables[column][index];
                        }
                    });
                    value
                });
                Polynomial::new(values.collect())
            })
            .collect();
        let point: Vec<F> = (0..rounds as u64)
            .map(|i| F::from_u64(900 + 53 * i))
            .collect();
        let mut eq = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let mut family = StreamedFoldedRa::new(
            tables,
            TestSource,
            BundleStore::Retained(rows),
            rounds,
            size_of::<F>() << (rounds - target),
        );
        for depth in 0..target {
            assert!(
                family
                    .materialized::<LookupIndexChunks>()
                    .unwrap()
                    .is_none(),
                "rounds {rounds}, target {target}: materialized at depth {depth}"
            );
            let (e_out, e_in) = (eq.e_out_current(), eq.e_in_current());
            let mut expected = vec![F::zero(); 2];
            for y in 0..dense[0].len() / 2 {
                let weight = e_out[y / e_in.len()] * e_in[y % e_in.len()];
                let [lo, hi] =
                    [2 * y, 2 * y + 1].map(|j| [dense[0].evals()[j], dense[1].evals()[j]]);
                expected[0] += weight * lo[0] * hi[1];
                expected[1] += weight * (hi[0] + lo[1]);
            }
            let sums = family
                .sum_pair_terms(
                    &eq,
                    2,
                    || (),
                    |(), lo, hi, terms| {
                        terms[0] = lo[0] * hi[1];
                        terms[1] = hi[0] + lo[1];
                    },
                )
                .unwrap();
            assert_eq!(
                sums, expected,
                "rounds {rounds}, target {target}, depth {depth}"
            );
            let challenge = F::from_u64(31 + 17 * depth as u64);
            family.bind(challenge);
            eq.round_mut().unwrap().bind(challenge);
            for column in &mut dense {
                column.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
        }
        let materialized = family.materialized::<LookupIndexChunks>().unwrap().unwrap();
        let len = (1 << rounds) >> target;
        assert_eq!(materialized.prefix_len(), len);
        for (column, dense) in dense.iter().enumerate() {
            assert_eq!(dense.len(), len);
            for (j, value) in dense.evals().iter().enumerate() {
                assert_eq!(
                    materialized.value(column, j),
                    *value,
                    "rounds {rounds}, target {target}, column {column}, index {j}"
                );
            }
            assert_eq!(materialized.tail_values()[column], dense.evals()[len - 1]);
        }
    }

    /// Every materializing bind of domains up to 2^7 cycles: the last bind of
    /// domains shorter than four binds, and the fourth through the last.
    #[test]
    fn pair_sums_and_materialized_rows_match_dense_binds() {
        for (rounds, targets) in [(1, 1..=1), (2, 2..=2), (3, 3..=3), (5, 4..=5), (7, 4..=7)] {
            for target in targets {
                check_family::<Fr>(rounds, target);
                #[cfg(feature = "akita")]
                check_family::<AkitaField>(rounds, target);
            }
        }
    }

    /// 2^16 cycles on one pool thread, so that a task holds more rows than a
    /// callback: materializing at the fourth bind, every pass and the gather
    /// visit a task's rows in several callbacks; at the thirteenth, the last
    /// passes' pairs and the gather's bound rows each fill a callback.
    #[test]
    fn tasks_of_several_callbacks_match_dense_binds() {
        let check = || {
            check_family::<Fr>(16, 4);
            check_family::<Fr>(16, 13);
        };
        #[cfg(feature = "parallel")]
        ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(check);
        #[cfg(not(feature = "parallel"))]
        check();
    }

    /// The materializing bind is exact at the budget's boundary, never
    /// earlier than the default family's fourth, the last bind where no
    /// earlier one fits or the domain is shorter, and does not wrap when the
    /// polynomials exceed `usize` bytes (a round count one below the word
    /// size).
    #[test]
    fn materialize_depth_is_exact_at_the_budget_and_past_usize_bytes() {
        let entry = size_of::<Fr>();
        assert_eq!(materialize_depth::<Fr>(20, entry << 10), 10);
        assert_eq!(materialize_depth::<Fr>(20, (entry << 10) - 1), 11);
        assert_eq!(materialize_depth::<Fr>(20, usize::MAX), 4);
        assert_eq!(materialize_depth::<Fr>(20, 0), 20);
        assert_eq!(materialize_depth::<Fr>(3, usize::MAX), 3);
        assert_eq!(materialize_depth::<Fr>(0, MATERIALIZE_BYTES), 0);
        assert_eq!(materialize_depth::<Fr>(28, MATERIALIZE_BYTES), 7);
        let wide = usize::BITS as usize - 1;
        assert_eq!(materialize_depth::<Fr>(wide, entry << 20), wide - 20);
        assert_eq!(materialize_depth::<Fr>(wide, usize::MAX), 5);
    }

    /// A row store that ends inside the domain fails a pass and the
    /// materializing gather typed instead of reading past it.
    #[test]
    fn rows_missing_from_the_domain_fail_typed() {
        let rounds = 4;
        let mut rows = test_rows(rounds);
        let _ = rows.pop();
        let mut family = StreamedFoldedRa::new(
            test_tables::<Fr>(),
            TestSource,
            BundleStore::Retained(rows),
            rounds,
            0,
        );
        let eq = GruenSplitEqPolynomial::new(&[Fr::from_u64(5); 4], BindingOrder::LowToHigh);
        assert!(matches!(
            family.sum_pair_terms(&eq, 1, || (), |(), _, _, _| ()),
            Err(WitnessError::InvalidDimensions { .. })
        ));
        for depth in 0..rounds as u64 {
            family.bind(Fr::from_u64(3 + depth));
        }
        assert!(matches!(
            family.materialized::<LookupIndexChunks>(),
            Err(WitnessError::InvalidDimensions { .. })
        ));
    }
}
