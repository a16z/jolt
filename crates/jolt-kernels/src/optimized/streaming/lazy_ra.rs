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
//! pattern's eq weight ([`LazyFoldedRa`] documents the derivation, and the
//! products a polynomial of several columns gathers). The family's
//! materializing bind ([`Schedule`]) gathers every polynomial dense, and the
//! kernel continues over the default family's dense state. A gather combines
//! the table entries the default family combines, so every value equals its
//! value.
//!
//! Like the default family, this one stops at the rows' constant suffix. A
//! bound index whose cycles all lie in the suffix holds the polynomial's
//! value there at every depth, because the branch weights sum to one. A pass
//! therefore visits the pair rows that hold a cycle before the suffix and
//! weighs the others once ([`GruenSplitEqPolynomial::suffix_weight`]), and
//! the dense state holds the rows that hold such a cycle, with the suffix
//! values as its tail.

use std::ops::Range;

use jolt_field::{Accumulator, JoltField};
use jolt_poly::GruenSplitEqPolynomial;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::{WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{dense_crossover, task_ranges, VISIT_ROWS};
use crate::optimized::lazy_ra::{double_branches, LazyBind, LazyFoldedRa, DEFAULT_MAX_WIDTH};
use crate::optimized::support::BundleStore;

/// Bytes of each dense polynomial a [`StreamedFoldedRa::new`] family
/// materializes ([`materialize_depth`]).
pub(crate) const MATERIALIZE_BYTES: usize = 1 << 26;

/// The default family's materializing bind, its fourth.
const DEFAULT_MATERIALIZE_DEPTH: usize = (2 * DEFAULT_MAX_WIDTH).ilog2() as usize;

/// The bind depth at which a family over `2^rounds` cycles materializes: the
/// first at which a polynomial over the whole domain, `2^rounds >> depth`
/// entries, fits `bytes`, so one that stops at a constant suffix fits too; at
/// least the default family's, whose polynomials are no shorter where they
/// stop at the same suffix and the domain has that many binds; and at most
/// the last bind, so the final claims are always dense entries. Sizes are
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

/// The hot scale-table index of each point-mass column in one cycle's typed
/// witness row.
pub(crate) trait RowIndexSource: Send + Sync {
    type Row: WitnessBundle + Copy + PartialEq + Send + Sync;

    /// Calls `visit(column, index)` once for every column that is hot in
    /// `row`, with that column's scale-table index.
    fn for_each_hot(&self, row: &Self::Row, visit: impl FnMut(usize, usize));
}

/// The bind at which a family materializes.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Schedule {
    /// The bind that reaches this depth, after doubling the branch tables.
    Depth(usize),
    /// The bind at which the default family of this widest branch set
    /// materializes, fused where that family fuses ([`LazyBind`]).
    Default { max_width: usize },
}

/// Selector polynomials bound `LowToHigh` before they materialize.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, S: RowIndexSource + allocative::Allocative")
)]
pub(crate) struct StreamedFoldedRa<F: JoltField, S: RowIndexSource> {
    /// Per column, its base table; polynomial `i` is the product of columns
    /// `i · factors..(i + 1) · factors`. A polynomial's first column holds
    /// its base table pre-scaled by each bound-bit pattern's eq weight,
    /// branch-major: `tables[c][branch · K_c + k]` for a `K_c`-entry base
    /// table.
    tables: Vec<Vec<F>>,
    factors: usize,
    /// Binds so far: every polynomial has `2^depth` branches.
    depth: usize,
    schedule: Schedule,
    /// log2 of the cycle domain `rows` span.
    rounds: usize,
    /// The first cycle of the constant suffix, at most the domain's
    /// `2^rounds` cycles: every row from it on gathers `tail`.
    prefix: usize,
    /// Every polynomial's value on the constant suffix, at any depth.
    tail: Vec<F>,
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
    ) -> Result<Self, WitnessError> {
        let schedule = Schedule::Depth(materialize_depth::<F>(rounds, materialize_bytes));
        Self::over(tables, 1, schedule, source, rows, rounds, 0)
    }

    /// One base table per column, in `source`'s column order, over the
    /// `2^rounds` cycles of `rows`; each run of `factors` consecutive columns
    /// is one polynomial. The family materializes at the bind at which
    /// [`LazyFoldedRa::factored`] over the same tables does, and holds that
    /// family's branch tables until then. Its constant suffix starts no
    /// earlier than cycle `min_prefix`: a caller whose pair terms read other
    /// tables by pair row passes the first cycle from which those are
    /// constant ([`Self::sum_pair_terms`]).
    pub(crate) fn factored(
        tables: Vec<Vec<F>>,
        factors: usize,
        max_width: usize,
        source: S,
        rows: BundleStore<S::Row>,
        rounds: usize,
        min_prefix: usize,
    ) -> Result<Self, WitnessError> {
        debug_assert!(factors > 0 && tables.len().is_multiple_of(factors));
        let schedule = Schedule::Default { max_width };
        Self::over(tables, factors, schedule, source, rows, rounds, min_prefix)
    }

    /// The family at depth zero. Its suffix values are the gather of the
    /// domain's last cycle. Its constant suffix starts where the rows' does
    /// ([`constant_suffix_start`](crate::optimized::support::BundleAccess::constant_suffix_start)),
    /// which is the domain's end where they establish none, or at
    /// `min_prefix` where that is later, and never past the domain's end.
    fn over(
        tables: Vec<Vec<F>>,
        factors: usize,
        schedule: Schedule,
        source: S,
        rows: BundleStore<S::Row>,
        rounds: usize,
        min_prefix: usize,
    ) -> Result<Self, WitnessError> {
        // `rounds` is below `usize::BITS`, as in [`materialize_depth`].
        let cycles = 1usize << rounds;
        let mut family = Self {
            tables,
            factors,
            depth: 0,
            schedule,
            rounds,
            prefix: cycles,
            tail: Vec::new(),
            source,
            rows,
        };
        let mut tail = vec![F::zero(); family.num_polys()];
        // This visit refuses a store that ends inside the domain, so the
        // suffix search below never indexes past one.
        family.rows.visit(cycles - 1..cycles, 1, |_, last| {
            family.gather(last, &mut vec![None; family.tables.len()], &mut tail);
            Ok(())
        })?;
        family.prefix = family
            .rows
            .access()
            .constant_suffix_start(cycles)?
            .max(min_prefix)
            .min(cycles);
        family.tail = tail;
        Ok(family)
    }

    fn num_polys(&self) -> usize {
        self.tables.len() / self.factors
    }

    /// Every polynomial's bound value at the index whose `2^depth` unbound
    /// cycles are `rows`. `hot` is scratch of one entry per column.
    #[inline]
    fn gather(&self, rows: &[S::Row], hot: &mut [Option<usize>], values: &mut [F]) {
        values.fill(F::zero());
        if self.factors == 1 {
            for (branch, row) in rows.iter().enumerate() {
                self.source.for_each_hot(row, |column, index| {
                    let table = &self.tables[column];
                    values[column] += table[branch * (table.len() >> self.depth) + index];
                });
            }
            return;
        }
        for (branch, row) in rows.iter().enumerate() {
            hot.fill(None);
            self.source
                .for_each_hot(row, |column, index| hot[column] = Some(index));
            for ((value, tables), hot) in values
                .iter_mut()
                .zip(self.tables.chunks_exact(self.factors))
                .zip(hot.chunks_exact(self.factors))
            {
                // A polynomial is cold on a cycle where any of its columns is.
                let product = hot[0].and_then(|lead| {
                    let lead = tables[0][branch * (tables[0].len() >> self.depth) + lead];
                    tables[1..]
                        .iter()
                        .zip(&hot[1..])
                        .try_fold(lead, |product, (table, index)| {
                            Some(product * table[(*index)?])
                        })
                });
                if let Some(product) = product {
                    *value += product;
                }
            }
        }
    }

    /// `Σ_y E(y) · pair_terms(y, lo(y), hi(y))` over the current pair rows
    /// `y`, where `lo(y)` and `hi(y)` hold every polynomial's bound values at
    /// indices `2y` and `2y + 1`, `pair_terms(scratch, y, lo, hi, terms)`
    /// writes the row's `terms` summands, and `E = e_out ⊗ e_in` are `eq`'s
    /// current tables. The pair rows that hold a cycle before the constant
    /// suffix split into [`task_ranges`] of whole pairs; each task visits its
    /// rows once, serially, and keeps one `scratch()`. Every callback holds
    /// whole pairs and a pair is weighted on its own, so no sum depends on
    /// where a callback or a task ends. Every later pair row holds the suffix
    /// values at both indices: `pair_terms` runs once for all of them, at the
    /// first, and [`GruenSplitEqPolynomial::suffix_weight`] weighs its
    /// summands. Whatever else `pair_terms` reads by `y` must therefore be
    /// constant from that row on ([`Self::factored`]'s `min_prefix`).
    pub(crate) fn sum_pair_terms<T>(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        terms: usize,
        scratch: impl Fn() -> T + Sync,
        pair_terms: impl Fn(&mut T, usize, &[F], &[F], &mut [F]) + Sync,
    ) -> Result<Vec<F>, WitnessError> {
        let e_out = eq.e_out_current();
        let e_in = eq.e_in_current();
        let pair = 2usize << self.depth;
        // `prefix` is at most the domain's cycles ([`Self::over`]), so these
        // are at most its pair rows and their cycles cannot overflow.
        let prefix_pairs = self.prefix.div_ceil(pair);
        let zero = || vec![F::Accumulator::default(); terms];
        let task = |range: &Range<usize>| -> Result<Vec<F::Accumulator>, WitnessError> {
            let mut scratch = scratch();
            let mut hot = vec![None; self.tables.len()];
            let mut lo = vec![F::zero(); self.num_polys()];
            let mut hi = vec![F::zero(); self.num_polys()];
            let mut values = vec![F::zero(); terms];
            let mut total = zero();
            self.rows
                .visit(range.clone(), callback_rows(pair), |start, block| {
                    for (y, rows) in (start / pair..).zip(block.chunks_exact(pair)) {
                        let (low, high) = rows.split_at(pair / 2);
                        self.gather(low, &mut hot, &mut lo);
                        self.gather(high, &mut hot, &mut hi);
                        pair_terms(&mut scratch, y, &lo, &hi, &mut values);
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
        let ranges = task_ranges(prefix_pairs * pair, pair);
        #[cfg(feature = "parallel")]
        let mut total = ranges
            .par_iter()
            .map(task)
            .try_reduce(zero, |left, right| Ok(merge(left, right)))?;
        #[cfg(not(feature = "parallel"))]
        let mut total = ranges.iter().try_fold(zero(), |total, range| {
            task(range).map(|part| merge(total, part))
        })?;
        if prefix_pairs < e_out.len() * e_in.len() {
            let mut values = vec![F::zero(); terms];
            pair_terms(
                &mut scratch(),
                prefix_pairs,
                &self.tail,
                &self.tail,
                &mut values,
            );
            let weight = eq.suffix_weight(prefix_pairs);
            for (total, value) in total.iter_mut().zip(values) {
                total.fmadd(weight, value);
            }
        }
        Ok(total.into_iter().map(Accumulator::reduce).collect())
    }

    /// Binds the next cycle variable, which the domain must still have. Until
    /// the materializing bind every polynomial's branch set doubles; that
    /// bind returns the family's dense state, and the family is spent.
    pub(crate) fn bind<I>(
        &mut self,
        challenge: F,
    ) -> Result<Option<LazyFoldedRa<F, I>>, WitnessError> {
        let step = match self.schedule {
            Schedule::Depth(depth) if self.depth + 1 < depth => LazyBind::Double,
            Schedule::Depth(_) => LazyBind::DoubleThenMaterialize,
            Schedule::Default { max_width } => LazyBind::new(
                1 << self.depth,
                max_width,
                self.factors,
                1 << self.rounds,
                self.tables.iter().map(Vec::len),
            ),
        };
        if step == LazyBind::FusedMaterialize {
            return self.materialize(Some(challenge)).map(Some);
        }
        self.tables = double_branches(std::mem::take(&mut self.tables), self.factors, challenge);
        self.depth += 1;
        match step {
            LazyBind::Double => Ok(None),
            _ => self.materialize(None).map(Some),
        }
    }

    /// The dense state of a one-cycle domain, which has no bind to
    /// materialize at; `None` for a longer domain.
    pub(crate) fn single_cycle<I>(&self) -> Result<Option<LazyFoldedRa<F, I>>, WitnessError> {
        if self.rounds != 0 {
            return Ok(None);
        }
        self.materialize(None).map(Some)
    }

    /// The default family's row-major dense state: every polynomial gathered
    /// at each bound index that holds a cycle before the constant suffix,
    /// with the suffix values as its tail. With `bind`, a challenge for the
    /// next cycle variable, each dense row is instead the pair of gathers
    /// `(lo, hi)` of the next depth's index bound once,
    /// `lo + bind · (hi − lo)`, so the branch tables never double for it. The
    /// storage is allocated once at its final size and split at the task
    /// boundaries; each task visits its cycles once, serially.
    fn materialize<I>(&self, bind: Option<F>) -> Result<LazyFoldedRa<F, I>, WitnessError> {
        let polys = self.num_polys();
        let width = 1usize << self.depth;
        let unit = if bind.is_some() { 2 * width } else { width };
        // `prefix` is at most the domain's cycles ([`Self::over`]), so these
        // are at most its `2^rounds / unit` rows.
        let len = self.prefix.div_ceil(unit);
        // A product past `usize` overflows the allocation's capacity check
        // instead of wrapping to a shorter vector.
        let mut values: Vec<F> = unsafe_allocate_zero_vec(polys.saturating_mul(len));
        let ranges = task_ranges(len * unit, unit);
        let mut rest = values.as_mut_slice();
        let slices: Vec<&mut [F]> = ranges
            .iter()
            .map(|range| {
                let (head, tail) =
                    std::mem::take(&mut rest).split_at_mut(polys * (range.len() / unit));
                rest = tail;
                head
            })
            .collect();
        let fill = |(range, slice): (&Range<usize>, &mut [F])| {
            let mut hot = vec![None; self.tables.len()];
            let mut high = vec![F::zero(); if bind.is_some() { polys } else { 0 }];
            let mut bound_rows = slice.chunks_exact_mut(polys);
            self.rows
                .visit(range.clone(), callback_rows(unit), |_, block| {
                    for (rows, bound_row) in block.chunks_exact(unit).zip(&mut bound_rows) {
                        let Some(challenge) = bind else {
                            self.gather(rows, &mut hot, bound_row);
                            continue;
                        };
                        let (lo, hi) = rows.split_at(width);
                        self.gather(lo, &mut hot, bound_row);
                        self.gather(hi, &mut hot, &mut high);
                        for (lo, hi) in bound_row.iter_mut().zip(&high) {
                            *lo += challenge * (*hi - *lo);
                        }
                    }
                    Ok(())
                })
        };
        #[cfg(feature = "parallel")]
        ranges.par_iter().zip(slices).try_for_each(fill)?;
        #[cfg(not(feature = "parallel"))]
        ranges.iter().zip(slices).try_for_each(fill)?;
        Ok(LazyFoldedRa::Rows {
            values,
            tail: self.tail.clone(),
        })
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
    use crate::optimized::lazy_ra::ChunkIndexSource;
    use crate::optimized::registers_read_write::test_support::structured_fixture;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
    struct TestRow {
        lookup_index: LookupIndex,
        remapped_ram_address: RemappedRamAddress,
    }

    /// Columns 0 and 3 are bit pairs of the lookup index, always hot; columns
    /// 1 and 2 are bit triples of the RAM address, cold on a cycle without
    /// one. In pairs, the first polynomial has a cold second column and the
    /// second a cold first column.
    struct TestSource;

    const TEST_COLUMNS: usize = 4;

    impl RowIndexSource for TestSource {
        type Row = TestRow;

        fn for_each_hot(&self, row: &TestRow, mut visit: impl FnMut(usize, usize)) {
            visit(0, (row.lookup_index.0 & 3) as usize);
            if let Some(address) = row.remapped_ram_address.0 {
                visit(1, (address & 7) as usize);
                visit(2, ((address >> 3) & 7) as usize);
            }
            visit(3, ((row.lookup_index.0 >> 2) & 3) as usize);
        }
    }

    /// [`TestSource`]'s columns by cycle, for the default family.
    struct IndexedRows(Vec<TestRow>);

    impl ChunkIndexSource for IndexedRows {
        fn num_columns(&self) -> usize {
            TEST_COLUMNS
        }

        fn cycles(&self) -> usize {
            self.0.len()
        }

        fn index(&self, i: usize, j: usize) -> Option<usize> {
            let mut hot = None;
            TestSource.for_each_hot(&self.0[j], |column, index| {
                if column == i {
                    hot = Some(index);
                }
            });
            hot
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

    /// [`test_rows`] with one row, hot in every column, from `prefix` on.
    fn padded_rows(rounds: usize, prefix: usize) -> Vec<TestRow> {
        let mut rows = test_rows(rounds);
        rows[prefix..].fill(TestRow {
            lookup_index: LookupIndex(6),
            remapped_ram_address: RemappedRamAddress(Some(0x2d)),
        });
        rows
    }

    /// The prefixes of a `2^rounds`-cycle domain the suffix cases cover: no
    /// padding row, one, every row, one or two rows before the padding, and
    /// suffixes that start inside a pair and a split-eq group, on a pair
    /// boundary and at the middle of the domain.
    fn prefixes(rounds: usize) -> Vec<usize> {
        let cycles = 1usize << rounds;
        let mut prefixes = vec![cycles, cycles - 1, 0, 1, 2, 37, 40, cycles / 2];
        prefixes.retain(|prefix| *prefix <= cycles);
        prefixes.sort_unstable();
        prefixes.dedup();
        prefixes
    }

    fn test_tables<F: JoltField>() -> Vec<Vec<F>> {
        [(4, 11, 7), (8, 101, 13), (8, 211, 17), (4, 307, 19)]
            .into_iter()
            .map(|(len, first, step)| (0..len).map(|k| F::from_u64(first + step * k)).collect())
            .collect()
    }

    /// Binds `family`, whose polynomials are products of `factors` columns
    /// over [`padded_rows`] and whose constant suffix starts at cycle
    /// `suffix`, beside dense multilinear binds of those products: before
    /// each bind the pair sums of two summands, one mixing two polynomials
    /// and both halves and one reading the pair row's index up to the first
    /// row of the suffix, equal the dense sums over the whole domain. Bind
    /// `target`, and none before it, returns the dense state: the bound
    /// polynomials' rows that hold a cycle before the suffix, and their
    /// values past those from its tail. Returns whether that bind doubled the
    /// branch tables.
    fn check_family<F: JoltField>(
        rounds: usize,
        prefix: usize,
        suffix: usize,
        factors: usize,
        target: usize,
        family: impl FnOnce(Vec<Vec<F>>, BundleStore<TestRow>) -> StreamedFoldedRa<F, TestSource>,
    ) -> bool {
        let rows = padded_rows(rounds, prefix);
        let tables = test_tables::<F>();
        let mut dense: Vec<Polynomial<F>> = tables
            .chunks_exact(factors)
            .enumerate()
            .map(|(poly, tables)| {
                let values = rows.iter().map(|row| {
                    let mut hot = [None; TEST_COLUMNS];
                    TestSource.for_each_hot(row, |column, index| hot[column] = Some(index));
                    tables
                        .iter()
                        .zip(&hot[poly * factors..])
                        .map(|(table, index)| index.map_or_else(F::zero, |index| table[index]))
                        .product()
                });
                Polynomial::new(values.collect())
            })
            .collect();
        let last = dense.len() - 1;
        let point: Vec<F> = (0..rounds as u64)
            .map(|i| F::from_u64(900 + 53 * i))
            .collect();
        let mut eq = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let mut family = family(tables, BundleStore::Retained(rows));
        let case = format!("rounds {rounds}, prefix {prefix}, factors {factors}, target {target}");
        assert!(
            family
                .single_cycle::<LookupIndexChunks>()
                .unwrap()
                .is_none(),
            "{case}"
        );
        for depth in 0..target {
            let (e_out, e_in) = (eq.e_out_current(), eq.e_in_current());
            let by_row = |y: usize| F::from_u64(y.min(suffix.div_ceil(2 << depth)) as u64);
            let mut expected = vec![F::zero(); 2];
            for y in 0..dense[0].len() / 2 {
                let weight = e_out[y / e_in.len()] * e_in[y % e_in.len()];
                let [lo, hi] =
                    [2 * y, 2 * y + 1].map(|j| [dense[0].evals()[j], dense[last].evals()[j]]);
                expected[0] += weight * lo[0] * hi[1];
                expected[1] += weight * (hi[0] + lo[1] + by_row(y));
            }
            let sums = family
                .sum_pair_terms(
                    &eq,
                    2,
                    || (),
                    |(), y, lo, hi, terms| {
                        terms[0] = lo[0] * hi[last];
                        terms[1] = hi[0] + lo[last] + by_row(y);
                    },
                )
                .unwrap();
            assert_eq!(sums, expected, "{case}, depth {depth}");
            let challenge = F::from_u64(31 + 17 * depth as u64);
            let materialized = family.bind::<LookupIndexChunks>(challenge).unwrap();
            eq.round_mut().unwrap().bind(challenge);
            for poly in &mut dense {
                poly.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
            let Some(materialized) = materialized else {
                assert!(depth + 1 < target, "{case}: lazy past bind {}", depth + 1);
                continue;
            };
            assert_eq!(depth + 1, target, "{case}");
            assert_eq!(
                materialized.prefix_len(),
                suffix.div_ceil(1 << target),
                "{case}"
            );
            for (poly, dense) in dense.iter().enumerate() {
                assert_eq!(dense.len(), (1 << rounds) >> target, "{case}");
                for (j, value) in dense.evals().iter().enumerate() {
                    assert_eq!(materialized.value(poly, j), *value, "{case}, {poly}, {j}");
                }
            }
        }
        family.depth == target
    }

    /// Where the constant suffix of [`padded_rows`] starts: at `prefix`, and
    /// at the last row, a suffix of its own, where no row is padding.
    fn suffix_start(rounds: usize, prefix: usize) -> usize {
        prefix.min((1 << rounds) - 1)
    }

    /// A [`StreamedFoldedRa::new`] family over [`padded_rows`] that
    /// materializes at bind `target`.
    fn check_budgeted<F: JoltField>(rounds: usize, prefix: usize, target: usize) {
        let suffix = suffix_start(rounds, prefix);
        let doubled = check_family::<F>(rounds, prefix, suffix, 1, target, |tables, rows| {
            StreamedFoldedRa::new(
                tables,
                TestSource,
                rows,
                rounds,
                size_of::<F>() << (rounds - target),
            )
            .unwrap()
        });
        assert!(doubled, "rounds {rounds}, prefix {prefix}, target {target}");
    }

    /// Every materializing bind of domains up to 2^7 cycles, the last bind of
    /// domains shorter than four binds and the fourth through the last, for
    /// every suffix of [`prefixes`].
    #[test]
    fn pair_sums_and_materialized_rows_match_dense_binds() {
        for (rounds, targets) in [(1, 1..=1), (2, 2..=2), (3, 3..=3), (5, 4..=5), (7, 4..=7)] {
            for target in targets {
                for prefix in prefixes(rounds) {
                    check_budgeted::<Fr>(rounds, prefix, target);
                    #[cfg(feature = "akita")]
                    check_budgeted::<AkitaField>(rounds, prefix, target);
                }
            }
        }
    }

    /// 2^16 cycles on one pool thread, so that a task holds more rows than a
    /// callback: materializing at the fourth bind, every pass and the gather
    /// visit a task's rows in several callbacks; at the thirteenth, the last
    /// passes' pairs and the gather's bound rows each fill a callback. The
    /// suffix starts at a callback's end and one row past it, at an eighth of
    /// the domain, where a task of the whole domain would end, where every
    /// task is one callback and where it is a callback and a part, and at the
    /// domain's last row.
    #[test]
    fn tasks_of_several_callbacks_match_dense_binds() {
        let check = || {
            for prefix in [1 << 12, (1 << 12) + 1, 1 << 13, 1 << 15, 9 << 12, 1 << 16] {
                check_budgeted::<Fr>(16, prefix, 4);
                check_budgeted::<Fr>(16, prefix, 13);
            }
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

    /// A [`StreamedFoldedRa::factored`] family materializes at the bind at
    /// which the default family over the same tables leaves its lazy state,
    /// with the dense binds' values, for polynomials of one, two and four
    /// columns and every suffix of [`prefixes`]. The cases cover a first, a
    /// second and a third materializing bind, each both fused and after
    /// doubling.
    #[test]
    fn factored_family_materializes_where_the_default_family_does() {
        fn check<F: JoltField>(rounds: usize, factors: usize, max_width: usize) -> (usize, bool) {
            let mut default = LazyFoldedRa::factored(
                test_tables::<F>(),
                factors,
                max_width,
                IndexedRows(test_rows(rounds)),
            );
            let target = (1..=rounds)
                .find(|bind| {
                    default.bind(F::from_u64(*bind as u64));
                    !matches!(default, LazyFoldedRa::Lazy { .. })
                })
                .unwrap();
            let doubled = prefixes(rounds).into_iter().map(|prefix| {
                let suffix = suffix_start(rounds, prefix);
                check_family::<F>(rounds, prefix, suffix, factors, target, |tables, rows| {
                    StreamedFoldedRa::factored(
                        tables, factors, max_width, TestSource, rows, rounds, 0,
                    )
                    .unwrap()
                })
            });
            let doubled: Vec<bool> = doubled.collect();
            assert!(doubled.iter().all(|step| *step == doubled[0]));
            (target, doubled[0])
        }
        for (rounds, factors, max_width, expected) in [
            (4, 1, 8, (1, false)),
            (7, 1, 1, (1, true)),
            (7, 1, 2, (2, true)),
            (7, 1, 8, (3, false)),
            (7, 2, 2, (2, true)),
            (7, 2, 8, (3, false)),
            (9, 2, 4, (3, true)),
            (7, 4, 8, (2, false)),
            (1, 4, 4, (1, false)),
        ] {
            let case = format!("rounds {rounds}, factors {factors}, max width {max_width}");
            assert_eq!(check::<Fr>(rounds, factors, max_width), expected, "{case}");
            #[cfg(feature = "akita")]
            assert_eq!(
                check::<AkitaField>(rounds, factors, max_width),
                expected,
                "{case}"
            );
        }
    }

    /// A family's suffix starts at its floor where the rows' own starts
    /// earlier, and at the domain's end, without a suffix, where the floor
    /// lies past it, up to the largest `usize`.
    #[test]
    fn suffix_starts_no_earlier_than_the_floor() {
        let (rounds, prefix) = (7, 37);
        for (floor, suffix) in [(0, 37), (36, 37), (38, 38), (128, 128), (usize::MAX, 128)] {
            let _ = check_family::<Fr>(rounds, prefix, suffix, 2, 3, |tables, rows| {
                StreamedFoldedRa::factored(tables, 2, 8, TestSource, rows, rounds, floor).unwrap()
            });
        }
    }

    /// The default schedule's fused switch is exact where the doubled branch
    /// tables reach the dense tables' element count, and a domain whose dense
    /// tables exceed `usize` elements (2^61 rows of 16 polynomials) keeps
    /// doubling instead of wrapping into the switch.
    #[test]
    fn fused_switch_is_exact_at_the_crossover_and_past_usize_elements() {
        let step = |cycles, len| LazyBind::new(2, 8, 1, cycles, [len; 16].into_iter());
        assert_eq!(step(1 << 10, 128), LazyBind::FusedMaterialize);
        assert_eq!(step(1 << 10, 127), LazyBind::Double);
        assert_eq!(step(1 << (usize::BITS - 1), 128), LazyBind::Double);
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

    /// Over a witness's resident rows the family stops at the trace's
    /// padding: 37 cycles in a domain of 64 leave `ceil(37 / 16)` dense rows
    /// at the fourth bind, and a trace that fills the domain leaves all four.
    #[test]
    fn dense_state_over_resident_rows_stops_at_the_padding() {
        let (rounds, depth) = (6, 4);
        for cycles in [37, 64] {
            structured_fixture(cycles).with_plane(rounds, |witness| {
                let rows = BundleStore::resident::<Fr>(witness, rounds, "test rows").unwrap();
                let mut family = StreamedFoldedRa::new(
                    test_tables::<Fr>(),
                    TestSource,
                    rows,
                    rounds,
                    usize::MAX,
                )
                .unwrap();
                let dense = (0..depth)
                    .find_map(|bind| {
                        family
                            .bind::<LookupIndexChunks>(Fr::from_u64(31 + bind as u64))
                            .unwrap()
                    })
                    .unwrap();
                assert_eq!(dense.prefix_len(), cycles.div_ceil(1 << depth));
            });
        }
    }

    /// A one-cycle domain has no bind: its dense state is every polynomial's
    /// value at that cycle.
    #[test]
    fn single_cycle_domain_is_dense_at_once() {
        let tables = test_tables::<Fr>();
        let row = test_rows(0)[0];
        let family = StreamedFoldedRa::factored(
            tables.clone(),
            2,
            4,
            TestSource,
            BundleStore::Retained(vec![row]),
            0,
            0,
        )
        .unwrap();
        let dense = family.single_cycle::<LookupIndexChunks>().unwrap().unwrap();
        let rows = IndexedRows(vec![row]);
        let expected: Vec<Fr> = tables
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(poly, tables)| {
                tables
                    .iter()
                    .enumerate()
                    .map(|(factor, table)| {
                        rows.index(2 * poly + factor, 0)
                            .map_or(Fr::from_u64(0), |index| table[index])
                    })
                    .product()
            })
            .collect();
        assert_eq!(dense.final_values(), expected);
    }

    /// A row store that ends inside the domain is refused typed at
    /// construction, which reads the domain's last cycle, instead of being
    /// read past.
    #[test]
    fn rows_missing_from_the_domain_fail_typed() {
        let rounds = 4;
        let mut rows = test_rows(rounds);
        let _ = rows.pop();
        assert!(matches!(
            StreamedFoldedRa::new(
                test_tables::<Fr>(),
                TestSource,
                BundleStore::Retained(rows),
                rounds,
                0,
            ),
            Err(WitnessError::InvalidDimensions { .. })
        ));
    }
}
