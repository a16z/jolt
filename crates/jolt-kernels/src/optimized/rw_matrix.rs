//! Sparse `(K × T)` read-write matrix for the RAM read-write-checking
//! kernel. RAM entries have no one-hot coefficient lookup tables.
//!
//! `ra(k, j)` and `val(k, j)` are conceptually `K × T` matrices, far too
//! large to materialize. One entry exists per RAM access; everything else is
//! implicit: `ra` is 0 off-entry, and `val` is the step function carried
//! between entries — `prev_val`/`next_val` checkpoints (raw `u64`s while
//! binding cycle variables, field elements once address binding starts)
//! recover any implicit coefficient from a neighbor.
//!
//! Cycle-major entries are sorted `(row, col)` and bind cycle variables
//! low-to-high by merging adjacent row pairs; address-major entries are
//! sorted `(col, row)` and bind address variables low-to-high by merging
//! adjacent column pairs against `val_init` checkpoints. Round messages come
//! out as the quadratic factor's evaluations, exactly like the legacy
//! prover; summation order differs from legacy where convenient (field
//! addition is exact, so the values are identical).
//!
//! The streaming RAM family starts both matrices from the [`RamEventLog`]
//! instead: one unbound entry per remapped access. While cycle variables
//! bind, its cycle-indexed `inc` column is a separate [`SparseCycleColumn`]
//! over the nonzero increments, which need not be accesses. The cycle-first
//! round-0 bind uses exactly sized destination allocations ([`exact_task_fill`]);
//! later cycle binds work in place.

use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use jolt_utils::par_concat;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::prefix_product::PrefixColumn;
use super::ram_events::{RamEvent, RamEventLog};
use super::ram_trace::{RamAccessColumns, NO_ACCESS};
use super::read_write::{AddressMaskValues, AddressMasks, ReadWriteBatch, ReadWriteCoefficients};
use super::streaming::{bind_groups_in_place, exact_task_fill, key_runs, FillMismatch, TaskSlots};

/// One explicit matrix entry while cycle variables bind (row-major order).
/// `prev_val`/`next_val` stay raw `u64`s: cycle binding always starts from
/// the unbound trace, so implicit neighbors are unbound memory values.
/// Indices are u32; prepare rejects larger domains.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct CycleMajorEntry<F> {
    /// Cycle index; in `[0, T)` before binding.
    pub row: u32,
    /// Address index; in `[0, K)` (columns never bind in this phase).
    pub col: u32,
    /// The unbound memory value right before this entry's row range.
    pub prev_val: u64,
    /// The unbound memory value right after this entry's row start.
    pub next_val: u64,
    pub val: F,
    pub ra: F,
}

/// One explicit matrix entry while address variables bind (column-major
/// order). Checkpoints are field elements: address binding interpolates
/// them across column pairs.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct AddressMajorEntry<F> {
    pub row: u32,
    pub col: u32,
    pub prev_val: F,
    pub next_val: F,
    pub val: F,
    pub ra: F,
}

impl<F: JoltField> CycleMajorEntry<F> {
    fn into_address_major(self) -> AddressMajorEntry<F> {
        AddressMajorEntry {
            row: self.row,
            col: self.col,
            prev_val: F::from_u64(self.prev_val),
            next_val: F::from_u64(self.next_val),
            val: self.val,
            ra: self.ra,
        }
    }

    fn bind(even: Option<&Self>, odd: Option<&Self>, r: F) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                Self {
                    row: even.row / 2,
                    col: even.col,
                    ra: even.ra + r * (odd.ra - even.ra),
                    val: even.val + r * (odd.val - even.val),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                }
            }
            (Some(even), None) => {
                let odd_val = F::from_u64(even.next_val);
                Self {
                    row: even.row / 2,
                    col: even.col,
                    ra: (F::one() - r) * even.ra,
                    val: even.val + r * (odd_val - even.val),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                }
            }
            (None, Some(odd)) => {
                let even_val = F::from_u64(odd.prev_val);
                Self {
                    row: odd.row / 2,
                    col: odd.col,
                    ra: r * odd.ra,
                    val: even_val + r * (odd.val - even_val),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                }
            }
            (None, None) => unreachable!("a merged pair has at least one side"),
        }
    }

    fn quadratic_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        coefficients: ReadWriteCoefficients<F>,
    ) -> [F; 2] {
        let (ra_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                ([even.ra, odd.ra - even.ra], [even.val, odd.val - even.val])
            }
            (Some(even), None) => {
                let odd_val = F::from_u64(even.next_val);
                ([even.ra, -even.ra], [even.val, odd_val - even.val])
            }
            (None, Some(odd)) => {
                let even_val = F::from_u64(odd.prev_val);
                return [
                    F::zero(),
                    odd.ra * (coefficients.slope(odd.val - even_val, inc_evals[1])),
                ];
            }
            (None, None) => unreachable!("a merged pair has at least one side"),
        };
        [
            ra_evals[0] * coefficients.value(val_evals[0], inc_evals[0]),
            ra_evals[1] * coefficients.slope(val_evals[1], inc_evals[1]),
        ]
    }
}

fn merge_bind_rows<F: JoltField>(
    even: &[CycleMajorEntry<F>],
    odd: &[CycleMajorEntry<F>],
    r: F,
    mut emit: impl FnMut(CycleMajorEntry<F>),
) {
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].col.cmp(&odd[j].col) {
            core::cmp::Ordering::Equal => {
                emit(CycleMajorEntry::bind(Some(&even[i]), Some(&odd[j]), r));
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                emit(CycleMajorEntry::bind(Some(&even[i]), None, r));
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                emit(CycleMajorEntry::bind(None, Some(&odd[j]), r));
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        emit(CycleMajorEntry::bind(Some(entry), None, r));
    }
    for entry in &odd[j..] {
        emit(CycleMajorEntry::bind(None, Some(entry), r));
    }
}

fn merge_quadratic_evals<F: JoltField>(
    even: &[CycleMajorEntry<F>],
    odd: &[CycleMajorEntry<F>],
    inc_evals: [F; 2],
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> [F; 2] {
    let mut acc = [F::zero(); 2];
    let mut add = |evals: [F; 2]| {
        acc[0] += evals[0];
        acc[1] += evals[1];
    };
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].col.cmp(&odd[j].col) {
            core::cmp::Ordering::Equal => {
                add(CycleMajorEntry::quadratic_evals(
                    Some(&even[i]),
                    Some(&odd[j]),
                    inc_evals,
                    batch.unbound(mask.contains_unbound(even[i].col as usize)),
                ));
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                add(CycleMajorEntry::quadratic_evals(
                    Some(&even[i]),
                    None,
                    inc_evals,
                    batch.unbound(mask.contains_unbound(even[i].col as usize)),
                ));
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                add(CycleMajorEntry::quadratic_evals(
                    None,
                    Some(&odd[j]),
                    inc_evals,
                    batch.unbound(mask.contains_unbound(odd[j].col as usize)),
                ));
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        add(CycleMajorEntry::quadratic_evals(
            Some(entry),
            None,
            inc_evals,
            batch.unbound(mask.contains_unbound(entry.col as usize)),
        ));
    }
    for entry in &odd[j..] {
        add(CycleMajorEntry::quadratic_evals(
            None,
            Some(entry),
            inc_evals,
            batch.unbound(mask.contains_unbound(entry.col as usize)),
        ));
    }
    acc
}

fn split_row_pair<F>(
    group: &[CycleMajorEntry<F>],
) -> (&[CycleMajorEntry<F>], &[CycleMajorEntry<F>]) {
    let odd_start = group.partition_point(|entry| entry.row % 2 == 0);
    group.split_at(odd_start)
}

fn same_row_pair<F>(a: &CycleMajorEntry<F>, b: &CycleMajorEntry<F>) -> bool {
    a.row / 2 == b.row / 2
}

/// One row pair's `[q(0), q_∞]` contribution: its entries' quadratic
/// evaluations against the pair's `inc` evaluations, times the Gruen head
/// weight.
fn pair_quadratic<F: JoltField>(
    group: &[CycleMajorEntry<F>],
    inc_evals: [F; 2],
    head: F,
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> [F; 2] {
    let (even, odd) = split_row_pair(group);
    let inner = merge_quadratic_evals(even, odd, inc_evals, batch, mask);
    [head * inner[0], head * inner[1]]
}

/// An odd-row entry's summand at `t = 1`, against `inc` at its row.
fn odd_entry_at_one<F: JoltField>(
    entry: &CycleMajorEntry<F>,
    inc: F,
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> F {
    CycleMajorEntry::quadratic_evals(
        Some(entry),
        None,
        [inc, F::zero()],
        batch.unbound(mask.contains_unbound(entry.col as usize)),
    )[0]
}

/// `q(1)` of the phase-1 round: every odd-row entry against `inc` at its row.
fn odd_rows_at_one<F: JoltField>(
    entries: &[CycleMajorEntry<F>],
    eq_head: impl Fn(usize) -> F,
    mut inc_at: impl FnMut(usize) -> F,
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> F {
    entries
        .iter()
        .filter(|entry| entry.row % 2 == 1)
        .fold(F::zero(), |sum, entry| {
            let row = entry.row as usize;
            sum + eq_head(row / 2) * odd_entry_at_one(entry, inc_at(row), batch, mask)
        })
}

/// The unbound entry of a remapped access: it reads `pre` and leaves `post`
/// behind.
fn unbound_entry<F: JoltField>(event: &RamEvent) -> Option<CycleMajorEntry<F>> {
    event.address.map(|col| CycleMajorEntry {
        row: event.cycle,
        col,
        prev_val: event.pre_value,
        next_val: event.post_value,
        val: F::from_u64(event.pre_value),
        ra: F::one(),
    })
}

/// The log's accesses as unbound entries in cycle order, in one exactly
/// sized allocation.
fn unbound_entries<F: JoltField, T: Copy + Send>(
    log: &RamEventLog,
    map: impl Fn(CycleMajorEntry<F>) -> T + Sync,
) -> Result<Vec<T>, FillMismatch> {
    exact_task_fill(
        &key_runs(log.events(), |event| event.cycle as usize),
        |events| {
            Ok(events
                .iter()
                .filter(|event| event.address.is_some())
                .count())
        },
        |events, slots: &mut TaskSlots<'_, T>| {
            events
                .iter()
                .filter_map(unbound_entry)
                .for_each(|entry| slots.push(map(entry)));
            Ok(())
        },
        |mismatch| mismatch,
    )
}

/// The cycle-major sparse matrix: entries sorted by `(row, col)`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct CycleMajorMatrix<F> {
    pub entries: Vec<CycleMajorEntry<F>>,
}

impl<F: JoltField> CycleMajorMatrix<F> {
    /// Bind one cycle variable low-to-high: merge every adjacent row pair.
    pub fn bind(&mut self, r: F) {
        #[cfg(feature = "parallel")]
        let bound: Vec<CycleMajorEntry<F>> = par_concat(
            &self
                .entries
                .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                .fold(Vec::new, |mut out, group| {
                    let (even, odd) = split_row_pair(group);
                    merge_bind_rows(even, odd, r, |entry| out.push(entry));
                    out
                })
                .collect::<Vec<_>>(),
        );
        #[cfg(not(feature = "parallel"))]
        let bound: Vec<CycleMajorEntry<F>> = {
            let mut out = Vec::with_capacity(self.entries.len());
            for group in self.entries.chunk_by(|a, b| a.row / 2 == b.row / 2) {
                let (even, odd) = split_row_pair(group);
                merge_bind_rows(even, odd, r, |entry| out.push(entry));
            }
            out
        };
        self.entries = bound;
    }

    /// The quadratic factor `[q(0), q_∞]` of the phase-1 round message:
    /// `q(t) = Σ_pairs eq_head(pair) · Σ_cols ra(t)·[(1+γ)val(t)+(γ+γ²C)inc(t)+γ³+γ⁴N]`,
    /// with `eq_head` supplied per pair index (the Gruen head weight) and
    /// `inc` the bound committed increment column.
    pub fn quadratic_coefficients(
        &self,
        eq_head: impl Fn(usize) -> F + Sync,
        inc: &PrefixColumn<F>,
        batch: ReadWriteBatch<F>,
        mask: &AddressMasks<F>,
    ) -> [F; 2] {
        // Each surviving entry adds one interval search and a coefficient selection
        // in the existing merge; count changes q(0), not the quadratic leading term.
        let per_group = |group: &[CycleMajorEntry<F>]| -> [F; 2] {
            let pair = (group[0].row / 2) as usize;
            let inc_0 = inc.value(2 * pair);
            let inc_evals = [inc_0, inc.value(2 * pair + 1) - inc_0];
            pair_quadratic(group, inc_evals, eq_head(pair), batch, mask)
        };

        #[cfg(feature = "parallel")]
        {
            self.entries
                .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                .map(per_group)
                .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.entries
                .chunk_by(|a, b| a.row / 2 == b.row / 2)
                .map(per_group)
                .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
        }
    }

    pub fn q_at_one(
        &self,
        eq_head: impl Fn(usize) -> F,
        inc: &PrefixColumn<F>,
        batch: ReadWriteBatch<F>,
        mask: &AddressMasks<F>,
    ) -> F {
        odd_rows_at_one(&self.entries, eq_head, |row| inc.value(row), batch, mask)
    }

    /// [`Self::bind`] in place ([`bind_groups_in_place`]), without a second
    /// entry vector.
    pub fn bind_in_place(&mut self, r: F) -> Result<(), FillMismatch> {
        bind_groups_in_place(
            &mut self.entries,
            |entry| entry.row as usize / 2,
            |group, emit| {
                let (even, odd) = split_row_pair(group);
                merge_bind_rows(even, odd, r, emit);
            },
        )
    }

    /// [`Self::quadratic_coefficients`] against a sparse `inc`: each run of
    /// whole row pairs seeks `inc` once and merge-walks it.
    pub fn sparse_quadratic_coefficients(
        &self,
        eq_head: impl Fn(usize) -> F + Sync,
        inc: &SparseCycleColumn<F>,
        batch: ReadWriteBatch<F>,
        mask: &AddressMasks<F>,
    ) -> [F; 2] {
        let task = |run: &&[CycleMajorEntry<F>]| -> [F; 2] {
            let mut cursor = inc.cursor(run.first().map_or(0, |entry| entry.row as usize / 2));
            run.chunk_by(same_row_pair)
                .fold([F::zero(); 2], |acc, group| {
                    let pair = (group[0].row / 2) as usize;
                    let [inc_0, inc_1] = cursor.pair(pair);
                    let [q_0, q_infty] =
                        pair_quadratic(group, [inc_0, inc_1 - inc_0], eq_head(pair), batch, mask);
                    [acc[0] + q_0, acc[1] + q_infty]
                })
        };
        let runs = key_runs(&self.entries, |entry| entry.row as usize / 2);
        let add = |a: [F; 2], b: [F; 2]| [a[0] + b[0], a[1] + b[1]];
        #[cfg(feature = "parallel")]
        return runs.par_iter().map(task).reduce(|| [F::zero(); 2], add);
        #[cfg(not(feature = "parallel"))]
        runs.iter().map(task).fold([F::zero(); 2], add)
    }

    /// [`Self::q_at_one`] against a sparse `inc`, merge-walked.
    pub fn sparse_q_at_one(
        &self,
        eq_head: impl Fn(usize) -> F,
        inc: &SparseCycleColumn<F>,
        batch: ReadWriteBatch<F>,
        mask: &AddressMasks<F>,
    ) -> F {
        let mut cursor = inc.cursor(0);
        odd_rows_at_one(
            &self.entries,
            eq_head,
            |row| cursor.pair(row / 2)[1],
            batch,
            mask,
        )
    }

    /// Reinterpret as address-major once every cycle variable is bound: all
    /// rows are 0, so `(row, col)` order IS `(col, row)` order and only the
    /// checkpoint representation changes.
    pub fn into_address_major(self) -> AddressMajorMatrix<F> {
        debug_assert!(self.entries.iter().all(|entry| entry.row == 0));
        #[cfg(feature = "parallel")]
        let entries = self
            .entries
            .into_par_iter()
            .map(CycleMajorEntry::into_address_major)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let entries = self
            .entries
            .into_iter()
            .map(CycleMajorEntry::into_address_major)
            .collect();
        AddressMajorMatrix { entries }
    }
}

#[inline]
fn round0_entry<F: JoltField>(
    columns: &RamAccessColumns,
    cycle: usize,
) -> Option<CycleMajorEntry<F>> {
    let address = columns.addresses[cycle];
    (address != NO_ACCESS).then(|| {
        let pre_value = columns.pre_value(cycle);
        CycleMajorEntry {
            row: cycle as u32,
            col: address,
            prev_val: pre_value,
            next_val: columns.post_value(cycle),
            val: F::from_u64(pre_value),
            ra: F::one(),
        }
    })
}

pub(crate) fn round0_q_at_one<F: JoltField>(
    columns: &RamAccessColumns,
    eq_head: impl Fn(usize) -> F,
    inc: &PrefixColumn<F>,
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> F {
    (1..columns.addresses.prefix_len())
        .step_by(2)
        .fold(F::zero(), |sum, row| {
            let Some(entry) = round0_entry::<F>(columns, row) else {
                return sum;
            };
            sum + eq_head(row / 2) * odd_entry_at_one(&entry, inc.value(row), batch, mask)
        })
}

/// An unbound row pair's quadratic evaluations: its two entries, which are
/// separate entries when their columns differ.
fn round0_pair_quadratic<F: JoltField>(
    even: Option<&CycleMajorEntry<F>>,
    odd: Option<&CycleMajorEntry<F>>,
    inc_evals: [F; 2],
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> [F; 2] {
    match (even, odd) {
        (Some(even), Some(odd)) if even.col != odd.col => {
            let lone_even = CycleMajorEntry::quadratic_evals(
                Some(even),
                None,
                inc_evals,
                batch.unbound(mask.contains_unbound(even.col as usize)),
            );
            let lone_odd = CycleMajorEntry::quadratic_evals(
                None,
                Some(odd),
                inc_evals,
                batch.unbound(mask.contains_unbound(odd.col as usize)),
            );
            [lone_even[0] + lone_odd[0], lone_even[1] + lone_odd[1]]
        }
        _ => {
            let col = even.or(odd).map_or(0, |entry| entry.col);
            CycleMajorEntry::quadratic_evals(
                even,
                odd,
                inc_evals,
                batch.unbound(mask.contains_unbound(col as usize)),
            )
        }
    }
}

/// An unbound row pair's bound entries in column order: two when the
/// columns differ.
fn round0_pair_bind<F: JoltField>(
    even: Option<&CycleMajorEntry<F>>,
    odd: Option<&CycleMajorEntry<F>>,
    r: F,
) -> [Option<CycleMajorEntry<F>>; 2] {
    match (even, odd) {
        (None, None) => [None, None],
        (Some(even), Some(odd)) if even.col != odd.col => {
            let lone_even = CycleMajorEntry::bind(Some(even), None, r);
            let lone_odd = CycleMajorEntry::bind(None, Some(odd), r);
            if even.col < odd.col {
                [Some(lone_even), Some(lone_odd)]
            } else {
                [Some(lone_odd), Some(lone_even)]
            }
        }
        _ => [Some(CycleMajorEntry::bind(even, odd, r)), None],
    }
}

pub(crate) fn round0_quadratic_coefficients<F: JoltField>(
    columns: &RamAccessColumns,
    eq_head: impl Fn(usize) -> F + Sync,
    inc: &PrefixColumn<F>,
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> [F; 2] {
    let pairs = columns.addresses.prefix_len().div_ceil(2);
    let per_pair = |pair: usize| -> [F; 2] {
        let even = round0_entry::<F>(columns, 2 * pair);
        let odd = round0_entry::<F>(columns, 2 * pair + 1);
        if even.is_none() && odd.is_none() {
            return [F::zero(); 2];
        }
        let inc_0 = inc.value(2 * pair);
        let inc_evals = [inc_0, inc.value(2 * pair + 1) - inc_0];
        let inner = round0_pair_quadratic(even.as_ref(), odd.as_ref(), inc_evals, batch, mask);
        let head = eq_head(pair);
        [head * inner[0], head * inner[1]]
    };

    #[cfg(feature = "parallel")]
    {
        (0..pairs)
            .into_par_iter()
            .map(per_pair)
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..pairs)
            .map(per_pair)
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
}

/// First bind, producing the first entry vector at half size.
/// Entries remain ordered by cycle, then column.
pub(crate) fn round0_bind<F: JoltField>(columns: &RamAccessColumns, r: F) -> CycleMajorMatrix<F> {
    let pairs = columns.addresses.prefix_len().div_ceil(2);
    let per_pair = |pair: usize| -> [Option<CycleMajorEntry<F>>; 2] {
        let even = round0_entry::<F>(columns, 2 * pair);
        let odd = round0_entry::<F>(columns, 2 * pair + 1);
        round0_pair_bind(even.as_ref(), odd.as_ref(), r)
    };

    #[cfg(feature = "parallel")]
    let entries: Vec<CycleMajorEntry<F>> = par_concat(
        &(0..pairs)
            .into_par_iter()
            .fold(Vec::new, |mut out, pair| {
                out.extend(per_pair(pair).into_iter().flatten());
                out
            })
            .collect::<Vec<_>>(),
    );
    #[cfg(not(feature = "parallel"))]
    let entries: Vec<CycleMajorEntry<F>> = (0..pairs)
        .flat_map(|pair| per_pair(pair).into_iter().flatten())
        .collect();
    CycleMajorMatrix { entries }
}

fn same_event_pair(a: &RamEvent, b: &RamEvent) -> bool {
    a.cycle / 2 == b.cycle / 2
}

/// A row pair's events as its index, the unbound entries of its two cycles
/// and their `inc` values.
fn event_pair<F: JoltField>(
    group: &[RamEvent],
) -> (usize, [Option<CycleMajorEntry<F>>; 2], [F; 2]) {
    let mut entries = [None, None];
    let mut inc = [F::zero(); 2];
    for event in group {
        let half = event.cycle as usize % 2;
        entries[half] = unbound_entry(event);
        inc[half] = F::from_i128(event.inc());
    }
    (group[0].cycle as usize / 2, entries, inc)
}

/// Runs of whole row pairs of the log's events.
fn event_pair_runs(log: &RamEventLog) -> Vec<&[RamEvent]> {
    key_runs(log.events(), |event| event.cycle as usize / 2)
}

/// Round 0's `[q(0), q_∞]` read straight off the event log, as
/// [`round0_quadratic_coefficients`] reads the access columns: neither the
/// unbound matrix nor the `inc` column is materialized.
pub(crate) fn round0_event_quadratic<F: JoltField>(
    log: &RamEventLog,
    eq_head: impl Fn(usize) -> F + Sync,
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> [F; 2] {
    let task = |run: &&[RamEvent]| -> [F; 2] {
        run.chunk_by(same_event_pair)
            .fold([F::zero(); 2], |acc, group| {
                let (pair, [even, odd], [inc_0, inc_1]) = event_pair::<F>(group);
                if even.is_none() && odd.is_none() {
                    return acc;
                }
                let inner = round0_pair_quadratic(
                    even.as_ref(),
                    odd.as_ref(),
                    [inc_0, inc_1 - inc_0],
                    batch,
                    mask,
                );
                let head = eq_head(pair);
                [acc[0] + head * inner[0], acc[1] + head * inner[1]]
            })
    };
    let runs = event_pair_runs(log);
    let add = |a: [F; 2], b: [F; 2]| [a[0] + b[0], a[1] + b[1]];
    #[cfg(feature = "parallel")]
    return runs.par_iter().map(task).reduce(|| [F::zero(); 2], add);
    #[cfg(not(feature = "parallel"))]
    runs.iter().map(task).fold([F::zero(); 2], add)
}

/// Round 0's `q(1)` read straight off the event log: every odd-cycle access
/// against its own increment.
pub(crate) fn round0_event_q_at_one<F: JoltField>(
    log: &RamEventLog,
    eq_head: impl Fn(usize) -> F,
    batch: ReadWriteBatch<F>,
    mask: &AddressMasks<F>,
) -> F {
    log.events()
        .iter()
        .filter(|event| event.cycle % 2 == 1)
        .fold(F::zero(), |sum, event| match unbound_entry::<F>(event) {
            Some(entry) => {
                let inc = F::from_i128(event.inc());
                sum + eq_head(event.cycle as usize / 2) * odd_entry_at_one(&entry, inc, batch, mask)
            }
            None => sum,
        })
}

/// Round 0's bind read straight off the event log: [`round0_bind`]'s
/// matrix and the bound sparse `inc`, each in one exactly sized allocation.
pub(crate) fn round0_event_bind<F: JoltField>(
    log: &RamEventLog,
    r: F,
) -> Result<(CycleMajorMatrix<F>, SparseCycleColumn<F>), FillMismatch> {
    let runs = event_pair_runs(log);
    let entries = exact_task_fill(
        &runs,
        |run| {
            Ok(run
                .chunk_by(same_event_pair)
                .map(|group| {
                    let mut cols = group.iter().filter_map(|event| event.address);
                    match (cols.next(), cols.next()) {
                        (None, _) => 0,
                        (Some(even), Some(odd)) if even != odd => 2,
                        _ => 1,
                    }
                })
                .sum())
        },
        |run, slots: &mut TaskSlots<'_, CycleMajorEntry<F>>| {
            for group in run.chunk_by(same_event_pair) {
                let (_, [even, odd], _) = event_pair::<F>(group);
                round0_pair_bind(even.as_ref(), odd.as_ref(), r)
                    .into_iter()
                    .flatten()
                    .for_each(|entry| slots.push(entry));
            }
            Ok(())
        },
        |mismatch| mismatch,
    )?;
    let incremented = |group: &&[RamEvent]| group.iter().any(|event| event.inc() != 0);
    let inc = exact_task_fill(
        &runs,
        |run| Ok(run.chunk_by(same_event_pair).filter(incremented).count()),
        |run, slots: &mut TaskSlots<'_, ColumnEntry<F>>| {
            for group in run.chunk_by(same_event_pair).filter(incremented) {
                let (pair, _, [lo, hi]) = event_pair::<F>(group);
                slots.push(ColumnEntry {
                    row: pair as u32,
                    value: lo + r * (hi - lo),
                });
            }
            Ok(())
        },
        |mismatch| mismatch,
    )?;
    Ok((
        CycleMajorMatrix { entries },
        SparseCycleColumn { entries: inc },
    ))
}

/// Bytes [`round0_bind`] requests at once at most over the `prefix_cycles`
/// cycles before the trace's constant suffix and `2^log_k` addresses, its
/// result included, or `None` past `usize::MAX`. A cycle pair emits one entry
/// per accessing cycle when their columns differ and one entry otherwise, so
/// at most `prefix_cycles` entries, or `⌈prefix_cycles / 2⌉` with one
/// address, over at most one split per pair ([`fold_concat_bytes`]).
pub(crate) fn round0_bind_bytes<F: JoltField>(prefix_cycles: usize, log_k: usize) -> Option<usize> {
    let pairs = prefix_cycles.div_ceil(2);
    let entries = if log_k == 0 { pairs } else { prefix_cycles };
    fold_concat_bytes::<CycleMajorEntry<F>>(entries, pairs.max(1))
}

/// Bytes a parallel `fold(Vec::new, ..)` over at most `splits` Rayon splits
/// that pushes at most `entries` `E` values in total, followed by `collect`
/// and [`par_concat`], requests at once at most, or `None` past `usize::MAX`.
/// A split's vector grows by doubling from four slots, and each reallocation
/// holds the old buffer beside the new one, so a split with `n > 0` values
/// holds at most `4n` slots at any time; the concatenated destination is
/// reserved at exactly the total. Each split also costs one `collect` list
/// node (a `Vec` header and two links) and its four-slot vector of headers,
/// one header in the collected vector, which reserves at least four, and one
/// `par_concat` slot reference. The serial builds stay within the same
/// bytes. These are Rust 1.95 `Vec` growth and Rayon 1.12 unindexed
/// `collect` facts, which the `tests` module measures with a counting
/// allocator.
fn fold_concat_bytes<E>(entries: usize, splits: usize) -> Option<usize> {
    let header = size_of::<Vec<E>>();
    let per_split = 6 * header + 4 * size_of::<usize>();
    entries
        .checked_mul(5)?
        .checked_mul(size_of::<E>())?
        .checked_add(splits.checked_mul(per_split)?)?
        .checked_add(3 * header)
}

impl<F: JoltField> AddressMajorEntry<F> {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.row, odd.row);
                Self {
                    row: even.row,
                    col: even.col / 2,
                    ra: even.ra + r * (odd.ra - even.ra),
                    val: even.val + r * (odd.val - even.val),
                    prev_val: even.prev_val + r * (odd.prev_val - even.prev_val),
                    next_val: even.next_val + r * (odd.next_val - even.next_val),
                }
            }
            (Some(even), None) => Self {
                row: even.row,
                col: even.col / 2,
                ra: (F::one() - r) * even.ra,
                val: even.val + r * (odd_checkpoint - even.val),
                prev_val: even.prev_val + r * (odd_checkpoint - even.prev_val),
                next_val: even.next_val + r * (odd_checkpoint - even.next_val),
            },
            (None, Some(odd)) => Self {
                row: odd.row,
                col: odd.col / 2,
                ra: r * odd.ra,
                val: even_checkpoint + r * (odd.val - even_checkpoint),
                prev_val: even_checkpoint + r * (odd.prev_val - even_checkpoint),
                next_val: even_checkpoint + r * (odd.next_val - even_checkpoint),
            },
            (None, None) => unreachable!("a merged pair has at least one side"),
        }
    }

    fn address_round_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        coefficients: [ReadWriteCoefficients<F>; 2],
    ) -> [F; 2] {
        let (ra_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.row, odd.row);
                (
                    [even.ra, odd.ra + odd.ra - even.ra],
                    [even.val, odd.val + odd.val - even.val],
                )
            }
            (Some(even), None) => (
                [even.ra, -even.ra],
                [even.val, odd_checkpoint + odd_checkpoint - even.val],
            ),
            (None, Some(odd)) => {
                return [
                    F::zero(),
                    eq_eval
                        * (odd.ra + odd.ra)
                        * coefficients[1].value(odd.val + odd.val - even_checkpoint, inc_eval),
                ];
            }
            (None, None) => unreachable!("a merged pair has at least one side"),
        };
        [
            eq_eval * ra_evals[0] * coefficients[0].value(val_evals[0], inc_eval),
            eq_eval * ra_evals[1] * coefficients[1].value(val_evals[1], inc_eval),
        ]
    }
}

fn split_col_pair<F>(
    group: &[AddressMajorEntry<F>],
) -> (&[AddressMajorEntry<F>], &[AddressMajorEntry<F>]) {
    let odd_start = group.partition_point(|entry| entry.col % 2 == 0);
    group.split_at(odd_start)
}

fn merge_bind_cols<F: JoltField>(
    even: &[AddressMajorEntry<F>],
    odd: &[AddressMajorEntry<F>],
    mut even_checkpoint: F,
    mut odd_checkpoint: F,
    r: F,
    out: &mut Vec<AddressMajorEntry<F>>,
) {
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].row.cmp(&odd[j].row) {
            core::cmp::Ordering::Equal => {
                out.push(AddressMajorEntry::bind(
                    Some(&even[i]),
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                ));
                even_checkpoint = even[i].next_val;
                odd_checkpoint = odd[j].next_val;
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                out.push(AddressMajorEntry::bind(
                    Some(&even[i]),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                ));
                even_checkpoint = even[i].next_val;
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                out.push(AddressMajorEntry::bind(
                    None,
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                ));
                odd_checkpoint = odd[j].next_val;
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        out.push(AddressMajorEntry::bind(
            Some(entry),
            None,
            even_checkpoint,
            odd_checkpoint,
            r,
        ));
        even_checkpoint = entry.next_val;
    }
    for entry in &odd[j..] {
        out.push(AddressMajorEntry::bind(
            None,
            Some(entry),
            even_checkpoint,
            odd_checkpoint,
            r,
        ));
        odd_checkpoint = entry.next_val;
    }
}

fn merge_address_round_evals<F: JoltField>(
    even: &[AddressMajorEntry<F>],
    odd: &[AddressMajorEntry<F>],
    mut even_checkpoint: F,
    mut odd_checkpoint: F,
    inc: &PrefixColumn<F>,
    eq: &[F],
    coefficients: [ReadWriteCoefficients<F>; 2],
) -> [F; 2] {
    let mut acc = [F::zero(); 2];
    let mut add = |evals: [F; 2]| {
        acc[0] += evals[0];
        acc[1] += evals[1];
    };
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].row.cmp(&odd[j].row) {
            core::cmp::Ordering::Equal => {
                add(AddressMajorEntry::address_round_evals(
                    Some(&even[i]),
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    inc.value(even[i].row as usize),
                    eq[even[i].row as usize],
                    coefficients,
                ));
                even_checkpoint = even[i].next_val;
                odd_checkpoint = odd[j].next_val;
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                add(AddressMajorEntry::address_round_evals(
                    Some(&even[i]),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    inc.value(even[i].row as usize),
                    eq[even[i].row as usize],
                    coefficients,
                ));
                even_checkpoint = even[i].next_val;
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                add(AddressMajorEntry::address_round_evals(
                    None,
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    inc.value(odd[j].row as usize),
                    eq[odd[j].row as usize],
                    coefficients,
                ));
                odd_checkpoint = odd[j].next_val;
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        add(AddressMajorEntry::address_round_evals(
            Some(entry),
            None,
            even_checkpoint,
            odd_checkpoint,
            inc.value(entry.row as usize),
            eq[entry.row as usize],
            coefficients,
        ));
        even_checkpoint = entry.next_val;
    }
    for entry in &odd[j..] {
        add(AddressMajorEntry::address_round_evals(
            None,
            Some(entry),
            even_checkpoint,
            odd_checkpoint,
            inc.value(entry.row as usize),
            eq[entry.row as usize],
            coefficients,
        ));
        odd_checkpoint = entry.next_val;
    }
    acc
}

/// The address-major sparse matrix: entries sorted by `(col, row)`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct AddressMajorMatrix<F> {
    pub entries: Vec<AddressMajorEntry<F>>,
}

impl<F: JoltField> AddressMajorMatrix<F> {
    pub fn from_columns(columns: &RamAccessColumns) -> Self {
        let mut entries: Vec<_> = (0..columns.addresses.prefix_len())
            .filter_map(|cycle| {
                round0_entry(columns, cycle).map(CycleMajorEntry::into_address_major)
            })
            .collect();
        #[cfg(feature = "parallel")]
        entries.par_sort_unstable_by_key(|entry| (entry.col, entry.row));
        #[cfg(not(feature = "parallel"))]
        entries.sort_unstable_by_key(|entry| (entry.col, entry.row));
        Self { entries }
    }

    /// The unbound matrix of the log's accesses, sorted `(col, row)`.
    pub fn from_events(log: &RamEventLog) -> Result<Self, FillMismatch> {
        let mut entries = unbound_entries(log, CycleMajorEntry::into_address_major)?;
        #[cfg(feature = "parallel")]
        entries.par_sort_unstable_by_key(|entry| (entry.col, entry.row));
        #[cfg(not(feature = "parallel"))]
        entries.sort_unstable_by_key(|entry| (entry.col, entry.row));
        Ok(Self { entries })
    }

    /// Entry bytes [`Self::from_columns`] keeps over the `prefix_cycles`
    /// cycles before the trace's constant suffix, or `None` past
    /// `usize::MAX`: at most one entry per cycle, collected sequentially, so
    /// the doubling growth leaves at most `max(4, 2 · prefix_cycles)` slots.
    pub fn collected_bytes(prefix_cycles: usize) -> Option<usize> {
        prefix_cycles
            .max(2)
            .checked_mul(2)?
            .checked_mul(size_of::<AddressMajorEntry<F>>())
    }

    /// Bytes the first [`Self::bind`] of a matrix built by
    /// [`Self::from_columns`] over the `prefix_cycles` cycles before the
    /// trace's constant suffix requests at once at most while it collects, or
    /// `None` past `usize::MAX`: the old entries at their collected capacity
    /// ([`Self::collected_bytes`]) beside the bind's collect of at most one
    /// entry per old entry, over at most one split per column pair
    /// ([`fold_concat_bytes`]). Binding `val_init` afterwards is not counted.
    pub fn first_bind_bytes(prefix_cycles: usize) -> Option<usize> {
        Self::collected_bytes(prefix_cycles)?.checked_add(
            fold_concat_bytes::<AddressMajorEntry<F>>(prefix_cycles, prefix_cycles.max(1))?,
        )
    }

    /// Entry bytes of the bound entries the first [`Self::bind`] keeps after
    /// dropping the old ones, or `None` past `usize::MAX`: at most one entry
    /// per cycle, at exact capacity.
    pub fn bound_bytes(prefix_cycles: usize) -> Option<usize> {
        prefix_cycles.checked_mul(size_of::<AddressMajorEntry<F>>())
    }

    /// After binding every address, retain the explicit cycle prefix and
    /// the final memory checkpoint as a constant value suffix.
    pub fn into_cycle_tables(self, initial: F) -> (PrefixColumn<F>, PrefixColumn<F>) {
        let prefix = self
            .entries
            .last()
            .map_or(0, |entry| entry.row as usize + 1);
        let mut ra = vec![F::zero(); prefix];
        let mut val = Vec::with_capacity(prefix);
        let mut checkpoint = initial;
        for entry in self.entries {
            debug_assert_eq!(entry.col, 0);
            let row = entry.row as usize;
            val.resize(row, checkpoint);
            val.push(entry.val);
            ra[row] = entry.ra;
            checkpoint = entry.next_val;
        }
        (
            PrefixColumn {
                values: ra,
                tail: [F::zero(); 2],
            },
            PrefixColumn {
                values: val,
                tail: [checkpoint; 2],
            },
        )
    }

    /// Bind one address variable low-to-high: merge every adjacent column
    /// pair against the `val_init` checkpoints, then bind `val_init` itself.
    pub fn bind(&mut self, r: F, val_init: &mut Polynomial<F>) {
        #[cfg(feature = "parallel")]
        let bound: Vec<AddressMajorEntry<F>> = par_concat(
            &self
                .entries
                .par_chunk_by(|a, b| a.col / 2 == b.col / 2)
                .fold(Vec::new, |mut out, group| {
                    let (even, odd) = split_col_pair(group);
                    let even_col = 2 * (group[0].col / 2) as usize;
                    merge_bind_cols(
                        even,
                        odd,
                        val_init.evals()[even_col],
                        val_init.evals()[even_col + 1],
                        r,
                        &mut out,
                    );
                    out
                })
                .collect::<Vec<_>>(),
        );
        #[cfg(not(feature = "parallel"))]
        let bound: Vec<AddressMajorEntry<F>> = {
            let mut out = Vec::with_capacity(self.entries.len());
            for group in self.entries.chunk_by(|a, b| a.col / 2 == b.col / 2) {
                let (even, odd) = split_col_pair(group);
                let even_col = 2 * (group[0].col / 2) as usize;
                merge_bind_cols(
                    even,
                    odd,
                    val_init.evals()[even_col],
                    val_init.evals()[even_col + 1],
                    r,
                    &mut out,
                );
            }
            out
        };
        self.entries = bound;
        val_init.bind_with_order(r, BindingOrder::LowToHigh);
    }

    /// The `[s(0), s(2)]` evaluations of the phase-2 round message over all
    /// column pairs.
    pub fn address_round_evals(
        &self,
        val_init: &Polynomial<F>,
        inc: &PrefixColumn<F>,
        eq: &[F],
        batch: ReadWriteBatch<F>,
        mask: &AddressMasks<F>,
    ) -> [F; 2] {
        let per_group = |group: &[AddressMajorEntry<F>]| -> [F; 2] {
            let (even, odd) = split_col_pair(group);
            let even_col = 2 * (group[0].col / 2) as usize;
            // Reuse the existing column-pair pass. The structural mask adds
            // O(s) public boundary lookups per pair, without a K-sized table.
            let mask_even = mask.value(even_col);
            let mask_odd = mask.value(even_col + 1);
            let coefficients = [
                batch.bound(mask_even),
                batch.bound(AddressMaskValues::at_two(mask_even, mask_odd)),
            ];
            merge_address_round_evals(
                even,
                odd,
                val_init.evals()[even_col],
                val_init.evals()[even_col + 1],
                inc,
                eq,
                coefficients,
            )
        };

        #[cfg(feature = "parallel")]
        {
            self.entries
                .par_chunk_by(|a, b| a.col / 2 == b.col / 2)
                .map(per_group)
                .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.entries
                .chunk_by(|a, b| a.col / 2 == b.col / 2)
                .map(per_group)
                .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
        }
    }

    /// The fully bound `(ra, val)` pair once every variable is bound: at
    /// most one entry remains (none when the trace makes no RAM access).
    pub fn final_values(&self, val_init: &Polynomial<F>) -> (F, F) {
        debug_assert!(self.entries.len() <= 1);
        debug_assert_eq!(val_init.len(), 1);
        match self.entries.first() {
            Some(entry) => (entry.ra, entry.val),
            None => (F::zero(), val_init.evals()[0]),
        }
    }
}

/// One row of a [`SparseCycleColumn`] that may be nonzero.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct ColumnEntry<F> {
    row: u32,
    value: F,
}

fn same_column_pair<F>(a: &ColumnEntry<F>, b: &ColumnEntry<F>) -> bool {
    a.row / 2 == b.row / 2
}

/// A row pair's index and `[lo, hi]` values; a missing half is zero.
fn column_pair<F: JoltField>(group: &[ColumnEntry<F>]) -> (usize, [F; 2]) {
    let mut values = [F::zero(); 2];
    for entry in group {
        values[entry.row as usize % 2] = entry.value;
    }
    ((group[0].row / 2) as usize, values)
}

/// A cycle-indexed column during the cycle rounds: its rows that may be
/// nonzero, sorted by row, at most one per row; an absent row is zero. The
/// support follows the trace, never a challenge: binding keeps an entry for
/// every row pair that had one, also when the challenge cancels it to zero.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SparseCycleColumn<F> {
    entries: Vec<ColumnEntry<F>>,
}

impl<F: JoltField> SparseCycleColumn<F> {
    /// The unbound `RamInc` column: each event's nonzero increment.
    pub fn increments(log: &RamEventLog) -> Result<Self, FillMismatch> {
        Self::from_events(
            log,
            |event| event.inc() != 0,
            |event| F::from_i128(event.inc()),
        )
    }

    /// The unbound address-folded `ra(j) = eq(r_address, address(j))`:
    /// `eq_address[address]` at every remapped access, zero-increment
    /// accesses included; a store whose address remaps to no word has no
    /// entry. `eq_address` spans the log's `ram_K` words, and the log checks
    /// every address against them at collection.
    pub fn folded_ra(log: &RamEventLog, eq_address: &[F]) -> Result<Self, FillMismatch> {
        Self::from_events(
            log,
            |event| event.address.is_some(),
            |event| {
                event
                    .address
                    .map_or(F::zero(), |address| eq_address[address as usize])
            },
        )
    }

    fn from_events(
        log: &RamEventLog,
        keep: impl Fn(&RamEvent) -> bool + Sync,
        value: impl Fn(&RamEvent) -> F + Sync,
    ) -> Result<Self, FillMismatch> {
        let entries = exact_task_fill(
            &key_runs(log.events(), |event| event.cycle as usize),
            |events| Ok(events.iter().filter(|event| keep(event)).count()),
            |events, slots: &mut TaskSlots<'_, ColumnEntry<F>>| {
                for event in events.iter().filter(|event| keep(event)) {
                    slots.push(ColumnEntry {
                        row: event.cycle,
                        value: value(event),
                    });
                }
                Ok(())
            },
            |mismatch| mismatch,
        )?;
        Ok(Self { entries })
    }

    /// Runs of whole row pairs for one pass's tasks ([`key_runs`]).
    pub fn pair_runs(&self) -> Vec<&[ColumnEntry<F>]> {
        key_runs(&self.entries, |entry| entry.row as usize / 2)
    }

    /// The row pairs of one [`Self::pair_runs`] run holding an entry, in pair
    /// order: `(pair, [value(2·pair), value(2·pair + 1)])`.
    pub fn run_pairs(run: &[ColumnEntry<F>]) -> impl Iterator<Item = (usize, [F; 2])> + '_ {
        run.chunk_by(same_column_pair).map(column_pair)
    }

    /// Binds one cycle variable low-to-high in place, `lo + r·(hi − lo)` per
    /// row pair with a missing half zero ([`bind_groups_in_place`]).
    pub fn bind(&mut self, r: F) -> Result<(), FillMismatch> {
        bind_groups_in_place(
            &mut self.entries,
            |entry| entry.row as usize / 2,
            |group, emit| {
                let (pair, [lo, hi]) = column_pair(group);
                emit(ColumnEntry {
                    row: pair as u32,
                    value: lo + r * (hi - lo),
                });
            },
        )
    }

    /// The column's value once every cycle variable is bound; zero without
    /// an entry.
    pub fn bound_value(&self) -> F {
        debug_assert!(self.entries.iter().all(|entry| entry.row == 0));
        self.entries.first().map_or(F::zero(), |entry| entry.value)
    }

    /// A merge-walk reader of this column's row pairs, starting at the first
    /// entry at or after `first_pair`; its queries must not decrease.
    pub fn cursor(&self, first_pair: usize) -> PairCursor<'_, F> {
        let start = self
            .entries
            .partition_point(|entry| (entry.row as usize) / 2 < first_pair);
        PairCursor {
            entries: &self.entries[start..],
        }
    }
}

/// A [`SparseCycleColumn`] reader for non-decreasing row-pair queries.
pub(crate) struct PairCursor<'a, F> {
    entries: &'a [ColumnEntry<F>],
}

impl<F: JoltField> PairCursor<'_, F> {
    /// `[value(2·pair), value(2·pair + 1)]`, dropping every earlier pair's
    /// entries for good.
    pub fn pair(&mut self, pair: usize) -> [F; 2] {
        let earlier = self
            .entries
            .iter()
            .take_while(|entry| (entry.row as usize) / 2 < pair)
            .count();
        self.entries = &self.entries[earlier..];
        let mut halves = [F::zero(); 2];
        for entry in self
            .entries
            .iter()
            .take_while(|entry| (entry.row as usize) / 2 == pair)
        {
            halves[entry.row as usize % 2] = entry.value;
        }
        halves
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use jolt_field::{Fr, Ring};
    use jolt_poly::Polynomial;

    use super::super::ram_trace::{RamAccessColumns, RamAddresses};
    use super::{round0_bind, round0_bind_bytes, AddressMajorEntry, AddressMajorMatrix};

    /// Live and peak heap bytes across every thread of this test process,
    /// with a reallocation counted as its new buffer beside the old one.
    struct CountingAllocator;

    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    impl CountingAllocator {
        fn add(bytes: usize) {
            let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
            let _ = PEAK.fetch_max(live, Ordering::Relaxed);
        }

        fn sub(bytes: usize) {
            let _ = LIVE.fetch_sub(bytes, Ordering::Relaxed);
        }
    }

    // SAFETY: every method forwards its request unchanged to `System`, which
    // upholds the `GlobalAlloc` contract; the counters only observe sizes.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            Self::add(layout.size());
            // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            Self::add(layout.size());
            // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the caller upholds `GlobalAlloc::dealloc`'s contract.
            unsafe { System.dealloc(ptr, layout) };
            Self::sub(layout.size());
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            Self::add(new_size);
            // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
            let moved = unsafe { System.realloc(ptr, layout, new_size) };
            Self::sub(if moved.is_null() {
                new_size
            } else {
                layout.size()
            });
            moved
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    /// `run`'s result and the most bytes it held at once beyond those live
    /// when it started. The caller warms the Rayon pool first, so that only
    /// `run`'s own requests move the counters.
    fn peak_above_start<R>(run: impl FnOnce() -> R) -> (R, usize) {
        let start = LIVE.load(Ordering::Relaxed);
        PEAK.store(start, Ordering::Relaxed);
        let result = run();
        (result, PEAK.load(Ordering::Relaxed) - start)
    }

    const CYCLES: usize = 1 << 12;
    const LOG_K: usize = 6;

    /// Every cycle accesses `address(cycle)`, so each bind meets its largest
    /// entry count when adjacent cycles differ.
    fn columns(address: impl Fn(usize) -> u32) -> RamAccessColumns {
        RamAccessColumns {
            addresses: Arc::new(RamAddresses::from(
                (0..CYCLES).map(address).collect::<Vec<_>>(),
            )),
            pre_values: (0..CYCLES as u64).collect(),
            post_values: (1..=CYCLES as u64).collect(),
        }
    }

    /// The round-0 collect stays within [`round0_bind_bytes`] at both entry
    /// counts it prices: one entry per cycle when adjacent cycles access
    /// different addresses, one per pair with a single address.
    #[test]
    fn round0_bind_stays_within_its_priced_bytes() {
        let r = Fr::from_u64(7);
        for (log_k, address, entries) in [
            (
                LOG_K,
                (|cycle| (cycle % (1 << LOG_K)) as u32) as fn(usize) -> u32,
                CYCLES,
            ),
            (0, |_| 0, CYCLES / 2),
        ] {
            let columns = columns(address);
            let _warm = round0_bind::<Fr>(&columns, r);
            let (matrix, peak) = peak_above_start(|| round0_bind::<Fr>(&columns, r));
            assert_eq!(matrix.entries.len(), entries);
            let priced = round0_bind_bytes::<Fr>(CYCLES, log_k).unwrap();
            assert!(
                peak <= priced,
                "log_k = {log_k}: round 0 held {peak} bytes, above the {priced} priced"
            );
        }
    }

    /// The first address-major bind stays within
    /// [`AddressMajorMatrix::first_bind_bytes`] while it collects, counting
    /// the old entries it starts with, and within the bound entries beside
    /// `val_init`'s half-length copy once it binds `val_init`, the window
    /// the RAM read-write kernel prices; it keeps at most
    /// [`AddressMajorMatrix::bound_bytes`].
    #[test]
    fn address_major_first_bind_stays_within_its_priced_bytes() {
        let entry = size_of::<AddressMajorEntry<Fr>>();
        let addresses = 1 << LOG_K;
        let columns = columns(|cycle| (cycle % addresses) as u32);
        let r = Fr::from_u64(7);
        let val_init = || Polynomial::new(vec![Fr::from_u64(3); addresses]);
        AddressMajorMatrix::<Fr>::from_columns(&columns).bind(r, &mut val_init());

        let mut matrix = AddressMajorMatrix::<Fr>::from_columns(&columns);
        let old = matrix.entries.capacity() * entry;
        assert!(old <= AddressMajorMatrix::<Fr>::collected_bytes(CYCLES).unwrap());
        let mut val_init = val_init();
        let ((), peak) = peak_above_start(|| matrix.bind(r, &mut val_init));
        let priced = AddressMajorMatrix::<Fr>::first_bind_bytes(CYCLES)
            .unwrap()
            .max(
                AddressMajorMatrix::<Fr>::bound_bytes(CYCLES).unwrap()
                    + addresses / 2 * size_of::<Fr>(),
            );
        assert!(
            old + peak <= priced,
            "first bind held {} bytes, above the {priced} priced",
            old + peak
        );
        assert!(
            matrix.entries.capacity() * entry
                <= AddressMajorMatrix::<Fr>::bound_bytes(CYCLES).unwrap()
        );
    }
}
