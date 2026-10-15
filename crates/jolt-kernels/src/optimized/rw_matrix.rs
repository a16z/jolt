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
//! Both start from the [`RamEventLog`]: one unbound entry per remapped
//! access. While cycle variables bind, the cycle-indexed `inc` column is the
//! separate [`SparseInc`] over the nonzero increments, which need not be
//! accesses.

use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::ram_trace::RamEventLog;

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
        gamma: F,
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
                    odd.ra * (val_slope_term(odd.val - even_val, inc_evals[1], gamma)),
                ];
            }
            (None, None) => unreachable!("a merged pair has at least one side"),
        };
        [
            ra_evals[0] * val_slope_term(val_evals[0], inc_evals[0], gamma),
            ra_evals[1] * val_slope_term(val_evals[1], inc_evals[1], gamma),
        ]
    }
}

#[inline]
pub(super) fn val_slope_term<F: JoltField>(val: F, inc: F, gamma: F) -> F {
    val + gamma * (inc + val)
}

fn merge_bind_rows<F: JoltField>(
    even: &[CycleMajorEntry<F>],
    odd: &[CycleMajorEntry<F>],
    r: F,
    out: &mut Vec<CycleMajorEntry<F>>,
) {
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].col.cmp(&odd[j].col) {
            core::cmp::Ordering::Equal => {
                out.push(CycleMajorEntry::bind(Some(&even[i]), Some(&odd[j]), r));
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                out.push(CycleMajorEntry::bind(Some(&even[i]), None, r));
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                out.push(CycleMajorEntry::bind(None, Some(&odd[j]), r));
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        out.push(CycleMajorEntry::bind(Some(entry), None, r));
    }
    for entry in &odd[j..] {
        out.push(CycleMajorEntry::bind(None, Some(entry), r));
    }
}

fn merge_quadratic_evals<F: JoltField>(
    even: &[CycleMajorEntry<F>],
    odd: &[CycleMajorEntry<F>],
    inc_evals: [F; 2],
    gamma: F,
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
                    gamma,
                ));
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                add(CycleMajorEntry::quadratic_evals(
                    Some(&even[i]),
                    None,
                    inc_evals,
                    gamma,
                ));
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                add(CycleMajorEntry::quadratic_evals(
                    None,
                    Some(&odd[j]),
                    inc_evals,
                    gamma,
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
            gamma,
        ));
    }
    for entry in &odd[j..] {
        add(CycleMajorEntry::quadratic_evals(
            None,
            Some(entry),
            inc_evals,
            gamma,
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

/// The unbound entries, in cycle order: an access reads `pre` and leaves
/// `post` behind.
fn unbound_entries<F: JoltField>(
    log: &RamEventLog,
) -> impl Iterator<Item = CycleMajorEntry<F>> + '_ {
    log.events().iter().filter_map(|event| {
        event.address.map(|col| CycleMajorEntry {
            row: event.cycle,
            col,
            prev_val: event.pre_value,
            next_val: event.post_value,
            val: F::from_u64(event.pre_value),
            ra: F::one(),
        })
    })
}

fn access_count(log: &RamEventLog) -> usize {
    log.events()
        .iter()
        .filter(|event| event.address.is_some())
        .count()
}

/// The cycle-major sparse matrix: entries sorted by `(row, col)`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct CycleMajorMatrix<F> {
    pub entries: Vec<CycleMajorEntry<F>>,
}

impl<F: JoltField> CycleMajorMatrix<F> {
    /// The unbound matrix: at most one access per cycle, so cycle order is
    /// `(row, col)` order.
    pub fn from_events(log: &RamEventLog) -> Self {
        let mut entries: Vec<CycleMajorEntry<F>> = Vec::with_capacity(access_count(log));
        entries.extend(unbound_entries(log));
        Self { entries }
    }

    /// Bind one cycle variable low-to-high: merge every adjacent row pair.
    pub fn bind(&mut self, r: F) {
        #[cfg(feature = "parallel")]
        let bound: Vec<CycleMajorEntry<F>> = self
            .entries
            .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
            .flat_map_iter(|group| {
                let (even, odd) = split_row_pair(group);
                let mut out = Vec::with_capacity(group.len());
                merge_bind_rows(even, odd, r, &mut out);
                out
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let bound: Vec<CycleMajorEntry<F>> = {
            let mut out = Vec::with_capacity(self.entries.len());
            for group in self.entries.chunk_by(|a, b| a.row / 2 == b.row / 2) {
                let (even, odd) = split_row_pair(group);
                merge_bind_rows(even, odd, r, &mut out);
            }
            out
        };
        self.entries = bound;
    }

    /// The quadratic factor `[q(0), q_∞]` of the phase-1 round message:
    /// `q(t) = Σ_pairs eq_head(pair) · Σ_cols ra(t)·(val(t) + γ(inc(t)+val(t)))`,
    /// with `eq_head` supplied per pair index (the Gruen head weight) and
    /// `inc` the bound committed increment column. Pairs without an entry
    /// contribute nothing: `ra` vanishes on them.
    pub fn quadratic_coefficients(
        &self,
        eq_head: impl Fn(usize) -> F + Sync,
        inc: &SparseInc<F>,
        gamma: F,
    ) -> [F; 2] {
        let per_group = |group: &[CycleMajorEntry<F>]| -> [F; 2] {
            let pair = (group[0].row / 2) as usize;
            let [inc_0, inc_1] = inc.pair(pair);
            let inc_evals = [inc_0, inc_1 - inc_0];
            let (even, odd) = split_row_pair(group);
            let inner = merge_quadratic_evals(even, odd, inc_evals, gamma);
            let head = eq_head(pair);
            [head * inner[0], head * inner[1]]
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

    pub fn q_at_one(&self, eq_head: impl Fn(usize) -> F, inc: &SparseInc<F>, gamma: F) -> F {
        self.entries
            .iter()
            .filter(|entry| entry.row % 2 == 1)
            .fold(F::zero(), |sum, entry| {
                let row = entry.row as usize;
                let constant = CycleMajorEntry::quadratic_evals(
                    Some(entry),
                    None,
                    [inc.at(row), F::zero()],
                    gamma,
                )[0];
                sum + eq_head(row / 2) * constant
            })
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

/// One `inc` row that may be nonzero while cycle variables bind.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct IncEntry<F> {
    row: u32,
    value: F,
}

/// The cycle-indexed `inc` column during the cycle rounds: its rows that may
/// be nonzero, sorted by row, at most one per row; an absent row is zero.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SparseInc<F> {
    entries: Vec<IncEntry<F>>,
}

impl<F: JoltField> SparseInc<F> {
    /// The unbound column: each event's exact `RamInc`, zeros omitted.
    pub fn from_events(log: &RamEventLog) -> Self {
        let increments = || log.events().iter().filter(|event| event.inc != 0);
        let mut entries = Vec::with_capacity(increments().count());
        entries.extend(increments().map(|event| IncEntry {
            row: event.cycle,
            value: F::from_i128(event.inc),
        }));
        Self { entries }
    }

    /// `[inc(2·pair), inc(2·pair + 1)]`.
    fn pair(&self, pair: usize) -> [F; 2] {
        let start = self
            .entries
            .partition_point(|entry| (entry.row as usize) < 2 * pair);
        let mut halves = [F::zero(); 2];
        for entry in &self.entries[start..] {
            if entry.row as usize / 2 != pair {
                break;
            }
            halves[entry.row as usize % 2] = entry.value;
        }
        halves
    }

    fn at(&self, row: usize) -> F {
        self.entries
            .binary_search_by_key(&row, |entry| entry.row as usize)
            .map_or(F::zero(), |index| self.entries[index].value)
    }

    /// Bind one cycle variable low-to-high: `lo + r·(hi − lo)` per row pair,
    /// a missing half being zero.
    pub fn bind(&mut self, r: F) {
        let bind_pair = |group: &[IncEntry<F>]| {
            let mut halves = [F::zero(); 2];
            for entry in group {
                halves[entry.row as usize % 2] = entry.value;
            }
            IncEntry {
                row: group[0].row / 2,
                value: halves[0] + r * (halves[1] - halves[0]),
            }
        };
        #[cfg(feature = "parallel")]
        let bound: Vec<IncEntry<F>> = self
            .entries
            .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
            .map(bind_pair)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let bound: Vec<IncEntry<F>> = self
            .entries
            .chunk_by(|a, b| a.row / 2 == b.row / 2)
            .map(bind_pair)
            .collect();
        self.entries = bound;
    }

    /// The column's value once every cycle variable is bound; zero without
    /// increments.
    pub fn bound_value(&self) -> F {
        debug_assert!(self.entries.iter().all(|entry| entry.row == 0));
        self.entries.first().map_or(F::zero(), |entry| entry.value)
    }
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
        gamma: F,
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
                        * val_slope_term(odd.val + odd.val - even_checkpoint, inc_eval, gamma),
                ];
            }
            (None, None) => unreachable!("a merged pair has at least one side"),
        };
        [
            eq_eval * ra_evals[0] * val_slope_term(val_evals[0], inc_eval, gamma),
            eq_eval * ra_evals[1] * val_slope_term(val_evals[1], inc_eval, gamma),
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
    inc: &Polynomial<F>,
    eq: &Polynomial<F>,
    gamma: F,
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
                    inc.evals()[even[i].row as usize],
                    eq.evals()[even[i].row as usize],
                    gamma,
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
                    inc.evals()[even[i].row as usize],
                    eq.evals()[even[i].row as usize],
                    gamma,
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
                    inc.evals()[odd[j].row as usize],
                    eq.evals()[odd[j].row as usize],
                    gamma,
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
            inc.evals()[entry.row as usize],
            eq.evals()[entry.row as usize],
            gamma,
        ));
        even_checkpoint = entry.next_val;
    }
    for entry in &odd[j..] {
        add(AddressMajorEntry::address_round_evals(
            None,
            Some(entry),
            even_checkpoint,
            odd_checkpoint,
            inc.evals()[entry.row as usize],
            eq.evals()[entry.row as usize],
            gamma,
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
    pub fn from_events(log: &RamEventLog) -> Self {
        let mut entries: Vec<AddressMajorEntry<F>> = Vec::with_capacity(access_count(log));
        entries.extend(unbound_entries(log).map(CycleMajorEntry::into_address_major));
        #[cfg(feature = "parallel")]
        entries.par_sort_unstable_by_key(|entry| (entry.col, entry.row));
        #[cfg(not(feature = "parallel"))]
        entries.sort_unstable_by_key(|entry| (entry.col, entry.row));
        Self { entries }
    }

    /// After binding every address, fill the remaining cycle tables using
    /// checkpoints for untouched cycles. This allocates O(T), never O(K*T).
    pub fn into_cycle_tables(self, cycles: usize, initial: F) -> (Polynomial<F>, Polynomial<F>) {
        let mut ra = vec![F::zero(); cycles];
        let mut val = Vec::with_capacity(cycles);
        let mut checkpoint = initial;
        for entry in self.entries {
            debug_assert_eq!(entry.col, 0);
            let row = entry.row as usize;
            val.resize(row, checkpoint);
            val.push(entry.val);
            ra[row] = entry.ra;
            checkpoint = entry.next_val;
        }
        val.resize(cycles, checkpoint);
        (Polynomial::new(ra), Polynomial::new(val))
    }

    /// Bind one address variable low-to-high: merge every adjacent column
    /// pair against the `val_init` checkpoints, then bind `val_init` itself.
    pub fn bind(&mut self, r: F, val_init: &mut Polynomial<F>) {
        #[cfg(feature = "parallel")]
        let bound: Vec<AddressMajorEntry<F>> = self
            .entries
            .par_chunk_by(|a, b| a.col / 2 == b.col / 2)
            .flat_map_iter(|group| {
                let (even, odd) = split_col_pair(group);
                let even_col = 2 * (group[0].col / 2) as usize;
                let mut out = Vec::with_capacity(group.len());
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
            .collect();
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
        inc: &Polynomial<F>,
        eq: &Polynomial<F>,
        gamma: F,
    ) -> [F; 2] {
        let per_group = |group: &[AddressMajorEntry<F>]| -> [F; 2] {
            let (even, odd) = split_col_pair(group);
            let even_col = 2 * (group[0].col / 2) as usize;
            merge_address_round_evals(
                even,
                odd,
                val_init.evals()[even_col],
                val_init.evals()[even_col + 1],
                inc,
                eq,
                gamma,
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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::ram::ram_inc;
    use jolt_field::{Fr, Ring};
    use jolt_poly::EqPolynomial;
    use jolt_witness::JoltWitnessOracle;

    use super::super::testing::{ram_fixture, FixtureShape, RamOp};
    use super::*;
    use crate::ProofSession;

    /// Binding keeps exact zeros (a pair cancelled by r = 2, then a retained
    /// zero half chosen by r = 0) and ends at the dense column's multilinear
    /// value.
    #[test]
    fn sparse_inc_binds_to_the_dense_mle() {
        let witness = ram_fixture(
            FixtureShape { log_t: 3, ram_k: 8 },
            Vec::new(),
            vec![
                RamOp::UnmappedWrite { pre: 0, post: 2 },
                RamOp::UnmappedWrite { pre: 0, post: 1 },
                RamOp::None,
                RamOp::UnmappedWrite { pre: 0, post: 5 },
                RamOp::UnmappedWrite { pre: 0, post: 1 },
            ],
        );
        let mut session = ProofSession::default();
        let log = RamEventLog::shared::<Fr>(&mut session, &witness, 3, 8).unwrap();
        let mut inc = SparseInc::<Fr>::from_events(log);
        let challenges = [2, 0, 1].map(Fr::from_u64);
        for (round, r) in challenges.into_iter().enumerate() {
            inc.bind(r);
            if round < 2 {
                let first = inc.entries.first().map(|entry| (entry.row, entry.value));
                assert_eq!(first, Some((0, Fr::from_u64(0))), "round {round}");
            }
        }
        let dense = JoltWitnessOracle::<Fr>::oracle_table(&witness, ram_inc().polynomial_id());
        let point: Vec<Fr> = challenges.into_iter().rev().collect();
        let expected: Fr = EqPolynomial::<Fr>::evals(&point, None)
            .iter()
            .zip(&dense.unwrap())
            .map(|(eq, value)| *eq * *value)
            .sum();
        assert_eq!(inc.bound_value(), expected);
    }
}
