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

use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::ram_trace::{RamAccessColumns, NO_ACCESS};

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

    /// Bind a `(row 2j, row 2j+1)` pair of same-column entries; a `None`
    /// side is implicit (`ra = 0`, `val` recovered from the present side's
    /// checkpoint).
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

    /// The pair's contribution to the phase-1 quadratic factor:
    /// `[q(0), q_∞]` (evaluation at 0 and the quadratic coefficient) of
    /// `ra(t) · (val(t) + γ·(inc(t) + val(t)))` over the pair's row
    /// variable, `inc_evals = [inc(0), inc_slope]`.
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

/// `val + γ·(inc + val)` — the shared value factor of the summand.
#[inline]
fn val_slope_term<F: JoltField>(val: F, inc: F, gamma: F) -> F {
    val + gamma * (inc + val)
}

/// Merge two sorted-by-column adjacent rows into `out`, binding entries
/// pairwise (the legacy `seq_bind_rows`).
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

/// The pair contribution of two sorted-by-column adjacent rows (the legacy
/// `seq_prover_message_contribution`).
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

/// Split a row-pair group (entries sharing `row / 2`) into its even and odd
/// row slices.
fn split_row_pair<F>(
    group: &[CycleMajorEntry<F>],
) -> (&[CycleMajorEntry<F>], &[CycleMajorEntry<F>]) {
    let odd_start = group.partition_point(|entry| entry.row % 2 == 0);
    group.split_at(odd_start)
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
    /// `inc` the bound committed increment column.
    pub fn quadratic_coefficients(
        &self,
        eq_head: impl Fn(usize) -> F + Sync,
        inc: &Polynomial<F>,
        gamma: F,
    ) -> [F; 2] {
        self.quadratic_coefficients_with(
            eq_head,
            |pair| {
                let inc_0 = inc.evals()[2 * pair];
                [inc_0, inc.evals()[2 * pair + 1] - inc_0]
            },
            gamma,
        )
    }

    pub(crate) fn quadratic_coefficients_with(
        &self,
        eq_head: impl Fn(usize) -> F + Sync,
        inc_pair: impl Fn(usize) -> [F; 2] + Sync,
        gamma: F,
    ) -> [F; 2] {
        let per_group = |group: &[CycleMajorEntry<F>]| -> [F; 2] {
            let pair = (group[0].row / 2) as usize;
            let inc_0 = inc.evals()[2 * pair];
            let inc_evals = [inc_0, inc.evals()[2 * pair + 1] - inc_0];
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

/// Reconstructs a round-0 entry from the access columns.
#[inline]
fn round0_entry<F: JoltField>(
    columns: &RamAccessColumns,
    cycle: usize,
) -> Option<CycleMajorEntry<F>> {
    let address = columns.addresses[cycle];
    (address != NO_ACCESS).then(|| {
        let pre_value = columns.pre_values[cycle];
        CycleMajorEntry {
            row: cycle as u32,
            col: address,
            prev_val: pre_value,
            next_val: columns.post_values[cycle],
            val: F::from_u64(pre_value),
            ra: F::one(),
        }
    })
}

/// Round-0 quadratic coefficients read directly from access columns.
pub(crate) fn round0_quadratic_coefficients<F: JoltField>(
    columns: &RamAccessColumns,
    eq_head: impl Fn(usize) -> F + Sync,
    inc: &Polynomial<F>,
    gamma: F,
) -> [F; 2] {
    let pairs = columns.addresses.len() / 2;
    let per_pair = |pair: usize| -> [F; 2] {
        let even = round0_entry::<F>(columns, 2 * pair);
        let odd = round0_entry::<F>(columns, 2 * pair + 1);
        if even.is_none() && odd.is_none() {
            return [F::zero(); 2];
        }
        let inc_0 = inc.evals()[2 * pair];
        let inc_evals = [inc_0, inc.evals()[2 * pair + 1] - inc_0];
        let inner = match (&even, &odd) {
            (Some(even), Some(odd)) if even.col != odd.col => {
                let lone_even =
                    CycleMajorEntry::quadratic_evals(Some(even), None, inc_evals, gamma);
                let lone_odd = CycleMajorEntry::quadratic_evals(None, Some(odd), inc_evals, gamma);
                [lone_even[0] + lone_odd[0], lone_even[1] + lone_odd[1]]
            }
            _ => CycleMajorEntry::quadratic_evals(even.as_ref(), odd.as_ref(), inc_evals, gamma),
        };
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
    let pairs = columns.addresses.len() / 2;
    let per_pair = |pair: usize| -> [Option<CycleMajorEntry<F>>; 2] {
        let even = round0_entry::<F>(columns, 2 * pair);
        let odd = round0_entry::<F>(columns, 2 * pair + 1);
        match (&even, &odd) {
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
            _ => [
                Some(CycleMajorEntry::bind(even.as_ref(), odd.as_ref(), r)),
                None,
            ],
        }
    };

    #[cfg(feature = "parallel")]
    let entries: Vec<CycleMajorEntry<F>> = (0..pairs)
        .into_par_iter()
        .flat_map_iter(|pair| per_pair(pair).into_iter().flatten())
        .collect();
    #[cfg(not(feature = "parallel"))]
    let entries: Vec<CycleMajorEntry<F>> = (0..pairs)
        .flat_map(|pair| per_pair(pair).into_iter().flatten())
        .collect();
    CycleMajorMatrix { entries }
}

impl<F: JoltField> AddressMajorEntry<F> {
    /// Bind a `(col 2k, col 2k+1)` pair of same-row entries; a `None` side
    /// is implicit (`ra = 0`, `val` recovered from its column checkpoint).
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

    /// The pair's contribution `[s(0), s(2)]` to the phase-2 (address)
    /// round message `eq · ra(t) · (val(t) + γ·(inc + val(t)))`, where the
    /// cycle-bound `eq` and `inc` are scalars per row.
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

/// Split a column-pair group (entries sharing `col / 2`) into its even and
/// odd column slices.
fn split_col_pair<F>(
    group: &[AddressMajorEntry<F>],
) -> (&[AddressMajorEntry<F>], &[AddressMajorEntry<F>]) {
    let odd_start = group.partition_point(|entry| entry.col % 2 == 0);
    group.split_at(odd_start)
}

/// Merge two sorted-by-row adjacent columns, binding entries pairwise and
/// walking the value checkpoints forward (the legacy `seq_bind_cols`).
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

/// The column pair's round contribution `[s(0), s(2)]` (the legacy
/// `seq_prover_message_contribution`), checkpoints walked forward per row.
fn merge_address_round_evals<F: JoltField>(
    even: &[AddressMajorEntry<F>],
    odd: &[AddressMajorEntry<F>],
    mut even_checkpoint: F,
    mut odd_checkpoint: F,
    inc_eval: F,
    eq_eval: F,
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
        self.address_round_evals_scalars(val_init, inc.evals()[0], eq.evals()[0], gamma)
    }

    pub(crate) fn address_round_evals_scalars(
        &self,
        val_init: &Polynomial<F>,
        inc_eval: F,
        eq_eval: F,
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
                inc_eval,
                eq_eval,
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
