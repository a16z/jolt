//! The sparse cycle-major entry machinery: u16 coefficient LUTs, entry
//! construction/merging, and the bind/quadratic walks. Representation and
//! algorithm contracts live on the items themselves.

use core::cmp::Ordering;
use core::mem::MaybeUninit;

use jolt_field::{AdditiveAccumulator, Field, OptimizedMul, RingAccumulator};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::rows::RegisterCycleRow;

/// Growing lookup table of the possible values of one one-hot coefficient
/// column (the legacy `OneHotCoeffLookupTable`). Seeded with the column's
/// initial coefficient values; on each cycle bind the table squares — entry
/// `(a ≪ bits) | b` holds `b + r·(a − b)` — so a `u16` per matrix entry keeps
/// addressing its bound coefficient until one more squaring would overflow
/// the index domain.
pub(crate) struct CoeffLut<F> {
    /// Power-of-two length; index 0 is always zero (zero seeds stay zero
    /// under `b + r·(a − b)`), which is what lets an absent merge partner
    /// keep index arithmetic pure.
    pub(crate) values: Vec<F>,
}

impl<F: Field> CoeffLut<F> {
    /// One-past the largest table an entry's `u16` index can address.
    const MAX_VALUES: usize = 1 << 16;

    pub(crate) fn new(values: Vec<F>) -> Self {
        debug_assert!(values.len().is_power_of_two());
        debug_assert!(values[0] == F::zero());
        Self { values }
    }

    fn bits(&self) -> u32 {
        self.values.len().trailing_zeros()
    }

    /// Whether one more bind would overflow the `u16` index domain.
    fn saturated(&self) -> bool {
        self.values.len() * self.values.len() > Self::MAX_VALUES
    }

    /// Square the table with `r`: the same pair combination
    /// `even + r·(odd − even)` the direct field representation applies, over
    /// every (odd, even) value pair.
    pub(crate) fn bind(&mut self, r: F) {
        debug_assert!(!self.saturated());
        let n = self.values.len();
        let old = &self.values;
        let square = |index: usize| {
            let a = old[index / n];
            let b = old[index % n];
            b + r * (a - b)
        };
        #[cfg(feature = "parallel")]
        let next: Vec<F> = (0..n * n).into_par_iter().map(square).collect();
        #[cfg(not(feature = "parallel"))]
        let next: Vec<F> = (0..n * n).map(square).collect();
        self.values = next;
    }
}

/// One-hot coefficient storage: either a direct field value or a `u16` index
/// into a [`CoeffLut`]. Both compute identical field values — the lookup
/// table pre-binds every possible value, the index arithmetic just selects —
/// so switching representations is memory-shape only, never wire-visible.
pub(super) trait OneHotCoeff<F: Field>: Copy + Send + Sync + 'static {
    /// Bind a vertically adjacent pair with `r`; a missing side is an
    /// implicit zero coefficient.
    fn bind(even: Option<Self>, odd: Option<Self>, r: F, lut: &CoeffLut<F>) -> Self;

    /// The pair's `[value at t = 0, slope]` sumcheck evaluations.
    fn eval_pair(even: Option<Self>, odd: Option<Self>, lut: &CoeffLut<F>) -> [F; 2];

    /// The coefficient's field value.
    fn value(self, lut: &CoeffLut<F>) -> F;
}

impl<F: Field> OneHotCoeff<F> for F {
    #[inline]
    fn bind(even: Option<Self>, odd: Option<Self>, r: F, _lut: &CoeffLut<F>) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => even + r.mul_0_optimized(odd - even),
            (Some(even), None) => (F::one() - r).mul_01_optimized(even),
            (None, Some(odd)) => r.mul_01_optimized(odd),
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    #[inline]
    fn eval_pair(even: Option<Self>, odd: Option<Self>, _lut: &CoeffLut<F>) -> [F; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => [even, odd - even],
            (Some(even), None) => [even, -even],
            (None, Some(odd)) => [F::zero(), odd],
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    #[inline]
    fn value(self, _lut: &CoeffLut<F>) -> F {
        self
    }
}

/// A `u16` index into a [`CoeffLut`] (newtype: a bare `u16` would collide
/// with the blanket field-value impl under coherence).
#[derive(Clone, Copy, Debug)]
pub(super) struct LutIndex(u16);

impl<F: Field> OneHotCoeff<F> for LutIndex {
    #[inline]
    fn bind(even: Option<Self>, odd: Option<Self>, _r: F, lut: &CoeffLut<F>) -> Self {
        // The table itself binds with `r` separately; index 0 is the zero
        // value, so an absent side combines as index 0.
        let bits = lut.bits();
        debug_assert!(bits <= 8, "coefficient LUT bound past u16 saturation");
        match (even, odd) {
            (Some(even), Some(odd)) => Self((odd.0 << bits) | even.0),
            (Some(even), None) => even,
            (None, Some(odd)) => Self(odd.0 << bits),
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    #[inline]
    fn eval_pair(even: Option<Self>, odd: Option<Self>, lut: &CoeffLut<F>) -> [F; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                let even = lut.values[even.0 as usize];
                [even, lut.values[odd.0 as usize] - even]
            }
            (Some(even), None) => {
                let even = lut.values[even.0 as usize];
                [even, -even]
            }
            (None, Some(odd)) => [F::zero(), lut.values[odd.0 as usize]],
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    #[inline]
    fn value(self, lut: &CoeffLut<F>) -> F {
        lut.values[self.0 as usize]
    }
}

/// One non-zero cell of the conceptual `K × T` register matrices: the bound
/// `Val` coefficient plus the γ-combined read and write coefficients of one
/// touched register slice, with the coefficient representation `C` chosen by
/// the round (indices while the LUTs can grow, field values after).
///
/// `prev_val`/`next_val` stay raw `u64`s: a register is constant between
/// touches, and a constant slice's bound coefficient is the constant itself,
/// so the values neighboring this entry's slice never need field form until
/// they participate in a merge.
#[derive(Clone, Copy, Debug)]
pub(super) struct SparseEntry<F, C> {
    /// Bound `Val(col, row-slice)` coefficient (value *before* the access).
    val: F,
    /// Register value just before this entry's row slice.
    prev_val: u64,
    /// Register value just after this entry's row slice.
    next_val: u64,
    /// Cycle-domain row index (before binding: the cycle).
    row: usize,
    /// Bound `γ·rs1_ra + γ²·rs2_ra` coefficient.
    ra: C,
    /// Bound `rd_wa` coefficient.
    wa: C,
    /// Register index.
    col: u8,
}

impl<F: Field, C: OneHotCoeff<F>> SparseEntry<F, C> {
    /// Bind two vertically adjacent cells (rows `2j`/`2j+1`, same column)
    /// with `r`. A missing side is an untouched slice: its `Val` is the
    /// neighbor's raw boundary value and its `ra`/`wa` are zero.
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                Self {
                    val: even.val + r.mul_0_optimized(odd.val - even.val),
                    ra: C::bind(Some(even.ra), Some(odd.ra), r, ra_lut),
                    wa: C::bind(Some(even.wa), Some(odd.wa), r, wa_lut),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                    row: even.row / 2,
                    col: even.col,
                }
            }
            (Some(even), None) => {
                let odd_val = F::from_u64(even.next_val);
                Self {
                    val: even.val + r.mul_0_optimized(odd_val - even.val),
                    ra: C::bind(Some(even.ra), None, r, ra_lut),
                    wa: C::bind(Some(even.wa), None, r, wa_lut),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                    row: even.row / 2,
                    col: even.col,
                }
            }
            (None, Some(odd)) => {
                let even_val = F::from_u64(odd.prev_val);
                Self {
                    val: even_val + r.mul_0_optimized(odd.val - even_val),
                    ra: C::bind(None, Some(odd.ra), r, ra_lut),
                    wa: C::bind(None, Some(odd.wa), r, wa_lut),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                    row: odd.row / 2,
                    col: odd.col,
                }
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    /// Accumulate this vertical pair's `[t = 0, t = ∞]` contributions to the
    /// quadratic inner factor: `ra_t·val_t + wa_t·(val_t + inc_t)`.
    fn accumulate_pair_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        acc: &mut [F::Accumulator; 2],
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                let ra = C::eval_pair(Some(even.ra), Some(odd.ra), ra_lut);
                let wa = C::eval_pair(Some(even.wa), Some(odd.wa), wa_lut);
                acc[0].fmadd(ra[0], even.val);
                acc[0].fmadd(wa[0], even.val + inc_evals[0]);
                let val_m = odd.val - even.val;
                acc[1].fmadd(ra[1], val_m);
                acc[1].fmadd(wa[1], val_m + inc_evals[1]);
            }
            (Some(even), None) => {
                let ra = C::eval_pair(Some(even.ra), None, ra_lut);
                let wa = C::eval_pair(Some(even.wa), None, wa_lut);
                let val_m = F::from_u64(even.next_val) - even.val;
                acc[0].fmadd(ra[0], even.val);
                acc[0].fmadd(wa[0], even.val + inc_evals[0]);
                acc[1].fmadd(ra[1], val_m);
                acc[1].fmadd(wa[1], val_m + inc_evals[1]);
            }
            (None, Some(odd)) => {
                // The even side has zero ra/wa, so the t = 0 term vanishes.
                let ra = C::eval_pair(None, Some(odd.ra), ra_lut);
                let wa = C::eval_pair(None, Some(odd.wa), wa_lut);
                let val_m = odd.val - F::from_u64(odd.prev_val);
                acc[1].fmadd(ra[1], val_m);
                acc[1].fmadd(wa[1], val_m + inc_evals[1]);
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }

    /// Merged length of two adjacent sorted-by-column rows (a bind dry run —
    /// the count is value-independent).
    fn merge_count(evens: &[SparseEntry<F, C>], odds: &[SparseEntry<F, C>]) -> usize {
        let mut i = 0;
        let mut j = 0;
        let mut produced = 0;
        while i < evens.len() && j < odds.len() {
            match evens[i].col.cmp(&odds[j].col) {
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
            produced += 1;
        }
        produced + (evens.len() - i) + (odds.len() - j)
    }

    /// Merge-bind two adjacent sorted-by-column rows into `out` (sized by
    /// [`Self::merge_count`]), keeping column order.
    fn merge_fill(
        evens: &[SparseEntry<F, C>],
        odds: &[SparseEntry<F, C>],
        r: F,
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
        out: &mut [MaybeUninit<SparseEntry<F, C>>],
    ) {
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        while i < evens.len() && j < odds.len() {
            let bound = match evens[i].col.cmp(&odds[j].col) {
                Ordering::Equal => {
                    let entry =
                        SparseEntry::bind(Some(&evens[i]), Some(&odds[j]), r, ra_lut, wa_lut);
                    i += 1;
                    j += 1;
                    entry
                }
                Ordering::Less => {
                    let entry = SparseEntry::bind(Some(&evens[i]), None, r, ra_lut, wa_lut);
                    i += 1;
                    entry
                }
                Ordering::Greater => {
                    let entry = SparseEntry::bind(None, Some(&odds[j]), r, ra_lut, wa_lut);
                    j += 1;
                    entry
                }
            };
            out[k] = MaybeUninit::new(bound);
            k += 1;
        }
        for even in &evens[i..] {
            out[k] = MaybeUninit::new(SparseEntry::bind(Some(even), None, r, ra_lut, wa_lut));
            k += 1;
        }
        for odd in &odds[j..] {
            out[k] = MaybeUninit::new(SparseEntry::bind(None, Some(odd), r, ra_lut, wa_lut));
            k += 1;
        }
        debug_assert_eq!(k, out.len());
    }

    /// Split a row-pair group (entries sharing `row / 2`) into its even and odd
    /// rows. Entries are sorted by `(row, col)`, so the evens form the prefix.
    #[expect(
        clippy::type_complexity,
        reason = "the (evens, odds) slice pair, spelled in full"
    )]
    fn split_pair_group(
        group: &[SparseEntry<F, C>],
    ) -> (&[SparseEntry<F, C>], &[SparseEntry<F, C>]) {
        let odd_start = group.partition_point(|entry| entry.row % 2 == 0);
        group.split_at(odd_start)
    }
}

/// ra seed indices: `[0, γ, γ², γ + γ²]` — rs1 hot, rs2 hot, both.
const RA_ZERO: LutIndex = LutIndex(0);
const RA_RS1: LutIndex = LutIndex(1);
const RA_RS2: LutIndex = LutIndex(2);
const RA_BOTH: LutIndex = LutIndex(3);
/// wa seed indices: `[0, 1]`.
const WA_ZERO: LutIndex = LutIndex(0);
const WA_HOT: LutIndex = LutIndex(1);

impl RegisterCycleRow {
    /// The entry count [`Self::entries`] will produce for one cycle — the
    /// counting pass's cheap twin (kept adjacent so the merge rules stay in
    /// sync: rs2 folds into rs1's entry, rd into either read's). Option-typed
    /// comparisons mirror the write twin's scan of emitted columns exactly, so
    /// no sentinel value can collide with a raw index (`rd == rs2` with a folded
    /// rs2 implies `rs2 == rs1`, which the `rd == rs1` test already catches).
    #[cfg(feature = "parallel")]
    pub(super) fn entry_count(&self) -> usize {
        let rs1 = self.rs1.map(|(register, _)| register);
        let rs2 = self.rs2.map(|(register, _)| register);
        let rd = self.rd.map(|(register, ..)| register);
        let mut count = usize::from(rs1.is_some());
        if rs2.is_some() && rs2 != rs1 {
            count += 1;
        }
        if rd.is_some() && rd != rs1 && rd != rs2 {
            count += 1;
        }
        count
    }

    /// Build the (sorted-by-column) sparse entries of one cycle as seed-table
    /// indices. Returns the filled prefix length (0–3).
    pub(super) fn entries<F: Field>(&self, row: usize) -> ([SparseEntry<F, LutIndex>; 3], usize) {
        let empty = SparseEntry {
            val: F::zero(),
            ra: RA_ZERO,
            wa: WA_ZERO,
            prev_val: 0,
            next_val: 0,
            row,
            col: 0,
        };
        let mut out = [empty; 3];
        let mut len = 0usize;

        if let Some((rs1, rs1_val)) = self.rs1 {
            out[len] = SparseEntry {
                col: rs1,
                prev_val: rs1_val,
                next_val: rs1_val,
                val: F::from_u64(rs1_val),
                ra: RA_RS1,
                ..empty
            };
            len += 1;
        }
        if let Some((rs2, rs2_val)) = self.rs2 {
            if let Some(entry) = out[..len].iter_mut().find(|entry| entry.col == rs2) {
                entry.ra = RA_BOTH;
            } else {
                out[len] = SparseEntry {
                    col: rs2,
                    prev_val: rs2_val,
                    next_val: rs2_val,
                    val: F::from_u64(rs2_val),
                    ra: RA_RS2,
                    ..empty
                };
                len += 1;
            }
        }
        if let Some((rd, rd_pre, rd_post)) = self.rd {
            if let Some(entry) = out[..len].iter_mut().find(|entry| entry.col == rd) {
                entry.wa = WA_HOT;
                entry.next_val = rd_post;
            } else {
                out[len] = SparseEntry {
                    col: rd,
                    prev_val: rd_pre,
                    next_val: rd_post,
                    val: F::from_u64(rd_pre),
                    wa: WA_HOT,
                    ..empty
                };
                len += 1;
            }
        }

        // Sort by column; len ≤ 3.
        out[..len].sort_unstable_by_key(|entry| entry.col);
        (out, len)
    }
}

/// The sparse entries in their round-dependent coefficient representation:
/// `u16` LUT indices while the tables can still square (the first four cycle
/// rounds — the peak-memory window), direct field values after.
pub(super) enum SparseEntries<F: Field> {
    Indexed {
        entries: Vec<SparseEntry<F, LutIndex>>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
    },
    Direct(Vec<SparseEntry<F, F>>),
}

impl<F: Field> SparseEntries<F> {
    /// A placeholder table for the direct representation, which ignores it.
    fn unused_lut() -> CoeffLut<F> {
        CoeffLut { values: Vec::new() }
    }

    /// Dereference every index through the bound tables — the exact field
    /// values the direct representation would have accumulated.
    fn deref(
        entries: Vec<SparseEntry<F, LutIndex>>,
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) -> Vec<SparseEntry<F, F>> {
        let deref_entry = |entry: &SparseEntry<F, LutIndex>| SparseEntry {
            val: entry.val,
            prev_val: entry.prev_val,
            next_val: entry.next_val,
            row: entry.row,
            ra: entry.ra.value(ra_lut),
            wa: entry.wa.value(wa_lut),
            col: entry.col,
        };
        #[cfg(feature = "parallel")]
        {
            entries.par_iter().map(deref_entry).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            entries.iter().map(deref_entry).collect()
        }
    }

    /// The cycle-round quadratic inner factor `[q(0), leading coefficient]`
    /// over the remaining sparse rows.
    pub(super) fn quadratic(&self, e_in: &[F], e_out: &[F], inc: &[F]) -> [F; 2] {
        match self {
            Self::Indexed {
                entries,
                ra_lut,
                wa_lut,
            } => sparse_quadratic(entries, ra_lut, wa_lut, e_in, e_out, inc),
            Self::Direct(entries) => {
                let unused = Self::unused_lut();
                sparse_quadratic(entries, &unused, &unused, e_in, e_out, inc)
            }
        }
    }

    /// Bind one cycle variable in place.
    pub(super) fn bind(&mut self, r: F) {
        // Dereference to direct field coefficients when one more table
        // squaring would overflow the u16 index domain (after the third
        // cycle bind under the seed sizes) — by then the entry count has
        // started merging down, so the wider entries no longer set the peak.
        let saturated = matches!(
            self,
            Self::Indexed { ra_lut, wa_lut, .. }
                if ra_lut.saturated() || wa_lut.saturated()
        );
        if saturated {
            if let Self::Indexed {
                entries,
                ra_lut,
                wa_lut,
            } = std::mem::replace(self, Self::Direct(Vec::new()))
            {
                *self = Self::Direct(Self::deref(entries, &ra_lut, &wa_lut));
            }
        }
        match self {
            Self::Indexed {
                entries,
                ra_lut,
                wa_lut,
            } => {
                // Entries combine indices against the CURRENT table widths;
                // the tables then square so the combined indices address the
                // bound values.
                bind_sparse_entries(entries, r, ra_lut, wa_lut);
                ra_lut.bind(r);
                wa_lut.bind(r);
            }
            Self::Direct(entries) => {
                let unused = Self::unused_lut();
                bind_sparse_entries(entries, r, &unused, &unused);
            }
        }
    }

    /// Scatter the fully cycle-bound single row into K-sized dense
    /// `(ra, wa, val)` arrays — the cycle→address transition state.
    pub(super) fn into_dense(self, k: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
        let mut ra = vec![F::zero(); k];
        let mut wa = vec![F::zero(); k];
        let mut val = vec![F::zero(); k];
        match self {
            Self::Indexed {
                entries,
                ra_lut,
                wa_lut,
            } => {
                for entry in entries {
                    debug_assert_eq!(entry.row, 0);
                    ra[entry.col as usize] = entry.ra.value(&ra_lut);
                    wa[entry.col as usize] = entry.wa.value(&wa_lut);
                    val[entry.col as usize] = entry.val;
                }
            }
            Self::Direct(entries) => {
                for entry in entries {
                    debug_assert_eq!(entry.row, 0);
                    ra[entry.col as usize] = entry.ra;
                    wa[entry.col as usize] = entry.wa;
                    val[entry.col as usize] = entry.val;
                }
            }
        }
        (ra, wa, val)
    }
}

/// Bind one cycle variable of the sparse matrix in place: merge every
/// adjacent row pair, exact-sized by a dry-run count pass.
fn bind_sparse_entries<F, C>(
    entries: &mut Vec<SparseEntry<F, C>>,
    r: F,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
) where
    F: Field,
    C: OneHotCoeff<F>,
{
    let pair_predicate = |a: &SparseEntry<F, C>, b: &SparseEntry<F, C>| a.row / 2 == b.row / 2;

    // Pair-aligned block decomposition: fixed-size blocks advanced to the
    // next row-pair edge, so no merge group straddles a block. Per-group
    // metadata (one length pair and two slice splits per group — tens of
    // millions of groups in the early rounds, built on the walking thread)
    // collapses to a handful of per-block counts.
    const BLOCK_TARGET: usize = 1 << 14;
    let len = entries.len();
    let block_count = len.div_ceil(BLOCK_TARGET).max(1);
    let mut bounds: Vec<usize> = Vec::with_capacity(block_count + 1);
    bounds.push(0);
    for block in 1..block_count {
        let mut index = block * len / block_count;
        while index < len && index > 0 && entries[index].row / 2 == entries[index - 1].row / 2 {
            index += 1;
        }
        #[expect(clippy::unwrap_used, reason = "bounds starts non-empty")]
        if index > *bounds.last().unwrap() && index < len {
            bounds.push(index);
        }
    }
    bounds.push(len);
    let blocks = bounds.len() - 1;

    let count_block = |block: usize| -> usize {
        entries[bounds[block]..bounds[block + 1]]
            .chunk_by(pair_predicate)
            .map(|group| {
                let (evens, odds) = SparseEntry::split_pair_group(group);
                SparseEntry::merge_count(evens, odds)
            })
            .sum()
    };
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = (0..blocks).into_par_iter().map(count_block).collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = (0..blocks).map(count_block).collect();

    let bound_length: usize = counts.iter().sum();
    let mut bound: Vec<SparseEntry<F, C>> = Vec::with_capacity(bound_length);
    let mut out_slices = Vec::with_capacity(blocks);
    let mut out_rest = bound.spare_capacity_mut();
    for &count in &counts {
        let (out_slice, next_out) = out_rest.split_at_mut(count);
        out_rest = next_out;
        out_slices.push(out_slice);
    }

    let entries_ref: &[SparseEntry<F, C>] = entries;
    let fill_block = |(block, out): (usize, &mut [MaybeUninit<SparseEntry<F, C>>])| {
        let mut written = 0usize;
        for group in entries_ref[bounds[block]..bounds[block + 1]].chunk_by(pair_predicate) {
            let (evens, odds) = SparseEntry::split_pair_group(group);
            let take = SparseEntry::merge_count(evens, odds);
            SparseEntry::merge_fill(
                evens,
                odds,
                r,
                ra_lut,
                wa_lut,
                &mut out[written..written + take],
            );
            written += take;
        }
        debug_assert_eq!(written, out.len());
    };
    #[cfg(feature = "parallel")]
    out_slices.into_par_iter().enumerate().for_each(fill_block);
    #[cfg(not(feature = "parallel"))]
    out_slices.into_iter().enumerate().for_each(fill_block);

    // SAFETY: the count pass sized every block's output slice exactly (the
    // fill pass re-derives the same per-group counts), the slices partition
    // `bound`'s spare capacity up to `bound_length`, and `merge_fill`
    // writes each slot of its slice exactly once.
    unsafe {
        bound.set_len(bound_length);
    }
    *entries = bound;
}

/// The cycle-round quadratic inner factor `[q(0), leading coefficient]` over
/// the sparse entries in either coefficient representation — the summand
/// values are representation-independent by construction.
fn sparse_quadratic<F, C>(
    entries: &[SparseEntry<F, C>],
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    e_in: &[F],
    e_out: &[F],
    inc: &[F],
) -> [F; 2]
where
    F: Field,
    C: OneHotCoeff<F>,
{
    let e_in_len = e_in.len();
    let in_bits = if e_in_len <= 1 {
        0
    } else {
        e_in_len.trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let group_contribution = |group: &[SparseEntry<F, C>]| -> [F; 2] {
        let x_out = (group[0].row / 2) >> in_bits;
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        for pair_group in group.chunk_by(|a, b| a.row / 2 == b.row / 2) {
            let z = pair_group[0].row / 2;
            let e_in_eval = if e_in_len <= 1 {
                F::one()
            } else {
                e_in[z & mask]
            };
            let j_prime = 2 * z;
            let inc_0 = inc[j_prime];
            let inc_evals = [inc_0, inc[j_prime + 1] - inc_0];

            let mut inner = [F::Accumulator::default(), F::Accumulator::default()];
            let (evens, odds) = SparseEntry::split_pair_group(pair_group);
            let mut i = 0;
            let mut j = 0;
            while i < evens.len() && j < odds.len() {
                match evens[i].col.cmp(&odds[j].col) {
                    Ordering::Equal => {
                        SparseEntry::accumulate_pair_evals(
                            Some(&evens[i]),
                            Some(&odds[j]),
                            inc_evals,
                            &mut inner,
                            ra_lut,
                            wa_lut,
                        );
                        i += 1;
                        j += 1;
                    }
                    Ordering::Less => {
                        SparseEntry::accumulate_pair_evals(
                            Some(&evens[i]),
                            None,
                            inc_evals,
                            &mut inner,
                            ra_lut,
                            wa_lut,
                        );
                        i += 1;
                    }
                    Ordering::Greater => {
                        SparseEntry::accumulate_pair_evals(
                            None,
                            Some(&odds[j]),
                            inc_evals,
                            &mut inner,
                            ra_lut,
                            wa_lut,
                        );
                        j += 1;
                    }
                }
            }
            for even in &evens[i..] {
                SparseEntry::accumulate_pair_evals(
                    Some(even),
                    None,
                    inc_evals,
                    &mut inner,
                    ra_lut,
                    wa_lut,
                );
            }
            for odd in &odds[j..] {
                SparseEntry::accumulate_pair_evals(
                    None,
                    Some(odd),
                    inc_evals,
                    &mut inner,
                    ra_lut,
                    wa_lut,
                );
            }

            acc[0].fmadd(e_in_eval, inner[0].reduce());
            acc[1].fmadd(e_in_eval, inner[1].reduce());
        }
        let e_out_eval = e_out[x_out];
        [e_out_eval * acc[0].reduce(), e_out_eval * acc[1].reduce()]
    };

    let group_predicate = |a: &SparseEntry<F, C>, b: &SparseEntry<F, C>| {
        (a.row / 2) >> in_bits == (b.row / 2) >> in_bits
    };
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
