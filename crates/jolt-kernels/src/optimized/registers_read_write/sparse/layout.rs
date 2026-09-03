use core::{
    cmp::Ordering,
    mem::{size_of, MaybeUninit},
};

use jolt_field::{Accumulator, JoltField};

use super::{CoeffLut, LutIndex, OneHotCoeff};
use crate::optimized::registers_read_write::rows::RegisterCycleRow;
use crate::optimized::support::mul_0_optimized;

/// Row/column access shared by all entry layouts.
pub(super) trait Cell: Copy + Send + Sync + 'static {
    fn row(&self) -> usize;
    fn col(&self) -> u8;
}

/// Packed SoA metadata: 25 bytes instead of the aligned layout's 32.
/// `row` fits `u32` because [`CollectRegisterEntries::collect`] rejects wider
/// domains.
///
/// WARNING: copy packed fields by value; references may be unaligned.
///
/// [`CollectRegisterEntries::collect`]: crate::optimized::registers_read_write::rows::CollectRegisterEntries::collect
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub(super) struct IndexedMeta {
    pub(super) prev_val: u64,
    pub(super) next_val: u64,
    pub(super) row: u32,
    pub(super) ra: u16,
    pub(super) wa: u16,
    pub(super) col: u8,
}

/// Pin the packed metadata size.
const _: () = assert!(size_of::<IndexedMeta>() == 25);

impl Cell for IndexedMeta {
    #[inline]
    fn row(&self) -> usize {
        self.row as usize
    }

    #[inline]
    fn col(&self) -> u8 {
        self.col
    }
}

/// Index-parallel value and metadata slices.
pub(super) type SoaRow<'a, F> = (&'a [F], &'a [IndexedMeta]);

/// One block's uninitialized output span across both SoA columns.
pub(super) type SoaSpareBlock<'a, F> =
    (&'a mut [MaybeUninit<F>], &'a mut [MaybeUninit<IndexedMeta>]);

/// Reassemble indexed entry `i`.
#[inline]
pub(super) fn load_indexed<F: JoltField>(
    (vals, metas): SoaRow<'_, F>,
    i: usize,
) -> SparseEntry<F, LutIndex> {
    let meta = metas[i];
    SparseEntry {
        val: vals[i],
        prev_val: meta.prev_val,
        next_val: meta.next_val,
        row: meta.row as usize,
        ra: LutIndex(meta.ra),
        wa: LutIndex(meta.wa),
        col: meta.col,
    }
}

/// Split a working entry into its SoA columns (inverse of [`load_indexed`]).
#[inline]
pub(super) fn split_indexed<F: JoltField>(entry: SparseEntry<F, LutIndex>) -> (F, IndexedMeta) {
    debug_assert!(u32::try_from(entry.row).is_ok());
    (
        entry.val,
        IndexedMeta {
            prev_val: entry.prev_val,
            next_val: entry.next_val,
            row: entry.row as u32,
            ra: entry.ra.0,
            wa: entry.wa.0,
            col: entry.col,
        },
    )
}

/// Split a SoA row-pair group (entries sharing `row / 2`) into its even and
/// odd rows — [`split_pair_group`] over both columns at once.
pub(super) fn split_soa_pair_group<'a, F>(
    vals: &'a [F],
    metas: &'a [IndexedMeta],
) -> (SoaRow<'a, F>, SoaRow<'a, F>) {
    let odd_start = metas.partition_point(|meta| meta.row % 2 == 0);
    (
        (&vals[..odd_start], &metas[..odd_start]),
        (&vals[odd_start..], &metas[odd_start..]),
    )
}

/// Merge adjacent SoA rows in column order.
#[inline]
pub(super) fn merge_soa<F: JoltField>(
    evens: SoaRow<'_, F>,
    odds: SoaRow<'_, F>,
    mut visit: impl FnMut(Option<SparseEntry<F, LutIndex>>, Option<SparseEntry<F, LutIndex>>),
) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < evens.1.len() && j < odds.1.len() {
        match evens.1[i].col.cmp(&odds.1[j].col) {
            Ordering::Equal => {
                visit(Some(load_indexed(evens, i)), Some(load_indexed(odds, j)));
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                visit(Some(load_indexed(evens, i)), None);
                i += 1;
            }
            Ordering::Greater => {
                visit(None, Some(load_indexed(odds, j)));
                j += 1;
            }
        }
    }
    while i < evens.1.len() {
        visit(Some(load_indexed(evens, i)), None);
        i += 1;
    }
    while j < odds.1.len() {
        visit(None, Some(load_indexed(odds, j)));
        j += 1;
    }
}

/// Quadratic-round accumulation shared by each sparse-entry layout.
pub(super) trait MatrixEntry<F: JoltField>: Cell {
    /// Accumulate this vertical pair's `[t = 0, t = ∞]` contributions to the
    /// quadratic inner factor: `ra_t·val_t + wa_t·(val_t + inc_t)`.
    fn accumulate_pair_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        acc: &mut [F::Accumulator; 2],
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    );
}

/// Round-0 entry. `val = F::from_u64(prev_val)` stays implicit until the
/// first bind, cutting the peak layout from 64 to 24 bytes.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(in crate::optimized::registers_read_write) struct SeedEntry {
    pub(super) prev_val: u64,
    pub(super) next_val: u64,
    /// `u32` is safe because collection rejects wider domains.
    ///
    /// [`CollectRegisterEntries::collect`]: crate::optimized::registers_read_write::rows::CollectRegisterEntries::collect
    pub(super) row: u32,
    /// Seed `γ·rs1_ra + γ²·rs2_ra` index.
    pub(super) ra: u8,
    pub(super) wa: u8,
    pub(super) col: u8,
}

/// Pin the peak layout size.
const _: () = assert!(size_of::<SeedEntry>() == 24);

impl SeedEntry {
    #[inline]
    fn ra(&self) -> LutIndex {
        LutIndex(u16::from(self.ra))
    }

    #[inline]
    fn wa(&self) -> LutIndex {
        LutIndex(u16::from(self.wa))
    }
}

impl Cell for SeedEntry {
    #[inline]
    fn row(&self) -> usize {
        self.row as usize
    }

    #[inline]
    fn col(&self) -> u8 {
        self.col
    }
}

impl SeedEntry {
    /// Bind adjacent rows into the indexed layout (materializing `val`); a
    /// missing side has zero `ra`/`wa`.
    #[inline]
    pub(super) fn bind<F: JoltField>(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) -> SparseEntry<F, LutIndex> {
        let coeff_bind = <LutIndex as OneHotCoeff<F>>::bind;
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                let even_val = F::from_u64(even.prev_val);
                let odd_val = F::from_u64(odd.prev_val);
                SparseEntry {
                    val: even_val + mul_0_optimized(r, odd_val - even_val),
                    ra: coeff_bind(Some(even.ra()), Some(odd.ra()), r, ra_lut),
                    wa: coeff_bind(Some(even.wa()), Some(odd.wa()), r, wa_lut),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                    row: (even.row / 2) as usize,
                    col: even.col,
                }
            }
            (Some(even), None) => {
                let even_val = F::from_u64(even.prev_val);
                let odd_val = F::from_u64(even.next_val);
                SparseEntry {
                    val: even_val + mul_0_optimized(r, odd_val - even_val),
                    ra: coeff_bind(Some(even.ra()), None, r, ra_lut),
                    wa: coeff_bind(Some(even.wa()), None, r, wa_lut),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                    row: (even.row / 2) as usize,
                    col: even.col,
                }
            }
            (None, Some(odd)) => {
                // The missing even side equals odd's implicit boundary value.
                SparseEntry {
                    val: F::from_u64(odd.prev_val),
                    ra: coeff_bind(None, Some(odd.ra()), r, ra_lut),
                    wa: coeff_bind(None, Some(odd.wa()), r, wa_lut),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                    row: (odd.row / 2) as usize,
                    col: odd.col,
                }
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }
}

impl<F: JoltField> MatrixEntry<F> for SeedEntry {
    #[inline]
    fn accumulate_pair_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        acc: &mut [F::Accumulator; 2],
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) {
        let coeff_evals = <LutIndex as OneHotCoeff<F>>::eval_pair;
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert_eq!(even.col, odd.col);
                let ra = coeff_evals(Some(even.ra()), Some(odd.ra()), ra_lut);
                let wa = coeff_evals(Some(even.wa()), Some(odd.wa()), wa_lut);
                let even_val = F::from_u64(even.prev_val);
                acc[0].fmadd(ra[0], even_val);
                acc[0].fmadd(wa[0], even_val + inc_evals[0]);
                let val_m = F::from_u64(odd.prev_val) - even_val;
                acc[1].fmadd(ra[1], val_m);
                acc[1].fmadd(wa[1], val_m + inc_evals[1]);
            }
            (Some(even), None) => {
                let ra = coeff_evals(Some(even.ra()), None, ra_lut);
                let wa = coeff_evals(Some(even.wa()), None, wa_lut);
                let even_val = F::from_u64(even.prev_val);
                let val_m = F::from_u64(even.next_val) - even_val;
                acc[0].fmadd(ra[0], even_val);
                acc[0].fmadd(wa[0], even_val + inc_evals[0]);
                acc[1].fmadd(ra[1], val_m);
                acc[1].fmadd(wa[1], val_m + inc_evals[1]);
            }
            (None, Some(odd)) => {
                // Missing even coefficients make the t=0 term zero.
                let ra = coeff_evals(None, Some(odd.ra()), ra_lut);
                let wa = coeff_evals(None, Some(odd.wa()), wa_lut);
                acc[1].fmadd(ra[1], F::zero());
                acc[1].fmadd(wa[1], inc_evals[1]);
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }
}

/// Nonzero register-matrix cell. `C` is a LUT index until saturation, then a
/// field value; boundary values stay raw until merged.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct SparseEntry<F, C> {
    pub(super) val: F,
    pub(super) prev_val: u64,
    pub(super) next_val: u64,
    pub(super) row: usize,
    pub(super) ra: C,
    pub(super) wa: C,
    pub(super) col: u8,
}

impl<F: JoltField, C: OneHotCoeff<F>> Cell for SparseEntry<F, C> {
    #[inline]
    fn row(&self) -> usize {
        self.row
    }

    #[inline]
    fn col(&self) -> u8 {
        self.col
    }
}

impl<F: JoltField, C: OneHotCoeff<F>> SparseEntry<F, C> {
    /// Bind adjacent rows; a missing side has zero `ra`/`wa`.
    pub(super) fn bind(
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
                    val: even.val + mul_0_optimized(r, odd.val - even.val),
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
                    val: even.val + mul_0_optimized(r, odd_val - even.val),
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
                    val: even_val + mul_0_optimized(r, odd.val - even_val),
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
}

impl<F: JoltField, C: OneHotCoeff<F>> MatrixEntry<F> for SparseEntry<F, C> {
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
                // Missing even coefficients make the t=0 term zero.
                let ra = C::eval_pair(None, Some(odd.ra), ra_lut);
                let wa = C::eval_pair(None, Some(odd.wa), wa_lut);
                let val_m = odd.val - F::from_u64(odd.prev_val);
                acc[1].fmadd(ra[1], val_m);
                acc[1].fmadd(wa[1], val_m + inc_evals[1]);
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
    }
}

/// ra seed indices: `[0, γ, γ², γ + γ²]` — rs1 hot, rs2 hot, both.
const RA_ZERO: u8 = 0;
const RA_RS1: u8 = 1;
const RA_RS2: u8 = 2;
const RA_BOTH: u8 = 3;
/// wa seed indices: `[0, 1]`.
const WA_ZERO: u8 = 0;
const WA_HOT: u8 = 1;

impl RegisterCycleRow {
    /// Count distinct touched registers without constructing entries.
    #[cfg(feature = "parallel")]
    pub(in crate::optimized::registers_read_write) fn entry_count(&self) -> usize {
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

    /// Build up to three column-sorted seed entries.
    pub(in crate::optimized::registers_read_write) fn entries(
        &self,
        row: u32,
    ) -> ([SeedEntry; 3], usize) {
        let empty = SeedEntry {
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
            out[len] = SeedEntry {
                col: rs1,
                prev_val: rs1_val,
                next_val: rs1_val,
                ra: RA_RS1,
                ..empty
            };
            len += 1;
        }
        if let Some((rs2, rs2_val)) = self.rs2 {
            if let Some(entry) = out[..len].iter_mut().find(|entry| entry.col == rs2) {
                entry.ra = RA_BOTH;
            } else {
                out[len] = SeedEntry {
                    col: rs2,
                    prev_val: rs2_val,
                    next_val: rs2_val,
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
                out[len] = SeedEntry {
                    col: rd,
                    prev_val: rd_pre,
                    next_val: rd_post,
                    wa: WA_HOT,
                    ..empty
                };
                len += 1;
            }
        }

        out[..len].sort_unstable_by_key(|entry| entry.col);
        (out, len)
    }
}

/// Output length of merging two column-sorted rows.
pub(super) fn merge_count<E: Cell>(evens: &[E], odds: &[E]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut produced = 0;
    while i < evens.len() && j < odds.len() {
        match evens[i].col().cmp(&odds[j].col()) {
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

/// Merge-bind adjacent column-sorted rows.
#[inline]
pub(super) fn merge_bind<E: Cell, B>(
    evens: &[E],
    odds: &[E],
    bind: &(impl Fn(Option<&E>, Option<&E>) -> B + ?Sized),
    mut sink: impl FnMut(B),
) {
    let mut i = 0;
    let mut j = 0;
    while i < evens.len() && j < odds.len() {
        let bound = match evens[i].col().cmp(&odds[j].col()) {
            Ordering::Equal => {
                let entry = bind(Some(&evens[i]), Some(&odds[j]));
                i += 1;
                j += 1;
                entry
            }
            Ordering::Less => {
                let entry = bind(Some(&evens[i]), None);
                i += 1;
                entry
            }
            Ordering::Greater => {
                let entry = bind(None, Some(&odds[j]));
                j += 1;
                entry
            }
        };
        sink(bound);
    }
    for even in &evens[i..] {
        sink(bind(Some(even), None));
    }
    for odd in &odds[j..] {
        sink(bind(None, Some(odd)));
    }
}

/// Split a sorted row-pair group into even and odd rows.
pub(super) fn split_pair_group<E: Cell>(group: &[E]) -> (&[E], &[E]) {
    let odd_start = group.partition_point(|entry| entry.row() % 2 == 0);
    group.split_at(odd_start)
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    /// The counting pass and the write pass must agree on every operand
    /// pattern, or the parallel collector's second pass overruns its window.
    #[test]
    fn cycle_entry_count_matches_cycle_entries() {
        let candidates: [Option<u8>; 5] = [None, Some(0), Some(5), Some(127), Some(255)];
        for rs1 in candidates {
            for rs2 in candidates {
                for rd in candidates {
                    let cycle = RegisterCycleRow {
                        rs1: rs1.map(|register| (register, 11)),
                        rs2: rs2.map(|register| (register, 22)),
                        rd: rd.map(|register| (register, 33, 44)),
                    };
                    let (_, len) = cycle.entries(0);
                    assert_eq!(
                        cycle.entry_count(),
                        len,
                        "count/write divergence for {cycle:?}"
                    );
                }
            }
        }
    }
}
