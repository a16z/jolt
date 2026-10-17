use core::{cmp::Ordering, mem::size_of};

use jolt_field::{Accumulator, JoltField};

use super::{CoeffLut, LutIndex, OneHotCoeff};
use crate::optimized::registers_read_write::rows::RegisterCycleRow;
use crate::optimized::support::mul_0_optimized;

pub(crate) trait Cell: Copy + Send + Sync + 'static {
    fn row(&self) -> usize;
    fn col(&self) -> u8;
}

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
    /// [`RegisterColumns::collect`]: crate::optimized::registers_read_write::rows::RegisterColumns::collect
    pub(super) row: u32,
    pub(super) ra: u8,
    pub(super) wa: u8,
    pub(super) col: u8,
}

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
    /// This cell as a field-coefficient entry: `val` materialized and both
    /// coefficients read from the level-0 LUTs.
    #[inline]
    pub(super) fn to_field<F: JoltField>(
        self,
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) -> SparseEntry<F, F> {
        SparseEntry {
            val: F::from_u64(self.prev_val),
            prev_val: self.prev_val,
            next_val: self.next_val,
            row: self.row as usize,
            ra: self.ra().value(ra_lut),
            wa: self.wa().value(wa_lut),
            col: self.col,
        }
    }

    /// Bind adjacent rows into LUT-index entries (materializing `val`); a
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

impl<F: JoltField> SparseEntry<F, LutIndex> {
    /// This entry with both coefficients read from its level's LUTs.
    #[inline]
    pub(super) fn to_field(self, ra_lut: &CoeffLut<F>, wa_lut: &CoeffLut<F>) -> SparseEntry<F, F> {
        SparseEntry {
            val: self.val,
            prev_val: self.prev_val,
            next_val: self.next_val,
            row: self.row,
            ra: self.ra.value(ra_lut),
            wa: self.wa.value(wa_lut),
            col: self.col,
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

/// Merge-bind adjacent column-sorted rows.
#[inline]
pub(crate) fn merge_bind<E: Cell, B>(
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
pub(crate) fn split_pair_group<E: Cell>(group: &[E]) -> (&[E], &[E]) {
    let odd_start = group.partition_point(|entry| entry.row() % 2 == 0);
    group.split_at(odd_start)
}
