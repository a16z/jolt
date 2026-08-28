//! The sparse cycle-major entry machinery: u16 coefficient LUTs, the
//! round-dependent entry layouts (24-byte seed, SoA-split indexed, direct
//! field), the in-place/fused bind walks, and the quadratic round walks.
//! Representation and algorithm contracts live on the items themselves.

use core::{cmp::Ordering, mem::MaybeUninit};

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::rows::RegisterCycleRow;

/// Growing lookup table of the possible values of one one-hot coefficient
/// column (the legacy `OneHotCoeffLookupTable`). Seeded with the column's
/// initial coefficient values; on each cycle bind the table squares — entry
/// `(a ≪ bits) | b` holds `b + r·(a − b)` — so a `u16` per matrix entry keeps
/// addressing its bound coefficient until one more squaring would overflow
/// the index domain.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F")
)]
pub(super) struct CoeffLut<F> {
    /// Power-of-two length; index 0 is always zero (zero seeds stay zero
    /// under `b + r·(a − b)`), which is what lets an absent merge partner
    /// keep index arithmetic pure.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    pub(super) values: Vec<F>,
}

impl<F: JoltField> CoeffLut<F> {
    /// One-past the largest table an entry's `u16` index can address.
    const MAX_VALUES: usize = 1 << 16;

    pub(super) fn new(values: Vec<F>) -> Self {
        debug_assert!(values.len().is_power_of_two());
        debug_assert!(values[0] == F::zero());
        Self { values }
    }

    fn bits(&self) -> u32 {
        self.values.len().trailing_zeros()
    }

    /// Whether one more bind would overflow the `u16` index domain.
    pub(super) fn saturated(&self) -> bool {
        self.values.len() * self.values.len() > Self::MAX_VALUES
    }

    /// Square the table with `r`: the same pair combination
    /// `even + r·(odd − even)` the direct field representation applies, over
    /// every (odd, even) value pair.
    pub(super) fn bind(&mut self, r: F) {
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

/// `left * right`, skipping the multiply when either side is zero.
#[inline(always)]
fn mul_0_optimized<F: JoltField>(left: F, right: F) -> F {
    if left.is_zero() || right.is_zero() {
        F::zero()
    } else {
        left * right
    }
}

/// `left * right`, skipping the multiply when either side is zero or one.
#[inline(always)]
fn mul_01_optimized<F: JoltField>(left: F, right: F) -> F {
    if left.is_zero() || right.is_zero() {
        F::zero()
    } else if left.is_one() {
        right
    } else if right.is_one() {
        left
    } else {
        left * right
    }
}

/// One-hot coefficient storage: either a direct field value or a `u16` index
/// into a [`CoeffLut`]. Both compute identical field values — the lookup
/// table pre-binds every possible value, the index arithmetic just selects —
/// so switching representations is memory-shape only, never wire-visible.
pub(super) trait OneHotCoeff<F: JoltField>: Copy + Send + Sync + 'static {
    /// Bind a vertically adjacent pair with `r`; a missing side is an
    /// implicit zero coefficient.
    fn bind(even: Option<Self>, odd: Option<Self>, r: F, lut: &CoeffLut<F>) -> Self;

    /// The pair's `[value at t = 0, slope]` sumcheck evaluations.
    fn eval_pair(even: Option<Self>, odd: Option<Self>, lut: &CoeffLut<F>) -> [F; 2];

    /// The coefficient's field value.
    fn value(self, lut: &CoeffLut<F>) -> F;
}

impl<F: JoltField> OneHotCoeff<F> for F {
    #[inline]
    fn bind(even: Option<Self>, odd: Option<Self>, r: F, _lut: &CoeffLut<F>) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => even + mul_0_optimized(r, odd - even),
            (Some(even), None) => mul_01_optimized(F::one() - r, even),
            (None, Some(odd)) => mul_01_optimized(r, odd),
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
pub(super) struct LutIndex(pub(super) u16);

impl<F: JoltField> OneHotCoeff<F> for LutIndex {
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

/// Row/column addressing shared by the entry layouts, so the pair-merge
/// machinery is written once.
pub(super) trait Cell: Copy + Send + Sync + 'static {
    fn row(&self) -> usize;
    fn col(&self) -> u8;
}

/// The non-value fields of one indexed-layout entry: the SoA twin of
/// `SparseEntry<F, LutIndex>` minus `val`, which lives in an index-parallel
/// (and 8-aligned) `Vec<F>` column. Packed: 25 bytes instead of 32, so the
/// split stores 57 bytes per entry against the 64-byte AoS layout across the
/// second-largest generation of the stage. `row` narrows to `u32` — indexed
/// rows are bound cycle indices, and [`CollectRegisterEntries::collect`]
/// rejects cycle counts past the u32 domain up front.
///
/// WARNING: packed fields are read/written by value only (unaligned loads,
/// free on the targets we care about); taking a reference to one is a
/// compile error, so keep accesses copy-shaped.
///
/// [`CollectRegisterEntries::collect`]: super::rows::CollectRegisterEntries::collect
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub(super) struct IndexedMeta {
    /// Register value just before this entry's row slice.
    pub(super) prev_val: u64,
    /// Register value just after this entry's row slice.
    pub(super) next_val: u64,
    /// Cycle-domain row index.
    pub(super) row: u32,
    /// Bound `γ·rs1_ra + γ²·rs2_ra` LUT index.
    pub(super) ra: u16,
    /// Bound `rd_wa` LUT index.
    pub(super) wa: u16,
    /// Register index.
    pub(super) col: u8,
}

/// The SoA split only pays if the meta column actually shrinks — its size is
/// load-bearing.
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

/// Borrowed view of one row slice of the SoA indexed columns
/// (index-parallel `vals`/`metas`).
type SoaRow<'a, F> = (&'a [F], &'a [IndexedMeta]);

/// One block's uninitialized output span across both SoA columns.
type SoaSpareBlock<'a, F> = (&'a mut [MaybeUninit<F>], &'a mut [MaybeUninit<IndexedMeta>]);

/// Reassemble the working entry at `i` — pure representation change, the
/// field values are the stored ones.
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
fn split_indexed<F: JoltField>(entry: SparseEntry<F, LutIndex>) -> (F, IndexedMeta) {
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
fn split_soa_pair_group<'a, F>(
    vals: &'a [F],
    metas: &'a [IndexedMeta],
) -> (SoaRow<'a, F>, SoaRow<'a, F>) {
    let odd_start = metas.partition_point(|meta| meta.row % 2 == 0);
    (
        (&vals[..odd_start], &metas[..odd_start]),
        (&vals[odd_start..], &metas[odd_start..]),
    )
}

/// Column-merge two vertically adjacent SoA rows, visiting every merged cell
/// as the reassembled working pair in column order — the SoA twin of
/// [`merge_bind`] (a missing side is an untouched slice, exactly as there).
#[inline]
fn merge_soa<F: JoltField>(
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

/// A sparse matrix cell in one of the round-dependent layouts, bound
/// pairwise against the coefficient LUTs.
pub(super) trait MatrixEntry<F: JoltField>: Cell {
    /// The layout after one cycle bind (the seed layout materializes `val`).
    type Bound: Cell;

    /// Bind two vertically adjacent cells (rows `2j`/`2j+1`, same column)
    /// with `r`. A missing side is an untouched slice: its `Val` is the
    /// neighbor's raw boundary value and its `ra`/`wa` are zero.
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) -> Self::Bound;

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

/// The round-0 entry layout: [`RegisterCycleRow::entries`]' output before any
/// bind. The `Val` coefficient is implicit — at construction it is always
/// `F::from_u64(prev_val)` (a read's value, or a write's pre-value; folding
/// rd into a read entry keeps the read's value) — so the entry is 24 bytes
/// instead of 64 exactly where the entry count peaks (≤ 3·T, the stage's
/// resident maximum). The first bind materializes `val` with the same
/// `F::from_u64` + bind ops the eager layout applied, so every bound value
/// is bit-identical.
#[derive(Clone, Copy, Debug)]
pub(super) struct SeedEntry {
    /// Register value just before this entry's cycle.
    pub(super) prev_val: u64,
    /// Register value just after this entry's cycle.
    pub(super) next_val: u64,
    /// The cycle. `u32`: [`CollectRegisterEntries::collect`] rejects wider
    /// domains.
    ///
    /// [`CollectRegisterEntries::collect`]: super::rows::CollectRegisterEntries::collect
    pub(super) row: u32,
    /// Seed `γ·rs1_ra + γ²·rs2_ra` index ([`RA_ZERO`]–[`RA_BOTH`]); `u8`
    /// (widened to [`LutIndex`] at the first bind) to keep the seed at 24
    /// bytes.
    pub(super) ra: u8,
    /// Seed `rd_wa` index ([`WA_ZERO`]/[`WA_HOT`]).
    pub(super) wa: u8,
    /// Register index.
    pub(super) col: u8,
}

/// The seed layout is the peak-memory shape — its size is load-bearing.
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

impl<F: JoltField> MatrixEntry<F> for SeedEntry {
    type Bound = SparseEntry<F, LutIndex>;

    #[inline]
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F,
        ra_lut: &CoeffLut<F>,
        wa_lut: &CoeffLut<F>,
    ) -> Self::Bound {
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
                // The eager layout's inferred even side is `from_u64(odd.prev_val)`
                // — exactly odd's implicit val — so `odd.val − even_val` is the
                // canonical zero and the bind collapses to the constant.
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
                // The even side has zero ra/wa, so the t = 0 term vanishes;
                // odd's implicit val equals `from_u64(odd.prev_val)`, so the
                // eager layout's `val_m` here is the canonical zero.
                let ra = coeff_evals(None, Some(odd.ra()), ra_lut);
                let wa = coeff_evals(None, Some(odd.wa()), wa_lut);
                acc[1].fmadd(ra[1], F::zero());
                acc[1].fmadd(wa[1], inc_evals[1]);
            }
            (None, None) => unreachable!("merge visits only represented cells"),
        }
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
    pub(super) val: F,
    /// Register value just before this entry's row slice.
    pub(super) prev_val: u64,
    /// Register value just after this entry's row slice.
    pub(super) next_val: u64,
    /// Cycle-domain row index (before binding: the cycle).
    pub(super) row: usize,
    /// Bound `γ·rs1_ra + γ²·rs2_ra` coefficient.
    pub(super) ra: C,
    /// Bound `rd_wa` coefficient.
    pub(super) wa: C,
    /// Register index.
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

impl<F: JoltField, C: OneHotCoeff<F>> MatrixEntry<F> for SparseEntry<F, C> {
    type Bound = Self;

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
    /// indices. Returns the filled prefix length (0–3). The `Val` coefficient
    /// is implicit in `prev_val` — see [`SeedEntry`].
    pub(super) fn entries(&self, row: u32) -> ([SeedEntry; 3], usize) {
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

        // Sort by column; len ≤ 3.
        out[..len].sort_unstable_by_key(|entry| entry.col);
        (out, len)
    }
}

/// Merged length of two adjacent sorted-by-column rows (a bind dry run —
/// the count is value-independent).
fn merge_count<E: Cell>(evens: &[E], odds: &[E]) -> usize {
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

/// Merge-bind two adjacent sorted-by-column rows, emitting the bound entries
/// to `sink` in column order.
#[inline]
fn merge_bind<E: Cell, B>(
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

/// Split a row-pair group (entries sharing `row / 2`) into its even and odd
/// rows. Entries are sorted by `(row, col)`, so the evens form the prefix.
fn split_pair_group<E: Cell>(group: &[E]) -> (&[E], &[E]) {
    let odd_start = group.partition_point(|entry| entry.row() % 2 == 0);
    group.split_at(odd_start)
}

/// The sparse entries in their round-dependent representation: the 24-byte
/// seed layout with implicit `Val` at round 0 (the peak-memory window), then
/// SoA-split `u16` LUT indices while the tables can still square (through
/// the fourth cycle round), direct field values after.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(super) enum SparseEntries<F: JoltField> {
    Seed {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        entries: Vec<SeedEntry>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
    },
    /// The first cycle bind received, nothing materialized: the 24-byte seed
    /// entries carry both early rounds. Round 1 and the second bind rebuild
    /// the would-be `T/2` intermediates per 4-row group in per-thread
    /// scratch (the canonical [`SeedEntry`] bind against the pre-square
    /// `seed_*` tables), and the squared tables serve their value lookups —
    /// the full-size intermediate generation the two sequential binds would
    /// materialize between them never exists.
    SeedBound {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        entries: Vec<SeedEntry>,
        /// Pre-square tables: the level-1 combine bits.
        seed_ra_lut: CoeffLut<F>,
        seed_wa_lut: CoeffLut<F>,
        /// Squared-once tables: the intermediates' value domain.
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        r1: F,
    },
    Indexed {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        vals: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        metas: Vec<IndexedMeta>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
    },
    Direct(
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        Vec<SparseEntry<F, F>>,
    ),
}

/// The `rd_inc` column in its round-dependent representation (the RAM
/// value-check kernel's pattern). Round 0 serves `inc(j)` straight from the
/// raw signed deltas collected alongside the seed entries — the oracle's
/// exact `F::from_i128` op — and round 1 serves the composed pair
/// `lo + r1·(hi − lo)` from the same deltas, so the dense bound column only
/// materializes at `T/4` on the second bind. Neither the `T`- nor the
/// `T/2`-sized field table ever exists; the raw column (16 B/cycle, exactly
/// a `T/2` field table's footprint) covers both early rounds.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(super) enum IncColumn<F: JoltField> {
    Raw(Vec<i128>),
    RawBound {
        raw: Vec<i128>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        r1: F,
    },
    Bound(Polynomial<F>),
}

/// The composed first-bind value `inc(y)` on the half domain: the exact
/// `lo + r·(hi − lo)` op the eager `T/2` materialization applied.
#[inline]
fn raw_bound_inc<F: JoltField>(raw: &[i128], r1: F, y: usize) -> F {
    let lo = F::from_i128(raw[2 * y]);
    let hi = F::from_i128(raw[2 * y + 1]);
    lo + mul_0_optimized(r1, hi - lo)
}

impl<F: JoltField> IncColumn<F> {
    /// Bind one cycle variable. The first bind is free (round 1 serves the
    /// composed pairs straight from the raw deltas); the second materializes
    /// directly at `T/4`, each level the same pair op `lo + r·(hi − lo)` as
    /// `Polynomial::bind_with_order` (the RAM value-check kernel's exact
    /// pattern, composed one round deeper).
    pub(super) fn bind(&mut self, r: F) {
        if let IncColumn::Bound(inc) = self {
            inc.bind_with_order(r, BindingOrder::LowToHigh);
            return;
        }
        *self = match std::mem::replace(self, IncColumn::Raw(Vec::new())) {
            IncColumn::Raw(raw) => IncColumn::RawBound { raw, r1: r },
            IncColumn::RawBound { raw, r1 } => {
                let quarter = raw.len() / 4;
                let pair = |z: usize| {
                    let lo = raw_bound_inc(&raw, r1, 2 * z);
                    let hi = raw_bound_inc(&raw, r1, 2 * z + 1);
                    lo + mul_0_optimized(r, hi - lo)
                };
                #[cfg(feature = "parallel")]
                let bound: Vec<F> = (0..quarter).into_par_iter().map(pair).collect();
                #[cfg(not(feature = "parallel"))]
                let bound: Vec<F> = (0..quarter).map(pair).collect();
                IncColumn::Bound(Polynomial::new(bound))
            }
            IncColumn::Bound(_) => unreachable!("bound inc binds in place above"),
        };
    }

    /// The fully cycle-bound `rd_inc` scalar at the cycle→address transition.
    pub(super) fn final_scalar(&self) -> F {
        match self {
            IncColumn::Bound(inc) => inc.evals()[0],
            // log_t = 1: the single cycle bind left the pair composed.
            IncColumn::RawBound { raw, r1 } => raw_bound_inc(raw, *r1, 0),
            // `prepare` requires log_t ≥ 1.
            IncColumn::Raw(_) => unreachable!("a cycle bind precedes the collapse"),
        }
    }
}

/// Pair-aligned block decomposition: fixed-size blocks advanced to the next
/// merge-group edge (`row >> pair_bits`; 1 for row pairs, 2 for the fused
/// 4-row groups), so no merge group straddles a block. Per-group metadata
/// (one length pair and two slice splits per group — tens of millions of
/// groups in the early rounds, built on the walking thread) collapses to a
/// handful of per-block counts.
fn pair_aligned_bounds<E: Cell>(entries: &[E], pair_bits: u32) -> Vec<usize> {
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

/// Bind one cycle variable in place: same-layout rounds write each merge
/// group's bound entries back into the input allocation at a write cursor,
/// then compact the per-block runs left and truncate. The entry vector is
/// the stage's largest allocation — the fresh output vector the out-of-place
/// path allocates every round is what set the prover's stage-4 peak.
///
/// Safety of the forward cursor: a merged pair produces at most one entry
/// per input entry ([`merge_count`] ≤ `evens.len() + odds.len()`), so each
/// block's write cursor never overtakes its read position, and entry order
/// is preserved by construction (groups are visited in order, merges keep
/// column order). Within one group the output may reach into the group's own
/// unread cells (an unpaired odd merges ahead of a later even), so a group
/// binds straight into the vacated prefix only when its whole output span
/// provably ends before the group — the common case once merges have opened
/// a gap — and stages through a small scratch otherwise.
pub(super) fn bind_sparse_entries_in_place<E>(
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

    // Left-compact the per-block runs; dest ≤ src throughout because every
    // run fits its own block.
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

/// In-place cycle bind of the SoA indexed columns — the SoA twin of
/// [`bind_sparse_entries_in_place`]: same pair-aligned blocks, same forward
/// write cursor (moved in lockstep across both columns, so the safety
/// argument there carries over verbatim), same left-compaction.
pub(super) fn bind_indexed_in_place_soa<F: JoltField>(
    vals: &mut Vec<F>,
    metas: &mut Vec<IndexedMeta>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r: F,
) {
    debug_assert_eq!(vals.len(), metas.len());
    let bounds = pair_aligned_bounds(metas, 1);
    let blocks = bounds.len() - 1;

    let bind_block = |scratch: &mut (Vec<F>, Vec<IndexedMeta>),
                      vals_block: &mut [F],
                      metas_block: &mut [IndexedMeta]|
     -> usize {
        let len = metas_block.len();
        let mut write = 0usize;
        let mut group_start = 0usize;
        while group_start < len {
            let pair_row = metas_block[group_start].row() / 2;
            let mut group_end = group_start + 1;
            while group_end < len && metas_block[group_end].row() / 2 == pair_row {
                group_end += 1;
            }
            let group_len = group_end - group_start;
            let bind_pair = |even: Option<SparseEntry<F, LutIndex>>,
                             odd: Option<SparseEntry<F, LutIndex>>| {
                split_indexed(<SparseEntry<F, LutIndex> as MatrixEntry<F>>::bind(
                    even.as_ref(),
                    odd.as_ref(),
                    r,
                    ra_lut,
                    wa_lut,
                ))
            };
            if write + group_len <= group_start {
                let (vals_out, vals_tail) = vals_block.split_at_mut(group_start);
                let (metas_out, metas_tail) = metas_block.split_at_mut(group_start);
                let (evens, odds) =
                    split_soa_pair_group(&vals_tail[..group_len], &metas_tail[..group_len]);
                merge_soa(evens, odds, |even, odd| {
                    let (val, meta) = bind_pair(even, odd);
                    vals_out[write] = val;
                    metas_out[write] = meta;
                    write += 1;
                });
            } else {
                scratch.0.clear();
                scratch.1.clear();
                let (evens, odds) = split_soa_pair_group(
                    &vals_block[group_start..group_end],
                    &metas_block[group_start..group_end],
                );
                merge_soa(evens, odds, |even, odd| {
                    let (val, meta) = bind_pair(even, odd);
                    scratch.0.push(val);
                    scratch.1.push(meta);
                });
                debug_assert!(write + scratch.0.len() <= group_end);
                vals_block[write..write + scratch.0.len()].copy_from_slice(&scratch.0);
                metas_block[write..write + scratch.1.len()].copy_from_slice(&scratch.1);
                write += scratch.0.len();
            }
            group_start = group_end;
        }
        write
    };

    let mut block_slices = Vec::with_capacity(blocks);
    let mut vals_rest: &mut [F] = vals;
    let mut metas_rest: &mut [IndexedMeta] = metas;
    for block in 0..blocks {
        let take = bounds[block + 1] - bounds[block];
        let (vals_head, vals_tail) = vals_rest.split_at_mut(take);
        let (metas_head, metas_tail) = metas_rest.split_at_mut(take);
        block_slices.push((vals_head, metas_head));
        vals_rest = vals_tail;
        metas_rest = metas_tail;
    }
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = block_slices
        .into_par_iter()
        .map_init(
            || {
                (
                    Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
                    Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
                )
            },
            |scratch, (vals_block, metas_block)| bind_block(scratch, vals_block, metas_block),
        )
        .collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = {
        let mut scratch = (
            Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
            Vec::with_capacity(2 << REGISTER_ADDRESS_BITS),
        );
        block_slices
            .into_iter()
            .map(|(vals_block, metas_block)| bind_block(&mut scratch, vals_block, metas_block))
            .collect()
    };

    // Left-compact the per-block runs of both columns; dest ≤ src throughout
    // because every run fits its own block.
    let mut total = counts[0];
    for block in 1..blocks {
        let src = bounds[block];
        if src != total {
            vals.copy_within(src..src + counts[block], total);
            metas.copy_within(src..src + counts[block], total);
        }
        total += counts[block];
    }
    vals.truncate(total);
    metas.truncate(total);
}

/// The Indexed → Direct layout transition from the SoA columns: dereference
/// each side's LUT indices during the merge and bind as direct field
/// coefficients — the exact deref-then-bind values of the AoS path, sized by
/// the same dry-run count pass.
pub(super) fn bind_indexed_to_direct<F: JoltField>(
    vals: &[F],
    metas: &[IndexedMeta],
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r: F,
) -> Vec<SparseEntry<F, F>> {
    let pair_predicate = |a: &IndexedMeta, b: &IndexedMeta| a.row() / 2 == b.row() / 2;
    let bounds = pair_aligned_bounds(metas, 1);
    let blocks = bounds.len() - 1;

    let count_block = |block: usize| -> usize {
        metas[bounds[block]..bounds[block + 1]]
            .chunk_by(pair_predicate)
            .map(|group| {
                let (evens, odds) = split_pair_group(group);
                merge_count(evens, odds)
            })
            .sum()
    };
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = (0..blocks).into_par_iter().map(count_block).collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = (0..blocks).map(count_block).collect();

    let bound_length: usize = counts.iter().sum();
    let mut bound: Vec<SparseEntry<F, F>> = Vec::with_capacity(bound_length);
    let mut out_slices = Vec::with_capacity(blocks);
    let mut out_rest = bound.spare_capacity_mut();
    for &count in &counts {
        let (out_slice, next_out) = out_rest.split_at_mut(count);
        out_rest = next_out;
        out_slices.push(out_slice);
    }

    let unused = SparseEntries::<F>::unused_lut();
    let deref = |entry: SparseEntry<F, LutIndex>| SparseEntry::<F, F> {
        val: entry.val,
        prev_val: entry.prev_val,
        next_val: entry.next_val,
        row: entry.row,
        ra: entry.ra.value(ra_lut),
        wa: entry.wa.value(wa_lut),
        col: entry.col,
    };
    let fill_block = |(block, out): (usize, &mut [MaybeUninit<SparseEntry<F, F>>])| {
        let mut written = 0usize;
        let mut cursor = bounds[block];
        for group in metas[bounds[block]..bounds[block + 1]].chunk_by(pair_predicate) {
            let group_vals = &vals[cursor..cursor + group.len()];
            cursor += group.len();
            let (evens, odds) = split_soa_pair_group(group_vals, group);
            merge_soa(evens, odds, |even, odd| {
                out[written] = MaybeUninit::new(<SparseEntry<F, F> as MatrixEntry<F>>::bind(
                    even.map(&deref).as_ref(),
                    odd.map(&deref).as_ref(),
                    r,
                    &unused,
                    &unused,
                ));
                written += 1;
            });
        }
        debug_assert_eq!(written, out.len());
    };
    #[cfg(feature = "parallel")]
    out_slices.into_par_iter().enumerate().for_each(fill_block);
    #[cfg(not(feature = "parallel"))]
    out_slices.into_iter().enumerate().for_each(fill_block);

    // SAFETY: the count pass sized every block's output slice exactly, the
    // slices partition `bound`'s spare capacity up to `bound_length`, and
    // the merge writes each slot exactly once.
    unsafe {
        bound.set_len(bound_length);
    }
    bound
}

/// Column-merge one vertical pair group and accumulate every merged cell's
/// `[t = 0, t = ∞]` contributions to the quadratic inner factor — the shared
/// inner loop of the AoS walk and the fused round-1 recompute.
fn accumulate_pair_group<F, E>(
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

/// The cycle-round quadratic inner factor `[q(0), leading coefficient]` over
/// the sparse entries in either coefficient representation — the summand
/// values are representation-independent by construction.
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

/// [`sparse_quadratic`] over the SoA indexed columns: identical grouping and
/// accumulation order (pair groups reassembled through [`merge_soa`]), so
/// every summand matches the AoS walk bit-for-bit.
pub(super) fn sparse_quadratic_soa<F: JoltField>(
    vals: &[F],
    metas: &[IndexedMeta],
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    e_in: &[F],
    e_out: &[F],
    inc_evals_at: impl Fn(usize) -> [F; 2] + Sync,
) -> [F; 2] {
    let e_in_len = e_in.len();
    let in_bits = if e_in_len <= 1 {
        0
    } else {
        e_in_len.trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let group_contribution = |group: &[IndexedMeta]| -> [F; 2] {
        // SAFETY: `group` is a sub-slice of `metas`, so the offset is the
        // group's start index; `vals` is index-parallel to `metas`.
        let start = unsafe { group.as_ptr().offset_from(metas.as_ptr()) } as usize;
        let group_vals = &vals[start..start + group.len()];
        let x_out = (group[0].row() / 2) >> in_bits;
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        let mut offset = 0usize;
        while offset < group.len() {
            let z = group[offset].row() / 2;
            let mut end = offset + 1;
            while end < group.len() && group[end].row() / 2 == z {
                end += 1;
            }
            let e_in_eval = if e_in_len <= 1 {
                F::one()
            } else {
                e_in[z & mask]
            };
            let inc_evals = inc_evals_at(z);

            let mut inner = [F::Accumulator::default(), F::Accumulator::default()];
            let (evens, odds) = split_soa_pair_group(&group_vals[offset..end], &group[offset..end]);
            merge_soa(evens, odds, |even, odd| {
                <SparseEntry<F, LutIndex> as MatrixEntry<F>>::accumulate_pair_evals(
                    even.as_ref(),
                    odd.as_ref(),
                    inc_evals,
                    &mut inner,
                    ra_lut,
                    wa_lut,
                );
            });

            acc[0].fmadd(e_in_eval, inner[0].reduce());
            acc[1].fmadd(e_in_eval, inner[1].reduce());
            offset = end;
        }
        let e_out_eval = e_out[x_out];
        [e_out_eval * acc[0].reduce(), e_out_eval * acc[1].reduce()]
    };

    let group_predicate =
        |a: &IndexedMeta, b: &IndexedMeta| (a.row() / 2) >> in_bits == (b.row() / 2) >> in_bits;
    #[cfg(feature = "parallel")]
    {
        metas
            .par_chunk_by(group_predicate)
            .map(group_contribution)
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
    #[cfg(not(feature = "parallel"))]
    {
        metas
            .chunk_by(group_predicate)
            .map(group_contribution)
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
}

/// Per-thread scratch for the fused paths: the two intermediate rows of one
/// 4-row group (each ≤ `K` entries).
type FusedScratch<F> = (Vec<SparseEntry<F, LutIndex>>, Vec<SparseEntry<F, LutIndex>>);

fn fused_scratch<F: JoltField>() -> FusedScratch<F> {
    (
        Vec::with_capacity(1 << REGISTER_ADDRESS_BITS),
        Vec::with_capacity(1 << REGISTER_ADDRESS_BITS),
    )
}

/// Rebuild one 4-row group's two `T/2`-domain intermediate rows: each half
/// pair merge-binds with `r1` through the canonical [`SeedEntry`] bind
/// against the pre-square tables — entry for entry what the sequential
/// first bind materialized at full size.
#[inline]
fn fused_intermediates<F: JoltField>(
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
        <SeedEntry as MatrixEntry<F>>::bind(even, odd, r1, seed_ra_lut, seed_wa_lut)
    };
    let (evens, odds) = split_pair_group(first);
    merge_bind(evens, odds, &bind, |entry| scratch.0.push(entry));
    let (evens, odds) = split_pair_group(second);
    merge_bind(evens, odds, &bind, |entry| scratch.1.push(entry));
}

/// The round-1 quadratic inner factor served straight from the seed
/// entries: [`sparse_quadratic`]'s walk over the `T/2`-domain intermediates,
/// which are rebuilt per 4-row group in per-thread scratch instead of read
/// from a materialized generation. Same pair groups, same merge order, same
/// accumulation — every summand is the sequential path's value.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors sparse_quadratic plus the two table generations"
)]
pub(super) fn sparse_quadratic_fused<F: JoltField>(
    entries: &[SeedEntry],
    seed_ra_lut: &CoeffLut<F>,
    seed_wa_lut: &CoeffLut<F>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r1: F,
    e_in: &[F],
    e_out: &[F],
    inc_evals_at: impl Fn(usize) -> [F; 2] + Sync,
) -> [F; 2] {
    let e_in_len = e_in.len();
    let in_bits = if e_in_len <= 1 {
        0
    } else {
        e_in_len.trailing_zeros() as usize
    };
    let mask = (1usize << in_bits) - 1;

    let group_contribution = |scratch: &mut FusedScratch<F>, group: &[SeedEntry]| -> [F; 2] {
        let x_out = (group[0].row() / 4) >> in_bits;
        let mut acc = [F::Accumulator::default(), F::Accumulator::default()];
        for pair_group in group.chunk_by(|a, b| a.row() / 4 == b.row() / 4) {
            let z = pair_group[0].row() / 4;
            let e_in_eval = if e_in_len <= 1 {
                F::one()
            } else {
                e_in[z & mask]
            };
            let inc_evals = inc_evals_at(z);

            fused_intermediates(pair_group, seed_ra_lut, seed_wa_lut, r1, scratch);
            let mut inner = [F::Accumulator::default(), F::Accumulator::default()];
            accumulate_pair_group(
                &scratch.0, &scratch.1, inc_evals, &mut inner, ra_lut, wa_lut,
            );

            acc[0].fmadd(e_in_eval, inner[0].reduce());
            acc[1].fmadd(e_in_eval, inner[1].reduce());
        }
        let e_out_eval = e_out[x_out];
        [e_out_eval * acc[0].reduce(), e_out_eval * acc[1].reduce()]
    };

    let group_predicate =
        |a: &SeedEntry, b: &SeedEntry| (a.row() / 4) >> in_bits == (b.row() / 4) >> in_bits;
    #[cfg(feature = "parallel")]
    {
        entries
            .par_chunk_by(group_predicate)
            .map_init(fused_scratch, |scratch, group| {
                group_contribution(scratch, group)
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = fused_scratch();
        entries
            .chunk_by(group_predicate)
            .map(|group| group_contribution(&mut scratch, group))
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }
}

/// The fused Seed → Indexed transition: one pass merges every 4-row group —
/// the two half-pair merge-binds with `r1`, then the level-2 merge-bind with
/// `r2` — straight into the SoA columns at a quarter of the cycle domain.
/// Identical values by construction (both levels are the canonical binds
/// the sequential rounds applied); only the full-size intermediate
/// generation between them disappears.
pub(super) fn bind_seed_entries_fused<F: JoltField>(
    entries: &[SeedEntry],
    seed_ra_lut: &CoeffLut<F>,
    seed_wa_lut: &CoeffLut<F>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    r1: F,
    r2: F,
) -> (Vec<F>, Vec<IndexedMeta>) {
    // The count pass's u128 column bitmap is exact only while the register
    // domain fits it.
    const _: () = assert!(REGISTER_ADDRESS_BITS <= 7);
    let group_predicate = |a: &SeedEntry, b: &SeedEntry| a.row() / 4 == b.row() / 4;
    let bounds = pair_aligned_bounds(entries, 2);
    let blocks = bounds.len() - 1;

    // Count pass: a merged 4-row group emits one entry per distinct column
    // (each level's merge unions its sides' column sets).
    let count_block = |block: usize| -> usize {
        entries[bounds[block]..bounds[block + 1]]
            .chunk_by(group_predicate)
            .map(|group| {
                group
                    .iter()
                    .fold(0u128, |mask, entry| mask | (1u128 << entry.col))
                    .count_ones() as usize
            })
            .sum()
    };
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = (0..blocks).into_par_iter().map(count_block).collect();
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = (0..blocks).map(count_block).collect();

    let bound_length: usize = counts.iter().sum();
    let mut vals: Vec<F> = Vec::with_capacity(bound_length);
    let mut metas: Vec<IndexedMeta> = Vec::with_capacity(bound_length);
    let mut out_slices = Vec::with_capacity(blocks);
    let mut vals_rest = vals.spare_capacity_mut();
    let mut metas_rest = metas.spare_capacity_mut();
    for &count in &counts {
        let (vals_slice, next_vals) = vals_rest.split_at_mut(count);
        let (metas_slice, next_metas) = metas_rest.split_at_mut(count);
        vals_rest = next_vals;
        metas_rest = next_metas;
        out_slices.push((vals_slice, metas_slice));
    }

    let fill_block =
        |scratch: &mut FusedScratch<F>,
         (block, (vals_out, metas_out)): (usize, SoaSpareBlock<'_, F>)| {
            let mut written = 0usize;
            for group in entries[bounds[block]..bounds[block + 1]].chunk_by(group_predicate) {
                fused_intermediates(group, seed_ra_lut, seed_wa_lut, r1, scratch);
                merge_bind(
                    &scratch.0,
                    &scratch.1,
                    &|even, odd| {
                        split_indexed(<SparseEntry<F, LutIndex> as MatrixEntry<F>>::bind(
                            even, odd, r2, ra_lut, wa_lut,
                        ))
                    },
                    |(val, meta)| {
                        vals_out[written] = MaybeUninit::new(val);
                        metas_out[written] = MaybeUninit::new(meta);
                        written += 1;
                    },
                );
            }
            debug_assert_eq!(written, vals_out.len());
        };
    #[cfg(feature = "parallel")]
    out_slices
        .into_par_iter()
        .enumerate()
        .for_each_init(fused_scratch, |scratch, item| fill_block(scratch, item));
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = fused_scratch();
        out_slices
            .into_iter()
            .enumerate()
            .for_each(|item| fill_block(&mut scratch, item));
    }

    // SAFETY: the count pass sized every block's output span exactly (each
    // 4-row group emits exactly its distinct-column count, which the
    // two-level merge reproduces), the spans partition both spare capacities
    // up to `bound_length`, and the merge writes each slot of both columns
    // exactly once.
    unsafe {
        vals.set_len(bound_length);
        metas.set_len(bound_length);
    }
    (vals, metas)
}

impl<F: JoltField> SparseEntries<F> {
    /// A placeholder table for the direct representation, which ignores it.
    pub(super) fn unused_lut() -> CoeffLut<F> {
        CoeffLut { values: Vec::new() }
    }

    /// The cycle-round quadratic inner factor `[q(0), leading coefficient]`
    /// over the remaining sparse rows. Raw `inc` serves rounds 0 (plain
    /// deltas) and 1 (composed pairs); `from_i128` per element is the
    /// oracle's exact op and the field composition is the exact bind op, so
    /// the pair evals match the dense table's in either representation.
    pub(super) fn quadratic(&self, e_in: &[F], e_out: &[F], inc: &IncColumn<F>) -> [F; 2] {
        match (self, inc) {
            (
                Self::Seed {
                    entries,
                    ra_lut,
                    wa_lut,
                },
                IncColumn::Raw(raw),
            ) => sparse_quadratic(entries, ra_lut, wa_lut, e_in, e_out, |z| {
                let inc_0 = F::from_i128(raw[2 * z]);
                [inc_0, F::from_i128(raw[2 * z + 1]) - inc_0]
            }),
            // Round 1: both columns still raw — entries rebuild the composed
            // intermediates per 4-row group, inc composes its first-bind
            // pairs; every served value is exactly what the eager `T/2`
            // materializations held.
            (
                Self::SeedBound {
                    entries,
                    seed_ra_lut,
                    seed_wa_lut,
                    ra_lut,
                    wa_lut,
                    r1,
                },
                IncColumn::RawBound { raw, r1: inc_r1 },
            ) => sparse_quadratic_fused(
                entries,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                *r1,
                e_in,
                e_out,
                |z| {
                    let inc_0 = raw_bound_inc(raw, *inc_r1, 2 * z);
                    [inc_0, raw_bound_inc(raw, *inc_r1, 2 * z + 1) - inc_0]
                },
            ),
            (
                Self::Indexed {
                    vals,
                    metas,
                    ra_lut,
                    wa_lut,
                },
                IncColumn::Bound(inc),
            ) => {
                let inc = inc.evals();
                sparse_quadratic_soa(vals, metas, ra_lut, wa_lut, e_in, e_out, |z| {
                    let inc_0 = inc[2 * z];
                    [inc_0, inc[2 * z + 1] - inc_0]
                })
            }
            (Self::Direct(entries), IncColumn::Bound(inc)) => {
                let unused = Self::unused_lut();
                let inc = inc.evals();
                sparse_quadratic(entries, &unused, &unused, e_in, e_out, |z| {
                    let inc_0 = inc[2 * z];
                    [inc_0, inc[2 * z + 1] - inc_0]
                })
            }
            _ => unreachable!("entry and inc representations advance in lockstep"),
        }
    }

    /// Bind one cycle variable. Returns whether this bind freed an entry
    /// generation (the fused SeedBound→Indexed transition or Indexed→Direct)
    /// — the rounds whose fresh output vector strands the previous multi-GiB
    /// generation in the allocator's cache.
    pub(super) fn bind(&mut self, r: F) -> bool {
        // Same-layout rounds bind in place — the entry vector is reused, not
        // reallocated.
        match self {
            Self::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
            } if !ra_lut.saturated() && !wa_lut.saturated() => {
                bind_indexed_in_place_soa(vals, metas, ra_lut, wa_lut, r);
                ra_lut.bind(r);
                wa_lut.bind(r);
                return false;
            }
            Self::Direct(entries) => {
                let unused = Self::unused_lut();
                bind_sparse_entries_in_place(entries, |even, odd| {
                    <SparseEntry<F, F> as MatrixEntry<F>>::bind(even, odd, r, &unused, &unused)
                });
                return false;
            }
            _ => {}
        }
        let state = std::mem::replace(self, Self::Direct(Vec::new()));
        let (next, freed_generation) = match state {
            // The first bind is deferred: nothing materializes — the seed
            // entries serve round 1 through the per-group intermediate
            // rebuild, so only the squared tables (and the pre-square ones,
            // for the level-1 combine bits) are prepared here.
            Self::Seed {
                entries,
                ra_lut,
                wa_lut,
            } => {
                let mut bound_ra = CoeffLut::new(ra_lut.values.clone());
                let mut bound_wa = CoeffLut::new(wa_lut.values.clone());
                bound_ra.bind(r);
                bound_wa.bind(r);
                (
                    Self::SeedBound {
                        entries,
                        seed_ra_lut: ra_lut,
                        seed_wa_lut: wa_lut,
                        ra_lut: bound_ra,
                        wa_lut: bound_wa,
                        r1: r,
                    },
                    false,
                )
            }
            // The second bind fuses both levels: the indexed layout
            // materializes `val` directly at a quarter of the cycle domain;
            // the tables then square (again) so the combined indices address
            // the twice-bound values.
            Self::SeedBound {
                entries,
                seed_ra_lut,
                seed_wa_lut,
                mut ra_lut,
                mut wa_lut,
                r1,
            } => {
                let (vals, metas) = bind_seed_entries_fused(
                    &entries,
                    &seed_ra_lut,
                    &seed_wa_lut,
                    &ra_lut,
                    &wa_lut,
                    r1,
                    r,
                );
                drop(entries);
                ra_lut.bind(r);
                wa_lut.bind(r);
                (
                    Self::Indexed {
                        vals,
                        metas,
                        ra_lut,
                        wa_lut,
                    },
                    true,
                )
            }
            // One more table squaring would overflow the u16 index domain
            // (after the third cycle bind under the seed sizes), so move to
            // direct field coefficients: dereference during the merge — the
            // exact deref-then-bind values, without the dense dereferenced
            // copy that used to double this round's transient.
            Self::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
            } => (
                Self::Direct(bind_indexed_to_direct(&vals, &metas, &ra_lut, &wa_lut, r)),
                true,
            ),
            Self::Direct(_) => unreachable!("direct entries bind in place above"),
        };
        *self = next;
        freed_generation
    }

    /// Scatter the fully cycle-bound single row into K-sized dense
    /// `(ra, wa, val)` arrays — the cycle→address transition state.
    pub(super) fn into_dense(self, k: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
        let mut ra = vec![F::zero(); k];
        let mut wa = vec![F::zero(); k];
        let mut val = vec![F::zero(); k];
        match self {
            Self::Seed { .. } => {
                unreachable!("prepare requires log_t ≥ 1, so a cycle bind precedes the collapse")
            }
            // log_t = 1: the single cycle bind is still pending — merge rows
            // {0, 1} with it on the way into the dense arrays.
            Self::SeedBound {
                entries,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
            } => {
                let (evens, odds) = split_pair_group(&entries);
                merge_bind(
                    evens,
                    odds,
                    &|even, odd| {
                        <SeedEntry as MatrixEntry<F>>::bind(
                            even,
                            odd,
                            r1,
                            &seed_ra_lut,
                            &seed_wa_lut,
                        )
                    },
                    |entry| {
                        debug_assert_eq!(entry.row, 0);
                        ra[entry.col as usize] = entry.ra.value(&ra_lut);
                        wa[entry.col as usize] = entry.wa.value(&wa_lut);
                        val[entry.col as usize] = entry.val;
                    },
                );
            }
            Self::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
            } => {
                for (value, meta) in vals.into_iter().zip(metas) {
                    debug_assert_eq!(meta.row(), 0);
                    ra[meta.col as usize] = LutIndex(meta.ra).value(&ra_lut);
                    wa[meta.col as usize] = LutIndex(meta.wa).value(&wa_lut);
                    val[meta.col as usize] = value;
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
