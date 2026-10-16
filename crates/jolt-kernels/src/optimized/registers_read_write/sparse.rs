use jolt_field::{Accumulator, JoltField};
use jolt_poly::Polynomial;
use jolt_sumcheck::SumcheckError;
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::rows::RegisterCycleRow;
use super::ROWS_LABEL;
use crate::optimized::support::{mul_0_optimized, KernelRows};

/// Bound one-hot coefficient values indexed by `u16`. Each bind squares the
/// table: `(a ≪ bits) | b` maps to `b + r·(a − b)`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct CoeffLut<F> {
    /// Power-of-two table with zero fixed at index 0.
    pub(super) values: Vec<F>,
}

impl<F: JoltField> CoeffLut<F> {
    pub(super) fn new(values: Vec<F>) -> Self {
        debug_assert!(values.len().is_power_of_two());
        debug_assert!(values[0] == F::zero());
        Self { values }
    }

    fn bits(&self) -> u32 {
        self.values.len().trailing_zeros()
    }

    pub(super) fn bind(&mut self, r: F) {
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

pub(super) trait OneHotCoeff<F: JoltField>: Copy + Send + Sync + 'static {
    /// Bind a vertically adjacent pair with `r`; a missing side is an
    /// implicit zero coefficient.
    fn bind(even: Option<Self>, odd: Option<Self>, r: F, lut: &CoeffLut<F>) -> Self;

    /// The pair's `[value at t = 0, slope]` sumcheck evaluations.
    fn eval_pair(even: Option<Self>, odd: Option<Self>, lut: &CoeffLut<F>) -> [F; 2];

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

/// Newtype avoids overlap with the blanket field-value implementation.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct LutIndex(pub(super) u16);

impl<F: JoltField> OneHotCoeff<F> for LutIndex {
    #[inline]
    fn bind(even: Option<Self>, odd: Option<Self>, _r: F, lut: &CoeffLut<F>) -> Self {
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

pub(crate) mod layout;
pub(crate) mod ops;
mod stream;

use layout::{merge_bind, split_pair_group, MatrixEntry, SparseEntry};
use ops::{bind_sparse_entries_in_place, sparse_quadratic};

/// What round 0's pass leaves for the next cycle bind.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) enum Level1<F: JoltField> {
    /// Each round-0 Gruen group's level-1 entries: one per column of either
    /// row of each of its pairs, whatever the coefficients. Round 1's pass
    /// fills exactly these.
    Counts(Vec<usize>),
    /// A one-round domain's only pair, which the final cycle bind merges.
    Final(FinalPair<F>),
}

/// Rows 0 and 1 of a one-round domain: their field-coefficient cells and
/// increments.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct FinalPair<F: JoltField> {
    evens: Vec<SparseEntry<F, F>>,
    odds: Vec<SparseEntry<F, F>>,
    inc: [F; 2],
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) enum CyclePhase<F: JoltField> {
    /// Round 0, whose pass regenerates the cells from the rows and keeps
    /// `level1`.
    Seed {
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        level1: Option<Level1<F>>,
    },
    /// Round 1, whose pass merges the cells bound by `r1` and fills the
    /// counted level-1 entries.
    Fill {
        seed_ra_lut: CoeffLut<F>,
        seed_wa_lut: CoeffLut<F>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        r1: F,
        counts: Vec<usize>,
    },
    Direct {
        entries: Vec<SparseEntry<F, F>>,
        inc: Polynomial<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct CycleState<F: JoltField>(pub(super) CyclePhase<F>);

/// The bound cycle row as dense `(ra, wa, val)` address arrays and its `inc`.
type DenseCycleRow<F> = (Vec<F>, Vec<F>, Vec<F>, F);

/// A row failed to extract, or round 1's rows or tasks disagree with round
/// 0's count.
fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS_LABEL }
}

/// A cycle-state step called out of round order.
fn out_of_order<F: JoltField>() -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS_LABEL }
}

impl<F: JoltField> CycleState<F> {
    pub(super) fn new(ra_lut: CoeffLut<F>, wa_lut: CoeffLut<F>) -> Self {
        Self(CyclePhase::Seed {
            ra_lut,
            wa_lut,
            level1: None,
        })
    }

    pub(super) fn unused_lut() -> CoeffLut<F> {
        CoeffLut { values: Vec::new() }
    }

    /// The round's `[q(0), leading coefficient]`. Round 0's pass also keeps
    /// its level-1 counts, or a one-round domain's pair; round 1's pass fills
    /// the counted entries and their increments.
    pub(super) fn quadratic(
        &mut self,
        rows: &KernelRows<RegisterCycleRow>,
        e_in: &[F],
        e_out: &[F],
    ) -> Result<[F; 2], SumcheckError<F>> {
        Ok(match &mut self.0 {
            CyclePhase::Seed {
                ra_lut,
                wa_lut,
                level1,
            } => {
                let (quadratic, next) = stream::count(rows, e_in, e_out, [&*ra_lut, &*wa_lut])
                    .map_err(rows_unavailable)?;
                *level1 = Some(next);
                quadratic
            }
            CyclePhase::Fill {
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                counts,
            } => {
                let (quadratic, entries, inc) = stream::fill(
                    rows,
                    e_in,
                    e_out,
                    *r1,
                    [[&*seed_ra_lut, &*seed_wa_lut], [&*ra_lut, &*wa_lut]],
                    counts,
                )
                .map_err(rows_unavailable)?;
                self.0 = CyclePhase::Direct {
                    entries,
                    inc: Polynomial::new(inc),
                };
                quadratic
            }
            CyclePhase::Direct { entries, inc } => {
                let unused = Self::unused_lut();
                let inc = inc.evals();
                sparse_quadratic(entries, &unused, &unused, e_in, e_out, |z| {
                    let inc_0 = inc[2 * z];
                    [inc_0, inc[2 * z + 1] - inc_0]
                })
            }
        })
    }

    pub(super) fn q_at_one(
        &self,
        rows: &KernelRows<RegisterCycleRow>,
        e_in: &[F],
        e_out: &[F],
    ) -> Result<F, SumcheckError<F>> {
        Ok(match &self.0 {
            CyclePhase::Seed { ra_lut, wa_lut, .. } => {
                stream::at_one(rows, e_in, e_out, &[], &[[ra_lut, wa_lut]])
                    .map_err(rows_unavailable)?
            }
            // Round 1's `quadratic` fills the entries before its `q(1)`.
            CyclePhase::Fill { .. } => return Err(out_of_order()),
            CyclePhase::Direct { entries, inc } => {
                let unused = Self::unused_lut();
                sparse_at_one(
                    entries.iter().copied(),
                    &unused,
                    &unused,
                    e_in,
                    e_out,
                    |row| inc.evals()[row],
                )
            }
        })
    }

    /// Binds a cycle challenge other than the last.
    pub(super) fn bind(&mut self, r: F) -> Result<(), SumcheckError<F>> {
        if let CyclePhase::Direct { entries, inc } = &mut self.0 {
            let unused = Self::unused_lut();
            bind_sparse_entries_in_place(entries, |even, odd| {
                SparseEntry::<F, F>::bind(even, odd, r, &unused, &unused)
            });
            let _ = inc.bind_low_to_high_in_place(r);
            return Ok(());
        }
        let CyclePhase::Seed {
            ra_lut,
            wa_lut,
            level1: Some(Level1::Counts(counts)),
        } = &mut self.0
        else {
            return Err(out_of_order());
        };
        let mut bound_ra = CoeffLut::new(ra_lut.values.clone());
        let mut bound_wa = CoeffLut::new(wa_lut.values.clone());
        bound_ra.bind(r);
        bound_wa.bind(r);
        self.0 = CyclePhase::Fill {
            seed_ra_lut: std::mem::replace(ra_lut, Self::unused_lut()),
            seed_wa_lut: std::mem::replace(wa_lut, Self::unused_lut()),
            ra_lut: bound_ra,
            wa_lut: bound_wa,
            r1: r,
            counts: std::mem::take(counts),
        };
        Ok(())
    }

    /// Binds the last cycle challenge `r`: the final pair — the filled
    /// entries' two remaining rows, or a one-round domain's pair from round
    /// 0's pass — merges into dense `(ra, wa, val)` address arrays and the
    /// bound increment, and the cycle state is dropped.
    pub(super) fn take_dense(
        &mut self,
        r: F,
        k: usize,
    ) -> Result<DenseCycleRow<F>, SumcheckError<F>> {
        let (evens, odds, [low, high]) = match &self.0 {
            CyclePhase::Seed {
                level1: Some(Level1::Final(pair)),
                ..
            } => (&pair.evens[..], &pair.odds[..], pair.inc),
            CyclePhase::Direct { entries, inc } => {
                let (evens, odds) = split_pair_group(entries);
                (evens, odds, [inc.evals()[0], inc.evals()[1]])
            }
            CyclePhase::Seed { .. } | CyclePhase::Fill { .. } => return Err(out_of_order()),
        };
        let mut ra = vec![F::zero(); k];
        let mut wa = vec![F::zero(); k];
        let mut val = vec![F::zero(); k];
        let unused = Self::unused_lut();
        merge_bind(
            evens,
            odds,
            &|even, odd| SparseEntry::<F, F>::bind(even, odd, r, &unused, &unused),
            |entry| {
                debug_assert_eq!(entry.row, 0);
                ra[entry.col as usize] = entry.ra;
                wa[entry.col as usize] = entry.wa;
                val[entry.col as usize] = entry.val;
            },
        );
        let inc = low + mul_0_optimized(r, high - low);
        self.0 = CyclePhase::Direct {
            entries: Vec::new(),
            inc: Polynomial::new(Vec::new()),
        };
        Ok((ra, wa, val, inc))
    }
}

fn sparse_at_one<F: JoltField, E: MatrixEntry<F>>(
    entries: impl IntoIterator<Item = E>,
    ra_lut: &CoeffLut<F>,
    wa_lut: &CoeffLut<F>,
    e_in: &[F],
    e_out: &[F],
    inc_at: impl Fn(usize) -> F,
) -> F {
    let in_bits = e_in.len().trailing_zeros() as usize;
    let mask = e_in.len() - 1;
    let mut sum = F::Accumulator::default();
    for entry in entries {
        if entry.row().is_multiple_of(2) {
            continue;
        }
        let pair = entry.row() / 2;
        let weight = e_out[pair >> in_bits] * e_in[pair & mask];
        let mut lanes = [F::Accumulator::default(), F::Accumulator::default()];
        E::accumulate_pair_evals(
            Some(&entry),
            None,
            [inc_at(entry.row()), F::zero()],
            &mut lanes,
            ra_lut,
            wa_lut,
        );
        sum.fmadd(weight, lanes[0].reduce());
    }
    sum.reduce()
}
