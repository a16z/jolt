use core::ops::Range;

use jolt_field::{Accumulator, JoltField};
use jolt_poly::Polynomial;
use jolt_sumcheck::SumcheckError;
#[cfg(feature = "parallel")]
use jolt_witness::TraceRows;
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::rows::RegisterCycleRow;
use super::ROWS_LABEL;
use crate::optimized::support::{bind_raw_twice, bound_pair, mul_0_optimized, KernelRows};

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

pub(super) use layout::SeedEntry;
use layout::{merge_bind, split_pair_group, Cell, MatrixEntry, SparseEntry};
use ops::{bind_seed_entries_fused, bind_sparse_entries_in_place, sparse_quadratic};

/// The round-0 register cells that the second bind, or a single-round
/// handoff, merges; round messages regenerate theirs from row visits.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) enum Seeds {
    /// Row-major cells, columns sorted within each row.
    Collected(Vec<SeedEntry>),
    /// Rows `prepare` extracted without error; each pass regenerates their
    /// cells. The witness plane owns the rows and reports them itself.
    #[cfg(feature = "parallel")]
    #[cfg_attr(feature = "allocative", allocative(skip))]
    Rows(TraceRows),
}

impl Seeds {
    /// Visits the nonempty `2^span_bits`-row groups of the span-aligned
    /// `rows` in row order, with their global group index.
    pub(super) fn for_each_group(
        &self,
        rows: Range<usize>,
        span_bits: u32,
        mut visit: impl FnMut(usize, &[SeedEntry]),
    ) -> Result<(), WitnessError> {
        match self {
            Self::Collected(entries) => {
                let start = entries.partition_point(|entry| entry.row() < rows.start);
                let end = entries.partition_point(|entry| entry.row() < rows.end);
                for group in entries[start..end]
                    .chunk_by(|a, b| a.row() >> span_bits == b.row() >> span_bits)
                {
                    visit(group[0].row() >> span_bits, group);
                }
            }
            #[cfg(feature = "parallel")]
            Self::Rows(access) => {
                let mut cells = Vec::with_capacity(3 << span_bits);
                for group in rows.start >> span_bits..rows.end >> span_bits {
                    cells.clear();
                    for row in group << span_bits..(group + 1) << span_bits {
                        let (row_cells, len) =
                            access.window::<RegisterCycleRow>(row)?.entries(row as u32);
                        cells.extend_from_slice(&row_cells[..len]);
                    }
                    if !cells.is_empty() {
                        visit(group, &cells);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum CyclePhase<F: JoltField> {
    Seed {
        seeds: Seeds,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        rd_inc: Vec<i128>,
    },
    SeedBound {
        seeds: Seeds,
        seed_ra_lut: CoeffLut<F>,
        seed_wa_lut: CoeffLut<F>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        r1: F,
        rd_inc: Vec<i128>,
    },
    Direct {
        entries: Vec<SparseEntry<F, F>>,
        inc: Polynomial<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct CycleState<F: JoltField>(CyclePhase<F>);

#[inline]
fn raw_bound_inc<F: JoltField>(raw: &[i128], r1: F, y: usize) -> F {
    bound_pair(|j| F::from_i128(raw[j]), r1, y)
}

/// The bound cycle row as dense `(ra, wa, val)` address arrays and its `inc`.
type DenseCycleRow<F> = (Vec<F>, Vec<F>, Vec<F>, F);

/// A row failed to re-extract after `prepare` extracted every row once.
fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource { kind: ROWS_LABEL }
}

impl<F: JoltField> CycleState<F> {
    pub(super) fn new(
        seeds: Seeds,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        rd_inc: Vec<i128>,
    ) -> Self {
        Self(CyclePhase::Seed {
            seeds,
            ra_lut,
            wa_lut,
            rd_inc,
        })
    }

    pub(super) fn unused_lut() -> CoeffLut<F> {
        CoeffLut { values: Vec::new() }
    }

    pub(super) fn quadratic(
        &self,
        rows: &KernelRows<RegisterCycleRow>,
        e_in: &[F],
        e_out: &[F],
    ) -> Result<[F; 2], SumcheckError<F>> {
        Ok(match &self.0 {
            CyclePhase::Seed { ra_lut, wa_lut, .. } => {
                stream::quadratic(rows, e_in, e_out, &[], &[[ra_lut, wa_lut]])
                    .map_err(rows_unavailable)?
            }
            CyclePhase::SeedBound {
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                ..
            } => stream::quadratic(
                rows,
                e_in,
                e_out,
                &[*r1],
                &[[seed_ra_lut, seed_wa_lut], [ra_lut, wa_lut]],
            )
            .map_err(rows_unavailable)?,
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
            CyclePhase::SeedBound {
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                ..
            } => stream::at_one(
                rows,
                e_in,
                e_out,
                &[*r1],
                &[[seed_ra_lut, seed_wa_lut], [ra_lut, wa_lut]],
            )
            .map_err(rows_unavailable)?,
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

    /// Bind once; return whether a full entry generation was replaced.
    pub(super) fn bind(&mut self, r: F) -> Result<bool, SumcheckError<F>> {
        if let CyclePhase::Direct { entries, inc } = &mut self.0 {
            let unused = Self::unused_lut();
            bind_sparse_entries_in_place(entries, |even, odd| {
                SparseEntry::<F, F>::bind(even, odd, r, &unused, &unused)
            });
            let _ = inc.bind_low_to_high_in_place(r);
            return Ok(false);
        }
        let state = std::mem::replace(
            &mut self.0,
            CyclePhase::Direct {
                entries: Vec::new(),
                inc: Polynomial::new(Vec::new()),
            },
        );
        let (next, freed_generation) = match state {
            CyclePhase::Seed {
                seeds,
                ra_lut,
                wa_lut,
                rd_inc,
            } => {
                let mut bound_ra = CoeffLut::new(ra_lut.values.clone());
                let mut bound_wa = CoeffLut::new(wa_lut.values.clone());
                bound_ra.bind(r);
                bound_wa.bind(r);
                (
                    CyclePhase::SeedBound {
                        seeds,
                        seed_ra_lut: ra_lut,
                        seed_wa_lut: wa_lut,
                        ra_lut: bound_ra,
                        wa_lut: bound_wa,
                        r1: r,
                        rd_inc,
                    },
                    false,
                )
            }
            CyclePhase::SeedBound {
                seeds,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                rd_inc,
            } => {
                let entries = bind_seed_entries_fused(
                    &seeds,
                    rd_inc.len(),
                    &seed_ra_lut,
                    &seed_wa_lut,
                    &ra_lut,
                    &wa_lut,
                    r1,
                    r,
                )
                .map_err(rows_unavailable)?;
                drop(seeds);
                (
                    CyclePhase::Direct {
                        entries,
                        inc: bind_raw_twice(|j| F::from_i128(rd_inc[j]), rd_inc.len(), r1, r),
                    },
                    true,
                )
            }
            CyclePhase::Direct { .. } => unreachable!("direct entries bind in place above"),
        };
        self.0 = next;
        Ok(freed_generation)
    }

    /// Scatter the final cycle row into dense `(ra, wa, val)` arrays.
    pub(super) fn take_dense(&mut self, k: usize) -> Result<DenseCycleRow<F>, SumcheckError<F>> {
        let phase = std::mem::replace(
            &mut self.0,
            CyclePhase::Direct {
                entries: Vec::new(),
                inc: Polynomial::new(Vec::new()),
            },
        );
        let mut ra = vec![F::zero(); k];
        let mut wa = vec![F::zero(); k];
        let mut val = vec![F::zero(); k];
        let inc = match phase {
            CyclePhase::Seed { .. } => {
                unreachable!("prepare requires log_t ≥ 1, so a cycle bind precedes the collapse")
            }
            CyclePhase::SeedBound {
                seeds,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                rd_inc,
            } => {
                seeds
                    .for_each_group(0..rd_inc.len(), 1, |_, pair_group| {
                        let (evens, odds) = split_pair_group(pair_group);
                        merge_bind(
                            evens,
                            odds,
                            &|even, odd| SeedEntry::bind(even, odd, r1, &seed_ra_lut, &seed_wa_lut),
                            |entry| {
                                debug_assert_eq!(entry.row, 0);
                                ra[entry.col as usize] = entry.ra.value(&ra_lut);
                                wa[entry.col as usize] = entry.wa.value(&wa_lut);
                                val[entry.col as usize] = entry.val;
                            },
                        );
                    })
                    .map_err(rows_unavailable)?;
                raw_bound_inc(&rd_inc, r1, 0)
            }
            CyclePhase::Direct { entries, inc } => {
                for entry in entries {
                    debug_assert_eq!(entry.row, 0);
                    ra[entry.col as usize] = entry.ra;
                    wa[entry.col as usize] = entry.wa;
                    val[entry.col as usize] = entry.val;
                }
                inc.evals()[0]
            }
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
