use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::optimized::support::{bind_raw_twice, bound_pair, mul_0_optimized};

/// Bound one-hot coefficient values indexed by `u16`. Each bind squares the
/// table: `(a ≪ bits) | b` maps to `b + r·(a − b)`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct CoeffLut<F> {
    /// Power-of-two table with zero fixed at index 0.
    pub(super) values: Vec<F>,
}

impl<F: JoltField> CoeffLut<F> {
    const MAX_VALUES: usize = 1 << 16;

    pub(super) fn new(values: Vec<F>) -> Self {
        debug_assert!(values.len().is_power_of_two());
        debug_assert!(values[0] == F::zero());
        Self { values }
    }

    fn bits(&self) -> u32 {
        self.values.len().trailing_zeros()
    }

    pub(super) fn saturated(&self) -> bool {
        self.values.len() * self.values.len() > Self::MAX_VALUES
    }

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

pub(super) use layout::SeedEntry;
use layout::{
    load_indexed, merge_bind, split_pair_group, Cell, IndexedMeta, MatrixEntry, SparseEntry,
};
use ops::{
    bind_indexed_in_place_soa, bind_indexed_to_direct, bind_seed_entries_fused,
    bind_sparse_entries_in_place, sparse_quadratic, sparse_quadratic_fused, sparse_quadratic_soa,
};
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum CyclePhase<F: JoltField> {
    Seed {
        entries: Vec<SeedEntry>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        rd_inc: Vec<i128>,
    },
    SeedBound {
        entries: Vec<SeedEntry>,
        seed_ra_lut: CoeffLut<F>,
        seed_wa_lut: CoeffLut<F>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        r1: F,
        rd_inc: Vec<i128>,
    },
    Indexed {
        vals: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
        metas: Vec<IndexedMeta>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        inc: Polynomial<F>,
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

impl<F: JoltField> CycleState<F> {
    pub(super) fn new(
        entries: Vec<SeedEntry>,
        ra_lut: CoeffLut<F>,
        wa_lut: CoeffLut<F>,
        rd_inc: Vec<i128>,
    ) -> Self {
        Self(CyclePhase::Seed {
            entries,
            ra_lut,
            wa_lut,
            rd_inc,
        })
    }

    pub(super) fn unused_lut() -> CoeffLut<F> {
        CoeffLut { values: Vec::new() }
    }

    pub(super) fn quadratic(&self, e_in: &[F], e_out: &[F]) -> [F; 2] {
        match &self.0 {
            CyclePhase::Seed {
                entries,
                ra_lut,
                wa_lut,
                rd_inc,
            } => sparse_quadratic(entries, ra_lut, wa_lut, e_in, e_out, |z| {
                let inc_0 = F::from_i128(rd_inc[2 * z]);
                [inc_0, F::from_i128(rd_inc[2 * z + 1]) - inc_0]
            }),
            CyclePhase::SeedBound {
                entries,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                rd_inc,
            } => sparse_quadratic_fused(
                entries,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                *r1,
                e_in,
                e_out,
                |z| {
                    let inc_0 = raw_bound_inc(rd_inc, *r1, 2 * z);
                    [inc_0, raw_bound_inc(rd_inc, *r1, 2 * z + 1) - inc_0]
                },
            ),
            CyclePhase::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
                inc,
            } => {
                let inc = inc.evals();
                sparse_quadratic_soa(vals, metas, ra_lut, wa_lut, e_in, e_out, |z| {
                    let inc_0 = inc[2 * z];
                    [inc_0, inc[2 * z + 1] - inc_0]
                })
            }
            CyclePhase::Direct { entries, inc } => {
                let unused = Self::unused_lut();
                let inc = inc.evals();
                sparse_quadratic(entries, &unused, &unused, e_in, e_out, |z| {
                    let inc_0 = inc[2 * z];
                    [inc_0, inc[2 * z + 1] - inc_0]
                })
            }
        }
    }

    pub(super) fn q_at_one(&self, e_in: &[F], e_out: &[F]) -> F {
        match &self.0 {
            CyclePhase::Seed {
                entries,
                ra_lut,
                wa_lut,
                rd_inc,
            } => sparse_at_one(
                entries.iter().copied(),
                ra_lut,
                wa_lut,
                e_in,
                e_out,
                |row| F::from_i128(rd_inc[row]),
            ),
            CyclePhase::SeedBound {
                entries,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                rd_inc,
            } => {
                let mut scratch = ops::fused_scratch();
                let mut sum = F::zero();
                for group in entries.chunk_by(|a, b| a.row() / 4 == b.row() / 4) {
                    ops::fused_intermediates(group, seed_ra_lut, seed_wa_lut, *r1, &mut scratch);
                    sum += sparse_at_one(
                        scratch.1.iter().copied(),
                        ra_lut,
                        wa_lut,
                        e_in,
                        e_out,
                        |row| raw_bound_inc(rd_inc, *r1, row),
                    );
                }
                sum
            }
            CyclePhase::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
                inc,
            } => sparse_at_one(
                (0..metas.len()).map(|i| load_indexed((vals, metas), i)),
                ra_lut,
                wa_lut,
                e_in,
                e_out,
                |row| inc.evals()[row],
            ),
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
        }
    }

    /// Bind once; return whether a full entry generation was replaced.
    pub(super) fn bind(&mut self, r: F) -> bool {
        match &mut self.0 {
            CyclePhase::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
                inc,
            } if !ra_lut.saturated() && !wa_lut.saturated() => {
                bind_indexed_in_place_soa(vals, metas, ra_lut, wa_lut, r);
                ra_lut.bind(r);
                wa_lut.bind(r);
                inc.bind_with_order(r, BindingOrder::LowToHigh);
                return false;
            }
            CyclePhase::Direct { entries, inc } => {
                let unused = Self::unused_lut();
                bind_sparse_entries_in_place(entries, |even, odd| {
                    SparseEntry::<F, F>::bind(even, odd, r, &unused, &unused)
                });
                inc.bind_with_order(r, BindingOrder::LowToHigh);
                return false;
            }
            _ => {}
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
                entries,
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
                        entries,
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
                entries,
                seed_ra_lut,
                seed_wa_lut,
                mut ra_lut,
                mut wa_lut,
                r1,
                rd_inc,
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
                crate::mem::drop_in_background_thread(entries);
                ra_lut.bind(r);
                wa_lut.bind(r);
                (
                    CyclePhase::Indexed {
                        vals,
                        metas,
                        ra_lut,
                        wa_lut,
                        inc: bind_raw_twice(|j| F::from_i128(rd_inc[j]), rd_inc.len(), r1, r),
                    },
                    true,
                )
            }
            // Dereference during the bind before the LUT index overflows.
            CyclePhase::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
                mut inc,
            } => (
                CyclePhase::Direct {
                    entries: bind_indexed_to_direct(&vals, &metas, &ra_lut, &wa_lut, r),
                    inc: {
                        inc.bind_with_order(r, BindingOrder::LowToHigh);
                        inc
                    },
                },
                true,
            ),
            CyclePhase::Direct { .. } => unreachable!("direct entries bind in place above"),
        };
        self.0 = next;
        freed_generation
    }

    /// Scatter the final cycle row into dense `(ra, wa, val)` arrays.
    pub(super) fn take_dense(&mut self, k: usize) -> (Vec<F>, Vec<F>, Vec<F>, F) {
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
                entries,
                seed_ra_lut,
                seed_wa_lut,
                ra_lut,
                wa_lut,
                r1,
                rd_inc,
            } => {
                let (evens, odds) = split_pair_group(&entries);
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
                raw_bound_inc(&rd_inc, r1, 0)
            }
            CyclePhase::Indexed {
                vals,
                metas,
                ra_lut,
                wa_lut,
                inc,
            } => {
                for (value, meta) in vals.into_iter().zip(metas) {
                    debug_assert_eq!(meta.row(), 0);
                    ra[meta.col as usize] = LutIndex(meta.ra).value(&ra_lut);
                    wa[meta.col as usize] = LutIndex(meta.wa).value(&wa_lut);
                    val[meta.col as usize] = value;
                }
                inc.evals()[0]
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
        (ra, wa, val, inc)
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
