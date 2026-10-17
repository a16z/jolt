//! The stage-6b bytecode cycle coefficient tables
//! `C(j) = Σ_s w_s · eq(r_s, j) + e · [j = 0]`, bound `LowToHigh` as weighted
//! eq sums until the lazy RA family's crossover, so no `T`-entry table is
//! ever allocated.
//!
//! A `LowToHigh` bind with challenge `ρ` fixes the index LSB, which pairs with
//! each big-endian point's last coordinate `r`: the bound table is the same sum
//! over the points' remaining prefixes, each weight multiplied by
//! `eq(r, ρ) = (1 − r)(1 − ρ) + r·ρ` and the entry spike by `1 − ρ`. The
//! update only multiplies, so challenges 0 and 1 take no special case. A lazy
//! round reads entry `j = (j_hi, j_lo)` as `Σ_s H_s[j_hi] · L_s[j_lo]` from
//! split eq tables of the remaining `v` variables (`2^⌈v/2⌉ + 2^⌊v/2⌋` entries
//! per term, `H_s` carrying the weight), rebuilt at every bind.
//!
//! Byte parity: the multilinear extension of an eq table is the eq polynomial,
//! so every lazy entry and the materialized table are the field elements the
//! iterated dense `lo + ρ·(hi − lo)` binds produce.

use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::optimized::support::{eq_table, pair, scaled_eq_table};

/// One coefficient table: lazy until the bind that folds `materialize_branches`
/// cycles into each entry, then a plain dense multilinear of
/// `T / materialize_branches` entries. A domain shorter than that stays lazy.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) enum LazyEqSum<F: JoltField> {
    Lazy(EqSum<F>),
    Dense(Polynomial<F>),
}

/// The lazy table over `vars` unbound cycle variables. Entry
/// `j = (hi << lo_bits) | lo` is `Σ_s high[hi·S + s] · low[lo·S + s]` over the
/// `S` terms, plus `entry` at `j = 0`; the term-interleaved rows make one entry
/// two contiguous reads.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct EqSum<F: JoltField> {
    terms: Vec<EqTerm<F>>,
    entry: F,
    vars: usize,
    /// `log2(T / materialize_branches)`, the unbound-variable count at which
    /// the table materializes; `None` for a domain shorter than the crossover.
    dense_vars: Option<usize>,
    lo_bits: usize,
    high: Vec<F>,
    low: Vec<F>,
}

/// A weight scaled by every bound coordinate's eq factor, and the big-endian
/// point whose first `vars` coordinates are still unbound.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct EqTerm<F: JoltField> {
    weight: F,
    point: Vec<F>,
}

impl<F: JoltField> LazyEqSum<F> {
    /// `Σ weight · eq(point, ·) + entry · [· = 0]` over `log_t` variables; each
    /// point has `log_t` coordinates (the cycle kernel's `prepare` checks every
    /// point it passes). Zero-weight terms contribute nothing and are dropped.
    pub(super) fn new<'a>(
        terms: impl IntoIterator<Item = (F, &'a [F])>,
        entry: F,
        log_t: usize,
        materialize_branches: usize,
    ) -> Self {
        let mut sum = EqSum {
            terms: terms
                .into_iter()
                .filter(|(weight, _)| !weight.is_zero())
                .map(|(weight, point)| EqTerm {
                    weight,
                    point: point.to_vec(),
                })
                .collect(),
            entry,
            vars: log_t,
            dense_vars: ((1usize << log_t) / materialize_branches)
                .checked_ilog2()
                .map(|vars| vars as usize),
            lo_bits: 0,
            high: Vec::new(),
            low: Vec::new(),
        };
        sum.split();
        Self::Lazy(sum)
    }

    /// The `(lo, hi)` pair at bound row `y`: entries `2y` and `2y + 1`.
    #[inline]
    pub(super) fn lo_hi(&self, y: usize) -> (F, F) {
        match self {
            Self::Lazy(sum) => (sum.value(2 * y), sum.value(2 * y + 1)),
            Self::Dense(table) => pair(table, y),
        }
    }

    pub(super) fn bind(&mut self, challenge: F) {
        match self {
            Self::Lazy(sum) => {
                sum.bind(challenge);
                if Some(sum.vars) == sum.dense_vars {
                    let table = sum.materialize();
                    *self = Self::Dense(table);
                }
            }
            Self::Dense(table) => table.bind_with_order(challenge, BindingOrder::LowToHigh),
        }
    }
}

impl<F: JoltField> EqSum<F> {
    #[inline]
    fn value(&self, j: usize) -> F {
        let terms = self.terms.len();
        let high = &self.high[(j >> self.lo_bits) * terms..][..terms];
        let low = &self.low[(j & ((1 << self.lo_bits) - 1)) * terms..][..terms];
        let mut sum = F::Accumulator::default();
        for (high, low) in high.iter().zip(low) {
            sum.fmadd(*high, *low);
        }
        if j == 0 {
            sum.add(self.entry);
        }
        sum.reduce()
    }

    fn bind(&mut self, challenge: F) {
        let one_minus = F::one() - challenge;
        self.vars -= 1;
        for term in &mut self.terms {
            let r = term.point[self.vars];
            term.weight *= one_minus * (F::one() - r) + challenge * r;
        }
        self.entry *= one_minus;
        self.split();
    }

    fn split(&mut self) {
        let terms = self.terms.len();
        self.lo_bits = self.vars / 2;
        let hi_bits = self.vars - self.lo_bits;
        self.high = vec![F::zero(); terms << hi_bits];
        self.low = vec![F::zero(); terms << self.lo_bits];
        for (s, term) in self.terms.iter().enumerate() {
            let (high, low) = term.point[..self.vars].split_at(hi_bits);
            for (row, value) in scaled_eq_table(high, term.weight).into_iter().enumerate() {
                self.high[row * terms + s] = value;
            }
            for (row, value) in eq_table(low).into_iter().enumerate() {
                self.low[row * terms + s] = value;
            }
        }
    }

    /// Every bound entry, written once into the final table.
    fn materialize(&self) -> Polynomial<F> {
        let entries = 1usize << self.vars;
        #[cfg(feature = "parallel")]
        let evals: Vec<F> = (0..entries)
            .into_par_iter()
            .map(|j| self.value(j))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let evals: Vec<F> = (0..entries).map(|j| self.value(j)).collect();
        Polynomial::new(evals)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField};
    use jolt_poly::{BindingOrder, Polynomial};

    use super::LazyEqSum;
    use crate::optimized::parity::ExceptionalEq;
    use crate::optimized::registers_read_write::test_support::challenge_sequence;
    use crate::optimized::support::eq_table;

    /// `terms` and `entry` over `log_t` variables against their dense table
    /// bound by jolt-poly: every round's pairs, the materializing bind
    /// (`2^(log_t − bind)` entries, never earlier) and the fully bound value.
    fn check<F: JoltField>(
        log_t: usize,
        branches: usize,
        materialize_at: Option<usize>,
        terms: &[(F, Vec<F>)],
        entry: F,
        challenges: &[F],
    ) {
        let mut dense = vec![F::zero(); 1 << log_t];
        for (weight, point) in terms {
            for (value, eq) in dense.iter_mut().zip(eq_table(point)) {
                *value += *weight * eq;
            }
        }
        dense[0] += entry;
        let mut dense = Polynomial::new(dense);
        let mut lazy = LazyEqSum::new(
            terms
                .iter()
                .map(|(weight, point)| (*weight, point.as_slice())),
            entry,
            log_t,
            branches,
        );
        for round in 0..=log_t {
            if round > 0 {
                lazy.bind(challenges[round - 1]);
                dense.bind_with_order(challenges[round - 1], BindingOrder::LowToHigh);
            }
            match &lazy {
                LazyEqSum::Dense(table) => {
                    let bind = materialize_at.unwrap();
                    assert!(round >= bind, "materialized at round {round}");
                    if round == bind {
                        assert_eq!(table.len(), 1 << (log_t - bind));
                    }
                }
                LazyEqSum::Lazy(_) => {
                    assert!(
                        materialize_at.is_none_or(|bind| round < bind),
                        "round {round}"
                    );
                }
            }
            for y in 0..dense.len() / 2 {
                let expected = (dense.evals()[2 * y], dense.evals()[2 * y + 1]);
                assert_eq!(lazy.lo_hi(y), expected, "round {round}, row {y}");
            }
        }
        let bound = match &lazy {
            LazyEqSum::Lazy(sum) => sum.value(0),
            LazyEqSum::Dense(table) => table.evals()[0],
        };
        assert_eq!(bound, dense.evals()[0]);
    }

    /// Generic, exceptional (0/1 coordinates, a weight vanishing at the first
    /// bind) and zero-weight terms under generic, literal 0/1-first, and
    /// all-Boolean challenges; with dense rounds after the crossover, the
    /// crossover at the final bind, a domain that ends lazy, and the entry
    /// spike alone.
    fn check_shapes<F: JoltField>() {
        for (log_t, materialize_at) in [(8, Some(4)), (4, Some(4)), (3, None)] {
            let generic = challenge_sequence::<F>(log_t, 3);
            let mut first_zero_one = generic.clone();
            first_zero_one[0] = F::zero();
            first_zero_one[1] = F::one();
            let boolean: Vec<F> = (0..log_t).map(|bit| F::from_u64(bit as u64 % 2)).collect();
            for challenges in [generic, first_zero_one, boolean] {
                let mut terms: Vec<(F, Vec<F>)> = (0..3)
                    .map(|seed| (F::from_u64(seed + 2), challenge_sequence(log_t, 11 + seed)))
                    .collect();
                terms.extend(
                    ExceptionalEq::ALL
                        .into_iter()
                        .zip(7..)
                        .map(|(case, weight)| {
                            (F::from_u64(weight), case.point(log_t, challenges[0]))
                        }),
                );
                terms.push((F::zero(), challenge_sequence(log_t, 19)));
                let entry = F::from_u64(23);
                check(log_t, 16, materialize_at, &terms, entry, &challenges);
                check(log_t, 16, materialize_at, &[], entry, &challenges);
            }
        }
    }

    #[test]
    fn eq_sum_pairs_match_dense_binds() {
        check_shapes::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn eq_sum_pairs_match_dense_binds_fp128() {
        check_shapes::<Prime128OffsetA7F7>();
    }
}
