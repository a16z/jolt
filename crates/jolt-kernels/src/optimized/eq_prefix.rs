//! A materialized prefix with an implicit linear combination of equality
//! polynomials beyond it. Low-bit-first binding preserves that representation.

use super::support::map_indices;
use jolt_field::JoltField;
use jolt_poly::{bind_prefix_in_place, eq_index_msb, EqPolynomial, TensorEqTable};

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct EqPrefix<F: JoltField> {
    values: Vec<F>,
    terms: Vec<(Vec<F>, F)>,
    num_vars: usize,
}

/// An [`EqPrefix`]'s entries past the explicit prefix at one multiplication
/// per term, for a caller that reads many of them before the next bind: every
/// term's weights as the product of two tables of about the square root of
/// the domain.
pub(crate) struct EqPrefixEntries<'a, F: JoltField> {
    values: &'a [F],
    tables: Vec<TensorEqTable<F>>,
}

impl<F: JoltField> EqPrefixEntries<'_, F> {
    #[inline]
    fn value(&self, index: usize) -> F {
        match self.values.get(index) {
            Some(value) => *value,
            None if self.tables.len() < 2 => self
                .tables
                .iter()
                .map(|table| table.evaluate_index(index))
                .sum(),
            None => TensorEqTable::sum_at_index(&self.tables, index),
        }
    }

    /// [`EqPrefix::pair`].
    #[inline]
    pub(crate) fn pair(&self, row: usize) -> (F, F) {
        (self.value(2 * row), self.value(2 * row + 1))
    }
}

impl<F: JoltField> EqPrefix<F> {
    pub(crate) fn new(num_vars: usize, prefix_len: usize, terms: Vec<(Vec<F>, F)>) -> Self {
        assert!(terms.iter().all(|(point, _)| point.len() == num_vars));
        let mut prefix = Self {
            values: Vec::new(),
            terms: terms
                .into_iter()
                .filter(|(_, weight)| !weight.is_zero())
                .collect(),
            num_vars,
        };
        prefix.extend(prefix_len);
        prefix
    }

    pub(crate) fn entries(&self) -> EqPrefixEntries<'_, F> {
        EqPrefixEntries {
            values: &self.values,
            tables: self
                .terms
                .iter()
                .map(|(point, weight)| TensorEqTable::new_with_scaling(point, *weight))
                .collect(),
        }
    }

    /// Makes the first `prefix_len` entries explicit, which must be no fewer
    /// than already are.
    pub(crate) fn extend(&mut self, prefix_len: usize) {
        assert!(self.values.len() <= prefix_len && prefix_len <= 1usize << self.num_vars);
        let entries = self.entries();
        let values = map_indices(prefix_len, |index| entries.value(index));
        self.values = values;
    }

    pub(crate) fn add_to_first(&mut self, value: F) {
        // The explicit prefix retains index zero through every bind.
        self.values[0] += value;
    }

    pub(crate) fn prefix_len(&self) -> usize {
        self.values.len()
    }

    fn implicit_value(&self, index: usize) -> F {
        self.terms
            .iter()
            .map(|(point, weight)| *weight * eq_index_msb(point, index as u128))
            .sum()
    }

    pub(crate) fn pair(&self, row: usize) -> (F, F) {
        let index = 2 * row;
        (
            self.values
                .get(index)
                .copied()
                .unwrap_or_else(|| self.implicit_value(index)),
            self.values
                .get(index + 1)
                .copied()
                .unwrap_or_else(|| self.implicit_value(index + 1)),
        )
    }

    pub(crate) fn bind(&mut self, challenge: F) {
        assert!(self.num_vars > 0);
        let boundary = if self.values.len() % 2 == 1 {
            self.implicit_value(self.values.len())
        } else {
            F::zero()
        };
        let _ = bind_prefix_in_place(&mut self.values, challenge, boundary);
        for (point, weight) in &mut self.terms {
            let r = point[self.num_vars - 1];
            *weight *= (F::one() - r) * (F::one() - challenge) + r * challenge;
            let _ = point.pop();
        }
        self.num_vars -= 1;
    }

    /// Endpoint sums over whole pairs beyond the explicit prefix.
    pub(crate) fn suffix_pair_sums(&self, start: usize) -> (F, F) {
        assert!(self.num_vars > 0);
        assert!(start * 2 >= self.values.len());
        self.terms
            .iter()
            .fold((F::zero(), F::zero()), |(lo, hi), (point, weight)| {
                let last = self.num_vars - 1;
                let mass = *weight * EqPolynomial::suffix_sum(&point[..last], start);
                let one = mass * point[last];
                (lo + mass - one, hi + one)
            })
    }

    #[cfg(test)]
    pub(crate) fn from_evals(values: Vec<F>) -> Self {
        assert!(values.len().is_power_of_two());
        Self {
            num_vars: values.len().ilog2() as usize,
            values,
            terms: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128Offset6A23801F;
    use jolt_poly::{BindingOrder, Polynomial};

    fn check<F: JoltField>(term_count: usize) {
        for bits in [6, 12] {
            let len = 1 << bits;
            let terms: Vec<_> = (0..term_count)
                .map(|term| {
                    let point: Vec<_> = (0..bits)
                        .map(|bit| F::from_u64((term * 7 + bit) as u64))
                        .collect();
                    (point, F::from_u64(11 + term as u64))
                })
                .collect();
            let full: Vec<F> = terms
                .iter()
                .map(|(point, weight)| EqPolynomial::evals(point, Some(*weight)))
                .fold(vec![F::zero(); len], |mut sum, term| {
                    for (a, b) in sum.iter_mut().zip(term) {
                        *a += b;
                    }
                    sum
                });
            for prefix in [0, 1, 3, len / 4, len / 4 + 1, len / 2 + 1, len - 1, len] {
                let mut actual = EqPrefix::new(bits, prefix, terms.clone());
                // Explicit at its first entry alone until the bind after
                // which it extends to `actual`'s prefix.
                let mut late = EqPrefix::new(bits, prefix.min(1), terms.clone());
                let mut expected_values = full.clone();
                if prefix > 0 {
                    actual.add_to_first(F::from_u64(19));
                    late.add_to_first(F::from_u64(19));
                    expected_values[0] += F::from_u64(19);
                }
                let mut expected = Polynomial::new(expected_values);
                for round in 0..bits {
                    let half = expected.len() / 2;
                    if round == bits / 3 {
                        late.extend(actual.prefix_len());
                        assert_eq!(late.values, actual.values);
                    }
                    let late_entries = late.entries();
                    for row in 0..half {
                        let pair = (expected.evals()[2 * row], expected.evals()[2 * row + 1]);
                        assert_eq!(actual.pair(row), pair);
                        assert_eq!(late_entries.pair(row), pair);
                    }
                    let start = actual.prefix_len().div_ceil(2);
                    let lo: F = (start..half).map(|row| expected.evals()[2 * row]).sum();
                    let hi: F = (start..half).map(|row| expected.evals()[2 * row + 1]).sum();
                    assert_eq!(actual.suffix_pair_sums(start), (lo, hi));
                    let challenge = F::from_u64([0, 1, 23, 41, 17, 99][round % 6]);
                    actual.bind(challenge);
                    late.bind(challenge);
                    expected.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                let final_value = actual
                    .values
                    .first()
                    .copied()
                    .unwrap_or_else(|| actual.implicit_value(0));
                assert_eq!(final_value, expected.evals()[0]);
            }
        }
    }
    #[test]
    fn structured_suffix_binding_and_round_sums_match_dense_multilinears() {
        for terms in [0, 1, 3, 9] {
            check::<Fr>(terms);
            #[cfg(feature = "akita")]
            check::<Prime128Offset6A23801F>(terms);
        }
    }
}
