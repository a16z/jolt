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

impl<F: JoltField> EqPrefix<F> {
    pub(crate) fn new(num_vars: usize, prefix_len: usize, terms: Vec<(Vec<F>, F)>) -> Self {
        assert!(prefix_len <= 1usize << num_vars);
        assert!(terms.iter().all(|(point, _)| point.len() == num_vars));
        let terms: Vec<_> = terms
            .into_iter()
            .filter(|(_, weight)| !weight.is_zero())
            .collect();
        let tables: Vec<_> = terms
            .iter()
            .map(|(point, weight)| TensorEqTable::new_with_scaling(point, *weight))
            .collect();
        let values = map_indices(prefix_len, |index| {
            tables.iter().map(|table| table.evaluate_index(index)).sum()
        });
        Self {
            values,
            terms,
            num_vars,
        }
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
    use jolt_field::{Fr, Ring, Zero};
    use jolt_poly::{BindingOrder, Polynomial};

    #[test]
    fn structured_suffix_binding_and_round_sums_match_dense_multilinears() {
        for bits in [6, 12] {
            let len = 1 << bits;
            let terms: Vec<_> = (0..3)
                .map(|term| {
                    let point: Vec<_> = (0..bits)
                        .map(|bit| Fr::from_u64((term * 7 + bit) as u64))
                        .collect();
                    (point, Fr::from_u64(11 + term as u64))
                })
                .collect();
            let full: Vec<Fr> = terms
                .iter()
                .map(|(point, weight)| EqPolynomial::evals(point, Some(*weight)))
                .fold(vec![Fr::zero(); len], |mut sum, term| {
                    for (a, b) in sum.iter_mut().zip(term) {
                        *a += b;
                    }
                    sum
                });
            for prefix in [0, 1, 3, len / 4, len / 4 + 1, len / 2 + 1, len - 1, len] {
                let mut actual = EqPrefix::new(bits, prefix, terms.clone());
                let mut expected_values = full.clone();
                if prefix > 0 {
                    actual.add_to_first(Fr::from_u64(19));
                    expected_values[0] += Fr::from_u64(19);
                }
                let mut expected = Polynomial::new(expected_values);
                for round in 0..bits {
                    let half = expected.len() / 2;
                    for row in 0..half {
                        assert_eq!(
                            actual.pair(row),
                            (expected.evals()[2 * row], expected.evals()[2 * row + 1])
                        );
                    }
                    let start = actual.prefix_len().div_ceil(2);
                    let lo: Fr = (start..half).map(|row| expected.evals()[2 * row]).sum();
                    let hi: Fr = (start..half).map(|row| expected.evals()[2 * row + 1]).sum();
                    assert_eq!(actual.suffix_pair_sums(start), (lo, hi));
                    let challenge = Fr::from_u64([0, 1, 23, 41, 17, 99][round % 6]);
                    actual.bind(challenge);
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
}
