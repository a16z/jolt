use jolt_field::{Accumulator, JoltField};
use jolt_poly::{bind_prefix_in_place, GruenSplitEqPolynomial};

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct PrefixColumn<F: JoltField> {
    pub(crate) values: Vec<F>,
    // A two-periodic suffix becomes constant after one low-bit bind.
    pub(crate) tail: [F; 2],
}

impl<F: JoltField> PrefixColumn<F> {
    #[inline]
    pub(crate) fn value(&self, index: usize) -> F {
        self.values
            .get(index)
            .copied()
            .unwrap_or(self.tail[index & 1])
    }

    pub(crate) fn pair(&self, row: usize) -> (F, F) {
        (
            self.values.get(2 * row).copied().unwrap_or(self.tail[0]),
            self.values
                .get(2 * row + 1)
                .copied()
                .unwrap_or(self.tail[1]),
        )
    }

    /// Binds the prefix in place and folds the two-periodic tail with it.
    /// Returns whether backing storage shrank.
    pub(crate) fn bind(&mut self, challenge: F) -> bool {
        let shrunk = bind_prefix_in_place(&mut self.values, challenge, self.tail[1]);
        let bound = self.tail[0] + challenge * (self.tail[1] - self.tail[0]);
        self.tail = [bound; 2];
        shrunk
    }
    pub(crate) fn product_at_one(&self, other: &Self, eq: &GruenSplitEqPolynomial<F>) -> F {
        let prefix_pairs = self.values.len().div_ceil(2);
        eq.par_fold_out_in_prefix(
            prefix_pairs,
            F::Accumulator::default,
            |sum, row, _, weight| {
                sum.fmadd(weight * self.pair(row).1, other.pair(row).1);
            },
            |_, weight, sum| weight * sum.reduce(),
            |left, right| left + right,
        ) + eq.suffix_weight(prefix_pairs) * self.tail[1] * other.tail[1]
    }

    pub(crate) fn product_endpoints(&self, other: &Self, eq: &GruenSplitEqPolynomial<F>) -> (F, F) {
        let prefix_pairs = self.values.len().div_ceil(2);
        let [zero, infinity] = eq.par_fold_out_in_prefix(
            prefix_pairs,
            || [F::Accumulator::default(); 2],
            |sum, row, _, weight| {
                let (a0, a1) = self.pair(row);
                let (b0, b1) = other.pair(row);
                sum[0].fmadd(weight * a0, b0);
                sum[1].fmadd(weight * (a1 - a0), b1 - b0);
            },
            |_, weight, sum| [weight * sum[0].reduce(), weight * sum[1].reduce()],
            |a, b| [a[0] + b[0], a[1] + b[1]],
        );
        let mass = eq.suffix_weight(prefix_pairs);
        (
            zero + mass * self.tail[0] * other.tail[0],
            infinity + mass * (self.tail[1] - self.tail[0]) * (other.tail[1] - other.tail[0]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};

    fn check<F: JoltField>() {
        for bits in [1, 6, 12] {
            let len = 1 << bits;
            let point: Vec<F> = (0..bits).map(|i| F::from_u64(17 + 7 * i as u64)).collect();
            for prefix in [0, 1, len / 2 + 1, len] {
                let make = |seed: u64, tail: [u64; 2]| PrefixColumn {
                    values: (0..prefix)
                        .map(|i| {
                            let value = F::from_u64(seed + i as u64 * (i as u64 + 3));
                            if i % 2 == 0 {
                                value
                            } else {
                                -value
                            }
                        })
                        .collect(),
                    tail: tail.map(F::from_u64),
                };
                let mut left = make(11, [19, 23]);
                let mut right = make(29, [31, 37]);
                let dense = |column: &PrefixColumn<F>| {
                    Polynomial::new((0..len).map(|i| column.value(i)).collect())
                };
                let mut dense_left = dense(&left);
                let mut dense_right = dense(&right);
                let mut eq = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
                for round in 0..bits {
                    let weights: Vec<F> = EqPolynomial::evals(&point[..bits - round - 1], None);
                    let mut expected = [F::zero(); 3];
                    for (row, weight) in weights.into_iter().enumerate() {
                        let [a0, a1] =
                            [dense_left.evals()[2 * row], dense_left.evals()[2 * row + 1]];
                        let [b0, b1] = [
                            dense_right.evals()[2 * row],
                            dense_right.evals()[2 * row + 1],
                        ];
                        expected[0] += weight * a0 * b0;
                        expected[1] += weight * (a1 - a0) * (b1 - b0);
                        expected[2] += weight * a1 * b1;
                    }
                    assert_eq!(
                        left.product_endpoints(&right, &eq),
                        (expected[0], expected[1])
                    );
                    assert_eq!(left.product_at_one(&right, &eq), expected[2]);
                    let challenge = F::from_u64([0, 1, 41][round % 3]);
                    let _ = left.bind(challenge);
                    let _ = right.bind(challenge);
                    dense_left.bind_with_order(challenge, BindingOrder::LowToHigh);
                    dense_right.bind_with_order(challenge, BindingOrder::LowToHigh);
                    eq.bind(challenge);
                }
            }
        }
    }

    #[test]
    fn product_sums_match_dense_multilinears() {
        check::<Fr>();
        #[cfg(feature = "akita")]
        check::<Prime128OffsetA7F7>();
    }
}
