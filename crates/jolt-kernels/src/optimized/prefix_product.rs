use jolt_field::JoltField;
use jolt_poly::GruenSplitEqPolynomial;

use super::support::bind_prefix_pairs;

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

    pub(crate) fn bind(&mut self, challenge: F) -> bool {
        let capacity = self.values.capacity();
        bind_prefix_pairs(&mut self.values, challenge, self.tail[1]);
        let bound = self.tail[0] + challenge * (self.tail[1] - self.tail[0]);
        self.tail = [bound; 2];
        self.values.capacity() < capacity
    }
    pub(crate) fn product_at_one(&self, other: &Self, eq: &GruenSplitEqPolynomial<F>) -> F {
        let prefix_pairs = self.values.len().div_ceil(2);
        eq.par_fold_out_in_prefix(
            prefix_pairs,
            F::zero,
            |sum, row, _, weight| {
                *sum += weight * self.pair(row).1 * other.pair(row).1;
            },
            |_, weight, sum| weight * sum,
            |left, right| left + right,
        ) + eq.suffix_weight(prefix_pairs) * self.tail[1] * other.tail[1]
    }

    pub(crate) fn product_endpoints(&self, other: &Self, eq: &GruenSplitEqPolynomial<F>) -> (F, F) {
        let prefix_pairs = self.values.len().div_ceil(2);
        let [zero, infinity] = eq.par_fold_out_in_prefix(
            prefix_pairs,
            || [F::zero(); 2],
            |sum, row, _, weight| {
                let (a0, a1) = self.pair(row);
                let (b0, b1) = other.pair(row);
                sum[0] += weight * a0 * b0;
                sum[1] += weight * (a1 - a0) * (b1 - b0);
            },
            |_, weight, sum| [weight * sum[0], weight * sum[1]],
            |a, b| [a[0] + b[0], a[1] + b[1]],
        );
        let mass = eq.suffix_weight(prefix_pairs);
        (
            zero + mass * self.tail[0] * other.tail[0],
            infinity + mass * (self.tail[1] - self.tail[0]) * (other.tail[1] - other.tail[0]),
        )
    }
}
