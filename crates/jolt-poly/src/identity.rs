//! Identity polynomial evaluating to the integer index on the Boolean hypercube.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

/// Identity polynomial: $\widetilde{I}(x) = \sum_{i=0}^{2^n - 1} i \cdot \widetilde{eq}(x, i)$.
///
/// At each Boolean hypercube point $b \in \{0,1\}^n$, this polynomial evaluates to the
/// integer whose binary representation is $b$ (most-significant bit first). Its multilinear
/// extension at an arbitrary point $r \in \mathbb{F}^n$ is:
/// $$\widetilde{I}(r_1, \ldots, r_n) = \sum_{i=1}^{n} r_i \cdot 2^{n-i}$$
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityPolynomial {
    num_vars: usize,
}

impl IdentityPolynomial {
    /// Creates an identity polynomial over $n$ variables.
    pub fn new(num_vars: usize) -> Self {
        Self { num_vars }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn try_evaluate<F: Field>(&self, point: &[F]) -> Option<F> {
        if point.len() != self.num_vars {
            return None;
        }
        Some(evaluate_identity(point.iter().copied(), self.num_vars))
    }

    pub fn try_evaluate_projected<F: Field>(
        &self,
        point: &[F],
        offset: usize,
        stride: usize,
    ) -> Option<F> {
        if self.num_vars == 0 {
            return Some(F::zero());
        }
        if stride == 0 {
            return None;
        }
        let last_index = stride
            .checked_mul(self.num_vars - 1)
            .and_then(|last_offset| offset.checked_add(last_offset))?;
        if last_index >= point.len() {
            return None;
        }
        Some(evaluate_identity(
            (0..self.num_vars).map(|index| point[offset + stride * index]),
            self.num_vars,
        ))
    }

    /// Evaluates $\widetilde{I}(r) = \sum_{i=1}^{n} r_i \cdot 2^{n-i}$.
    ///
    /// Time: $O(n)$. No heap allocation.
    #[inline]
    pub fn evaluate<F: Field>(&self, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension must match num_vars"
        );
        evaluate_identity(point.iter().copied(), self.num_vars)
    }
}

fn evaluate_identity<F: Field>(values: impl Iterator<Item = F>, num_vars: usize) -> F {
    values.enumerate().fold(F::zero(), |acc, (index, value)| {
        acc + value.mul_pow_2(num_vars - 1 - index)
    })
}

impl<F: Field> crate::MultilinearEvaluation<F> for IdentityPolynomial {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn len(&self) -> usize {
        1 << self.num_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        IdentityPolynomial::evaluate(self, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    #[test]
    fn evaluate_at_boolean_points_returns_index() {
        let n = 4;
        let id = IdentityPolynomial::new(n);

        for idx in 0..(1usize << n) {
            let bits: Vec<Fr> = (0..n)
                .map(|i| {
                    if (idx >> (n - 1 - i)) & 1 == 1 {
                        Fr::one()
                    } else {
                        Fr::zero()
                    }
                })
                .collect();
            assert_eq!(
                id.evaluate(&bits),
                Fr::from_u64(idx as u64),
                "mismatch at index {idx}"
            );
        }
    }

    #[test]
    fn zero_vars() {
        let id = IdentityPolynomial::new(0);
        assert!(id.evaluate::<Fr>(&[]).is_zero());
    }

    #[test]
    fn single_var() {
        let id = IdentityPolynomial::new(1);
        assert!(id.evaluate(&[Fr::zero()]).is_zero());
        assert_eq!(id.evaluate(&[Fr::one()]), Fr::one());
    }

    #[test]
    fn try_evaluate_rejects_dimension_mismatch() {
        let id = IdentityPolynomial::new(2);
        assert_eq!(id.try_evaluate::<Fr>(&[Fr::one()]), None);
    }

    #[test]
    fn projected_identity_uses_offset_and_stride() {
        let id = IdentityPolynomial::new(3);
        let point = [
            Fr::from_u64(1),
            Fr::from_u64(9),
            Fr::from_u64(0),
            Fr::from_u64(8),
            Fr::from_u64(1),
        ];

        assert_eq!(
            id.try_evaluate_projected(&point, 0, 2),
            Some(Fr::from_u64(5))
        );
    }

    #[test]
    fn projected_identity_rejects_out_of_range_projection() {
        let id = IdentityPolynomial::new(3);
        assert_eq!(id.try_evaluate_projected::<Fr>(&[Fr::one()], 0, 2), None);
        assert_eq!(
            id.try_evaluate_projected::<Fr>(&[Fr::one(), Fr::one()], 0, 0),
            None
        );
    }
}
