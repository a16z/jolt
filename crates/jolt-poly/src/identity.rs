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

    /// Number of variables `n`.
    pub fn num_vars(&self) -> usize {
        self.num_vars
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
        let n = self.num_vars;
        point
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (i, &r_i)| acc + r_i.mul_pow_2(n - 1 - i))
    }
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
    use jolt_field::Field;
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
}
