//! Lagrange basis evaluation and interpolation over integer domains.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::univariate::UnivariatePoly;

/// Lagrange basis evaluation and interpolation over the integer domain $\{0, 1, \ldots, n-1\}$.
///
/// Provides utilities for computing individual Lagrange basis polynomials and
/// for interpolating a polynomial from evaluations at consecutive integer points.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LagrangePolynomial;

impl LagrangePolynomial {
    /// Evaluates the $i$-th Lagrange basis polynomial at `point` over the domain
    /// $\{0, 1, \ldots, n-1\}$:
    /// $$L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n-1} \frac{x - j}{i - j}$$
    ///
    /// # Panics
    /// Panics if `index >= domain_size`.
    pub fn evaluate_basis<F: Field>(domain_size: usize, index: usize, point: F) -> F {
        assert!(
            index < domain_size,
            "index {index} out of domain of size {domain_size}"
        );

        let mut numer = F::one();
        let mut denom = F::one();

        for j in 0..domain_size {
            if j == index {
                continue;
            }
            let j_f = F::from_u64(j as u64);
            let i_f = F::from_u64(index as u64);
            numer *= point - j_f;
            denom *= i_f - j_f;
        }

        numer * denom.inverse().expect("Lagrange denominator is zero")
    }

    /// Interpolates a polynomial from evaluations at $\{0, 1, \ldots, n-1\}$.
    ///
    /// Given evaluations $[f(0), f(1), \ldots, f(n-1)]$, returns the unique polynomial
    /// of degree $\le n-1$ matching those values.
    ///
    /// # Panics
    /// Panics if `evals` is empty.
    pub fn interpolate<F: Field>(evals: &[F]) -> UnivariatePoly<F> {
        assert!(!evals.is_empty(), "cannot interpolate zero evaluations");
        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(i, &y)| (F::from_u64(i as u64), y))
            .collect();
        UnivariatePoly::interpolate(&points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    #[test]
    fn basis_is_one_at_own_index() {
        let n = 5;
        for i in 0..n {
            let val = LagrangePolynomial::evaluate_basis::<Fr>(n, i, Fr::from_u64(i as u64));
            assert_eq!(val, Fr::one(), "L_{i}({i}) should be 1");
        }
    }

    #[test]
    fn basis_is_zero_at_other_indices() {
        let n = 5;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let val = LagrangePolynomial::evaluate_basis::<Fr>(n, i, Fr::from_u64(j as u64));
                assert!(val.is_zero(), "L_{i}({j}) should be 0");
            }
        }
    }

    #[test]
    fn interpolate_matches_evaluations() {
        let evals: Vec<Fr> = vec![
            Fr::from_u64(7),
            Fr::from_u64(3),
            Fr::from_u64(11),
            Fr::from_u64(2),
        ];
        let poly = LagrangePolynomial::interpolate(&evals);

        for (i, &expected) in evals.iter().enumerate() {
            let x = Fr::from_u64(i as u64);
            assert_eq!(poly.evaluate(x), expected, "mismatch at x={i}");
        }
    }

    #[test]
    fn interpolate_constant() {
        let c = Fr::from_u64(42);
        let evals = vec![c; 4];
        let poly = LagrangePolynomial::interpolate(&evals);
        assert_eq!(poly.evaluate(Fr::from_u64(100)), c);
    }
}
