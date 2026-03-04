//! Univariate polynomial in coefficient form.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::serde_canonical::vec_canonical;

/// Univariate polynomial in coefficient form: $p(x) = \sum_{i=0}^{d} c_i x^i$.
///
/// Coefficients are stored in ascending degree order: `coefficients[i]` is the
/// coefficient of $x^i$. An empty coefficient vector represents the zero polynomial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct UnivariatePoly<F: Field> {
    #[serde(with = "vec_canonical")]
    coefficients: Vec<F>,
}

impl<F: Field> UnivariatePoly<F> {
    /// Creates a polynomial from coefficients in ascending degree order.
    pub fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    /// The zero polynomial.
    pub fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
        }
    }

    /// Degree of the polynomial, or 0 for the zero polynomial.
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Evaluates $p(\text{point})$ using Horner's method.
    ///
    /// Computes $c_d + x(c_{d-1} + x(c_{d-2} + \cdots))$ in $O(d)$ multiplications.
    #[inline]
    pub fn evaluate(&self, point: F) -> F {
        if self.coefficients.is_empty() {
            return F::zero();
        }
        self.coefficients
            .iter()
            .rev()
            .copied()
            .reduce(|acc, c| acc * point + c)
            .unwrap()
    }

    /// Lagrange interpolation from a set of $(x_i, y_i)$ pairs.
    ///
    /// Given $n$ distinct points, produces the unique polynomial of degree $\le n-1$
    /// passing through all of them using the standard $O(n^2)$ algorithm.
    ///
    /// # Panics
    /// Panics if `points` is empty.
    pub fn interpolate(points: &[(F, F)]) -> Self {
        assert!(!points.is_empty(), "cannot interpolate zero points");

        let n = points.len();
        let mut result = vec![F::zero(); n];

        for j in 0..n {
            let mut basis = vec![F::zero(); n];
            basis[0] = F::one();

            let mut basis_len = 1;
            for m in 0..n {
                if m == j {
                    continue;
                }
                let denom = (points[j].0 - points[m].0)
                    .inverse()
                    .expect("interpolation points must be distinct");
                let neg_xm = -points[m].0;

                // Multiply polynomial by (x - x_m): shift up and add
                for k in (1..=basis_len).rev() {
                    basis[k] = basis[k - 1] + basis[k] * neg_xm;
                }
                basis[0] *= neg_xm;
                basis_len += 1;

                for coeff in basis.iter_mut().take(basis_len) {
                    *coeff *= denom;
                }
            }

            for k in 0..n {
                result[k] += points[j].1 * basis[k];
            }
        }

        Self {
            coefficients: result,
        }
    }

    /// Coefficients in ascending degree order: index $i$ holds the coefficient of $x^i$.
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_field::Field;
    use num_traits::Zero;

    #[test]
    fn horner_known_polynomial() {
        // p(x) = 3 + 2x + x^2
        let p = UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(2), Fr::from_u64(1)]);
        assert_eq!(p.evaluate(Fr::from_u64(0)), Fr::from_u64(3));
        assert_eq!(p.evaluate(Fr::from_u64(1)), Fr::from_u64(6));
        assert_eq!(p.evaluate(Fr::from_u64(2)), Fr::from_u64(11));
    }

    #[test]
    fn interpolate_round_trip() {
        let points = vec![
            (Fr::from_u64(0), Fr::from_u64(1)),
            (Fr::from_u64(1), Fr::from_u64(4)),
            (Fr::from_u64(2), Fr::from_u64(11)),
        ];
        let p = UnivariatePoly::interpolate(&points);

        for &(x, y) in &points {
            assert_eq!(p.evaluate(x), y);
        }
    }

    #[test]
    fn degree_is_correct() {
        let p = UnivariatePoly::<Fr>::zero();
        assert_eq!(p.degree(), 0);

        let p = UnivariatePoly::new(vec![Fr::from_u64(5)]);
        assert_eq!(p.degree(), 0);

        let p = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(1), Fr::from_u64(1)]);
        assert_eq!(p.degree(), 2);
    }

    #[test]
    fn zero_polynomial_evaluates_to_zero() {
        let p = UnivariatePoly::<Fr>::zero();
        assert!(p.evaluate(Fr::from_u64(42)).is_zero());
    }

    #[test]
    fn interpolate_linear() {
        // (0, 1), (1, 3) -> p(x) = 1 + 2x
        let points = vec![
            (Fr::from_u64(0), Fr::from_u64(1)),
            (Fr::from_u64(1), Fr::from_u64(3)),
        ];
        let p = UnivariatePoly::interpolate(&points);
        assert_eq!(p.evaluate(Fr::from_u64(5)), Fr::from_u64(11));
    }

    #[test]
    fn serde_round_trip() {
        let p = UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(2), Fr::from_u64(1)]);
        let bytes = bincode::serialize(&p).unwrap();
        let recovered: UnivariatePoly<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(p, recovered);
    }

    #[test]
    fn serde_round_trip_zero() {
        let p = UnivariatePoly::<Fr>::zero();
        let bytes = bincode::serialize(&p).unwrap();
        let recovered: UnivariatePoly<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(p, recovered);
    }
}
