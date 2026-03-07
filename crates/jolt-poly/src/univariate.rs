//! Univariate polynomial in coefficient form.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use jolt_field::Field;
use serde::{Deserialize, Serialize};

/// Shared interface for univariate polynomial types.
///
/// Provides the minimal common API between [`UnivariatePoly`] (full coefficient
/// form) and [`CompressedPoly`](crate::CompressedPoly) (linear term omitted). Evaluation and coefficient
/// access are deliberately left as inherent methods because the two representations
/// require different calling conventions (compressed evaluation needs an external
/// hint value).
pub trait UnivariatePolynomial<F: Field>: Send + Sync {
    /// Degree of the polynomial, or 0 for the zero polynomial.
    fn degree(&self) -> usize;
}

/// Univariate polynomial in coefficient form: $p(x) = \sum_{i=0}^{d} c_i x^i$.
///
/// Coefficients are stored in ascending degree order: `coefficients[i]` is the
/// coefficient of $x^i$. An empty coefficient vector represents the zero polynomial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct UnivariatePoly<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> UnivariatePolynomial<F> for UnivariatePoly<F> {
    fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }
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

    /// Consumes the polynomial and returns the coefficient vector.
    pub fn into_coefficients(self) -> Vec<F> {
        self.coefficients
    }

    /// Evaluates the $i$-th Lagrange basis polynomial at `point` over the domain
    /// $\{0, 1, \ldots, n-1\}$:
    /// $$L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n-1} \frac{x - j}{i - j}$$
    ///
    /// # Panics
    /// Panics if `index >= domain_size`.
    pub fn evaluate_basis(domain_size: usize, index: usize, point: F) -> F {
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
    /// of degree $\le n-1$ matching those values. This is a convenience wrapper around
    /// [`interpolate`](Self::interpolate) for the common integer-domain case.
    ///
    /// # Panics
    /// Panics if `evals` is empty.
    pub fn interpolate_over_integers(evals: &[F]) -> Self {
        assert!(!evals.is_empty(), "cannot interpolate zero evaluations");
        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(i, &y)| (F::from_u64(i as u64), y))
            .collect();
        Self::interpolate(&points)
    }

    /// Compresses the polynomial by omitting the linear term.
    ///
    /// The resulting [`CompressedPoly`](crate::CompressedPoly) stores
    /// `[c0, c2, c3, ...]`, saving one field element in proof serialization.
    /// The linear term can be recovered given the hint value `f(0) + f(1)`.
    ///
    /// # Panics
    /// Panics if the polynomial has degree < 1 (no linear term to omit).
    pub fn compress(&self) -> crate::CompressedPoly<F> {
        assert!(
            self.coefficients.len() >= 2,
            "cannot compress a polynomial of degree < 1"
        );
        let coeffs = [&self.coefficients[..1], &self.coefficients[2..]].concat();
        debug_assert_eq!(coeffs.len() + 1, self.coefficients.len());
        crate::CompressedPoly::new(coeffs)
    }

    /// Interpolates from evaluations at `0, 1, 2, ..., n-1` using Gaussian elimination
    /// on the Vandermonde system. Equivalent to `interpolate_over_integers` but uses a
    /// direct matrix solve instead of the Lagrange formula.
    pub fn from_evals(evals: &[F]) -> Self {
        Self {
            coefficients: gaussian_elimination_vandermonde(evals),
        }
    }

    /// Interpolates from evaluations at `[0, 2, 3, ..., n-1]` with the hint `p(0) + p(1)`.
    ///
    /// Recovers `p(1) = hint - p(0)` and then interpolates over the full set `{0, 1, ..., n-1}`.
    pub fn from_evals_and_hint(hint: F, evals: &[F]) -> Self {
        let mut full = evals.to_vec();
        let eval_at_1 = hint - full[0];
        full.insert(1, eval_at_1);
        Self::from_evals(&full)
    }

    /// Interpolates from evaluations at `[0, 1, ..., degree-1, ∞]`.
    ///
    /// The last entry is interpreted as the evaluation at infinity, i.e., the leading
    /// coefficient of the polynomial. Uses Gaussian elimination on the augmented
    /// Vandermonde-plus-infinity system.
    pub fn from_evals_toom(evals: &[F]) -> Self {
        let n = evals.len();
        let mut matrix: Vec<Vec<F>> = Vec::with_capacity(n);

        // Rows for finite x values: x = 0, 1, ..., n-2
        for (i, &eval) in evals[..n - 1].iter().enumerate() {
            let mut row = Vec::with_capacity(n + 1);
            let x = F::from_u64(i as u64);
            let mut power = F::one();
            for _ in 0..n {
                row.push(power);
                power *= x;
            }
            row.push(eval);
            matrix.push(row);
        }

        // Row for x = ∞: only the leading coefficient survives
        let mut row = vec![F::zero(); n];
        row[n - 1] = F::one();
        row.push(evals[n - 1]);
        matrix.push(row);

        Self {
            coefficients: gaussian_elimination_augmented(&mut matrix),
        }
    }

    /// Computes the cubic polynomial `s(X) = l(X) * q(X)`, where `l(X)` is linear
    /// and `q(X)` is quadratic, given partial information and a hint.
    ///
    /// - `linear_coeffs = [l(0), l(∞)]` (constant and leading coefficient)
    /// - `quadratic_coeff_0 = q(0)` (constant)
    /// - `quadratic_coeff_2 = q(∞)` (quadratic coefficient, i.e., leading coeff)
    /// - `hint = s(0) + s(1)`
    ///
    /// Used by the split-eq evaluator to construct round polynomials.
    pub fn from_linear_times_quadratic_with_hint(
        linear_coeffs: [F; 2],
        quadratic_coeff_0: F,
        quadratic_coeff_2: F,
        hint: F,
    ) -> Self {
        let linear_eval_one = linear_coeffs[0] + linear_coeffs[1];
        let cubic_coeff_0 = linear_coeffs[0] * quadratic_coeff_0;

        // s(0) + s(1) = l(0)*q(0) + l(1)*q(1) = hint
        // l(1) = l(0) + l(∞) = linear_eval_one
        // q(1) = q(0) + q(1_coeff) + q(2_coeff)
        // Solve for the linear coefficient of q:
        let quadratic_coeff_1 =
            (hint - cubic_coeff_0) / linear_eval_one - quadratic_coeff_0 - quadratic_coeff_2;

        // s(X) = (a + bX)(c + dX + eX^2) = ac + (ad+bc)X + (ae+bd)X^2 + beX^3
        let coefficients = vec![
            cubic_coeff_0,
            linear_coeffs[0] * quadratic_coeff_1 + linear_coeffs[1] * quadratic_coeff_0,
            linear_coeffs[0] * quadratic_coeff_2 + linear_coeffs[1] * quadratic_coeff_1,
            linear_coeffs[1] * quadratic_coeff_2,
        ];
        Self { coefficients }
    }

    /// Returns `true` if all coefficients are zero (or the vector is empty).
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.iter().all(|c| *c == F::zero())
    }

    /// The leading (highest-degree) coefficient, or `None` for the zero polynomial.
    pub fn leading_coefficient(&self) -> Option<&F> {
        self.coefficients.last()
    }

    /// Polynomial long division: `self = quotient * divisor + remainder`.
    ///
    /// Returns `Some((quotient, remainder))`, or `None` if `divisor` is the
    /// zero polynomial.
    pub fn divide_with_remainder(&self, divisor: &Self) -> Option<(Self, Self)> {
        if self.is_zero() {
            return Some((Self::zero(), Self::zero()));
        }
        if divisor.is_zero() {
            return None;
        }
        if self.coefficients.len() < divisor.coefficients.len() {
            return Some((Self::zero(), self.clone()));
        }

        let divisor_leading_inv = divisor
            .leading_coefficient()
            .unwrap()
            .inverse()
            .expect("leading coefficient must be invertible");

        let mut remainder = self.clone();
        let mut quotient =
            vec![F::zero(); self.coefficients.len() - divisor.coefficients.len() + 1];

        while !remainder.is_zero() && remainder.coefficients.len() >= divisor.coefficients.len() {
            let cur_q_coeff = *remainder.leading_coefficient().unwrap() * divisor_leading_inv;
            let cur_q_degree = remainder.coefficients.len() - divisor.coefficients.len();
            quotient[cur_q_degree] = cur_q_coeff;

            for (i, div_coeff) in divisor.coefficients.iter().enumerate() {
                remainder.coefficients[cur_q_degree + i] -= cur_q_coeff * *div_coeff;
            }

            // Strip trailing zeros
            while remainder
                .coefficients
                .last()
                .is_some_and(|c| *c == F::zero())
            {
                let _ = remainder.coefficients.pop();
            }
        }

        Some((Self::new(quotient), remainder))
    }
}

impl<F: Field> Neg for UnivariatePoly<F> {
    type Output = Self;

    fn neg(mut self) -> Self {
        for c in &mut self.coefficients {
            *c = -*c;
        }
        self
    }
}

impl<F: Field> Add for UnivariatePoly<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;
        self
    }
}

impl<F: Field> Add for &UnivariatePoly<F> {
    type Output = UnivariatePoly<F>;

    fn add(self, rhs: Self) -> UnivariatePoly<F> {
        let (longer, shorter) = if self.coefficients.len() >= rhs.coefficients.len() {
            (&self.coefficients, &rhs.coefficients)
        } else {
            (&rhs.coefficients, &self.coefficients)
        };
        let mut coeffs = longer.clone();
        for (a, b) in coeffs.iter_mut().zip(shorter) {
            *a += *b;
        }
        UnivariatePoly::new(coeffs)
    }
}

impl<F: Field> AddAssign<&Self> for UnivariatePoly<F> {
    fn add_assign(&mut self, rhs: &Self) {
        if rhs.coefficients.len() > self.coefficients.len() {
            self.coefficients.resize(rhs.coefficients.len(), F::zero());
        }
        for (a, b) in self.coefficients.iter_mut().zip(&rhs.coefficients) {
            *a += *b;
        }
    }
}

impl<F: Field> Sub for UnivariatePoly<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= &rhs;
        self
    }
}

impl<F: Field> Sub for &UnivariatePoly<F> {
    type Output = UnivariatePoly<F>;

    fn sub(self, rhs: Self) -> UnivariatePoly<F> {
        let max_len = self.coefficients.len().max(rhs.coefficients.len());
        let mut coeffs = vec![F::zero(); max_len];
        for (i, c) in self.coefficients.iter().enumerate() {
            coeffs[i] += *c;
        }
        for (i, c) in rhs.coefficients.iter().enumerate() {
            coeffs[i] -= *c;
        }
        UnivariatePoly::new(coeffs)
    }
}

impl<F: Field> SubAssign<&Self> for UnivariatePoly<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        if rhs.coefficients.len() > self.coefficients.len() {
            self.coefficients.resize(rhs.coefficients.len(), F::zero());
        }
        for (a, b) in self.coefficients.iter_mut().zip(&rhs.coefficients) {
            *a -= *b;
        }
    }
}

/// Scalar multiplication: `poly * scalar`.
impl<F: Field> Mul<F> for UnivariatePoly<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        self *= rhs;
        self
    }
}

/// Scalar multiplication by reference: `&poly * scalar`.
impl<F: Field> Mul<F> for &UnivariatePoly<F> {
    type Output = UnivariatePoly<F>;

    fn mul(self, rhs: F) -> UnivariatePoly<F> {
        UnivariatePoly::new(self.coefficients.iter().map(|c| *c * rhs).collect())
    }
}

impl<F: Field> MulAssign<F> for UnivariatePoly<F> {
    fn mul_assign(&mut self, rhs: F) {
        for c in &mut self.coefficients {
            *c *= rhs;
        }
    }
}

/// Gaussian elimination on a Vandermonde system for evaluations at `0, 1, ..., n-1`.
fn gaussian_elimination_vandermonde<F: Field>(evals: &[F]) -> Vec<F> {
    let n = evals.len();
    let xs: Vec<F> = (0..n).map(|x| F::from_u64(x as u64)).collect();

    let mut matrix: Vec<Vec<F>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        let x = xs[i];
        let mut power = F::one();
        for _ in 0..n {
            row.push(power);
            power *= x;
        }
        row.push(evals[i]);
        matrix.push(row);
    }

    gaussian_elimination_augmented(&mut matrix)
}

/// Gaussian elimination on an augmented matrix `[A | b]` where `A` is `n × n`.
///
/// Returns the solution vector `x` such that `A x = b`.
fn gaussian_elimination_augmented<F: Field>(matrix: &mut [Vec<F>]) -> Vec<F> {
    let size = matrix.len();
    debug_assert_eq!(size, matrix[0].len() - 1);

    // Forward elimination (row echelon form)
    for i in 0..size.saturating_sub(1) {
        for j in i..size - 1 {
            if matrix[i][i] != F::zero() {
                let factor = matrix[j + 1][i] / matrix[i][i];
                for k in i..=size {
                    let tmp = matrix[i][k];
                    matrix[j + 1][k] -= factor * tmp;
                }
            }
        }
    }

    // Back substitution
    for i in (1..size).rev() {
        if matrix[i][i] != F::zero() {
            for j in (1..=i).rev() {
                let factor = matrix[j - 1][i] / matrix[i][i];
                for k in (0..=size).rev() {
                    let tmp = matrix[i][k];
                    matrix[j - 1][k] -= factor * tmp;
                }
            }
        }
    }

    let mut result = vec![F::zero(); size];
    for i in 0..size {
        result[i] = matrix[i][size] / matrix[i][i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

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

    #[test]
    fn basis_is_one_at_own_index() {
        let n = 5;
        for i in 0..n {
            let val = UnivariatePoly::<Fr>::evaluate_basis(n, i, Fr::from_u64(i as u64));
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
                let val = UnivariatePoly::<Fr>::evaluate_basis(n, i, Fr::from_u64(j as u64));
                assert!(val.is_zero(), "L_{i}({j}) should be 0");
            }
        }
    }

    #[test]
    fn interpolate_over_integers_matches_evaluations() {
        let evals: Vec<Fr> = vec![
            Fr::from_u64(7),
            Fr::from_u64(3),
            Fr::from_u64(11),
            Fr::from_u64(2),
        ];
        let poly = UnivariatePoly::interpolate_over_integers(&evals);

        for (i, &expected) in evals.iter().enumerate() {
            let x = Fr::from_u64(i as u64);
            assert_eq!(poly.evaluate(x), expected, "mismatch at x={i}");
        }
    }

    #[test]
    fn interpolate_over_integers_constant() {
        let c = Fr::from_u64(42);
        let evals = vec![c; 4];
        let poly = UnivariatePoly::interpolate_over_integers(&evals);
        assert_eq!(poly.evaluate(Fr::from_u64(100)), c);
    }

    #[test]
    fn interpolate_single_point_constant() {
        let c = Fr::from_u64(7);
        let poly = UnivariatePoly::interpolate(&[(Fr::from_u64(0), c)]);
        // Degree-0 polynomial: evaluates to c everywhere
        assert_eq!(poly.evaluate(Fr::from_u64(0)), c);
        assert_eq!(poly.evaluate(Fr::from_u64(99)), c);
        assert_eq!(poly.degree(), 0);
    }

    #[test]
    fn compress_then_evaluate_with_hint() {
        // p(x) = 1 + 3x + 2x^2  =>  p(0)=1, p(1)=6
        let p = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(2)]);
        let hint = p.evaluate(Fr::zero()) + p.evaluate(Fr::one());

        let compressed = p.compress();
        let x = Fr::from_u64(5);
        assert_eq!(compressed.evaluate_with_hint(hint, x), p.evaluate(x));
    }

    #[test]
    fn add_polynomials() {
        // (1 + 2x) + (3 + x + 5x^2) = 4 + 3x + 5x^2
        let a = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let b = UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(1), Fr::from_u64(5)]);
        let sum = &a + &b;
        assert_eq!(
            sum,
            UnivariatePoly::new(vec![Fr::from_u64(4), Fr::from_u64(3), Fr::from_u64(5)])
        );
    }

    #[test]
    fn add_assign_extends_shorter() {
        let mut a = UnivariatePoly::new(vec![Fr::from_u64(1)]);
        let b = UnivariatePoly::new(vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)]);
        a += &b;
        assert_eq!(
            a,
            UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(3), Fr::from_u64(4)])
        );
    }

    #[test]
    fn sub_polynomials() {
        let a = UnivariatePoly::new(vec![Fr::from_u64(5), Fr::from_u64(3)]);
        let b = UnivariatePoly::new(vec![Fr::from_u64(2), Fr::from_u64(1)]);
        let diff = a - b;
        assert_eq!(
            diff,
            UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(2)])
        );
    }

    #[test]
    fn neg_polynomial() {
        let p = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let neg_p = -p.clone();
        let sum = p + neg_p;
        assert!(sum.is_zero());
    }

    #[test]
    fn scalar_mul() {
        // (1 + 2x) * 3 = 3 + 6x
        let p = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let scaled = &p * Fr::from_u64(3);
        assert_eq!(
            scaled,
            UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(6)])
        );
    }

    #[test]
    fn scalar_mul_assign() {
        let mut p = UnivariatePoly::new(vec![Fr::from_u64(2), Fr::from_u64(4)]);
        p *= Fr::from_u64(5);
        assert_eq!(
            p,
            UnivariatePoly::new(vec![Fr::from_u64(10), Fr::from_u64(20)])
        );
    }

    #[test]
    fn add_then_scalar_mul_pattern() {
        // Mimics sumcheck batching: batched += &(round_poly * coeff)
        let mut batched = UnivariatePoly::<Fr>::zero();
        let poly_a = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)]);
        let poly_b = UnivariatePoly::new(vec![Fr::from_u64(4), Fr::from_u64(5), Fr::from_u64(6)]);
        let coeff_a = Fr::from_u64(2);
        let coeff_b = Fr::from_u64(3);

        batched += &(&poly_a * coeff_a);
        batched += &(&poly_b * coeff_b);

        // 2*(1+2x+3x^2) + 3*(4+5x+6x^2) = (2+12) + (4+15)x + (6+18)x^2
        for x_val in 0..5u64 {
            let x = Fr::from_u64(x_val);
            let expected = poly_a.evaluate(x) * coeff_a + poly_b.evaluate(x) * coeff_b;
            assert_eq!(batched.evaluate(x), expected, "mismatch at x={x_val}");
        }
    }

    #[test]
    fn divide_exact() {
        // (x^2 - 1) / (x - 1) = (x + 1), remainder 0
        let dividend = UnivariatePoly::new(vec![-Fr::one(), Fr::zero(), Fr::one()]);
        let divisor = UnivariatePoly::new(vec![-Fr::one(), Fr::one()]);
        let (q, r) = dividend.divide_with_remainder(&divisor).unwrap();
        assert_eq!(q, UnivariatePoly::new(vec![Fr::one(), Fr::one()]));
        assert!(r.is_zero());
    }

    #[test]
    fn divide_with_remainder_nonzero() {
        // (x^2 + 1) / (x - 1): quotient = x + 1, remainder = 2
        let dividend = UnivariatePoly::new(vec![Fr::one(), Fr::zero(), Fr::one()]);
        let divisor = UnivariatePoly::new(vec![-Fr::one(), Fr::one()]);
        let (q, r) = dividend.divide_with_remainder(&divisor).unwrap();

        // Verify: q * divisor + r == dividend
        for x_val in 0..5u64 {
            let x = Fr::from_u64(x_val);
            assert_eq!(
                q.evaluate(x) * divisor.evaluate(x) + r.evaluate(x),
                dividend.evaluate(x),
            );
        }
    }

    #[test]
    fn divide_by_zero_returns_none() {
        let p = UnivariatePoly::new(vec![Fr::one()]);
        assert!(p.divide_with_remainder(&UnivariatePoly::zero()).is_none());
    }

    #[test]
    fn divide_lower_degree_returns_self_as_remainder() {
        let dividend = UnivariatePoly::new(vec![Fr::from_u64(3)]);
        let divisor = UnivariatePoly::new(vec![Fr::one(), Fr::one()]);
        let (q, r) = dividend.divide_with_remainder(&divisor).unwrap();
        assert!(q.is_zero());
        assert_eq!(r, dividend);
    }

    #[test]
    fn from_evals_quadratic() {
        // p(x) = 2x^2 + 3x + 1 → p(0)=1, p(1)=6, p(2)=15
        let evals = vec![Fr::from_u64(1), Fr::from_u64(6), Fr::from_u64(15)];
        let poly = UnivariatePoly::from_evals(&evals);
        assert_eq!(poly.coefficients[0], Fr::from_u64(1));
        assert_eq!(poly.coefficients[1], Fr::from_u64(3));
        assert_eq!(poly.coefficients[2], Fr::from_u64(2));
    }

    #[test]
    fn from_evals_cubic() {
        // p(x) = x^3 + 2x^2 + 3x + 1
        let evals = vec![
            Fr::from_u64(1),
            Fr::from_u64(7),
            Fr::from_u64(23),
            Fr::from_u64(55),
        ];
        let poly = UnivariatePoly::from_evals(&evals);
        assert_eq!(poly.coefficients[0], Fr::from_u64(1));
        assert_eq!(poly.coefficients[1], Fr::from_u64(3));
        assert_eq!(poly.coefficients[2], Fr::from_u64(2));
        assert_eq!(poly.coefficients[3], Fr::from_u64(1));
    }

    #[test]
    fn from_evals_matches_interpolate_over_integers() {
        let evals: Vec<Fr> = vec![
            Fr::from_u64(7),
            Fr::from_u64(3),
            Fr::from_u64(11),
            Fr::from_u64(2),
        ];
        let via_evals = UnivariatePoly::from_evals(&evals);
        let via_lagrange = UnivariatePoly::interpolate_over_integers(&evals);

        for i in 0..10u64 {
            let x = Fr::from_u64(i);
            assert_eq!(
                via_evals.evaluate(x),
                via_lagrange.evaluate(x),
                "mismatch at x={i}"
            );
        }
    }

    #[test]
    fn from_evals_and_hint() {
        // p(x) = 2x^2 + 3x + 1 → p(0)=1, p(1)=6
        // hint = p(0) + p(1) = 7
        // Given evals at [0, 2] = [1, 15], recover p(1) = 7 - 1 = 6
        let hint = Fr::from_u64(7);
        let evals = vec![Fr::from_u64(1), Fr::from_u64(15)];
        let poly = UnivariatePoly::from_evals_and_hint(hint, &evals);

        assert_eq!(poly.evaluate(Fr::from_u64(0)), Fr::from_u64(1));
        assert_eq!(poly.evaluate(Fr::from_u64(1)), Fr::from_u64(6));
        assert_eq!(poly.evaluate(Fr::from_u64(2)), Fr::from_u64(15));
    }

    #[test]
    fn from_evals_toom_cubic() {
        // p(x) = 9x^3 + 3x^2 + x + 5
        let gt_poly = UnivariatePoly::new(vec![
            Fr::from_u64(5),
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(9),
        ]);
        let degree = 3;
        let mut toom_evals: Vec<Fr> = (0..degree)
            .map(|x| gt_poly.evaluate(Fr::from_u64(x)))
            .collect();
        // eval at ∞ = leading coefficient
        toom_evals.push(*gt_poly.coefficients().last().unwrap());

        let poly = UnivariatePoly::from_evals_toom(&toom_evals);
        assert_eq!(gt_poly, poly);
    }

    #[test]
    fn from_linear_times_quadratic_with_hint() {
        // s(x) = (x + 1) * (x^2 + 2x + 3) = x^3 + 3x^2 + 5x + 3
        // hint = s(0) + s(1) = 3 + 12 = 15
        let linear_coeffs = [Fr::from_u64(1), Fr::from_u64(1)];
        let q0 = Fr::from_u64(3);
        let q2 = Fr::from_u64(1);
        let hint = Fr::from_u64(15);
        let poly = UnivariatePoly::from_linear_times_quadratic_with_hint(linear_coeffs, q0, q2, hint);

        let expected = UnivariatePoly::new(vec![
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(3),
            Fr::from_u64(1),
        ]);
        assert_eq!(poly, expected);
    }
}
