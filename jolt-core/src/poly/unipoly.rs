#![allow(dead_code)]
use crate::field::JoltField;
use std::cmp::Ordering;
use std::ops::{AddAssign, Index, IndexMut, Mul, MulAssign, Sub};

use crate::utils::gaussian_elimination::gaussian_elimination;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_serialize::*;
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;

use super::compact_polynomial::SmallScalar;
use super::multilinear_polynomial::MultilinearPolynomial;

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(Debug, Clone, PartialEq)]
pub struct UniPoly<F> {
    pub coeffs: Vec<F>,
}

// ax^2 + bx + c stored as vec![c,a]
// ax^3 + bx^2 + cx + d stored as vec![d,b,a]
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct CompressedUniPoly<F: JoltField> {
    pub coeffs_except_linear_term: Vec<F>,
}

impl<F: JoltField> UniPoly<F> {
    pub fn from_coeff(coeffs: Vec<F>) -> Self {
        UniPoly { coeffs }
    }

    /// Interpolate a polynomial from its evaluations at the points 0, 1, 2, ..., n-1.
    pub fn from_evals(evals: &[F]) -> Self {
        UniPoly {
            coeffs: Self::vandermonde_interpolation(evals),
        }
    }

    fn vandermonde_interpolation(evals: &[F]) -> Vec<F> {
        let n = evals.len();
        let xs: Vec<F> = (0..n).map(|x| F::from_u64(x as u64)).collect();

        let mut vandermonde: Vec<Vec<F>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            let x = xs[i];
            row.push(F::one());
            row.push(x);
            for j in 2..n {
                row.push(row[j - 1] * x);
            }
            row.push(evals[i]);
            vandermonde.push(row);
        }

        gaussian_elimination(&mut vandermonde)
    }

    /// Divide self by another polynomial, and returns the
    /// quotient and remainder.
    #[tracing::instrument(skip_all, name = "UniPoly::divide_with_remainder")]
    pub fn divide_with_remainder(&self, divisor: &Self) -> Option<(Self, Self)> {
        if self.is_zero() {
            Some((Self::zero(), Self::zero()))
        } else if divisor.is_zero() {
            None
        } else if self.degree() < divisor.degree() {
            Some((Self::zero(), self.clone()))
        } else {
            // Now we know that self.degree() >= divisor.degree();
            let mut quotient = vec![F::zero(); self.degree() - divisor.degree() + 1];
            let mut remainder: Self = self.clone();
            // Can unwrap here because we know self is not zero.
            let divisor_leading_inv = divisor.leading_coefficient().unwrap().inverse().unwrap();
            while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
                let cur_q_coeff = *remainder.leading_coefficient().unwrap() * divisor_leading_inv;
                let cur_q_degree = remainder.degree() - divisor.degree();
                quotient[cur_q_degree] = cur_q_coeff;

                for (i, div_coeff) in divisor.coeffs.iter().enumerate() {
                    remainder.coeffs[cur_q_degree + i] -= cur_q_coeff * *div_coeff;
                }

                while let Some(true) = remainder.coeffs.last().map(|c| c == &F::zero()) {
                    remainder.coeffs.pop();
                }
            }
            Some((Self::from_coeff(quotient), remainder))
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|c| c == &F::zero())
    }

    fn leading_coefficient(&self) -> Option<&F> {
        self.coeffs.last()
    }

    pub fn zero() -> Self {
        Self::from_coeff(Vec::new())
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn as_vec(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    pub fn eval_at_zero(&self) -> F {
        self.coeffs[0]
    }

    pub fn eval_at_one(&self) -> F {
        (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
    }

    #[tracing::instrument(skip_all, name = "UniPoly::evaluate")]
    pub fn evaluate(&self, r: &F) -> F {
        Self::eval_with_coeffs(&self.coeffs, r)
    }

    #[tracing::instrument(skip_all, name = "UniPoly::eval_with_coeffs")]
    pub fn eval_with_coeffs(coeffs: &[F], r: &F) -> F {
        let mut eval = coeffs[0];
        let mut power = *r;
        for i in 1..coeffs.len() {
            eval += power * coeffs[i];
            power *= *r;
        }
        eval
    }

    #[tracing::instrument(skip_all, name = "UniPoly::eval_as_univariate")]
    pub fn eval_as_univariate(poly: &MultilinearPolynomial<F>, r: &F) -> F {
        match poly {
            MultilinearPolynomial::LargeScalars(poly) => {
                let mut eval = poly.Z[0];
                let mut power = *r;
                for coeff in poly.evals_ref()[1..].iter() {
                    eval += power * coeff;
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U8Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            MultilinearPolynomial::I64Scalars(poly) => {
                let mut eval = F::zero();
                let mut power = F::one();
                for coeff in poly.coeffs.iter() {
                    eval += coeff.field_mul(power);
                    power *= *r;
                }
                eval
            }
            _ => unimplemented!("Unsupported MultilinearPolynomial variant"),
        }
    }

    pub fn compress(&self) -> CompressedUniPoly<F> {
        let coeffs_except_linear_term = [&self.coeffs[..1], &self.coeffs[2..]].concat();
        debug_assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
        CompressedUniPoly {
            coeffs_except_linear_term,
        }
    }

    pub fn random<R: RngCore + CryptoRng>(num_vars: usize, mut rng: &mut R) -> Self {
        Self::from_coeff(
            std::iter::from_fn(|| Some(F::random(&mut rng)))
                .take(num_vars)
                .collect(),
        )
    }

    pub fn shift_coefficients(&mut self, rhs: &F) {
        self.coeffs.par_iter_mut().for_each(|c| *c += *rhs);
    }

    /// This function computes a cubic polynomial s(X), given the following conditions:
    /// - s(X) = l(X) * t(X), where l(X) is linear and t(X) is quadratic,
    /// - l(X) = a + bX is given by l(0) = a and l(\infty) = b,
    /// - t(X) = c + dX + eX^2 is given by t(0) = c and t(\infty) = e (but d is missing),
    /// - s(0) + s(1) = hint,
    ///
    /// This is used in the optimized sum-check evaluation with split eq polynomial.
    pub fn from_linear_times_quadratic_with_hint(
        linear_coeffs: [F; 2],
        quadratic_coeff_0: F,
        quadratic_coeff_2: F,
        hint: F,
    ) -> Self {
        let linear_eval_one = linear_coeffs[0] + linear_coeffs[1];

        let cubic_coeff_0 = linear_coeffs[0] * quadratic_coeff_0;

        // Compute the linear coefficient of the quadratic polynomial from the hint
        // Given that s(0) + s(1) = hint, we can rewrite this as:
        // a * c + (a + b) * (c + d + e) = hint, which means we can solve for d as:
        // d = (hint - a * c) / (a + b) - c - e
        let quadratic_coeff_1 =
            (hint - cubic_coeff_0) / linear_eval_one - quadratic_coeff_0 - quadratic_coeff_2;

        // Now derive the coefficients of the cubic polynomial from the evaluations
        // We have s(X) = (a + bX) * (c + dX + eX^2) = ac + (ad + bc)X + (ae + bd)X^2 + beX^3
        let coeffs = [
            cubic_coeff_0,
            linear_coeffs[0] * quadratic_coeff_1 + linear_coeffs[1] * quadratic_coeff_0,
            linear_coeffs[0] * quadratic_coeff_2 + linear_coeffs[1] * quadratic_coeff_1,
            linear_coeffs[1] * quadratic_coeff_2,
        ];
        Self::from_coeff(coeffs.to_vec())
    }
}

impl<F: JoltField> AddAssign<&Self> for UniPoly<F> {
    fn add_assign(&mut self, rhs: &Self) {
        let ordering = self.coeffs.len().cmp(&rhs.coeffs.len());
        #[allow(clippy::disallowed_methods)]
        for (lhs, rhs) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *lhs += *rhs;
        }
        if matches!(ordering, Ordering::Less) {
            self.coeffs
                .extend(rhs.coeffs[self.coeffs.len()..].iter().cloned());
        }
    }
}

impl<F: JoltField> Sub for UniPoly<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let ordering = self.coeffs.len().cmp(&rhs.coeffs.len());
        #[allow(clippy::disallowed_methods)]
        for (lhs, rhs) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *lhs -= *rhs;
        }
        if matches!(ordering, Ordering::Less) {
            self.coeffs
                .extend(rhs.coeffs[self.coeffs.len()..].iter().map(|v| v.neg()));
        }
        self
    }
}

impl<F: JoltField> Mul<F> for UniPoly<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        let iter = self.coeffs.into_par_iter();
        Self::from_coeff(iter.map(|c| c * rhs).collect::<Vec<_>>())
    }
}

impl<F: JoltField> Mul<&F> for UniPoly<F> {
    type Output = Self;

    fn mul(self, rhs: &F) -> Self {
        let iter = self.coeffs.into_par_iter();
        Self::from_coeff(iter.map(|c| c * *rhs).collect::<Vec<_>>())
    }
}

impl<F: JoltField> Mul<&F> for &UniPoly<F> {
    type Output = UniPoly<F>;

    fn mul(self, rhs: &F) -> UniPoly<F> {
        UniPoly::from_coeff(self.coeffs.iter().map(|c| *c * *rhs).collect::<Vec<_>>())
    }
}

impl<F: JoltField> Index<usize> for UniPoly<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}

impl<F: JoltField> IndexMut<usize> for UniPoly<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coeffs[index]
    }
}

impl<F: JoltField> MulAssign<&F> for UniPoly<F> {
    fn mul_assign(&mut self, rhs: &F) {
        self.coeffs.par_iter_mut().for_each(|c| *c *= *rhs);
    }
}

impl<F: JoltField> CompressedUniPoly<F> {
    // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
    // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
    pub fn decompress(&self, hint: &F) -> UniPoly<F> {
        let mut linear_term =
            *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term -= self.coeffs_except_linear_term[i];
        }

        let mut coeffs = vec![self.coeffs_except_linear_term[0], linear_term];
        coeffs.extend(&self.coeffs_except_linear_term[1..]);
        assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
        UniPoly { coeffs }
    }

    // In the verifier we do not have to check that f(0) + f(1) = hint as we can just
    // recover the linear term assuming the prover did it right, then eval the poly
    pub fn eval_from_hint(&self, hint: &F, x: &F) -> F {
        let mut linear_term =
            *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term -= self.coeffs_except_linear_term[i];
        }

        let mut running_point = *x;
        let mut running_sum = self.coeffs_except_linear_term[0] + *x * linear_term;
        for i in 1..self.coeffs_except_linear_term.len() {
            running_point = running_point * x;
            running_sum += self.coeffs_except_linear_term[i] * running_point;
        }
        running_sum
    }

    pub fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }
}

impl<F: JoltField> AppendToTranscript for CompressedUniPoly<F> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"UniPoly_begin");
        for i in 0..self.coeffs_except_linear_term.len() {
            transcript.append_scalar(&self.coeffs_except_linear_term[i]);
        }
        transcript.append_message(b"UniPoly_end");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn test_from_evals_quad() {
        test_from_evals_quad_helper::<Fr>()
    }

    fn test_from_evals_quad_helper<F: JoltField>() {
        // polynomial is 2x^2 + 3x + 1
        let e0 = F::one();
        let e1 = F::from_u64(6u64);
        let e2 = F::from_u64(15u64);
        let evals = vec![e0, e1, e2];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 3);
        assert_eq!(poly.coeffs[0], F::one());
        assert_eq!(poly.coeffs[1], F::from_u64(3u64));
        assert_eq!(poly.coeffs[2], F::from_u64(2u64));

        let hint = e0 + e1;
        let compressed_poly = poly.compress();
        let decompressed_poly = compressed_poly.decompress(&hint);
        for i in 0..decompressed_poly.coeffs.len() {
            assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
        }

        let e3 = F::from_u64(28u64);
        assert_eq!(poly.evaluate(&F::from_u64(3u64)), e3);
    }

    #[test]
    fn test_from_evals_cubic() {
        test_from_evals_cubic_helper::<Fr>()
    }
    fn test_from_evals_cubic_helper<F: JoltField>() {
        // polynomial is x^3 + 2x^2 + 3x + 1
        let e0 = F::one();
        let e1 = F::from_u64(7u64);
        let e2 = F::from_u64(23u64);
        let e3 = F::from_u64(55u64);
        let evals = vec![e0, e1, e2, e3];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 4);
        assert_eq!(poly.coeffs[0], F::one());
        assert_eq!(poly.coeffs[1], F::from_u64(3u64));
        assert_eq!(poly.coeffs[2], F::from_u64(2u64));
        assert_eq!(poly.coeffs[3], F::one());

        let hint = e0 + e1;
        let compressed_poly = poly.compress();
        let decompressed_poly = compressed_poly.decompress(&hint);
        for i in 0..decompressed_poly.coeffs.len() {
            assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
        }

        let e4 = F::from_u64(109u64);
        assert_eq!(poly.evaluate(&F::from_u64(4u64)), e4);
    }

    pub fn naive_mul<F: JoltField>(ours: &UniPoly<F>, other: &UniPoly<F>) -> UniPoly<F> {
        if ours.is_zero() || other.is_zero() {
            UniPoly::zero()
        } else {
            let mut result = vec![F::zero(); ours.degree() + other.degree() + 1];
            for (i, self_coeff) in ours.coeffs.iter().enumerate() {
                for (j, other_coeff) in other.coeffs.iter().enumerate() {
                    result[i + j] += *self_coeff * *other_coeff;
                }
            }
            UniPoly::from_coeff(result)
        }
    }

    #[test]
    fn test_divide_poly() {
        let rng = &mut ChaCha20Rng::from_seed([0u8; 32]);

        for a_degree in 0..50 {
            for b_degree in 0..50 {
                let dividend = UniPoly::<Fr>::random(a_degree, rng);
                let divisor = UniPoly::<Fr>::random(b_degree, rng);

                if let Some((quotient, remainder)) =
                    UniPoly::divide_with_remainder(&dividend, &divisor)
                {
                    let mut prod = naive_mul(&divisor, &quotient);
                    prod += &remainder;
                    assert_eq!(dividend, prod)
                }
            }
        }
    }

    #[test]
    fn test_from_linear_times_quadratic_with_hint() {
        // polynomial is s(x) = (x + 1) * (x^2 + 2x + 3) = x^3 + 3x^2 + 5x + 3
        // hint = s(0) + s(1) = 3 + (1 + 3 + 5 + 3) = 15
        let linear_coeffs = [Fr::from_u64(1u64), Fr::from_u64(1u64)];
        let quadratic_coeff_0 = Fr::from_u64(3u64);
        let quadratic_coeff_2 = Fr::from_u64(1u64);
        let true_poly = UniPoly::from_coeff(vec![
            Fr::from_u64(3u64),
            Fr::from_u64(5u64),
            Fr::from_u64(3u64),
            Fr::from_u64(1u64),
        ]);
        let hint = Fr::from_u64(15u64);
        let poly = UniPoly::from_linear_times_quadratic_with_hint(
            linear_coeffs,
            quadratic_coeff_0,
            quadratic_coeff_2,
            hint,
        );
        assert_eq!(poly.coeffs, true_poly.coeffs);
    }
}
