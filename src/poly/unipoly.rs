#![allow(dead_code)]

use std::ops::{Index, IndexMut};

use super::commitments::{Commitments, MultiCommitGens};
use crate::utils::gaussian_elimination::gaussian_elimination;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::{PrimeField, batch_inversion};
use ark_serialize::*;

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(Debug, Clone, PartialEq)]
pub struct UniPoly<F> {
  pub coeffs: Vec<F>,
}

// ax^2 + bx + c stored as vec![c,a]
// ax^3 + bx^2 + cx + d stored as vec![d,b,a]
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct CompressedUniPoly<F: PrimeField> {
  coeffs_except_linear_term: Vec<F>,
}

impl<F: PrimeField> UniPoly<F> {
  #[allow(dead_code)]
  pub fn from_coeff(coeffs: Vec<F>) -> Self {
    UniPoly { coeffs }
  }

  pub fn from_evals(evals: &[F]) -> Self {
    UniPoly {
      coeffs: Self::vandermonde_interpolation(evals),
    }
  }

  fn vandermonde_interpolation(evals: &[F]) -> Vec<F> {
    let n = evals.len();
    let xs: Vec<F> = (0..n).map(|x| F::from(x as u64)).collect();

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

  pub fn degree(&self) -> usize {
    self.coeffs.len() - 1
  }

  pub fn as_vec(&self) -> Vec<F> {
    self.coeffs.clone()
  }

  pub fn len(&self) -> usize {
    self.coeffs.len()
  }

  pub fn eval_at_zero(&self) -> F {
    self.coeffs[0]
  }

  pub fn eval_at_one(&self) -> F {
    (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
  }

  pub fn evaluate(&self, r: &F) -> F {
    let mut eval = self.coeffs[0];
    let mut power = *r;
    for i in 1..self.coeffs.len() {
      eval += power * self.coeffs[i];
      power *= r;
    }
    eval
  }

  pub fn compress(&self) -> CompressedUniPoly<F> {
    let coeffs_except_linear_term = [&self.coeffs[..1], &self.coeffs[2..]].concat();
    assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
    CompressedUniPoly {
      coeffs_except_linear_term,
    }
  }

  pub fn commit<G: CurveGroup<ScalarField = F>>(&self, gens: &MultiCommitGens<G>, blind: &F) -> G {
    Commitments::batch_commit(&self.coeffs, blind, gens)
  }

  fn factor_root(&mut self, root: &F) -> UniPoly<F> {
    let mut coeffs = self.coeffs.clone();
    if root.is_zero() {
      coeffs.rotate_left(1);
      //YUCK!!!
      coeffs = coeffs[..coeffs.len() - 1].to_vec();
    } else {
      //TODO: handle this unwrap somehow
      let root_inverse = -root.inverse().unwrap();
      let mut temp = F::zero();
      for coeff in &mut coeffs {
        temp = *coeff - temp;
        temp *= root_inverse;
        *coeff = temp;
      }
    }
    coeffs[self.coeffs.len() - 1] = F::zero();
    UniPoly { coeffs }
  }

  pub fn factor_roots(&mut self, roots: &[F]) -> UniPoly<F> {
    assert!(self.len() != 0);
    if roots.len() == 1 {
      return self.factor_root(&roots[0])
    }
    
    let num_roots = roots.len();
    assert!(num_roots < self.len());

    let new_size = self.len() - num_roots;
    let mut minus_root_inverses = vec![F::zero(); num_roots];

    let mut num_zero_roots = 0;
    for root in roots {
      if root.is_zero() {
        num_zero_roots += 1;
      } else {
        minus_root_inverses.push(root.neg());
      }
    }

    // If there are M zero roots, then the first M coefficients of poly must be zero
    for i in 0..num_zero_roots {
      assert!(self.coeffs[i].is_zero())
    }

    let zero_factored = self.coeffs[num_zero_roots..].to_vec();
    let num_non_zero_roots = minus_root_inverses.len();

    if num_non_zero_roots > 0 {
      batch_inversion(&mut minus_root_inverses); 
      let mut division_cache = vec![F::zero(); num_non_zero_roots];

      let mut temp = zero_factored[0];
      for root in minus_root_inverses.clone() {
        temp *= root;
        division_cache.push(temp);
      }

      //Note: we know this can't be 0
      self[0] = *division_cache.last().unwrap();

      // Compute resulting coeffs one by one
      for i in 1..(zero_factored.len() - num_non_zero_roots) {
        temp = zero_factored[i];

        // Compute the intermediate values for the coefficient and save in cache
        for j in 0..num_non_zero_roots {
          temp -= division_cache[j];
          temp *= minus_root_inverses[j];
          division_cache[j] = temp;
        }
        // Save the resulting coefficient
        self[i] = temp;
      }

    } else if num_zero_roots > 0 {
      self.coeffs.rotate_left(1);
      //YUCK!!!
      self.coeffs = self.coeffs[..self.coeffs.len() - 1].to_vec(); 
    }

    // Clear last coefficient
    for i in new_size..self.coeffs.len() {
      self[i] = F::zero();
    }

    todo!()
  }
}

impl<F: PrimeField> CompressedUniPoly<F> {
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
}

impl<F> Index<usize> for UniPoly<F> {
  type Output = F;

  #[inline(always)]
  fn index(&self, _index: usize) -> &F {
    &(self.coeffs[_index])
  }
}

impl<F> IndexMut<usize> for UniPoly<F> {
  #[inline(always)]
  fn index_mut(&mut self, index: usize) -> &mut F {
    &mut (self.coeffs[index])
  }
}

impl<G: CurveGroup> AppendToTranscript<G> for UniPoly<G::ScalarField> {
  fn append_to_transcript<T: ProofTranscript<G>>(&self, label: &'static [u8], transcript: &mut T) {
    transcript.append_message(label, b"UniPoly_begin");
    for i in 0..self.coeffs.len() {
      transcript.append_scalar(b"coeff", &self.coeffs[i]);
    }
    transcript.append_message(label, b"UniPoly_end");
  }
}

#[cfg(test)]
mod tests {

  use super::*;
  use ark_curve25519::Fr;
  use ark_ff::{batch_inversion, BigInt, Field};
  use ark_std::{ops::Neg, test_rng, One, UniformRand, Zero};

  fn interpolate(points: &[Fr], evals: &[Fr]) -> UniPoly<Fr> {
    let n = points.len();

    let numerator_polynomial = compute_linear_polynomial_product(&evals, points.len());

    let mut roots_and_denominators: Vec<Fr> = vec![Fr::zero(); 2 * n];
    let temp_src: Vec<Fr> = points.to_vec();

    for i in 0..n {
      roots_and_denominators[i] = -evals[i];

      // compute constant denominator
      roots_and_denominators[n + i] = Fr::one();
      for j in 0..n {
        if j == i {
          continue;
        }
        roots_and_denominators[n + i] *= evals[i] - evals[j];
      }
    }

    batch_inversion(&mut roots_and_denominators);

    let mut coeffs = vec![Fr::zero(); n];
    let mut temp = vec![Fr::zero(); n];
    let mut z;
    let mut mult;
    for i in 0..n {
      z = roots_and_denominators[i];
      mult = temp_src[i] * roots_and_denominators[n + i];
      temp[0] = mult * numerator_polynomial[0];
      temp[0] *= z;
      coeffs[0] += temp[0];

      for j in 1..n {
        temp[j] = mult * numerator_polynomial[j] - temp[j - 1];
        temp[j] *= z;
        coeffs[j] += temp[j];
      }
    }

    UniPoly::from_coeff(coeffs)
  }

  // This function computes the polynomial (x - a)(x - b)(x - c)... given n distinct roots (a, b, c, ...).
  fn compute_linear_polynomial_product(roots: &[Fr], n: usize) -> Vec<Fr> {
    let mut res = vec![Fr::zero(); n + 1];

    let mut scratch = roots.to_vec();
    res[n] = Fr::one();
    res[n - 1] = roots.into_iter().sum::<Fr>().neg();

    let mut temp;
    let mut constant = Fr::one();
    for i in 0..(n - 1) {
      temp = Fr::zero();
      for j in 0..(n - 1 - i) {
        scratch[j] = roots[j] * scratch[(j + 1)..].into_iter().take(n - 1 - i - j).sum::<Fr>();
        temp += scratch[j];
      }
      res[n - 2 - i] = temp * constant;
      constant *= Fr::one().neg();
    }

    res
  }

  fn compute_linear_polynomial_product_evaluation(roots: &[Fr], z: Fr, n: usize) -> Fr {
    let mut expected = Fr::one();
    for i in 0..n {
      expected *= z - roots[i];
    }
    expected
  }

  #[test]
  fn linear_poly_product() {
    let n = 64;
    let mut roots = vec![Fr::zero(); n];
    let mut rng = test_rng();

    let z = Fr::rand(&mut rng);
    let mut expected = Fr::one();
    for i in 0..n {
      roots[i] = Fr::rand(&mut rng);
      expected *= z - roots[i];
    }

    let res = UniPoly::from_coeff(compute_linear_polynomial_product(&roots, n)).evaluate(&z);
    assert_eq!(res, expected);
  }

  #[test]
  fn interpolate_poly() {
    let n = 250;
    let mut rng = test_rng();

    let poly =
      UniPoly::from_coeff((0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>());

    let mut src = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);

    for _ in 0..n {
      let val = Fr::rand(&mut rng);
      x.push(val);
      src.push(poly.evaluate(&val));
    }
    let res = interpolate(&src, &x);
    
    for i in 0..poly.len() {
      assert_eq!(res[i], poly[i]);
    }
  }

  #[test]
  fn factor_roots() {
    let n = 32;
    let mut rng = test_rng();

    let mut test_case = |num_zero_roots: usize, num_non_zero_roots: usize| {
      let num_roots = num_non_zero_roots + num_zero_roots;
      let mut poly = UniPoly::from_coeff((0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>());

      let mut non_zero_roots: Vec<Fr> = Vec::with_capacity(num_non_zero_roots);
      let mut non_zero_evaluations: Vec<Fr> = Vec::with_capacity(num_non_zero_roots);

      for _ in 0..num_non_zero_roots {
        let root = Fr::rand(&mut rng);
        non_zero_roots.push(root);
        let root_pow = root.pow(BigInt::<1>::from(num_zero_roots as u64));
        non_zero_evaluations.push(poly.evaluate(&root) / root_pow);
      }
      let mut roots = (0..n).map(|_| Fr::zero()).collect::<Vec<_>>();

      for i in 0..num_non_zero_roots {
        roots[num_zero_roots + i] = non_zero_roots[i];
      }

      if num_non_zero_roots > 0 {
        //create poly that interpolates given evaluations
        let interpolated = interpolate(&non_zero_roots, &non_zero_evaluations);
        assert_eq!(interpolated.len(), num_non_zero_roots);
        for (k, coeff) in interpolated.coeffs.iter().enumerate() {
          poly.coeffs[num_non_zero_roots + k] -= coeff;
        }
      }

      // Sanity check that all roots are actually roots
      for i in 0..num_roots {
        assert_eq!(poly.evaluate(&roots[i]), Fr::zero());
      }

      let quotient = poly.factor_roots(&roots);

      // check that (t-r)q(t) == p(t)
      let t = Fr::rand(&mut rng);
      let roots_eval = compute_linear_polynomial_product_evaluation(&roots, t, num_roots);
      let q_t = quotient.evaluate(&t);
      let p_t = poly.evaluate(&t);
      assert_eq!(roots_eval * q_t, p_t);

      for i in (n - num_roots)..n {
        assert_eq!(quotient[i], Fr::zero());
      }
      
      if num_roots == 0 {
        assert_eq!(poly, quotient);
      }

      if num_roots == 1 {
        let quotient_single = poly.factor_roots(&[roots[0]]);
        assert_eq!(quotient_single, quotient);
      }
    };

    test_case(1,0);
    test_case(0,1);
    test_case(1,0);
    test_case(1,1);
    test_case(2,0);
    test_case(0,2);
    test_case(3, 6);
  }

  #[test]
  fn test_from_evals_quad() {
    test_from_evals_quad_helper::<Fr>()
  }

  fn test_from_evals_quad_helper<F: PrimeField>() {
    // polynomial is 2x^2 + 3x + 1
    let e0 = F::one();
    let e1 = F::from(6u64);
    let e2 = F::from(15u64);
    let evals = vec![e0, e1, e2];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 3);
    assert_eq!(poly.coeffs[0], F::one());
    assert_eq!(poly.coeffs[1], F::from(3u64));
    assert_eq!(poly.coeffs[2], F::from(2u64));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e3 = F::from(28u64);
    assert_eq!(poly.evaluate(&F::from(3u64)), e3);
  }

  #[test]
  fn test_from_evals_cubic() {
    test_from_evals_cubic_helper::<Fr>()
  }
  fn test_from_evals_cubic_helper<F: PrimeField>() {
    // polynomial is x^3 + 2x^2 + 3x + 1
    let e0 = F::one();
    let e1 = F::from(7u64);
    let e2 = F::from(23u64);
    let e3 = F::from(55u64);
    let evals = vec![e0, e1, e2, e3];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 4);
    assert_eq!(poly.coeffs[0], F::one());
    assert_eq!(poly.coeffs[1], F::from(3u64));
    assert_eq!(poly.coeffs[2], F::from(2u64));
    assert_eq!(poly.coeffs[3], F::one());

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e4 = F::from(109u64);
    assert_eq!(poly.evaluate(&F::from(4u64)), e4);
  }
}
