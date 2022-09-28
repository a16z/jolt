use super::commitments::{Commitments, MultiCommitGens};
use super::group::GroupElement;
use super::scalar::Scalar;
use super::transcript::{AppendToTranscript, ProofTranscript};
use ark_ff::Field;
use ark_serialize::*;
use merlin::Transcript;

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(Debug)]
pub struct UniPoly {
  coeffs: Vec<Scalar>,
}

// ax^2 + bx + c stored as vec![c,a]
// ax^3 + bx^2 + cx + d stored as vec![d,b,a]
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct CompressedUniPoly {
  coeffs_except_linear_term: Vec<Scalar>,
}

impl UniPoly {
  pub fn from_evals(evals: &[Scalar]) -> Self {
    // we only support degree-2 or degree-3 univariate polynomials
    assert!(evals.len() == 3 || evals.len() == 4);
    let coeffs = if evals.len() == 3 {
      // ax^2 + bx + c
      let two_inv = Scalar::from(2u64).inverse().unwrap();

      let c = evals[0];
      let a = two_inv * (evals[2] - evals[1] - evals[1] + c);
      let b = evals[1] - c - a;
      vec![c, b, a]
    } else {
      // ax^3 + bx^2 + cx + d
      let two_inv = Scalar::from(2u64).inverse().unwrap();
      let six_inv = Scalar::from(6u64).inverse().unwrap();

      let d = evals[0];
      let a = six_inv
        * (evals[3] - evals[2] - evals[2] - evals[2] + evals[1] + evals[1] + evals[1] - evals[0]);
      let b = two_inv
        * (evals[0] + evals[0] - evals[1] - evals[1] - evals[1] - evals[1] - evals[1]
          + evals[2]
          + evals[2]
          + evals[2]
          + evals[2]
          - evals[3]);
      let c = evals[1] - d - a - b;
      vec![d, c, b, a]
    };

    UniPoly { coeffs }
  }

  pub fn degree(&self) -> usize {
    self.coeffs.len() - 1
  }

  pub fn as_vec(&self) -> Vec<Scalar> {
    self.coeffs.clone()
  }

  pub fn eval_at_zero(&self) -> Scalar {
    self.coeffs[0]
  }

  pub fn eval_at_one(&self) -> Scalar {
    (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
  }

  pub fn evaluate(&self, r: &Scalar) -> Scalar {
    let mut eval = self.coeffs[0];
    let mut power = *r;
    for i in 1..self.coeffs.len() {
      eval += power * self.coeffs[i];
      power *= r;
    }
    eval
  }

  pub fn compress(&self) -> CompressedUniPoly {
    let coeffs_except_linear_term = [&self.coeffs[..1], &self.coeffs[2..]].concat();
    assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
    CompressedUniPoly {
      coeffs_except_linear_term,
    }
  }

  pub fn commit(&self, gens: &MultiCommitGens, blind: &Scalar) -> GroupElement {
    self.coeffs.commit(blind, gens)
  }
}

impl CompressedUniPoly {
  // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
  // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
  pub fn decompress(&self, hint: &Scalar) -> UniPoly {
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

impl AppendToTranscript for UniPoly {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
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
  use ark_std::One;

  #[test]
  fn test_from_evals_quad() {
    // polynomial is 2x^2 + 3x + 1
    let e0 = Scalar::one();
    let e1 = Scalar::from(6u64);
    let e2 = Scalar::from(15u64);
    let evals = vec![e0, e1, e2];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 3);
    assert_eq!(poly.coeffs[0], Scalar::one());
    assert_eq!(poly.coeffs[1], Scalar::from(3u64));
    assert_eq!(poly.coeffs[2], Scalar::from(2u64));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e3 = Scalar::from(28u64);
    assert_eq!(poly.evaluate(&Scalar::from(3u64)), e3);
  }

  #[test]
  fn test_from_evals_cubic() {
    // polynomial is x^3 + 2x^2 + 3x + 1
    let e0 = Scalar::one();
    let e1 = Scalar::from(7u64);
    let e2 = Scalar::from(23u64);
    let e3 = Scalar::from(55u64);
    let evals = vec![e0, e1, e2, e3];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 4);
    assert_eq!(poly.coeffs[0], Scalar::one());
    assert_eq!(poly.coeffs[1], Scalar::from(3u64));
    assert_eq!(poly.coeffs[2], Scalar::from(2u64));
    assert_eq!(poly.coeffs[3], Scalar::one());

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e4 = Scalar::from(109u64);
    assert_eq!(poly.evaluate(&Scalar::from(4u64)), e4);
  }
}
