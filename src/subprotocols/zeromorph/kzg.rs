use ark_bn254::Bn254;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};
use std::{borrow::Borrow, marker::PhantomData};

use crate::msm::VariableBaseMSM;
use crate::poly::unipoly::UniPoly;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::PrimeField;
use ark_std::One;
use ark_std::{
  rand::{Rng, SeedableRng},
  UniformRand,
};
use rand_chacha::rand_core::RngCore;
use thiserror::Error;

/// `UniversalParams` are the universal parameters for the KZG10 scheme.
#[derive(Debug, Clone, Default)]
pub struct UNIVERSAL_KZG_SRS<P: Pairing> {
  /// Group elements of the form `{ β^i G }`, where `i` ranges from 0 to
  /// `degree`.
  pub g1_powers: Vec<P::G1Affine>,
  /// Group elements of the form `{ β^i H }`, where `i` ranges from 0 to
  /// `degree`.
  pub g2_powers: Vec<P::G2Affine>,
}

/// `UnivariateProverKey` is used to generate a proof
#[derive(Clone, Debug)]
pub struct KZGProverKey<P: Pairing> {
  /// generators
  pub g1_powers: Vec<P::G1Affine>,
}

/// `UVKZGVerifierKey` is used to check evaluation proofs for a given
/// commitment.
pub struct KZGVerifierKey<P: Pairing> {
  /// The generator of G1.
  pub g1: P::G1Affine,
  /// The generator of G2.
  pub g2: P::G2Affine,
  /// tau times the above generator of G2.
  pub tau_2: P::G2Affine,
}

impl<P: Pairing> UNIVERSAL_KZG_SRS<P> {
  /// Generates a new srs
  pub fn setup<R: RngCore>(
    toxic_waste: Option<&[u8]>,
    max_degree: usize,
    mut rng: &mut R,
  ) -> UNIVERSAL_KZG_SRS<P> {
    let tau = P::ScalarField::rand(rng);
    let g1 = P::G1::rand(rng);
    let g2 = P::G2::rand(rng);

    let tau_powers: Vec<_> = (0..=max_degree)
      .scan(tau, |state, _| {
        let val = *state;
        *state *= &tau;
        Some(val)
      })
      .collect();

    let window_size = FixedBase::get_mul_window_size(max_degree);
    let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;

    //TODO: gate with rayon

    let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);
    let g2_table = FixedBase::get_window_table(scalar_bits, window_size, g2);
    let g1_powers_projective = FixedBase::msm(scalar_bits, window_size, &g1_table, &tau_powers);
    let g2_powers_projective = FixedBase::msm(scalar_bits, window_size, &g2_table, &tau_powers);
    let g1_powers = P::G1::normalize_batch(&g1_powers_projective);
    let g2_powers = P::G2::normalize_batch(&g2_powers_projective);

    UNIVERSAL_KZG_SRS {
      g1_powers,
      g2_powers,
    }
  }
  /// Returns the maximum supported degree
  pub fn max_degree(&self) -> usize {
    self.g1_powers.len()
  }

  /// Returns the prover parameters
  ///
  /// # Panics
  /// if `supported_size` is greater than `self.max_degree()`
  pub fn extract_prover_key(&self, supported_size: usize) -> KZGProverKey<P> {
    let g1_powers = self.g1_powers[..=supported_size].to_vec();
    KZGProverKey { g1_powers }
  }

  /// Returns the verifier parameters
  ///
  /// # Panics
  /// If self.prover_params is empty.
  pub fn extract_verifier_key(&self, supported_size: usize) -> KZGVerifierKey<P> {
    assert!(
      self.g1_powers.len() >= supported_size,
      "supported_size is greater than self.max_degree()"
    );
    KZGVerifierKey {
      g1: self.g1_powers[0],
      g2: self.g2_powers[0],
      tau_2: self.g2_powers[1],
    }
  }

  /// Trim the universal parameters to specialize the public parameters
  /// for univariate polynomials to the given `supported_size`, and
  /// returns prover key and verifier key. `supported_size` should
  /// be in range `1..params.len()`
  ///
  /// # Panics
  /// If `supported_size` is greater than `self.max_degree()`, or `self.max_degree()` is zero.
  pub fn get_pk_vk(&self, supported_size: usize) -> (KZGProverKey<P>, KZGVerifierKey<P>) {
    let g1_powers = self.g1_powers[..=supported_size].to_vec();

    let pk = KZGProverKey { g1_powers };
    let vk = KZGVerifierKey {
      g1: self.g1_powers[0],
      g2: self.g2_powers[0],
      tau_2: self.g2_powers[1],
    };
    (pk, vk)
  }
}

/// Commitments

/// Polynomial Evaluation

#[derive(Error, Debug)]
pub enum KZGError {
  #[error("length error")]
  LengthError,
  #[error("Failed to compute quotient polynomial due to polynomial division")]
  PolyDivisionError,
}

/// KZG Polynomial Commitment Scheme on univariate polynomial.
/// Note: this is non-hiding, which is why we will implement traits on this token struct,
/// as we expect to have several impls for the trait pegged on the same instance of a pairing::Engine.
pub struct UVKZGPCS<P> {
  phantom: PhantomData<P>,
}

impl<P: Pairing> UVKZGPCS<P> {
  pub fn commit_offset(
    prover_param: impl Borrow<KZGProverKey<P>>,
    poly: &UniPoly<P::ScalarField>,
    offset: usize,
  ) -> Result<P::G1Affine, KZGError> {
    let prover_param = prover_param.borrow();

    if poly.degree() > prover_param.g1_powers.len() {
      return Err(KZGError::LengthError);
    }

    let scalars = poly.coeffs.as_slice();
    let bases = prover_param.g1_powers.as_slice();

    // We can avoid some scalar multiplications if 'scalars' contains a lot of leading zeroes using
    // offset, that points where non-zero scalars start.
    let C = <P::G1 as VariableBaseMSM>::msm(&bases[offset..scalars.len()], &poly.coeffs[offset..])
      .unwrap();

    Ok(C.into_affine())
  }

  /// Generate a commitment for a polynomial
  /// Note that the scheme is not hidding
  pub fn commit(
    prover_param: impl Borrow<KZGProverKey<P>>,
    poly: &UniPoly<P::ScalarField>,
  ) -> Result<P::G1Affine, KZGError> {
    let prover_param = prover_param.borrow();

    if poly.degree() > prover_param.g1_powers.len() {
      return Err(KZGError::LengthError);
    }
    let C = <P::G1 as VariableBaseMSM>::msm(
      &prover_param.g1_powers.as_slice()[..poly.coeffs.len()],
      poly.coeffs.as_slice(),
    )
    .unwrap();
    Ok(C.into_affine())
  }

  /// On input a polynomial `p` and a point `point`, outputs a proof for the
  /// same.
  pub fn open(
    prover_param: impl Borrow<KZGProverKey<P>>,
    polynomial: &UniPoly<P::ScalarField>,
    point: &P::ScalarField,
  ) -> Result<(P::G1Affine, P::ScalarField), KZGError> {
    let prover_param = prover_param.borrow();
    let divisor = UniPoly {
      coeffs: vec![-*point, P::ScalarField::one()],
    };
    let witness_polynomial = polynomial
      .divide_with_q_and_r(&divisor)
      .map(|(q, _r)| q)
      .ok_or(KZGError::PolyDivisionError)?;
    let proof = <P::G1 as VariableBaseMSM>::msm(
      &prover_param.g1_powers.as_slice()[..witness_polynomial.coeffs.len()],
      witness_polynomial.coeffs.as_slice(),
    )
    .unwrap();
    let evaluation = polynomial.evaluate(point);

    Ok((proof.into_affine(), evaluation))
  }

  /// Verifies that `value` is the evaluation at `x` of the polynomial
  /// committed inside `comm`.
  fn verify(
    vk: impl Borrow<KZGVerifierKey<P>>,
    commitment: &P::G1Affine,
    point: &P::ScalarField,
    proof: &P::G1Affine,
    evaluation: &P::ScalarField,
  ) -> Result<bool, KZGError> {
    let vk = vk.borrow();

    let lhs = P::pairing(
      commitment.into_group() - vk.g1.into_group() * evaluation,
      vk.g2,
    );
    let rhs = P::pairing(proof, vk.tau_2.into_group() - (vk.g2 * point));
    Ok(lhs == rhs)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::poly::unipoly::UniPoly;
  use ark_bn254::Bn254;
  use ark_std::{
    rand::{Rng, SeedableRng},
    UniformRand,
  };
  use rand_chacha::{rand_core::RngCore, ChaCha20Rng};

  fn random<P: Pairing, R: RngCore>(degree: usize, mut rng: &mut R) -> UniPoly<P::ScalarField> {
    let coeffs = (0..=degree)
      .map(|_| P::ScalarField::rand(&mut rng))
      .collect::<Vec<_>>();
    UniPoly::from_coeff(coeffs)
  }

  fn end_to_end_test_template<P: Pairing>() -> Result<(), KZGError> {
    let seed = b"11111111111111111111111111111111";
    for _ in 0..100 {
      let mut rng = &mut ChaCha20Rng::from_seed(*seed);
      let degree = rng.gen_range(2..20);

      let pp = UNIVERSAL_KZG_SRS::<P>::setup(None, degree, &mut rng);
      let (ck, vk) = pp.get_pk_vk(degree);
      let p = random::<P, ChaCha20Rng>(degree, rng);
      let comm = UVKZGPCS::<P>::commit(&ck, &p)?;
      let point = P::ScalarField::rand(rng);
      let (proof, value) = UVKZGPCS::<P>::open(&ck, &p, &point)?;
      assert!(
        UVKZGPCS::<P>::verify(&vk, &comm, &point, &proof, &value)?,
        "proof was incorrect for max_degree = {}, polynomial_degree = {}",
        degree,
        p.degree(),
      );
    }
    Ok(())
  }

  #[test]
  fn end_to_end_test() {
    end_to_end_test_template::<Bn254>().expect("test failed for Bn256");
  }
}
