use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;

use crate::utils::{errors::ProofVerifyError, random::RandomTape};

pub trait BatchablePolynomials {
  type Commitment;
  type BatchedPolynomials;

  fn batch(&self) -> Self::BatchedPolynomials;
  fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment;
}

pub trait StructuredOpeningProof<F, G, Polynomials>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
  Polynomials: BatchablePolynomials + ?Sized,
{
  type Openings;
  
  fn open(polynomials: &Polynomials, opening_point: &Vec<F>) -> Self::Openings;

  fn prove_openings(
    polynomials: &Polynomials::BatchedPolynomials,
    commitment: &Polynomials::Commitment,
    opening_point: &Vec<F>,
    openings: Self::Openings,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self;

  fn verify_openings(
    &self,
    commitment: &Polynomials::Commitment,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError>;
}
