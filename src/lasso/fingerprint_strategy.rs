use ark_ec::CurveGroup;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use merlin::Transcript;

use crate::{utils::{random::RandomTape, errors::ProofVerifyError}, poly::dense_mlpoly::DensePolynomial};

use super::memory_checking::GPEvals;

/// Trait which defines a strategy for creating opening proofs for multi-set fingerprints and verifies.
pub trait FingerprintStrategy<G: CurveGroup>:
  std::marker::Sync + CanonicalSerialize + CanonicalDeserialize
{
  type Polynomials;
  type Generators;
  type Commitments;

  fn prove(
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    polynomials: &Self::Polynomials,
    generators: &Self::Generators,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self;

  // TODO(sragss): simplify signature
  fn verify<F1: Fn(usize) -> usize, F2: Fn(usize, &[G::ScalarField]) -> G::ScalarField>(
    &self,
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    grand_product_claims: &[GPEvals<G::ScalarField>], // NUM_MEMORIES-sized
    memory_to_dimension_index: F1,
    evaluate_memory_mle: F2,
    commitments: &Self::Commitments,
    generators: &Self::Generators,
    r_hash: &G::ScalarField,
    r_multiset_check: &G::ScalarField,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError>;

  fn num_ops(polys: &Self::Polynomials) -> usize;
  fn num_memories(polys: &Self::Polynomials) -> usize;
  fn memory_size(polys: &Self::Polynomials) -> usize;

  // TODO(sragss): Move flags to grand products with a GrandProductCircuitLayer trait. Remove these from interfaces.
  // fn get_flags(polys: &Self::Polynomials) -> Option<Vec<DensePolynomial<G::ScalarField>>>;
}