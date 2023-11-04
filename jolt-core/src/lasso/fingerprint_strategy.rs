use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;

use crate::{
  subprotocols::grand_product::{BGPCInterpretable, GPEvals},
  utils::{errors::ProofVerifyError, random::RandomTape},
};

/// Trait which defines a strategy for creating opening proofs for multi-set fingerprints and verifies.
pub trait FingerprintStrategy<G: CurveGroup>:
  std::marker::Sync + CanonicalSerialize + CanonicalDeserialize
{
  type Polynomials: BGPCInterpretable<G::ScalarField>;
  type Generators;
  type Commitments;

  fn prove(
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    polynomials: &Self::Polynomials,
    generators: &Self::Generators,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self;

  // TODO(JOLT-47): simplify signature
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
}
