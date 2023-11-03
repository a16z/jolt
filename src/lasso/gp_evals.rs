use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;

use crate::utils::transcript::ProofTranscript;

/// Evaluations of a Grand Product Argument for the four required sets.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct GPEvals<F: PrimeField> {
  pub hash_init: F,
  pub hash_read: F,
  pub hash_write: F,
  pub hash_final: F,
}

impl<F: PrimeField> GPEvals<F> {
  pub fn new(hash_init: F, hash_read: F, hash_write: F, hash_final: F) -> Self {
    Self {
      hash_init,
      hash_read,
      hash_write,
      hash_final,
    }
  }

  pub fn append_to_transcript<G: CurveGroup<ScalarField = F>>(&self, transcript: &mut Transcript) {
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_hash_init",
      &self.hash_init,
    );
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_hash_read",
      &self.hash_read,
    );
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_hash_write",
      &self.hash_write,
    );
    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_hash_final",
      &self.hash_final,
    );
  }

  /// Flattens a vector of GPEvals to a vector of field elements alternating between init evals and final evals.
  pub fn flatten_init_final(evals: &[Self]) -> Vec<F> {
    evals
      .iter()
      .flat_map(|eval| [eval.hash_init, eval.hash_final])
      .collect()
  }

  /// Flattens a vector of GPEvals to a vector of field elements alternating between read evals and write evals.
  pub fn flatten_read_write(evals: &[Self]) -> Vec<F> {
    evals
      .iter()
      .flat_map(|eval| [eval.hash_read, eval.hash_write])
      .collect()
  }
}