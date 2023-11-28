use std::borrow::Borrow;

use ark_std::iterable::Iterable;
use merlin::Transcript;

use crate::utils::transcript;

pub trait CommitmentScheme {
  type Commitment;
  type Evaluation;
  type Polynomial;
  type Challenge;
  type Proof;
  type Error;

  type ProverKey;
  type VerifierKey;

  //TODO: convert to impl IntoIterator<Item = Self::Polynomial>
  fn commit(
    polys: &[Self::Polynomial],
    ck: &Self::ProverKey,
  ) -> Result<Vec<Self::Commitment>, Self::Error>;

  fn prove(
    polys: &[Self::Polynomial],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    pk: impl Borrow<Self::ProverKey>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error>;

  fn verify(
    commitments: &[Self::Commitment],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    vk: impl Borrow<Self::VerifierKey>,
    transcript: &mut Transcript,
    proof: Self::Proof,
  ) -> Result<bool, Self::Error>;
}
