use std::borrow::Borrow;

use merlin::Transcript;

pub trait PCS {
  type Commitment;
  type Evaluation;
  type Polynomial;
  type CommitPolynomial;
  type Challenge;
  type Proof;
  type Error;

  type ProverKey;
  type VerifierKey;

  //TODO: convert to impl IntoIterator<Item = Self::Polynomial>
  fn commit(
    polys: &[Self::CommitPolynomial],
    ck: &Self::ProverKey,
  ) -> Result<Vec<Self::Commitment>, Self::Error>;

  fn prove(
    polys: &[Self::Polynomial],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    pk: impl Borrow<Self::ProverKey>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error>;

  fn open(
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
