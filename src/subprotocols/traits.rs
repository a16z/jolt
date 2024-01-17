use merlin::Transcript;
use std::borrow::Borrow;

pub trait PolynomialCommitmentScheme {
  // Abstracting over Polynomial allows us to have batched and non-batched PCS
  type Polynomial;
  type Commitment;
  type Evaluation;
  type Challenge;
  type Proof;
  type Error;

  type ProverKey;
  type CommitmentKey;
  type VerifierKey;

  //TODO: convert to impl IntoIterator<Item = Self::Polynomial>
  fn commit(
    poly: Self::Polynomial,
    ck: impl Borrow<Self::CommitmentKey>,
  ) -> Result<Self::Commitment, Self::Error>;

  fn prove(
    poly: Self::Polynomial,
    evals: Self::Evaluation,
    challenges: Self::Challenge,
    pk: impl Borrow<Self::ProverKey>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error>;

  fn verify(
    commitments: Self::Commitment,
    evals: Self::Evaluation,
    challenges: Self::Challenge,
    vk: impl Borrow<Self::VerifierKey>,
    transcript: &mut Transcript,
    proof: Self::Proof,
  ) -> Result<(), Self::Error>;
}
