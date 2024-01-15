use std::borrow::Borrow;

use ark_ff::Field;
use merlin::Transcript;

use crate::poly::dense_mlpoly::DensePolynomial;

pub trait PCS<F: Field> {
  type Commitment;
  type Evaluation;
  type Challenge;
  type Proof;
  type Error;

  type ProverKey;
  type VerifierKey;

  //TODO: convert to impl IntoIterator<Item = Self::Polynomial>
  fn commit(
    polys: &[DensePolynomial<F>],
    ck: &Self::ProverKey,
  ) -> Result<Vec<Self::Commitment>, Self::Error>;

  fn prove(
    polys: &[DensePolynomial<F>],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    pk: impl Borrow<Self::ProverKey>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error>;

  fn open(
    polys: &[DensePolynomial<F>],
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
