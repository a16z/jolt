use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use merlin::Transcript;

use crate::transcript::ProofTranscript;

/// Wrapper around merlin_transcript that allows overriding
pub struct TestTranscript<F> {
  pub merlin_transcript: Transcript,

  pub scalars: Vec<F>,
  pub scalar_index: usize,

  pub vecs: Vec<Vec<F>>,
  pub vec_index: usize,
}

impl<F: PrimeField> TestTranscript<F> {
  pub fn new(scalar_responses: Vec<F>, vec_responses: Vec<Vec<F>>) -> Self {
    Self {
      merlin_transcript: Transcript::new(b"transcript"),
      scalars: scalar_responses,
      scalar_index: 0,
      vecs: vec_responses,
      vec_index: 0,
    }
  }
}

impl<G: CurveGroup> ProofTranscript<G> for TestTranscript<G::ScalarField> {
  fn challenge_scalar(&mut self, _label: &'static [u8]) -> G::ScalarField {
    assert!(self.scalar_index < self.scalars.len());

    let res = self.scalars[self.scalar_index];
    self.scalar_index += 1;
    res
  }

  fn challenge_vector(&mut self, _label: &'static [u8], len: usize) -> Vec<G::ScalarField> {
    assert!(self.vec_index < self.vecs.len());

    let res = self.vecs[self.vec_index].clone();

    assert_eq!(res.len(), len);

    self.vec_index += 1;
    res
  }

  // The following match impl ProofTranscript for Transcript, but do not affect challenge responses

  fn append_message(&mut self, label: &'static [u8], msg: &'static [u8]) {
    self.merlin_transcript.append_message(label, msg);
  }

  fn append_u64(&mut self, label: &'static [u8], x: u64) {
    self.merlin_transcript.append_u64(label, x);
  }

  fn append_protocol_name(&mut self, protocol_name: &'static [u8]) {
    self
      .merlin_transcript
      .append_message(b"protocol-name", protocol_name);
  }

  fn append_scalar(&mut self, label: &'static [u8], scalar: &G::ScalarField) {
    let mut buf = vec![];
    scalar.serialize_compressed(&mut buf).unwrap();
    self.merlin_transcript.append_message(label, &buf);
  }

  fn append_scalars(&mut self, label: &'static [u8], scalars: &[G::ScalarField]) {
    self
      .merlin_transcript
      .append_message(label, b"begin_append_vector");
    for item in scalars.iter() {
      <Self as ProofTranscript<G>>::append_scalar(self, label, item);
    }
    self
      .merlin_transcript
      .append_message(label, b"end_append_vector");
  }

  fn append_point(&mut self, label: &'static [u8], point: &G) {
    let mut buf = vec![];
    point.serialize_compressed(&mut buf).unwrap();
    self.merlin_transcript.append_message(label, &buf);
  }

  fn append_points(&mut self, label: &'static [u8], points: &[G]) {
    self
      .merlin_transcript
      .append_message(label, b"begin_append_vector");
    for item in points.iter() {
      self.merlin_transcript.append_point(label, item);
    }
    self
      .merlin_transcript
      .append_message(label, b"end_append_vector");
  }
}

#[cfg(test)]
mod test {
  use super::{ProofTranscript, TestTranscript};
  use ark_bls12_381::{Fr, G1Projective};
  use ark_ec::CurveGroup;
  use ark_ff::PrimeField;

  #[test]
  fn test_transcript() {
    let scalars = vec![Fr::from(10), Fr::from(20), Fr::from(30)];
    let vecs = vec![
      vec![Fr::from(10), Fr::from(20), Fr::from(30)],
      vec![Fr::from(40), Fr::from(50), Fr::from(60)],
    ];
    let mut transcript = TestTranscript::new(scalars.clone(), vecs);

    verify_scalars::<G1Projective, Fr, _>(&mut transcript, scalars);
  }

  fn verify_scalars<G: CurveGroup, F: PrimeField, T: ProofTranscript<G>>(
    transcript: &mut T,
    scalars: Vec<G::ScalarField>,
  ) {
    for scalar in scalars {
      let challenge: G::ScalarField = transcript.challenge_scalar(b"oi-mate");
      assert_eq!(challenge, scalar);
    }
  }

  fn verify_vecs<G: CurveGroup, F: PrimeField, T: ProofTranscript<G>>(
    transcript: &mut T,
    vecs: Vec<Vec<G::ScalarField>>,
  ) {
    for vec in vecs {
      let challenge: Vec<G::ScalarField> = transcript.challenge_vector(b"ahoy-mate", vec.len());
      assert_eq!(challenge, vec);
    }
  }
}
