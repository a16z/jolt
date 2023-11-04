#![allow(dead_code)] // zk is not yet used

use crate::poly::commitments::{Commitments, MultiCommitGens};
use crate::utils::errors::ProofVerifyError;
use crate::utils::random::RandomTape;
use crate::utils::transcript::ProofTranscript;
use ark_ec::CurveGroup;
use ark_serialize::*;
use merlin::Transcript;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct KnowledgeProof<G: CurveGroup> {
  alpha: G,
  z1: G::ScalarField,
  z2: G::ScalarField,
}

impl<G: CurveGroup> KnowledgeProof<G> {
  fn protocol_name() -> &'static [u8] {
    b"knowledge proof"
  }

  pub fn prove(
    gens_n: &MultiCommitGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
    x: &G::ScalarField,
    r: &G::ScalarField,
  ) -> (KnowledgeProof<G>, G) {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      KnowledgeProof::<G>::protocol_name(),
    );

    // produce two random Fs
    let t1 = random_tape.random_scalar(b"t1");
    let t2 = random_tape.random_scalar(b"t2");

    let C = x.commit(r, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"C", &C);

    let alpha = t1.commit(&t2, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"alpha", &alpha);

    let c = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");

    let z1 = *x * c + t1;
    let z2 = *r * c + t2;

    (KnowledgeProof { alpha, z1, z2 }, C)
  }

  pub fn verify(
    &self,
    gens_n: &MultiCommitGens<G>,
    transcript: &mut Transcript,
    C: &G,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      KnowledgeProof::<G>::protocol_name(),
    );

    <Transcript as ProofTranscript<G>>::append_point(transcript, b"C", C);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"alpha", &self.alpha);

    let c = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");

    let lhs = self.z1.commit(&self.z2, gens_n);
    let rhs = *C * c + self.alpha;

    (lhs == rhs)
      .then_some(())
      .ok_or(ProofVerifyError::InternalError)
  }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct EqualityProof<G: CurveGroup> {
  alpha: G,
  z: G::ScalarField,
}

impl<G: CurveGroup> EqualityProof<G> {
  fn protocol_name() -> &'static [u8] {
    b"equality proof"
  }

  pub fn prove(
    gens_n: &MultiCommitGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
    v1: &G::ScalarField,
    s1: &G::ScalarField,
    v2: &G::ScalarField,
    s2: &G::ScalarField,
  ) -> (Self, G, G) {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      EqualityProof::<G>::protocol_name(),
    );

    // produce a random F
    let r = random_tape.random_scalar(b"r");

    let C1 = v1.commit(s1, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"C1", &C1);

    let C2 = v2.commit(s2, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"C2", &C2);

    let alpha = gens_n.h * r;

    <Transcript as ProofTranscript<G>>::append_point(transcript, b"alpha", &alpha);

    let c = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");

    let z = c * (*s1 - *s2) + r;

    (EqualityProof { alpha, z }, C1, C2)
  }

  pub fn verify(
    &self,
    gens_n: &MultiCommitGens<G>,
    transcript: &mut Transcript,
    C1: &G,
    C2: &G,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      EqualityProof::<G>::protocol_name(),
    );

    <Transcript as ProofTranscript<G>>::append_point(transcript, b"C1", C1);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"C2", C2);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"alpha", &self.alpha);

    let c = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");

    let rhs = {
      let C = *C1 - *C2;
      C * c + self.alpha
    };

    let lhs = gens_n.h * self.z;

    if lhs == rhs {
      Ok(())
    } else {
      Err(ProofVerifyError::InternalError)
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProductProof<G: CurveGroup> {
  alpha: G,
  beta: G,
  delta: G,
  z: [G::ScalarField; 5],
}

impl<G: CurveGroup> ProductProof<G> {
  fn protocol_name() -> &'static [u8] {
    b"product proof"
  }

  pub fn prove(
    gens_n: &MultiCommitGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
    x: &G::ScalarField,
    rX: &G::ScalarField,
    y: &G::ScalarField,
    rY: &G::ScalarField,
    z: &G::ScalarField,
    rZ: &G::ScalarField,
  ) -> (Self, G, G, G) {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      ProductProof::<G>::protocol_name(),
    );

    // produce five random F
    let b1 = random_tape.random_scalar(b"b1");
    let b2 = random_tape.random_scalar(b"b2");
    let b3 = random_tape.random_scalar(b"b3");
    let b4 = random_tape.random_scalar(b"b4");
    let b5 = random_tape.random_scalar(b"b5");

    let X = x.commit(rX, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"X", &X);

    let Y = y.commit(rY, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"Y", &Y);

    let Z = z.commit(rZ, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"Z", &Z);

    let alpha = b1.commit(&b2, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"alpha", &alpha);

    let beta = b3.commit(&b4, gens_n);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"beta", &beta);

    let delta = {
      let gens_X = &MultiCommitGens {
        n: 1,
        G: vec![X],
        h: gens_n.h,
      };
      b3.commit(&b5, gens_X)
    };
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"delta", &delta);

    let c = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");

    let z1 = b1 + c * x;
    let z2 = b2 + c * rX;
    let z3 = b3 + c * y;
    let z4 = b4 + c * rY;
    let z5 = b5 + c * (*rZ - *rX * *y);
    let z = [z1, z2, z3, z4, z5];

    (
      ProductProof {
        alpha,
        beta,
        delta,
        z,
      },
      X,
      Y,
      Z,
    )
  }

  fn check_equality(
    P: &G,
    X: &G,
    c: &G::ScalarField,
    gens_n: &MultiCommitGens<G>,
    z1: &G::ScalarField,
    z2: &G::ScalarField,
  ) -> bool {
    let lhs = *P + *X * *c;
    let rhs = z1.commit(z2, gens_n);

    lhs == rhs
  }

  pub fn verify(
    &self,
    gens_n: &MultiCommitGens<G>,
    transcript: &mut Transcript,
    X: &G,
    Y: &G,
    Z: &G,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(
      transcript,
      ProductProof::<G>::protocol_name(),
    );

    <Transcript as ProofTranscript<G>>::append_point(transcript, b"X", X);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"Y", Y);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"Z", Z);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"alpha", &self.alpha);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"beta", &self.beta);
    <Transcript as ProofTranscript<G>>::append_point(transcript, b"delta", &self.delta);

    let z1 = self.z[0];
    let z2 = self.z[1];
    let z3 = self.z[2];
    let z4 = self.z[3];
    let z5 = self.z[4];

    let c = <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"c");

    if ProductProof::check_equality(&self.alpha, X, &c, gens_n, &z1, &z2)
      && ProductProof::check_equality(&self.beta, Y, &c, gens_n, &z3, &z4)
      && ProductProof::check_equality(
        &self.delta,
        Z,
        &c,
        &MultiCommitGens {
          n: 1,
          G: vec![*X],
          h: gens_n.h,
        },
        &z3,
        &z5,
      )
    {
      Ok(())
    } else {
      Err(ProofVerifyError::InternalError)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_curve25519::EdwardsProjective as G1Projective;
  use ark_std::test_rng;
  use ark_std::UniformRand;

  #[test]
  fn check_knowledgeproof() {
    check_knowledgeproof_helper::<G1Projective>()
  }

  fn check_knowledgeproof_helper<G: CurveGroup>() {
    let mut prng = test_rng();

    let gens_1 = MultiCommitGens::<G>::new(1, b"test-knowledgeproof");

    let x = G::ScalarField::rand(&mut prng);
    let r = G::ScalarField::rand(&mut prng);

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let (proof, committed_value) =
      KnowledgeProof::<G>::prove(&gens_1, &mut prover_transcript, &mut random_tape, &x, &r);

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&gens_1, &mut verifier_transcript, &committed_value)
      .is_ok());
  }

  #[test]
  fn check_equalityproof() {
    check_equalityproof_helper::<G1Projective>()
  }

  fn check_equalityproof_helper<G: CurveGroup>() {
    let mut prng = test_rng();

    let gens_1 = MultiCommitGens::<G>::new(1, b"test-equalityproof");
    let v1 = G::ScalarField::rand(&mut prng);
    let v2 = v1;
    let s1 = G::ScalarField::rand(&mut prng);
    let s2 = G::ScalarField::rand(&mut prng);

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let (proof, C1, C2) = EqualityProof::prove(
      &gens_1,
      &mut prover_transcript,
      &mut random_tape,
      &v1,
      &s1,
      &v2,
      &s2,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&gens_1, &mut verifier_transcript, &C1, &C2)
      .is_ok());
  }

  #[test]
  fn check_productproof() {
    check_productproof_helper::<G1Projective>()
  }

  fn check_productproof_helper<G: CurveGroup>() {
    let mut prng = test_rng();

    let gens_1 = MultiCommitGens::<G>::new(1, b"test-productproof");
    let x = G::ScalarField::rand(&mut prng);
    let rX = G::ScalarField::rand(&mut prng);
    let y = G::ScalarField::rand(&mut prng);
    let rY = G::ScalarField::rand(&mut prng);
    let z = x * y;
    let rZ = G::ScalarField::rand(&mut prng);

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let (proof, X, Y, Z) = ProductProof::prove(
      &gens_1,
      &mut prover_transcript,
      &mut random_tape,
      &x,
      &rX,
      &y,
      &rY,
      &z,
      &rZ,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&gens_1, &mut verifier_transcript, &X, &Y, &Z)
      .is_ok());
  }
}
