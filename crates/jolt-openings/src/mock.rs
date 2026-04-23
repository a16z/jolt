//! Mock PCS for testing. Truly homomorphic, no hiding or soundness.

use std::marker::PhantomData;

use jolt_crypto::Commitment;
use jolt_field::Field;
use jolt_poly::Polynomial;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Serialize};

use jolt_crypto::HomomorphicCommitment;

use crate::backend::CommitmentBackend;
use crate::error::OpeningsError;
use crate::schemes::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use crate::verification::{
    homomorphic_prove_batch, homomorphic_verify_batch_with_backend, OpeningVerification,
};
use crate::{OpeningClaim, ProverClaim};

#[derive(Clone, Debug)]
pub struct MockCommitmentScheme<F: Field>(PhantomData<F>);

/// Stores the full evaluation table so `combine` is truly homomorphic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockCommitment<F: Field> {
    evaluations: Vec<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockProof<F: Field> {
    evaluations: Vec<F>,
}

impl<F: Field> AppendToTranscript for MockCommitment<F> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = Vec::with_capacity(self.evaluations.len() * 32);
        for e in &self.evaluations {
            buf.extend_from_slice(&e.to_bytes());
        }
        buf.reverse();
        transcript.append_bytes(&buf);
    }

    fn serialized_len(&self) -> u64 {
        (self.evaluations.len() * 32) as u64
    }
}

impl<F: Field> Commitment for MockCommitmentScheme<F> {
    type Output = MockCommitment<F>;
}

impl<F: Field> CommitmentScheme for MockCommitmentScheme<F> {
    type Field = F;
    type Proof = MockProof<F>;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Polynomial = Polynomial<F>;
    type OpeningHint = ();
    type SetupParams = ();

    fn setup(_params: Self::SetupParams) -> ((), ()) {
        ((), ())
    }

    fn verifier_setup(_prover_setup: &()) {}

    fn commit<P: jolt_poly::MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, ()) {
        let mut evaluations = Vec::with_capacity(1 << poly.num_vars());
        poly.for_each_row(poly.num_vars(), &mut |_, row| {
            evaluations.extend_from_slice(row);
        });
        (MockCommitment { evaluations }, ())
    }

    fn open(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<()>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        MockProof {
            evaluations: poly.evaluations().to_vec(),
        }
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        if commitment.evaluations != proof.evaluations {
            return Err(OpeningsError::CommitmentMismatch {
                expected: format!("len={}", commitment.evaluations.len()),
                actual: format!("len={}", proof.evaluations.len()),
            });
        }

        let poly = Polynomial::new(proof.evaluations.clone());
        let actual_eval = poly.evaluate(point);
        if actual_eval != eval {
            return Err(OpeningsError::VerificationFailed);
        }

        Ok(())
    }
}

impl<F: Field> HomomorphicCommitment<F> for MockCommitment<F> {
    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self {
        let len = c1.evaluations.len().max(c2.evaluations.len());
        let mut result = vec![F::zero(); len];
        for (i, r) in result.iter_mut().enumerate() {
            let a = c1.evaluations.get(i).copied().unwrap_or_else(F::zero);
            let b = c2.evaluations.get(i).copied().unwrap_or_else(F::zero);
            *r = a + *scalar * b;
        }
        MockCommitment {
            evaluations: result,
        }
    }
}

impl<F: Field> AdditivelyHomomorphic for MockCommitmentScheme<F> {
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        let len = commitments.first().map_or(0, |c| c.evaluations.len());

        let mut result = vec![F::zero(); len];
        for (c, &s) in commitments.iter().zip(scalars.iter()) {
            for (r, &e) in result.iter_mut().zip(c.evaluations.iter()) {
                *r += s * e;
            }
        }

        MockCommitment {
            evaluations: result,
        }
    }
}

impl<F: Field> OpeningVerification for MockCommitmentScheme<F> {
    type BatchProof = Vec<MockProof<F>>;

    fn prove_batch<T: Transcript<Challenge = F>>(
        claims: Vec<ProverClaim<F>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut T,
    ) -> (Self::BatchProof, Vec<F>) {
        homomorphic_prove_batch::<Self, _>(claims, hints, setup, transcript)
    }

    fn verify_batch_with_backend<B>(
        backend: &mut B,
        vk: &Self::VerifierSetup,
        claims: Vec<OpeningClaim<B, Self>>,
        batch_proof: &Self::BatchProof,
        transcript: &mut B::Transcript,
    ) -> Result<(), OpeningsError>
    where
        B: CommitmentBackend<Self, F = Self::Field>,
    {
        homomorphic_verify_batch_with_backend::<Self, B>(backend, vk, claims, batch_proof, transcript)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockHidingCommitment<F: Field> {
    pub eval: F,
}

impl<F: Field> AppendToTranscript for MockHidingCommitment<F> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.eval.append_to_transcript(transcript);
    }
}

impl<F: Field> ZkOpeningScheme for MockCommitmentScheme<F> {
    type HidingCommitment = MockHidingCommitment<F>;
    type Blind = ();

    fn open_zk(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let proof = MockProof {
            evaluations: poly.evaluations().to_vec(),
        };
        let eval_commitment = MockHidingCommitment { eval };
        (proof, eval_commitment, ())
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval_commitment: &Self::HidingCommitment,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        if commitment.evaluations != proof.evaluations {
            return Err(OpeningsError::CommitmentMismatch {
                expected: format!("len={}", commitment.evaluations.len()),
                actual: format!("len={}", proof.evaluations.len()),
            });
        }

        let poly = Polynomial::new(proof.evaluations.clone());
        let actual_eval = poly.evaluate(point);
        if actual_eval != eval_commitment.eval {
            return Err(OpeningsError::VerificationFailed);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OpeningVerification, ProverClaim};
    use jolt_field::Field;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    type MockPCS = MockCommitmentScheme<Fr>;

    #[test]
    fn commit_open_verify_roundtrip() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());

        let mut transcript_p = Blake2bTranscript::new(b"test");
        let proof = MockPCS::open(&poly, &point, eval, &(), None, &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"test");
        MockPCS::verify(&commitment, &point, eval, &proof, &(), &mut transcript_v)
            .expect("valid proof should verify");
    }

    #[test]
    fn verify_rejects_wrong_eval() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let poly = Polynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let wrong_eval = eval + Fr::from_u64(1);

        let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());

        let mut transcript_p = Blake2bTranscript::new(b"test");
        let proof = MockPCS::open(&poly, &point, eval, &(), None, &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"test");
        let result = MockPCS::verify(
            &commitment,
            &point,
            wrong_eval,
            &proof,
            &(),
            &mut transcript_v,
        );
        assert!(result.is_err());
    }

    #[test]
    fn verify_rejects_wrong_commitment() {
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        let poly1 = Polynomial::<Fr>::random(3, &mut rng);
        let poly2 = Polynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();

        let (wrong_commitment, ()) = MockPCS::commit(poly2.evaluations(), &());
        let eval = poly1.evaluate(&point);

        let mut transcript_p = Blake2bTranscript::new(b"test");
        let proof = MockPCS::open(&poly1, &point, eval, &(), None, &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"test");
        let result = MockPCS::verify(
            &wrong_commitment,
            &point,
            eval,
            &proof,
            &(),
            &mut transcript_v,
        );
        assert!(result.is_err(), "wrong commitment should be rejected");
    }

    #[test]
    fn combine_is_homomorphic() {
        let mut rng = ChaCha20Rng::seed_from_u64(55);
        let poly_a = Polynomial::<Fr>::random(3, &mut rng);
        let poly_b = Polynomial::<Fr>::random(3, &mut rng);

        let (ca, ()) = MockPCS::commit(poly_a.evaluations(), &());
        let (cb, ()) = MockPCS::commit(poly_b.evaluations(), &());

        let sum_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let (c_sum_direct, ()) = MockPCS::commit(&sum_evals, &());
        let c_sum_combined = MockPCS::combine(&[ca, cb], &[Fr::from_u64(1), Fr::from_u64(1)]);

        assert_eq!(c_sum_direct, c_sum_combined);
    }

    /// Smoke-tests the prover side of the new fused trait surface:
    /// `prove_batch` should produce one PCS::Proof per distinct opening
    /// point and a parallel list of binding evals (the per-group RLC-
    /// combined evaluations). Verifier-side e2e is covered by per-PCS
    /// parity tests in `jolt-verifier-backend/tests/verification_parity.rs`.
    fn prove_batch_groups_match(
        prover_polys: &[(Polynomial<Fr>, Vec<Fr>)],
        expected_groups: usize,
    ) {
        let mut prover_claims = Vec::new();
        let mut hints = Vec::new();
        for (poly, point) in prover_polys {
            let eval = poly.evaluate(point);
            prover_claims.push(ProverClaim {
                polynomial: Polynomial::new(poly.evaluations().to_vec()),
                point: point.clone(),
                eval,
            });
            hints.push(());
        }

        let mut transcript = Blake2bTranscript::new(b"prove-batch");
        let (proofs, binding_evals) =
            MockPCS::prove_batch(prover_claims, hints, &(), &mut transcript);

        assert_eq!(
            proofs.len(),
            expected_groups,
            "expected {expected_groups} per-group proofs, got {}",
            proofs.len()
        );
        assert_eq!(
            binding_evals.len(),
            expected_groups,
            "binding_evals must be parallel to proofs",
        );
    }

    #[test]
    fn prove_batch_single_claim_one_group() {
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        prove_batch_groups_match(&[(poly, point)], 1);
    }

    #[test]
    fn prove_batch_groups_by_point() {
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let nv = 3;
        let r: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
        let s: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

        let p1 = Polynomial::<Fr>::random(nv, &mut rng);
        let p2 = Polynomial::<Fr>::random(nv, &mut rng);
        let p3 = Polynomial::<Fr>::random(nv, &mut rng);
        let p4 = Polynomial::<Fr>::random(nv, &mut rng);

        prove_batch_groups_match(&[(p1, r.clone()), (p2, r), (p3, s.clone()), (p4, s)], 2);
    }

    #[test]
    fn prove_batch_empty_returns_empty() {
        let mut transcript = Blake2bTranscript::new(b"empty");
        let (proofs, evals) =
            MockPCS::prove_batch(Vec::<ProverClaim<Fr>>::new(), Vec::new(), &(), &mut transcript);
        assert!(proofs.is_empty());
        assert!(evals.is_empty());
    }

    #[test]
    fn zk_open_verify_roundtrip() {
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());

        let mut transcript_p = Blake2bTranscript::new(b"zk-test");
        let (proof, eval_com, _blinding) =
            MockPCS::open_zk(&poly, &point, eval, &(), None, &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"zk-test");
        MockPCS::verify_zk(
            &commitment,
            &point,
            &eval_com,
            &proof,
            &(),
            &mut transcript_v,
        )
        .expect("valid ZK proof should verify");
    }

    #[test]
    fn zk_verify_rejects_wrong_eval_commitment() {
        let mut rng = ChaCha20Rng::seed_from_u64(501);
        let poly = Polynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());

        let mut transcript_p = Blake2bTranscript::new(b"zk-test");
        let (proof, _eval_com, _blinding) =
            MockPCS::open_zk(&poly, &point, eval, &(), None, &mut transcript_p);

        let wrong_eval_com = MockHidingCommitment {
            eval: eval + Fr::from_u64(1),
        };

        let mut transcript_v = Blake2bTranscript::new(b"zk-test");
        let result = MockPCS::verify_zk(
            &commitment,
            &point,
            &wrong_eval_com,
            &proof,
            &(),
            &mut transcript_v,
        );
        assert!(result.is_err(), "wrong eval commitment should be rejected");
    }
}
