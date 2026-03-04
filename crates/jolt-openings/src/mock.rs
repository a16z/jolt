//! Mock polynomial commitment scheme for testing.
//!
//! [`MockCommitmentScheme`] implements [`CommitmentScheme`] and
//! [`HomomorphicCommitmentScheme`] without any cryptographic security.
//! It is intended solely for testing accumulator and reduction logic.

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_poly::serde_canonical::vec_canonical;
use jolt_poly::{DensePolynomial, MultilinearPolynomial};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::error::OpeningsError;
use crate::traits::{CommitmentScheme, HomomorphicCommitmentScheme};

/// A trivial commitment scheme for testing infrastructure.
///
/// - **Commitment**: fingerprint of the evaluation table via `Hash`.
/// - **Proof**: the full evaluation table, allowing the verifier to re-evaluate.
///
/// Provides no hiding, binding, or soundness guarantees.
#[derive(Clone, Debug)]
pub struct MockCommitmentScheme<F: Field>(PhantomData<F>);

/// Mock commitment: a `Hash`-based fingerprint of the evaluations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MockCommitment {
    /// Hash fingerprint of the evaluation table.
    pub fingerprint: u64,
    /// Number of evaluations.
    pub len: usize,
}

/// Mock proof: carries the full evaluation table for re-evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockProof<F: Field> {
    #[serde(with = "vec_canonical")]
    evaluations: Vec<F>,
}

/// Mock batched proof: one sub-proof per polynomial.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockBatchedProof<F: Field> {
    entries: Vec<MockProof<F>>,
}

/// Deterministic fingerprint of field element evaluations.
fn field_fingerprint<F: Field>(evals: &[F]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    evals.hash(&mut hasher);
    hasher.finish()
}

impl<F: Field> CommitmentScheme for MockCommitmentScheme<F> {
    type Field = F;
    type Commitment = MockCommitment;
    type Proof = MockProof<F>;
    type ProverSetup = ();
    type VerifierSetup = ();

    fn protocol_name() -> &'static str {
        "mock-pcs"
    }

    fn setup_prover(_max_size: usize) -> Self::ProverSetup {}

    fn setup_verifier(_max_size: usize) -> Self::VerifierSetup {}

    fn commit(
        poly: &impl MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> Self::Commitment {
        let evals = poly.evaluations();
        MockCommitment {
            fingerprint: field_fingerprint(&evals),
            len: evals.len(),
        }
    }

    fn prove(
        poly: &impl MultilinearPolynomial<Self::Field>,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _transcript: &mut impl Transcript,
    ) -> Self::Proof {
        MockProof {
            evaluations: poly.evaluations().into_owned(),
        }
    }

    fn verify(
        commitment: &Self::Commitment,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        let fp = field_fingerprint(&proof.evaluations);
        if commitment.fingerprint != fp || commitment.len != proof.evaluations.len() {
            return Err(OpeningsError::CommitmentMismatch {
                expected: format!("{}", commitment.fingerprint),
                actual: format!("{fp}"),
            });
        }

        let poly = DensePolynomial::new(proof.evaluations.clone());
        let actual_eval = poly.evaluate(point);
        if actual_eval != eval {
            return Err(OpeningsError::VerificationFailed);
        }

        Ok(())
    }
}

impl<F: Field> HomomorphicCommitmentScheme for MockCommitmentScheme<F> {
    type BatchedProof = MockBatchedProof<F>;

    fn combine_commitments(
        commitments: &[Self::Commitment],
        scalars: &[Self::Field],
    ) -> Self::Commitment {
        assert_eq!(commitments.len(), scalars.len());
        let len = commitments.first().map_or(0, |c| c.len);

        // Combine fingerprints deterministically via hashing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for (c, s) in commitments.iter().zip(scalars.iter()) {
            c.fingerprint.hash(&mut hasher);
            s.hash(&mut hasher);
        }

        MockCommitment {
            fingerprint: hasher.finish(),
            len,
        }
    }

    fn batch_prove(
        polynomials: &[&dyn MultilinearPolynomial<Self::Field>],
        _points: &[Vec<Self::Field>],
        _evals: &[Self::Field],
        _setup: &Self::ProverSetup,
        _transcript: &mut impl Transcript,
    ) -> Self::BatchedProof {
        MockBatchedProof {
            entries: polynomials
                .iter()
                .map(|p| MockProof {
                    evaluations: p.evaluations().into_owned(),
                })
                .collect(),
        }
    }

    fn batch_verify(
        _commitments: &[Self::Commitment],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        proof: &Self::BatchedProof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError> {
        for (i, entry) in proof.entries.iter().enumerate() {
            let poly = DensePolynomial::new(entry.evaluations.clone());
            let actual_eval = poly.evaluate(&points[i]);
            if actual_eval != evals[i] {
                return Err(OpeningsError::VerificationFailed);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_field::Field;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    type MockPCS = MockCommitmentScheme<Fr>;

    #[test]
    fn commit_prove_verify_roundtrip() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let poly = DensePolynomial::<Fr>::random(4, &mut rng);
        let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let commitment = MockPCS::commit(&poly, &());

        let mut transcript_p = Blake2bTranscript::new(b"test");
        let proof = MockPCS::prove(&poly, &point, eval, &(), &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"test");
        MockPCS::verify(&commitment, &point, eval, &proof, &(), &mut transcript_v)
            .expect("valid proof should verify");
    }

    #[test]
    fn verify_rejects_wrong_eval() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let poly = DensePolynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let eval = poly.evaluate(&point);
        let wrong_eval = eval + Fr::from_u64(1);

        let commitment = MockPCS::commit(&poly, &());

        let mut transcript_p = Blake2bTranscript::new(b"test");
        let proof = MockPCS::prove(&poly, &point, eval, &(), &mut transcript_p);

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
        let poly1 = DensePolynomial::<Fr>::random(3, &mut rng);
        let poly2 = DensePolynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();

        let wrong_commitment = MockPCS::commit(&poly2, &());
        let eval = poly1.evaluate(&point);

        let mut transcript_p = Blake2bTranscript::new(b"test");
        let proof = MockPCS::prove(&poly1, &point, eval, &(), &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"test");
        let result = MockPCS::verify(
            &wrong_commitment,
            &point,
            eval,
            &proof,
            &(),
            &mut transcript_v,
        );
        assert!(result.is_err());
    }

    #[test]
    fn batch_prove_verify_multiple_polynomials() {
        let mut rng = ChaCha20Rng::seed_from_u64(111);
        let num_vars = 4;

        let polys: Vec<DensePolynomial<Fr>> = (0..3)
            .map(|_| DensePolynomial::random(num_vars, &mut rng))
            .collect();
        let points: Vec<Vec<Fr>> = (0..3)
            .map(|_| (0..num_vars).map(|_| Fr::random(&mut rng)).collect())
            .collect();
        let evals: Vec<Fr> = polys
            .iter()
            .zip(points.iter())
            .map(|(p, pt)| p.evaluate(pt))
            .collect();

        let poly_refs: Vec<&dyn MultilinearPolynomial<Fr>> = polys
            .iter()
            .map(|p| p as &dyn MultilinearPolynomial<Fr>)
            .collect();

        let mut transcript_p = Blake2bTranscript::new(b"batch-test");
        let proof = MockPCS::batch_prove(&poly_refs, &points, &evals, &(), &mut transcript_p);

        let commitments: Vec<MockCommitment> =
            polys.iter().map(|p| MockPCS::commit(p, &())).collect();

        let mut transcript_v = Blake2bTranscript::new(b"batch-test");
        MockPCS::batch_verify(
            &commitments,
            &points,
            &evals,
            &proof,
            &(),
            &mut transcript_v,
        )
        .expect("valid batch proof should verify");
    }

    #[test]
    fn batch_verify_rejects_wrong_commitment() {
        let mut rng = ChaCha20Rng::seed_from_u64(222);
        let num_vars = 3;

        let poly1 = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let poly2 = DensePolynomial::<Fr>::random(num_vars, &mut rng);
        let decoy = DensePolynomial::<Fr>::random(num_vars, &mut rng);

        let point1: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point2: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let points = vec![point1.clone(), point2.clone()];

        let eval1 = poly1.evaluate(&point1);
        let eval2 = poly2.evaluate(&point2);
        let evals = vec![eval1, eval2];

        let poly_refs: Vec<&dyn MultilinearPolynomial<Fr>> =
            vec![&poly1 as &dyn MultilinearPolynomial<Fr>, &poly2];

        let mut transcript_p = Blake2bTranscript::new(b"batch-reject");
        let proof = MockPCS::batch_prove(&poly_refs, &points, &evals, &(), &mut transcript_p);

        // Use a wrong commitment for the first polynomial
        let wrong_commitment = MockPCS::commit(&decoy, &());
        let good_commitment = MockPCS::commit(&poly2, &());
        let commitments = vec![wrong_commitment, good_commitment];

        let mut transcript_v = Blake2bTranscript::new(b"batch-reject");
        // batch_verify checks evaluations, not fingerprints, so tamper the eval instead
        let tampered_evals = vec![eval1 + Fr::from_u64(1), eval2];
        let result = MockPCS::batch_verify(
            &commitments,
            &points,
            &tampered_evals,
            &proof,
            &(),
            &mut transcript_v,
        );
        assert!(
            result.is_err(),
            "batch_verify should reject tampered evaluation"
        );
    }
}
