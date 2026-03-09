//! Mock polynomial commitment scheme for testing.
//!
//! [`MockCommitmentScheme`] implements [`CommitmentScheme`],
//! [`AdditivelyHomomorphic`], and [`ZkOpeningScheme`] with true additive
//! homomorphism. It is intended solely for testing reduction and trait logic.

use std::marker::PhantomData;

use jolt_crypto::Commitment;
use jolt_field::Field;
use jolt_poly::Polynomial;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Serialize};

use jolt_crypto::HomomorphicCommitment;

use crate::error::OpeningsError;
use crate::traits::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};

/// A trivial commitment scheme for testing infrastructure.
///
/// - **Commitment**: stores the full evaluation table.
/// - **Proof**: the full evaluation table, allowing the verifier to re-evaluate.
///
/// Truly homomorphic: `combine` computes the linear combination of evaluations.
/// Provides no hiding or soundness guarantees.
#[derive(Clone, Debug)]
pub struct MockCommitmentScheme<F: Field>(PhantomData<F>);

/// Mock commitment storing the full evaluation table.
///
/// This allows `combine` to compute the actual linear combination of
/// polynomials, making the mock truly additively homomorphic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockCommitment<F: Field> {
    evaluations: Vec<F>,
}

/// Mock proof: carries the full evaluation table for re-evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockProof<F: Field> {
    evaluations: Vec<F>,
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

    fn commit(evaluations: &[Self::Field], _setup: &Self::ProverSetup) -> (Self::Output, ()) {
        (
            MockCommitment {
                evaluations: evaluations.to_vec(),
            },
            (),
        )
    }

    fn open(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<()>,
        _transcript: &mut impl Transcript,
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
        _transcript: &mut impl Transcript,
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

/// Mock eval commitment: wraps the cleartext evaluation for testing.
///
/// In a real ZK scheme this would be a hiding commitment (e.g., Pedersen).
/// For testing, we just store the value directly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MockEvalCommitment<F: Field> {
    /// The evaluation value (stored in the clear for testing).
    pub eval: F,
}

impl<F: Field> AppendToTranscript for MockEvalCommitment<F> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.eval.append_to_transcript(transcript);
    }
}

impl<F: Field> ZkOpeningScheme for MockCommitmentScheme<F> {
    type EvalCommitment = MockEvalCommitment<F>;
    type EvalBlinding = ();

    fn open_zk(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        _transcript: &mut impl Transcript,
    ) -> (Self::Proof, Self::EvalCommitment, Self::EvalBlinding) {
        let proof = MockProof {
            evaluations: poly.evaluations().to_vec(),
        };
        let eval_commitment = MockEvalCommitment { eval };
        (proof, eval_commitment, ())
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval_commitment: &Self::EvalCommitment,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript,
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

    fn extract_eval_commitment(proof: &Self::Proof) -> Option<Self::EvalCommitment> {
        // Mock always returns Some — real schemes would check a flag or enum variant.
        let _ = proof;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OpeningReduction, ProverClaim, RlcReduction, VerifierClaim};
    use jolt_field::Field;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    type MockPCS = MockCommitmentScheme<Fr>;

    fn challenge_fn(c: u128) -> Fr {
        Fr::from_u128(c)
    }

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

    /// Builds prover and verifier claims, reduces via RLC, opens/verifies reduced claims.
    fn prove_and_verify(
        prover_polys: &[(Polynomial<Fr>, Vec<Fr>)],
        verifier_evals: Option<&[Fr]>,
    ) -> Result<(), OpeningsError> {
        let mut prover_claims = Vec::new();
        let mut verifier_claims = Vec::new();

        for (i, (poly, point)) in prover_polys.iter().enumerate() {
            let eval = poly.evaluate(point);
            prover_claims.push(ProverClaim {
                evaluations: poly.evaluations().to_vec(),
                point: point.clone(),
                eval,
            });

            let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());
            let v_eval = verifier_evals.map_or(eval, |overrides| overrides[i]);
            verifier_claims.push(VerifierClaim {
                commitment,
                point: point.clone(),
                eval: v_eval,
            });
        }

        // Prover: reduce + open
        let mut transcript_p = Blake2bTranscript::new(b"e2e-test");
        let (reduced_prover, ()) = <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(
            prover_claims,
            &mut transcript_p,
            challenge_fn,
        );
        let proofs: Vec<_> = reduced_prover
            .iter()
            .map(|claim| {
                let poly: Polynomial<Fr> = claim.evaluations.clone().into();
                MockPCS::open(
                    &poly,
                    &claim.point,
                    claim.eval,
                    &(),
                    None,
                    &mut transcript_p,
                )
            })
            .collect();

        // Verifier: reduce + verify
        let mut transcript_v = Blake2bTranscript::new(b"e2e-test");
        let reduced_verifier = <RlcReduction as OpeningReduction<MockPCS>>::reduce_verifier(
            verifier_claims,
            &(),
            &mut transcript_v,
            challenge_fn,
        )?;

        assert_eq!(reduced_verifier.len(), proofs.len());

        for (claim, proof) in reduced_verifier.iter().zip(proofs.iter()) {
            MockPCS::verify(
                &claim.commitment,
                &claim.point,
                claim.eval,
                proof,
                &(),
                &mut transcript_v,
            )?;
        }

        Ok(())
    }

    #[test]
    fn e2e_single_claim_roundtrip() {
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        let point: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();

        prove_and_verify(&[(poly, point)], None).expect("single claim e2e should verify");
    }

    #[test]
    fn e2e_multiple_claims_shared_and_distinct_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let num_vars = 3;

        let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_c = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_d = Polynomial::<Fr>::random(num_vars, &mut rng);

        let point1: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point2: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let claims = vec![
            (poly_a, point1.clone()),
            (poly_b, point1),
            (poly_c, point2.clone()),
            (poly_d, point2),
        ];

        prove_and_verify(&claims, None).expect("multi-claim e2e should verify");
    }

    #[test]
    fn e2e_rejects_tampered_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let num_vars = 3;

        let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_c = Polynomial::<Fr>::random(num_vars, &mut rng);

        let point1: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let point2: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let eval_a = poly_a.evaluate(&point1);
        let eval_b = poly_b.evaluate(&point1);
        let eval_c = poly_c.evaluate(&point2);

        let tampered_evals = [eval_a, eval_b + Fr::from_u64(1), eval_c];

        let claims = vec![(poly_a, point1.clone()), (poly_b, point1), (poly_c, point2)];

        let result = prove_and_verify(&claims, Some(&tampered_evals));
        assert!(
            result.is_err(),
            "tampered evaluation should cause rejection"
        );
    }

    #[test]
    fn reduction_groups_claims_at_same_point() {
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let nv = 3;
        let p1 = Polynomial::<Fr>::random(nv, &mut rng);
        let p2 = Polynomial::<Fr>::random(nv, &mut rng);
        let p3 = Polynomial::<Fr>::random(nv, &mut rng);
        let r: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
        let s: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();

        let claims = vec![
            ProverClaim {
                evaluations: p1.evaluations().to_vec(),
                point: r.clone(),
                eval: p1.evaluate(&r),
            },
            ProverClaim {
                evaluations: p2.evaluations().to_vec(),
                point: r.clone(),
                eval: p2.evaluate(&r),
            },
            ProverClaim {
                evaluations: p3.evaluations().to_vec(),
                point: s.clone(),
                eval: p3.evaluate(&s),
            },
        ];

        let mut transcript = Blake2bTranscript::new(b"grouping");
        let (reduced, ()) = <RlcReduction as OpeningReduction<MockPCS>>::reduce_prover(
            claims,
            &mut transcript,
            challenge_fn,
        );
        assert_eq!(reduced.len(), 2, "two distinct points → two reduced claims");
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

        let wrong_eval_com = MockEvalCommitment {
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

    #[test]
    fn extract_eval_commitment_returns_none_for_mock() {
        let proof = MockProof::<Fr> {
            evaluations: vec![Fr::from_u64(1)],
        };
        assert!(MockPCS::extract_eval_commitment(&proof).is_none());
    }
}
