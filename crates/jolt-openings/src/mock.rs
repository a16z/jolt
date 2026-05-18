//! Mock PCS for testing. Truly homomorphic, no hiding or soundness.

use std::marker::PhantomData;

use jolt_crypto::Commitment;
use jolt_field::Field;
use jolt_poly::Polynomial;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Serialize};

use jolt_crypto::HomomorphicCommitment;

use crate::claims::{OpeningClaim, ProverClaim};
use crate::error::OpeningsError;
use crate::homomorphic::{homomorphic_prove_batch, homomorphic_verify_batch};
use crate::schemes::{
    AdditivelyHomomorphic, AdditivelyHomomorphicVerifier, CommitmentScheme,
    CommitmentSchemeVerifier, LinearOpeningScheme, LinearOpeningSchemeVerifier,
    VerifierSetupFromPublicParams, ZkOpeningScheme, ZkOpeningSchemeVerifier,
};
use crate::sources::{materialize_source_evaluations, CommitmentSource};

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
        let mut buf = Vec::with_capacity(self.evaluations.len() * F::NUM_BYTES);
        for e in &self.evaluations {
            let start = buf.len();
            buf.resize(start + F::NUM_BYTES, 0);
            e.to_bytes_le(&mut buf[start..]);
        }
        buf.reverse();
        transcript.append_bytes(&buf);
    }
}

impl<F: Field> Commitment for MockCommitmentScheme<F> {
    type Output = MockCommitment<F>;
}

impl<F: Field> CommitmentSchemeVerifier for MockCommitmentScheme<F> {
    type Field = F;
    type Proof = MockProof<F>;
    type BatchProof = Vec<MockProof<F>>;
    type VerifierSetup = ();

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

    fn bind_opening_inputs(
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
        _point: &[Self::Field],
        _eval: &Self::Field,
    ) {
    }

    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        homomorphic_verify_batch::<Self, _>(claims, proof, setup, transcript)
    }
}

impl<F: Field> VerifierSetupFromPublicParams for MockCommitmentScheme<F> {
    type PublicParams = ();

    fn verifier_setup_from_public_params(_params: Self::PublicParams) -> Self::VerifierSetup {}
}

impl<F: Field> CommitmentScheme for MockCommitmentScheme<F> {
    type ProverSetup = ();
    type OpeningHint = ();
    type SetupParams = ();

    fn setup(_params: Self::SetupParams) -> ((), ()) {
        ((), ())
    }

    fn prover_to_verifier_setup(_prover_setup: &()) {}

    fn commit<S: CommitmentSource<Self::Field> + ?Sized>(
        source: &S,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, ()) {
        (
            MockCommitment {
                evaluations: materialize_source_evaluations(source),
            },
            (),
        )
    }

    fn open<S>(
        poly: &S,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<()>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        MockProof {
            evaluations: materialize_source_evaluations(poly),
        }
    }

    fn prove_batch<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof
    where
        S: CommitmentSource<Self::Field>,
    {
        homomorphic_prove_batch::<Self, _, _>(claims, hints, setup, transcript)
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

impl<F: Field> AdditivelyHomomorphicVerifier for MockCommitmentScheme<F> {
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

impl<F: Field> AdditivelyHomomorphic for MockCommitmentScheme<F> {
    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        assert_eq!(hints.len(), scalars.len());
    }
}

impl<F: Field> LinearOpeningSchemeVerifier for MockCommitmentScheme<F> {}

impl<F: Field> LinearOpeningScheme for MockCommitmentScheme<F> {}

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

impl<F: Field> ZkOpeningSchemeVerifier for MockCommitmentScheme<F> {
    type HidingCommitment = MockHidingCommitment<F>;

    fn verify_zk(
        commitment: &Self::Output,
        _point: &[Self::Field],
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
        Ok(())
    }

    fn verify_batch_zk(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        Self::verify_batch(claims, proof, setup, transcript)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        Self::bind_opening_inputs(transcript, point, &hiding_commitment.eval);
    }
}

impl<F: Field> ZkOpeningScheme for MockCommitmentScheme<F> {
    type Blind = ();

    fn commit_zk<S: CommitmentSource<Self::Field> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit(source, setup)
    }

    fn open_zk<S>(
        poly: &S,
        _point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Self::OpeningHint,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        let proof = MockProof {
            evaluations: materialize_source_evaluations(poly),
        };
        let eval_commitment = MockHidingCommitment { eval };
        (proof, eval_commitment, ())
    }

    fn prove_batch_zk<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::BatchProof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field>,
    {
        let eval = claims.first().map_or_else(F::zero, |claim| claim.eval);
        let proof = Self::prove_batch(claims, hints, setup, transcript);
        (proof, MockHidingCommitment { eval }, ())
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use crate::{
        BatchOpeningPoint, BatchOpeningSource, BatchOutputExpression, CommitmentSource,
        LinearCombinationOpeningSource, MaterializedLinearCombination, OneHotEntries, OneHotIndex,
        OneHotRow, OpeningClaim, ProverBatchOpeningTerm, ProverClaim, SourceRow,
        VerifierBatchOpeningTerm,
    };
    use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
    use jolt_poly::{MultilinearPoly, Polynomial};
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

    #[test]
    fn strided_source_rows_materialize_skipped_columns() {
        struct TestSource {
            rows: Vec<Vec<u64>>,
            dense: Polynomial<Fr>,
        }

        impl CommitmentSource<Fr> for TestSource {
            fn num_vars(&self) -> usize {
                self.dense.num_vars()
            }

            fn evaluate(&self, point: &[Fr]) -> Fr {
                self.dense.evaluate(point)
            }

            fn natural_chunk_len(&self) -> Option<usize> {
                Some(8)
            }

            fn for_each_row<V>(&self, _chunk_len: usize, mut visit: V)
            where
                V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
            {
                for (row_index, row) in self.rows.iter().enumerate() {
                    visit(
                        row_index,
                        SourceRow::StridedU64 {
                            values: row,
                            column_stride: 4,
                        },
                    );
                }
            }

            fn fold_rows(&self, left: &[Fr], chunk_len: usize) -> Vec<Fr> {
                let sigma = chunk_len.trailing_zeros() as usize;
                MultilinearPoly::fold_rows(&self.dense, left, sigma)
            }
        }

        let rows = vec![vec![3, 5], vec![7, 11]];
        let mut dense = vec![Fr::from_u64(0); 16];
        dense[0] = Fr::from_u64(3);
        dense[4] = Fr::from_u64(5);
        dense[8] = Fr::from_u64(7);
        dense[12] = Fr::from_u64(11);

        let source = TestSource {
            rows,
            dense: Polynomial::new(dense.clone()),
        };

        let (commitment, ()) = MockPCS::commit(&source, &());
        assert_eq!(commitment.evaluations, dense);
    }

    #[test]
    fn one_hot_source_rows_materialize_hot_coordinate_major() {
        struct TestSource {
            entries: Vec<Option<OneHotIndex>>,
            dense: Polynomial<Fr>,
        }

        impl CommitmentSource<Fr> for TestSource {
            fn num_vars(&self) -> usize {
                self.dense.num_vars()
            }

            fn evaluate(&self, point: &[Fr]) -> Fr {
                self.dense.evaluate(point)
            }

            fn natural_chunk_len(&self) -> Option<usize> {
                Some(self.entries.len())
            }

            fn for_each_row<V>(&self, _chunk_len: usize, mut visit: V)
            where
                V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
            {
                visit(
                    0,
                    SourceRow::OneHot(OneHotRow {
                        log_domain_size: 2,
                        entries: OneHotEntries::MaybeZero(&self.entries),
                    }),
                );
            }

            fn fold_rows(&self, left: &[Fr], chunk_len: usize) -> Vec<Fr> {
                let sigma = chunk_len.trailing_zeros() as usize;
                MultilinearPoly::fold_rows(&self.dense, left, sigma)
            }
        }

        let entries = vec![
            Some(OneHotIndex::new(1, 2).expect("valid index")),
            None,
            Some(OneHotIndex::new(3, 2).expect("valid index")),
            None,
        ];
        let mut dense = vec![Fr::from_u64(0); 16];
        dense[4] = Fr::from_u64(1);
        dense[14] = Fr::from_u64(1);

        let source = TestSource {
            entries,
            dense: Polynomial::new(dense.clone()),
        };

        let (commitment, ()) = MockPCS::commit(&source, &());
        assert_eq!(commitment.evaluations, dense);
    }

    #[test]
    fn one_hot_source_rows_materialize_multi_row_hot_coordinate_major() {
        struct TestSource {
            chunks: Vec<Vec<Option<OneHotIndex>>>,
            dense: Polynomial<Fr>,
        }

        impl CommitmentSource<Fr> for TestSource {
            fn num_vars(&self) -> usize {
                self.dense.num_vars()
            }

            fn evaluate(&self, point: &[Fr]) -> Fr {
                self.dense.evaluate(point)
            }

            fn natural_chunk_len(&self) -> Option<usize> {
                self.chunks.first().map(Vec::len)
            }

            fn for_each_row<V>(&self, _chunk_len: usize, mut visit: V)
            where
                V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
            {
                for (row_index, chunk) in self.chunks.iter().enumerate() {
                    visit(
                        row_index,
                        SourceRow::OneHot(OneHotRow {
                            log_domain_size: 2,
                            entries: OneHotEntries::MaybeZero(chunk),
                        }),
                    );
                }
            }

            fn fold_rows(&self, left: &[Fr], chunk_len: usize) -> Vec<Fr> {
                let sigma = chunk_len.trailing_zeros() as usize;
                MultilinearPoly::fold_rows(&self.dense, left, sigma)
            }
        }

        let chunks = vec![
            vec![
                Some(OneHotIndex::new(1, 2).expect("valid index")),
                None,
                Some(OneHotIndex::new(3, 2).expect("valid index")),
                None,
            ],
            vec![
                Some(OneHotIndex::new(0, 2).expect("valid index")),
                Some(OneHotIndex::new(2, 2).expect("valid index")),
                None,
                Some(OneHotIndex::new(1, 2).expect("valid index")),
            ],
        ];
        let mut dense = vec![Fr::from_u64(0); 32];
        dense[4] = Fr::from_u64(1);
        dense[8] = Fr::from_u64(1);
        dense[15] = Fr::from_u64(1);
        dense[21] = Fr::from_u64(1);
        dense[26] = Fr::from_u64(1);

        let source = TestSource {
            chunks,
            dense: Polynomial::new(dense.clone()),
        };

        let (commitment, ()) = MockPCS::commit(&source, &());
        assert_eq!(commitment.evaluations, dense);
    }

    fn prove_and_verify(
        prover_polys: &[(Polynomial<Fr>, Vec<Fr>)],
        verifier_evals: Option<&[Fr]>,
    ) -> Result<(), OpeningsError> {
        let mut prover_claims = Vec::new();
        let mut verifier_claims = Vec::new();

        for (i, (poly, point)) in prover_polys.iter().enumerate() {
            let eval = poly.evaluate(point);
            prover_claims.push(ProverClaim {
                polynomial: Polynomial::new(poly.evaluations().to_vec()),
                point: point.clone(),
                eval,
            });

            let (commitment, ()) = MockPCS::commit(poly.evaluations(), &());
            let v_eval = verifier_evals.map_or(eval, |overrides| overrides[i]);
            verifier_claims.push(OpeningClaim::<Fr, MockPCS> {
                commitment,
                point: point.clone(),
                eval: v_eval,
            });
        }

        let mut transcript_p = Blake2bTranscript::new(b"e2e-test");
        let hints = vec![(); prover_claims.len()];
        let proof = MockPCS::prove_batch(prover_claims, hints, &(), &mut transcript_p);

        let mut transcript_v = Blake2bTranscript::new(b"e2e-test");
        MockPCS::verify_batch(verifier_claims, &proof, &(), &mut transcript_v)
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
                polynomial: Polynomial::new(p1.evaluations().to_vec()),
                point: r.clone(),
                eval: p1.evaluate(&r),
            },
            ProverClaim {
                polynomial: Polynomial::new(p2.evaluations().to_vec()),
                point: r.clone(),
                eval: p2.evaluate(&r),
            },
            ProverClaim {
                polynomial: Polynomial::new(p3.evaluations().to_vec()),
                point: s.clone(),
                eval: p3.evaluate(&s),
            },
        ];

        let mut transcript = Blake2bTranscript::new(b"grouping");
        let hints = vec![(); claims.len()];
        let proofs = MockPCS::prove_batch(claims, hints, &(), &mut transcript);
        assert_eq!(proofs.len(), 2, "two distinct points → two batch proofs");
    }

    #[test]
    fn source_backed_opening_returns_public_relations() {
        struct TestOpeningBatch {
            polynomials: Vec<Polynomial<Fr>>,
            hints: Vec<()>,
        }

        impl BatchOpeningSource<Fr, ()> for TestOpeningBatch {
            type Id = usize;
            type Source<'a>
                = &'a Polynomial<Fr>
            where
                Self: 'a;

            fn source(&self, id: Self::Id) -> Self::Source<'_> {
                &self.polynomials[id]
            }

            fn opening_hint(&self, id: Self::Id) -> &() {
                &self.hints[id]
            }
        }

        impl LinearCombinationOpeningSource<Fr, ()> for TestOpeningBatch {
            type LinearCombination<'a>
                = MaterializedLinearCombination<Fr>
            where
                Self: 'a;

            fn linear_combination<'a>(
                &'a mut self,
                terms: &[crate::LinearSourceTerm<Fr, Self::Id>],
            ) -> Self::LinearCombination<'a> {
                MaterializedLinearCombination::new(self, terms)
            }
        }

        let mut rng = ChaCha20Rng::seed_from_u64(450);
        let p1 = Polynomial::<Fr>::random(3, &mut rng);
        let p2 = Polynomial::<Fr>::random(3, &mut rng);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let eval1 = p1.evaluate(&point);
        let eval2 = p2.evaluate(&point);
        let scale1 = Fr::from_u64(5);
        let scale2 = Fr::from_u64(7);

        let mut batch = TestOpeningBatch {
            polynomials: vec![p1.clone(), p2.clone()],
            hints: vec![(), ()],
        };
        let (c1, ()) = MockPCS::commit(&p1, &());
        let (c2, ()) = MockPCS::commit(&p2, &());

        let prover_terms = vec![
            ProverBatchOpeningTerm {
                claim_id: 10u8,
                source_id: 0usize,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval1,
                eval_scale: scale1,
            },
            ProverBatchOpeningTerm {
                claim_id: 11u8,
                source_id: 1usize,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval2,
                eval_scale: scale2,
            },
        ];

        let verifier_terms = vec![
            VerifierBatchOpeningTerm::<Fr, MockPCS, _, _> {
                claim_id: 10u8,
                source_id: 0usize,
                commitment: c1,
                point: BatchOpeningPoint::same(point.clone()),
                eval: eval1,
                eval_scale: scale1,
            },
            VerifierBatchOpeningTerm::<Fr, MockPCS, _, _> {
                claim_id: 11u8,
                source_id: 1usize,
                commitment: c2,
                point: BatchOpeningPoint::same(point),
                eval: eval2,
                eval_scale: scale2,
            },
        ];

        let mut prover_transcript = Blake2bTranscript::new(b"source-backed");
        let prover_result =
            MockPCS::prove_batch_opening(prover_terms, &mut batch, &(), &mut prover_transcript);

        let mut verifier_transcript = Blake2bTranscript::new(b"source-backed");
        let verifier_public = MockPCS::verify_batch_opening(
            verifier_terms,
            &prover_result.proof,
            &(),
            &mut verifier_transcript,
        )
        .expect("source-backed mock proof should verify");

        assert_eq!(prover_result.public, verifier_public);
        assert_eq!(verifier_public.outputs.len(), 2);
        assert_eq!(
            verifier_public.outputs[0].value.as_public(),
            Some(&(eval1 * scale1)),
        );
        assert_eq!(
            verifier_public.outputs[1].value.as_public(),
            Some(&(eval2 * scale2)),
        );
        assert!(matches!(
            &verifier_public.relations[0].expression,
            BatchOutputExpression::Linear(terms) if terms == &vec![(10u8, scale1)]
        ));
        assert!(matches!(
            &verifier_public.relations[1].expression,
            BatchOutputExpression::Linear(terms) if terms == &vec![(11u8, scale2)]
        ));
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
            MockPCS::open_zk(&poly, &point, eval, &(), (), &mut transcript_p);

        let _ = eval_com;
        let mut transcript_v = Blake2bTranscript::new(b"zk-test");
        MockPCS::verify_zk(&commitment, &point, &proof, &(), &mut transcript_v)
            .expect("valid ZK proof should verify");
    }
}
