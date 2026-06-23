#![expect(
    clippy::expect_used,
    reason = "tests assert successful batch proof results"
)]

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::{Fr, FromPrimitiveInt, Invertible};
use jolt_openings::{
    AdditivelyHomomorphic, BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement,
    CommitmentScheme, OpeningsError, PackingFamilyRef, PackingTerm, PhysicalView,
    ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Label, LabelWithCount, Transcript};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TestPcs;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct TestCommitment {
    evaluations: Vec<Fr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TestProof {
    evaluations: Vec<Fr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TestHidingCommitment {
    eval: Fr,
}

impl AppendToTranscript for TestHidingCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.eval.append_to_transcript(transcript);
    }
}

impl Commitment for TestPcs {
    type Output = TestCommitment;
}

impl CommitmentScheme for TestPcs {
    type Field = Fr;
    type Proof = TestProof;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Polynomial = Polynomial<Fr>;
    type OpeningHint = ();
    type SetupParams = ();

    fn setup(_params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        ((), ())
    }

    fn verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        let mut evaluations = Vec::with_capacity(1 << poly.num_vars());
        poly.for_each_row(poly.num_vars(), &mut |_, row| {
            evaluations.extend_from_slice(row);
        });
        (TestCommitment { evaluations }, ())
    }

    fn open(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        TestProof {
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
            return Err(OpeningsError::VerificationFailed);
        }
        let poly = Polynomial::new(proof.evaluations.clone());
        if poly.evaluate(point) != eval {
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
}

impl HomomorphicCommitment<Fr> for TestCommitment {
    fn add(c1: &Self, c2: &Self) -> Self {
        Self::linear_combine(c1, c2, &Fr::from_u64(1))
    }

    fn linear_combine(c1: &Self, c2: &Self, scalar: &Fr) -> Self {
        let len = c1.evaluations.len().max(c2.evaluations.len());
        let mut result = vec![Fr::from_u64(0); len];
        for (i, entry) in result.iter_mut().enumerate() {
            let a = c1
                .evaluations
                .get(i)
                .copied()
                .unwrap_or_else(|| Fr::from_u64(0));
            let b = c2
                .evaluations
                .get(i)
                .copied()
                .unwrap_or_else(|| Fr::from_u64(0));
            *entry = a + *scalar * b;
        }
        TestCommitment {
            evaluations: result,
        }
    }
}

impl AdditivelyHomomorphic for TestPcs {
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());
        let len = commitments
            .first()
            .map_or(0, |commitment| commitment.evaluations.len());
        let mut result = vec![Fr::from_u64(0); len];
        for (commitment, scalar) in commitments.iter().zip(scalars) {
            for (entry, eval) in result.iter_mut().zip(&commitment.evaluations) {
                *entry += *scalar * *eval;
            }
        }
        TestCommitment {
            evaluations: result,
        }
    }
}

impl ZkOpeningScheme for TestPcs {
    type HidingCommitment = TestHidingCommitment;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit(poly, setup)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        _point: &[Self::Field],
        eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Self::OpeningHint,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        (
            TestProof {
                evaluations: poly.evaluations().to_vec(),
            },
            TestHidingCommitment { eval },
            (),
        )
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        if commitment.evaluations != proof.evaluations {
            return Err(OpeningsError::VerificationFailed);
        }
        let poly = Polynomial::new(proof.evaluations.clone());
        Ok(TestHidingCommitment {
            eval: poly.evaluate(point),
        })
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        transcript.append(&LabelWithCount(
            b"test_zk_opening_point",
            point.len() as u64,
        ));
        for coordinate in point {
            coordinate.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"test_zk_eval_commitment"));
        hiding_commitment.append_to_transcript(transcript);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpeningId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationId {
    First,
    Second,
}

fn packing_term<F>(coefficient: F) -> PackingTerm<F> {
    PackingTerm::new(coefficient, PackingFamilyRef::new(0x6a6f_6c74, 1, 0), 0, 0)
}

fn packing_term_at<F>(coefficient: F, symbol: usize) -> PackingTerm<F> {
    PackingTerm::new(
        coefficient,
        PackingFamilyRef::new(0x6a6f_6c74, 1, 0),
        0,
        symbol,
    )
}

#[test]
fn batch_opening_statement_preserves_claim_metadata() {
    let statement = BatchOpeningStatement {
        logical_point: vec![3_u64, 5],
        pcs_point: vec![5_u64, 3],
        layout_digest: [7; 32],
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: 11_u64,
                claim: 13,
                view: PhysicalView::Direct,
                scale: 17,
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Second,
                commitment: 19_u64,
                claim: 23,
                view: PhysicalView::Packing {
                    layout_digest: [23; 32],
                    terms: vec![packing_term(29), packing_term_at(31, 1)],
                },
                scale: 37,
            },
        ],
    };

    assert_eq!(statement.logical_point, vec![3, 5]);
    assert_eq!(statement.pcs_point, vec![5, 3]);
    assert_eq!(statement.layout_digest, [7; 32]);
    assert_eq!(statement.claims[0].id, OpeningId::A);
    assert_eq!(statement.claims[0].relation, RelationId::First);
    assert_eq!(statement.claims[0].claim, 13);
    assert_eq!(statement.claims[1].id, OpeningId::B);
    assert_eq!(statement.claims[1].relation, RelationId::Second);
    assert_eq!(statement.claims[1].claim, 23);
}

#[test]
fn zk_batch_opening_statement_can_omit_claim_payload() {
    let statement: BatchOpeningStatement<_, _, _, _, ()> = BatchOpeningStatement {
        logical_point: vec![1_u64],
        pcs_point: vec![1_u64],
        layout_digest: [0; 32],
        claims: vec![BatchOpeningClaim {
            id: OpeningId::A,
            relation: RelationId::First,
            commitment: 2_u64,
            claim: (),
            view: PhysicalView::Direct,
            scale: 1,
        }],
    };

    let _: () = statement.claims[0].claim;
}

#[test]
fn physical_view_records_packing_layout_and_terms() {
    let view = PhysicalView::Packing {
        layout_digest: [41; 32],
        terms: vec![packing_term(43_u64), packing_term_at(47, 1)],
    };

    assert!(matches!(view, PhysicalView::Packing { .. }));
    if let PhysicalView::Packing {
        layout_digest,
        terms,
    } = view
    {
        assert_eq!(layout_digest, [41; 32]);
        assert_eq!(terms[0].coefficient, 43);
        assert_eq!(terms[1].coefficient, 47);
        assert_eq!(terms[1].symbol, 1);
    }
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn batch_polynomials() -> (Vec<Polynomial<Fr>>, Vec<Fr>) {
    let polynomials = vec![
        Polynomial::new((0..8).map(|value| fr(value + 1)).collect()),
        Polynomial::new((0..8).map(|value| fr(17 + 2 * value)).collect()),
    ];
    let point = vec![fr(2), fr(3), fr(5)];
    (polynomials, point)
}

fn clear_batch_statement(
    polynomials: &[Polynomial<Fr>],
    point: &[Fr],
) -> BatchOpeningStatement<Fr, TestCommitment, OpeningId, RelationId> {
    let commitments = polynomials
        .iter()
        .map(|polynomial| TestPcs::commit(polynomial.evaluations(), &()).0)
        .collect::<Vec<_>>();
    let first_scale = fr(2);
    let second_scale = fr(5);
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: [9; 32],
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: commitments[0].clone(),
                claim: polynomials[0].evaluate(point)
                    * first_scale.inverse().expect("scale is nonzero"),
                view: PhysicalView::Direct,
                scale: first_scale,
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Second,
                commitment: commitments[1].clone(),
                claim: polynomials[1].evaluate(point)
                    * second_scale.inverse().expect("scale is nonzero"),
                view: PhysicalView::Direct,
                scale: second_scale,
            },
        ],
    }
}

#[test]
fn homomorphic_batch_opening_roundtrip_clear() {
    let (polynomials, point) = batch_polynomials();
    let statement = clear_batch_statement(&polynomials, &point);
    let hints = vec![(); polynomials.len()];

    let mut prover_transcript = Blake2bTranscript::new(b"batch-clear");
    let proof = <TestPcs as BatchOpeningScheme>::prove_batch(
        &(),
        &mut prover_transcript,
        &statement,
        &polynomials,
        hints,
    )
    .expect("batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear");
    let result = <TestPcs as BatchOpeningScheme>::verify_batch(
        &(),
        &mut verifier_transcript,
        &statement,
        &proof,
    )
    .expect("batch proof should verify");

    assert_eq!(result.coefficients.len(), statement.claims.len());
    assert_eq!(
        result.reduced_opening,
        result
            .coefficients
            .iter()
            .zip(&statement.claims)
            .fold(Fr::from_u64(0), |acc, (coefficient, claim)| {
                acc + *coefficient * claim.claim
            })
    );
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn homomorphic_batch_opening_rejects_tampered_clear_claim() {
    let (polynomials, point) = batch_polynomials();
    let statement = clear_batch_statement(&polynomials, &point);

    let mut prover_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
    let proof = <TestPcs as BatchOpeningScheme>::prove_batch(
        &(),
        &mut prover_transcript,
        &statement,
        &polynomials,
        vec![(); polynomials.len()],
    )
    .expect("batch proof should be produced");

    let mut tampered = statement.clone();
    tampered.claims[1].claim += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
    let result = <TestPcs as BatchOpeningScheme>::verify_batch(
        &(),
        &mut verifier_transcript,
        &tampered,
        &proof,
    );
    assert!(result.is_err(), "tampered claim should fail");
}

#[test]
fn homomorphic_batch_opening_rejects_packing_view() {
    let (polynomials, point) = batch_polynomials();
    let mut statement = clear_batch_statement(&polynomials, &point);
    statement.claims[0].view = PhysicalView::Packing {
        layout_digest: [3; 32],
        terms: vec![packing_term(fr(1)), packing_term_at(fr(2), 1)],
    };

    let mut transcript = Blake2bTranscript::new(b"batch-packed");
    let result = <TestPcs as BatchOpeningScheme>::prove_batch(
        &(),
        &mut transcript,
        &statement,
        &polynomials,
        vec![(); polynomials.len()],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn homomorphic_batch_opening_rejects_mismatched_witness_count() {
    let (polynomials, point) = batch_polynomials();
    let statement = clear_batch_statement(&polynomials, &point);

    let mut transcript = Blake2bTranscript::new(b"batch-mismatch");
    let result = <TestPcs as BatchOpeningScheme>::prove_batch(
        &(),
        &mut transcript,
        &statement,
        &polynomials[..1],
        vec![()],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

fn zk_batch_statement(
    polynomials: &[Polynomial<Fr>],
    point: &[Fr],
) -> BatchOpeningStatement<Fr, TestCommitment, OpeningId, RelationId, ()> {
    let commitments = polynomials
        .iter()
        .map(|polynomial| <TestPcs as ZkOpeningScheme>::commit_zk(polynomial.evaluations(), &()).0)
        .collect::<Vec<_>>();
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: [10; 32],
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: commitments[0].clone(),
                claim: (),
                view: PhysicalView::Direct,
                scale: fr(3),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Second,
                commitment: commitments[1].clone(),
                claim: (),
                view: PhysicalView::Direct,
                scale: fr(7),
            },
        ],
    }
}

#[test]
fn homomorphic_batch_opening_roundtrip_zk() {
    let (polynomials, point) = batch_polynomials();
    let statement = zk_batch_statement(&polynomials, &point);
    let evals = polynomials
        .iter()
        .map(|polynomial| polynomial.evaluate(&point))
        .collect::<Vec<_>>();

    let mut prover_transcript = Blake2bTranscript::new(b"batch-zk");
    let (proof, hiding_commitment, _blind) = <TestPcs as ZkBatchOpeningScheme>::prove_batch_zk(
        &(),
        &mut prover_transcript,
        &statement,
        &evals,
        &polynomials,
        vec![(); polynomials.len()],
    )
    .expect("ZK batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-zk");
    let result = <TestPcs as ZkBatchOpeningScheme>::verify_batch_zk(
        &(),
        &mut verifier_transcript,
        &statement,
        &proof,
    )
    .expect("ZK batch proof should verify");

    assert_eq!(result.reduced_opening, hiding_commitment);
    assert_eq!(result.coefficients.len(), statement.claims.len());
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn homomorphic_zk_batch_opening_rejects_eval_count_mismatch() {
    let (polynomials, point) = batch_polynomials();
    let statement = zk_batch_statement(&polynomials, &point);
    let evals = [polynomials[0].evaluate(&point)];

    let mut transcript = Blake2bTranscript::new(b"batch-zk-mismatch");
    let result = <TestPcs as ZkBatchOpeningScheme>::prove_batch_zk(
        &(),
        &mut transcript,
        &statement,
        &evals,
        &polynomials,
        vec![(); polynomials.len()],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
