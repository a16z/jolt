use std::marker::PhantomData;

use jolt_crypto::Commitment;
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

use crate::{
    BatchOpeningClaim, BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement,
    CommitmentScheme, OpeningsError, PhysicalView, ZkBatchOpeningScheme, ZkOpeningScheme,
};

/// Lightweight packed-view coefficient adapter over an inner batch-opening PCS.
///
/// This adapter is intentionally a newtype so an additively homomorphic inner
/// PCS can still use the blanket [`BatchOpeningScheme`] implementation while
/// packed-view tests exercise a path that does not expose that bound to callers.
///
/// Use [`crate::PackedLinearBatch`] for packed-linear views that require the
/// selector/product-sumcheck reduction to a native packed-polynomial opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedCombine<PCS>(PhantomData<PCS>);

impl<PCS> PackedCombine<PCS> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<PCS> Default for PackedCombine<PCS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<PCS> Commitment for PackedCombine<PCS>
where
    PCS: CommitmentScheme,
{
    type Output = PCS::Output;
}

impl<PCS> CommitmentScheme for PackedCombine<PCS>
where
    PCS: CommitmentScheme,
{
    type Field = PCS::Field;
    type Proof = PCS::Proof;
    type ProverSetup = PCS::ProverSetup;
    type VerifierSetup = PCS::VerifierSetup;
    type Polynomial = PCS::Polynomial;
    type OpeningHint = PCS::OpeningHint;
    type SetupParams = PCS::SetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        PCS::setup(params)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        PCS::verifier_setup(prover_setup)
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        PCS::commit(poly, setup)
    }

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        PCS::open(poly, point, eval, setup, hint, transcript)
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        PCS::verify(commitment, point, eval, proof, setup, transcript)
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        PCS::bind_opening_inputs(transcript, point, eval);
    }
}

impl<PCS> ZkOpeningScheme for PackedCombine<PCS>
where
    PCS: ZkOpeningScheme,
{
    type HidingCommitment = PCS::HidingCommitment;
    type Blind = PCS::Blind;

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        PCS::commit_zk(poly, setup)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        PCS::open_zk(poly, point, eval, setup, hint, transcript)
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        PCS::verify_zk(commitment, point, proof, setup, transcript)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        PCS::bind_zk_opening_inputs(transcript, point, hiding_commitment);
    }
}

impl<PCS> BatchOpeningScheme for PackedCombine<PCS>
where
    PCS: BatchOpeningScheme,
{
    fn prove_batch<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<Self::Proof, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let physical_statement = packed_to_physical_statement(statement)?;
        bind_packed_batch_statement(transcript, statement);
        PCS::prove_batch(setup, transcript, &physical_statement, polynomials, hints)
    }

    fn verify_batch<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let physical_statement = packed_to_physical_statement(statement)?;
        bind_packed_batch_statement(transcript, statement);
        PCS::verify_batch(setup, transcript, &physical_statement, proof)
    }
}

impl<PCS> ZkBatchOpeningScheme for PackedCombine<PCS>
where
    PCS: ZkBatchOpeningScheme,
{
    fn prove_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::ProverSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        evals: &[Self::Field],
        polynomials: &[Self::Polynomial],
        hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let physical_statement = packed_to_physical_statement(statement)?;
        bind_packed_batch_statement(transcript, statement);
        PCS::prove_batch_zk(
            setup,
            transcript,
            &physical_statement,
            evals,
            polynomials,
            hints,
        )
    }

    fn verify_batch_zk<T, OpeningId, RelationId>(
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let physical_statement = packed_to_physical_statement(statement)?;
        bind_packed_batch_statement(transcript, statement);
        PCS::verify_batch_zk(setup, transcript, &physical_statement, proof)
    }
}

fn bind_packed_batch_statement<F, C, OpeningId, RelationId, Claim, T>(
    transcript: &mut T,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId, Claim>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"packed_batch_layout"));
    transcript.append_bytes(&statement.layout_digest);
    transcript.append(&LabelWithCount(
        b"packed_logical_point",
        statement.logical_point.len() as u64,
    ));
    for challenge in &statement.logical_point {
        challenge.append_to_transcript(transcript);
    }
    transcript.append(&LabelWithCount(
        b"packed_pcs_point",
        statement.pcs_point.len() as u64,
    ));
    for challenge in &statement.pcs_point {
        challenge.append_to_transcript(transcript);
    }
    transcript.append(&LabelWithCount(
        b"packed_batch_views",
        statement.claims.len() as u64,
    ));
    for claim in &statement.claims {
        claim.scale.append_to_transcript(transcript);
        match &claim.view {
            PhysicalView::Direct => transcript.append_bytes(&[0]),
            PhysicalView::PackedLinear {
                layout_digest,
                terms,
            } => {
                transcript.append_bytes(&[1]);
                transcript.append_bytes(layout_digest);
                transcript.append(&LabelWithCount(b"packed_view_terms", terms.len() as u64));
                for term in terms {
                    transcript.append(&U64Word(term.family.namespace));
                    transcript.append(&U64Word(term.family.id));
                    transcript.append(&U64Word(term.family.index));
                    transcript.append(&U64Word(term.limb as u64));
                    transcript.append(&U64Word(term.symbol as u64));
                    transcript.append(&LabelWithCount(
                        b"packed_view_row_point",
                        term.row_point.len() as u64,
                    ));
                    for challenge in &term.row_point {
                        challenge.append_to_transcript(transcript);
                    }
                    term.coefficient.append_to_transcript(transcript);
                }
            }
        }
    }
}

fn packed_to_physical_statement<F, C, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId, Claim>,
) -> Result<BatchOpeningStatement<F, C, (), (), Claim>, OpeningsError>
where
    F: Field,
    C: Clone,
    Claim: Copy,
{
    let claims = statement
        .claims
        .iter()
        .map(|claim| {
            let view_scale = physical_view_scale(&statement.layout_digest, &claim.view)?;
            Ok(BatchOpeningClaim {
                id: (),
                relation: (),
                commitment: claim.commitment.clone(),
                claim: claim.claim,
                view: PhysicalView::Direct,
                scale: claim.scale * view_scale,
            })
        })
        .collect::<Result<Vec<_>, OpeningsError>>()?;

    Ok(BatchOpeningStatement {
        logical_point: statement.logical_point.clone(),
        pcs_point: statement.pcs_point.clone(),
        layout_digest: statement.layout_digest,
        claims,
    })
}

fn physical_view_scale<F>(
    statement_layout_digest: &[u8; 32],
    view: &PhysicalView<F>,
) -> Result<F, OpeningsError>
where
    F: Field,
{
    match view {
        PhysicalView::Direct => Ok(F::one()),
        PhysicalView::PackedLinear {
            layout_digest,
            terms,
        } => {
            if layout_digest != statement_layout_digest {
                return Err(OpeningsError::InvalidBatch(
                    "packed view layout digest does not match statement layout digest".to_owned(),
                ));
            }
            if terms.is_empty() {
                return Err(OpeningsError::InvalidBatch(
                    "packed linear view requires at least one term".to_owned(),
                ));
            }
            Ok(terms
                .iter()
                .fold(F::zero(), |acc, term| acc + term.coefficient))
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "tests assert successful packed batch proof results"
)]
mod tests {
    use super::PackedCombine;
    use crate::{
        mock::MockCommitmentScheme, BatchOpeningClaim, BatchOpeningResult, BatchOpeningScheme,
        BatchOpeningStatement, CommitmentScheme, OpeningsError, PackedFamilyRef, PackedLinearTerm,
        PhysicalView,
    };
    use jolt_crypto::Commitment;
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};
    use jolt_poly::{MultilinearPoly, Polynomial};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use serde::{Deserialize, Serialize};

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

    type MockPCS = MockCommitmentScheme<Fr>;
    type PackedMockPCS = PackedCombine<MockPCS>;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NonHomomorphicBatchPCS;

    impl Commitment for NonHomomorphicBatchPCS {
        type Output = u64;
    }

    impl CommitmentScheme for NonHomomorphicBatchPCS {
        type Field = Fr;
        type Proof = NonHomomorphicProof;
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
            _poly: &P,
            _setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            (11, ())
        }

        fn open(
            _poly: &Self::Polynomial,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Option<Self::OpeningHint>,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
            NonHomomorphicProof { claim_count: 1 }
        }

        fn verify(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _eval: Self::Field,
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }

        fn bind_opening_inputs(
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
            _point: &[Self::Field],
            _eval: &Self::Field,
        ) {
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct NonHomomorphicProof {
        claim_count: usize,
    }

    impl BatchOpeningScheme for NonHomomorphicBatchPCS {
        fn prove_batch<T, OpeningId, RelationId>(
            _setup: &Self::ProverSetup,
            _transcript: &mut T,
            statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            _polynomials: &[Self::Polynomial],
            _hints: Vec<Self::OpeningHint>,
        ) -> Result<Self::Proof, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            Ok(NonHomomorphicProof {
                claim_count: statement.claims.len(),
            })
        }

        fn verify_batch<T, OpeningId, RelationId>(
            _setup: &Self::VerifierSetup,
            _transcript: &mut T,
            statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId>,
            proof: &Self::Proof,
        ) -> Result<BatchOpeningResult<Self::Field, Self::Output>, OpeningsError>
        where
            T: Transcript<Challenge = Self::Field>,
        {
            if proof.claim_count != statement.claims.len() {
                return Err(OpeningsError::VerificationFailed);
            }
            Ok(BatchOpeningResult {
                coefficients: vec![Fr::from_u64(1); statement.claims.len()],
                joint_commitment: statement.claims[0].commitment,
                reduced_opening: Fr::from_u64(0),
            })
        }
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn packed_term(coefficient: Fr) -> PackedLinearTerm<Fr> {
        packed_term_at(coefficient, 0)
    }

    fn packed_term_at(coefficient: Fr, symbol: usize) -> PackedLinearTerm<Fr> {
        PackedLinearTerm::new(
            coefficient,
            PackedFamilyRef::new(0x6a6f_6c74, 1, 0),
            0,
            symbol,
        )
    }

    fn batch_polynomials() -> (Vec<Polynomial<Fr>>, Vec<Fr>) {
        let polynomials = vec![
            Polynomial::new((0..8).map(|value| fr(value + 1)).collect()),
            Polynomial::new((0..8).map(|value| fr(17 + 2 * value)).collect()),
        ];
        let point = vec![fr(2), fr(3), fr(5)];
        (polynomials, point)
    }

    fn packed_batch_statement(
        polynomial: &Polynomial<Fr>,
        point: &[Fr],
    ) -> BatchOpeningStatement<Fr, <MockPCS as Commitment>::Output, OpeningId, RelationId> {
        let (commitment, ()) = MockPCS::commit(polynomial.evaluations(), &());
        let eval = polynomial.evaluate(point);
        let first_decode = fr(2);
        let second_decode = fr(5);

        BatchOpeningStatement {
            logical_point: point.to_vec(),
            pcs_point: point.to_vec(),
            layout_digest: [44; 32],
            claims: vec![
                BatchOpeningClaim {
                    id: OpeningId::A,
                    relation: RelationId::First,
                    commitment: commitment.clone(),
                    claim: eval * first_decode.inverse().expect("decode is nonzero"),
                    view: PhysicalView::PackedLinear {
                        layout_digest: [44; 32],
                        terms: vec![packed_term(first_decode)],
                    },
                    scale: fr(1),
                },
                BatchOpeningClaim {
                    id: OpeningId::B,
                    relation: RelationId::Second,
                    commitment,
                    claim: eval * second_decode.inverse().expect("decode is nonzero"),
                    view: PhysicalView::PackedLinear {
                        layout_digest: [44; 32],
                        terms: vec![packed_term(second_decode)],
                    },
                    scale: fr(1),
                },
            ],
        }
    }

    #[test]
    fn packed_combine_many_claims_one_commitment_roundtrip_clear() {
        let (polynomials, point) = batch_polynomials();
        let polynomial = polynomials[0].clone();
        let statement = packed_batch_statement(&polynomial, &point);
        let polynomials = vec![polynomial.clone(), polynomial];
        let hints = vec![(); polynomials.len()];

        let mut prover_transcript = Blake2bTranscript::new(b"packed-clear");
        let proof = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut prover_transcript,
            &statement,
            &polynomials,
            hints,
        )
        .expect("packed batch proof should be produced");

        let mut verifier_transcript = Blake2bTranscript::new(b"packed-clear");
        let result = <PackedMockPCS as BatchOpeningScheme>::verify_batch(
            &(),
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("packed batch proof should verify");

        assert_eq!(result.coefficients.len(), statement.claims.len());
        assert_eq!(
            result.reduced_opening,
            result
                .coefficients
                .iter()
                .zip(&statement.claims)
                .fold(fr(0), |acc, (coefficient, claim)| {
                    acc + *coefficient * claim.claim
                })
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn packed_combine_rejects_layout_digest_mismatch() {
        let (polynomials, point) = batch_polynomials();
        let mut statement = packed_batch_statement(&polynomials[0], &point);
        statement.claims[0].view = PhysicalView::PackedLinear {
            layout_digest: [45; 32],
            terms: vec![packed_term(fr(2))],
        };

        let mut transcript = Blake2bTranscript::new(b"packed-layout-mismatch");
        let result = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut transcript,
            &statement,
            &[polynomials[0].clone(), polynomials[0].clone()],
            vec![(), ()],
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
    }

    #[test]
    fn packed_combine_binds_layout_digest_before_inner_batch() {
        let (polynomials, point) = batch_polynomials();
        let polynomial = polynomials[0].clone();
        let statement = packed_batch_statement(&polynomial, &point);

        let mut prover_transcript = Blake2bTranscript::new(b"packed-layout-bound");
        let proof = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut prover_transcript,
            &statement,
            &[polynomial.clone(), polynomial.clone()],
            vec![(), ()],
        )
        .expect("packed batch proof should be produced");

        let mut tampered = statement.clone();
        tampered.layout_digest = [46; 32];
        for claim in &mut tampered.claims {
            if let PhysicalView::PackedLinear { layout_digest, .. } = &mut claim.view {
                *layout_digest = [46; 32];
            }
        }

        let mut verifier_transcript = Blake2bTranscript::new(b"packed-layout-bound");
        let result = <PackedMockPCS as BatchOpeningScheme>::verify_batch(
            &(),
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "changed layout digest should fail");
    }

    #[test]
    fn packed_combine_binds_view_coefficients_before_inner_batch() {
        let (polynomials, point) = batch_polynomials();
        let polynomial = polynomials[0].clone();
        let statement = packed_batch_statement(&polynomial, &point);

        let mut prover_transcript = Blake2bTranscript::new(b"packed-coeffs-bound");
        let proof = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut prover_transcript,
            &statement,
            &[polynomial.clone(), polynomial.clone()],
            vec![(), ()],
        )
        .expect("packed batch proof should be produced");

        let mut tampered = statement;
        tampered.claims[0].view = PhysicalView::PackedLinear {
            layout_digest: [44; 32],
            terms: vec![packed_term(fr(1)), packed_term_at(fr(1), 1)],
        };

        let mut verifier_transcript = Blake2bTranscript::new(b"packed-coeffs-bound");
        let result = <PackedMockPCS as BatchOpeningScheme>::verify_batch(
            &(),
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(
            result.is_err(),
            "changed coefficients should fail even when their sum is unchanged"
        );
    }

    #[test]
    fn packed_combine_binds_view_addresses_before_inner_batch() {
        let (polynomials, point) = batch_polynomials();
        let polynomial = polynomials[0].clone();
        let statement = packed_batch_statement(&polynomial, &point);

        let mut prover_transcript = Blake2bTranscript::new(b"packed-addresses-bound");
        let proof = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut prover_transcript,
            &statement,
            &[polynomial.clone(), polynomial.clone()],
            vec![(), ()],
        )
        .expect("packed batch proof should be produced");

        let mut tampered = statement;
        tampered.claims[0].view = PhysicalView::PackedLinear {
            layout_digest: [44; 32],
            terms: vec![packed_term_at(fr(2), 1)],
        };

        let mut verifier_transcript = Blake2bTranscript::new(b"packed-addresses-bound");
        let result = <PackedMockPCS as BatchOpeningScheme>::verify_batch(
            &(),
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(
            result.is_err(),
            "changed packed address should fail even when coefficients are unchanged"
        );
    }

    #[test]
    fn packed_combine_rejects_empty_linear_view() {
        let (polynomials, point) = batch_polynomials();
        let polynomial = polynomials[0].clone();
        let mut statement = packed_batch_statement(&polynomial, &point);
        statement.claims[0].view = PhysicalView::PackedLinear {
            layout_digest: [44; 32],
            terms: Vec::new(),
        };

        let mut transcript = Blake2bTranscript::new(b"packed-empty-view");
        let result = <PackedMockPCS as BatchOpeningScheme>::prove_batch(
            &(),
            &mut transcript,
            &statement,
            &[polynomial.clone(), polynomial],
            vec![(), ()],
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
    }

    #[test]
    fn packed_combine_can_wrap_non_homomorphic_batch_scheme() {
        fn assert_batch_scheme<PCS: BatchOpeningScheme>() {}
        assert_batch_scheme::<PackedCombine<NonHomomorphicBatchPCS>>();

        let polynomial = Polynomial::new(vec![fr(1), fr(2)]);
        let statement = BatchOpeningStatement {
            logical_point: vec![fr(3)],
            pcs_point: vec![fr(3)],
            layout_digest: [47; 32],
            claims: vec![BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: 11,
                claim: fr(5),
                view: PhysicalView::PackedLinear {
                    layout_digest: [47; 32],
                    terms: vec![packed_term(fr(7))],
                },
                scale: fr(1),
            }],
        };

        let mut prover_transcript = Blake2bTranscript::new(b"packed-nonhom");
        let proof = <PackedCombine<NonHomomorphicBatchPCS> as BatchOpeningScheme>::prove_batch(
            &(),
            &mut prover_transcript,
            &statement,
            &[polynomial],
            vec![()],
        )
        .expect("non-homomorphic inner batch scheme should prove");

        let mut verifier_transcript = Blake2bTranscript::new(b"packed-nonhom");
        let _result = <PackedCombine<NonHomomorphicBatchPCS> as BatchOpeningScheme>::verify_batch(
            &(),
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("non-homomorphic inner batch scheme should verify");
    }
}
