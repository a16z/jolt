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
