use std::marker::PhantomData;

use jolt_crypto::Commitment;
use jolt_field::Field;
use jolt_openings::{
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PhysicalView, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Label, LabelWithCount, Transcript};
use serde::{Deserialize, Serialize};

use crate::types::{
    append_field_slice, u64_field, AkitaBatchProof, AkitaCommitInput, AkitaCommitment,
    AkitaHidingCommitment, AkitaProverHint, AkitaProverSetup, AkitaSetup, AkitaSetupKey,
    AkitaSetupParams, AkitaVerifierSetup,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AkitaScheme<F: Field>(PhantomData<F>);

impl<F: Field> AkitaScheme<F> {
    pub fn commit_packed_witness(
        setup: &AkitaProverSetup,
        input: AkitaCommitInput<F>,
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        validate_setup_dimension(&setup.key, input.d_pack)?;
        validate_evaluation_shape(input.d_pack, input.evaluations.len())?;
        let commitment_digest = commitment_digest::<F>(
            &setup.key,
            &input.layout_digest,
            input.d_pack,
            &input.evaluations,
        );
        let commitment = AkitaCommitment {
            layout_digest: input.layout_digest,
            commitment_digest,
            d_pack: input.d_pack,
        };
        let hint = AkitaProverHint {
            layout_digest: commitment.layout_digest,
            commitment_digest,
            d_pack: commitment.d_pack,
        };
        Ok((commitment, hint))
    }
}

impl<F: Field> Commitment for AkitaScheme<F> {
    type Output = AkitaCommitment;
}

impl<F: Field> CommitmentScheme for AkitaScheme<F> {
    type Field = F;
    type Proof = AkitaBatchProof<F>;
    type ProverSetup = AkitaProverSetup;
    type VerifierSetup = AkitaVerifierSetup;
    type Polynomial = Polynomial<F>;
    type OpeningHint = AkitaProverHint;
    type SetupParams = AkitaSetupParams;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        let setup = AkitaSetup::new(params);
        (setup.clone(), setup)
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        prover_setup.clone()
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        let evaluations = polynomial_evaluations(poly);
        let commitment_digest = commitment_digest::<F>(
            &setup.key,
            &setup.default_layout_digest,
            poly.num_vars(),
            &evaluations,
        );
        let commitment = AkitaCommitment {
            layout_digest: setup.default_layout_digest,
            commitment_digest,
            d_pack: poly.num_vars(),
        };
        let hint = AkitaProverHint {
            layout_digest: commitment.layout_digest,
            commitment_digest,
            d_pack: commitment.d_pack,
        };
        (commitment, hint)
    }

    fn open(
        _poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let commitment = hint.map_or_else(AkitaCommitment::default, |hint| AkitaCommitment {
            layout_digest: hint.layout_digest,
            commitment_digest: hint.commitment_digest,
            d_pack: hint.d_pack,
        });
        let statement_digest =
            bind_single_opening_statement(&setup.key, &commitment, point, eval, transcript);
        AkitaBatchProof {
            setup_key: setup.key.clone(),
            packed_commitment: commitment,
            statement_digest,
            coefficients: vec![F::one()],
            reduced_opening: eval,
        }
    }

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        validate_setup_dimension(&setup.key, commitment.d_pack)?;
        if proof.setup_key != setup.key || proof.packed_commitment != *commitment {
            return Err(OpeningsError::VerificationFailed);
        }
        let statement_digest =
            bind_single_opening_statement(&setup.key, commitment, point, eval, transcript);
        if proof.statement_digest != statement_digest
            || proof.coefficients != [F::one()]
            || proof.reduced_opening != eval
        {
            return Err(OpeningsError::VerificationFailed);
        }
        Ok(())
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        transcript.append(&Label(b"akita_opening_inputs"));
        append_field_slice(transcript, b"akita_opening_point", point);
        eval.append_to_transcript(transcript);
    }
}

impl<F: Field> BatchOpeningScheme for AkitaScheme<F> {
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
        if polynomials.len() != 1 {
            return Err(OpeningsError::InvalidBatch(format!(
                "Akita mock expects exactly one PackedWitness polynomial, got {}",
                polynomials.len()
            )));
        }
        if hints.len() != 1 {
            return Err(OpeningsError::InvalidBatch(format!(
                "Akita mock expects exactly one PackedWitness opening hint, got {}",
                hints.len()
            )));
        }
        let normalized = normalize_clear_batch(&setup.key, statement, transcript)?;
        if !hints[0].matches_commitment(&normalized.packed_commitment) {
            return Err(OpeningsError::InvalidBatch(
                "Akita prover hint does not match PackedWitness commitment".to_owned(),
            ));
        }
        if polynomials[0].num_vars() != statement.pcs_point.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "PackedWitness polynomial has {} variables but opening point has {}",
                polynomials[0].num_vars(),
                statement.pcs_point.len()
            )));
        }
        validate_physical_claims(statement, &polynomials[0])?;
        Ok(AkitaBatchProof {
            setup_key: setup.key.clone(),
            packed_commitment: normalized.packed_commitment,
            statement_digest: normalized.statement_digest,
            coefficients: normalized.coefficients,
            reduced_opening: normalized.reduced_opening,
        })
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
        if proof.setup_key != setup.key {
            return Err(OpeningsError::VerificationFailed);
        }
        let normalized = normalize_clear_batch(&setup.key, statement, transcript)?;
        if proof.packed_commitment != normalized.packed_commitment
            || proof.statement_digest != normalized.statement_digest
            || proof.coefficients != normalized.coefficients
            || proof.reduced_opening != normalized.reduced_opening
        {
            return Err(OpeningsError::VerificationFailed);
        }
        Ok(BatchOpeningResult {
            coefficients: proof.coefficients.clone(),
            joint_commitment: proof.packed_commitment.clone(),
            reduced_opening: proof.reduced_opening,
        })
    }
}

impl<F: Field> ZkOpeningScheme for AkitaScheme<F> {
    type HidingCommitment = AkitaHidingCommitment<F>;
    type Blind = ();

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit(poly, setup)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let proof = Self::open(poly, point, eval, setup, Some(hint), transcript);
        (proof, AkitaHidingCommitment { commitment: eval }, ())
    }

    fn verify_zk(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        Err(transparent_zk_error())
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        transcript.append(&Label(b"akita_zk_opening_inputs"));
        append_field_slice(transcript, b"akita_zk_opening_point", point);
        hiding_commitment.append_to_transcript(transcript);
    }
}

impl<F: Field> ZkBatchOpeningScheme for AkitaScheme<F> {
    fn prove_batch_zk<T, OpeningId, RelationId>(
        _setup: &Self::ProverSetup,
        _transcript: &mut T,
        _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        _evals: &[Self::Field],
        _polynomials: &[Self::Polynomial],
        _hints: Vec<Self::OpeningHint>,
    ) -> Result<(Self::Proof, Self::HidingCommitment, Self::Blind), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }

    fn verify_batch_zk<T, OpeningId, RelationId>(
        _setup: &Self::VerifierSetup,
        _transcript: &mut T,
        _statement: &BatchOpeningStatement<Self::Field, Self::Output, OpeningId, RelationId, ()>,
        _proof: &Self::Proof,
    ) -> Result<BatchOpeningResult<Self::Field, Self::Output, Self::HidingCommitment>, OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        Err(transparent_zk_error())
    }
}

struct NormalizedBatch<F: Field> {
    packed_commitment: AkitaCommitment,
    statement_digest: F,
    coefficients: Vec<F>,
    reduced_opening: F,
}

fn normalize_clear_batch<F, C, OpeningId, RelationId, T>(
    setup_key: &AkitaSetupKey,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    transcript: &mut T,
) -> Result<NormalizedBatch<F>, OpeningsError>
where
    F: Field,
    C: Clone + IntoAkitaCommitment,
    T: Transcript<Challenge = F>,
{
    if statement.claims.is_empty() {
        return Err(OpeningsError::InvalidBatch(
            "Akita batch opening requires at least one claim".to_owned(),
        ));
    }
    let packed_commitment = statement.claims[0]
        .commitment
        .clone()
        .into_akita_commitment();
    validate_setup_dimension(setup_key, packed_commitment.d_pack)?;
    if statement.layout_digest != packed_commitment.layout_digest {
        return Err(OpeningsError::InvalidBatch(
            "statement layout digest does not match PackedWitness commitment".to_owned(),
        ));
    }

    let mut coefficients = Vec::with_capacity(statement.claims.len());
    for claim in &statement.claims {
        let commitment = claim.commitment.clone().into_akita_commitment();
        if commitment != packed_commitment {
            return Err(OpeningsError::InvalidBatch(
                "Akita batch statement must use exactly one PackedWitness commitment".to_owned(),
            ));
        }
        coefficients
            .push(claim.scale * physical_view_scale(&statement.layout_digest, &claim.view)?);
    }
    let reduced_opening = coefficients
        .iter()
        .zip(&statement.claims)
        .fold(F::zero(), |acc, (coefficient, claim)| {
            acc + *coefficient * claim.claim
        });

    bind_batch_statement(
        setup_key,
        &packed_commitment,
        statement,
        &coefficients,
        reduced_opening,
        transcript,
    );
    Ok(NormalizedBatch {
        packed_commitment,
        statement_digest: transcript.challenge_scalar(),
        coefficients,
        reduced_opening,
    })
}

trait IntoAkitaCommitment {
    fn into_akita_commitment(self) -> AkitaCommitment;
}

impl IntoAkitaCommitment for AkitaCommitment {
    fn into_akita_commitment(self) -> AkitaCommitment {
        self
    }
}

fn bind_batch_statement<F, C, OpeningId, RelationId, T>(
    setup_key: &AkitaSetupKey,
    packed_commitment: &AkitaCommitment,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    coefficients: &[F],
    reduced_opening: F,
    transcript: &mut T,
) where
    F: Field,
    C: Clone + IntoAkitaCommitment,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"akita_batch_statement"));
    setup_key.append_to_transcript(transcript);
    packed_commitment.append_to_transcript(transcript);
    transcript.append_bytes(&statement.layout_digest);
    append_field_slice(transcript, b"akita_logical_point", &statement.logical_point);
    append_field_slice(transcript, b"akita_pcs_point", &statement.pcs_point);
    transcript.append(&LabelWithCount(
        b"akita_claims",
        statement.claims.len() as u64,
    ));
    for claim in &statement.claims {
        claim
            .commitment
            .clone()
            .into_akita_commitment()
            .append_to_transcript(transcript);
        claim.claim.append_to_transcript(transcript);
        claim.scale.append_to_transcript(transcript);
        bind_physical_view(&statement.layout_digest, &claim.view, transcript);
    }
    append_field_slice(transcript, b"akita_coefficients", coefficients);
    reduced_opening.append_to_transcript(transcript);
}

fn bind_physical_view<F, T>(
    statement_layout_digest: &[u8; 32],
    view: &PhysicalView<F>,
    transcript: &mut T,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match view {
        PhysicalView::Direct => transcript.append_bytes(&[0]),
        PhysicalView::PackedLinear {
            layout_digest,
            coefficients,
        } => {
            transcript.append_bytes(&[1]);
            transcript.append_bytes(statement_layout_digest);
            transcript.append_bytes(layout_digest);
            append_field_slice(transcript, b"akita_view_coeffs", coefficients);
        }
    }
}

fn validate_physical_claims<F, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<F, AkitaCommitment, OpeningId, RelationId>,
    polynomial: &Polynomial<F>,
) -> Result<(), OpeningsError>
where
    F: Field,
{
    let physical_eval = polynomial.evaluate(&statement.pcs_point);
    for claim in &statement.claims {
        let scale = physical_view_scale(&statement.layout_digest, &claim.view)?;
        if claim.claim * scale != physical_eval {
            return Err(OpeningsError::VerificationFailed);
        }
    }
    Ok(())
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
            coefficients,
        } => {
            if layout_digest != statement_layout_digest {
                return Err(OpeningsError::InvalidBatch(
                    "packed view layout digest does not match statement layout digest".to_owned(),
                ));
            }
            if coefficients.is_empty() {
                return Err(OpeningsError::InvalidBatch(
                    "packed linear view requires at least one coefficient".to_owned(),
                ));
            }
            Ok(coefficients
                .iter()
                .copied()
                .fold(F::zero(), |acc, coefficient| acc + coefficient))
        }
    }
}

fn bind_single_opening_statement<F, T>(
    setup_key: &AkitaSetupKey,
    commitment: &AkitaCommitment,
    point: &[F],
    eval: F,
    transcript: &mut T,
) -> F
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"akita_single_opening"));
    setup_key.append_to_transcript(transcript);
    commitment.append_to_transcript(transcript);
    append_field_slice(transcript, b"akita_single_point", point);
    eval.append_to_transcript(transcript);
    transcript.challenge_scalar()
}

fn commitment_digest<F: Field>(
    setup_key: &AkitaSetupKey,
    layout_digest: &[u8; 32],
    d_pack: usize,
    evaluations: &[F],
) -> [u8; 32] {
    let mut transcript = Blake2bTranscript::<F>::new(b"akita-commit");
    setup_key.append_to_transcript(&mut transcript);
    transcript.append_bytes(layout_digest);
    transcript.append(&u64_field::<F>(d_pack as u64));
    append_field_slice(&mut transcript, b"akita_commit_evals", evaluations);
    transcript.state()
}

fn polynomial_evaluations<F, P>(polynomial: &P) -> Vec<F>
where
    F: Field,
    P: MultilinearPoly<F> + ?Sized,
{
    let capacity = if polynomial.num_vars() < usize::BITS as usize {
        1usize << polynomial.num_vars()
    } else {
        0
    };
    let mut evals = Vec::with_capacity(capacity);
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}

fn validate_setup_dimension(setup_key: &AkitaSetupKey, d_pack: usize) -> Result<(), OpeningsError> {
    if setup_key.accepts_dimension(d_pack) {
        Ok(())
    } else {
        Err(OpeningsError::InvalidSetup(format!(
            "Akita setup dimension {} does not accept PackedWitness dimension {d_pack}",
            setup_key.d_setup
        )))
    }
}

fn validate_evaluation_shape(d_pack: usize, evaluation_count: usize) -> Result<(), OpeningsError> {
    if d_pack >= usize::BITS as usize {
        return Err(OpeningsError::InvalidSetup(format!(
            "PackedWitness dimension {d_pack} exceeds usize bit width"
        )));
    }
    let expected = 1usize << d_pack;
    if evaluation_count != expected {
        return Err(OpeningsError::InvalidBatch(format!(
            "PackedWitness evaluation count {evaluation_count} does not match dimension {d_pack}"
        )));
    }
    Ok(())
}

fn transparent_zk_error() -> OpeningsError {
    OpeningsError::InvalidBatch(
        "Akita mock backend is transparent-only and does not support ZK openings".to_owned(),
    )
}
