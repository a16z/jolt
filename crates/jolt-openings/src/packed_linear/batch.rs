use jolt_crypto::Commitment;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::{
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    ZkBatchOpeningScheme, ZkOpeningScheme,
};

use super::{
    reduction::{
        has_packed_linear_view, invalid_batch, polynomial_evaluations,
        prove_packed_linear_reduction, validate_packed_linear_statement,
        verify_packed_linear_reduction,
    },
    types::{PackedLinearBatch, PackedLinearBatchBackend, PackedLinearBatchProof},
};

impl<PCS> Commitment for PackedLinearBatch<PCS>
where
    PCS: CommitmentScheme,
{
    type Output = PCS::Output;
}

impl<PCS> CommitmentScheme for PackedLinearBatch<PCS>
where
    PCS: CommitmentScheme,
{
    type Field = PCS::Field;
    type Proof = PackedLinearBatchProof<PCS::Proof>;
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
        PackedLinearBatchProof {
            reduction: None,
            native: PCS::open(poly, point, eval, setup, hint, transcript),
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
        if proof.reduction.is_some() {
            return Err(OpeningsError::VerificationFailed);
        }
        PCS::verify(commitment, point, eval, &proof.native, setup, transcript)
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        PCS::bind_opening_inputs(transcript, point, eval);
    }
}

impl<PCS> BatchOpeningScheme for PackedLinearBatch<PCS>
where
    PCS: PackedLinearBatchBackend,
    PCS::Output: AppendToTranscript,
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
        if !has_packed_linear_view(statement) {
            let native = PCS::prove_batch(setup, transcript, statement, polynomials, hints)?;
            return Ok(PackedLinearBatchProof {
                reduction: None,
                native,
            });
        }

        let layout = PCS::prover_layout(setup)
            .ok_or_else(|| invalid_batch("packed linear opening requires setup layout"))?;
        let commitment = validate_packed_linear_statement(layout, statement)?;
        PCS::validate_packed_prover_inputs(setup, layout, &commitment, polynomials, &hints)?;
        let hint = hints
            .into_iter()
            .next()
            .ok_or_else(|| invalid_batch("packed linear proof requires one opening hint"))?;
        PCS::bind_packed_prover_setup(setup, transcript);
        let reduction = prove_packed_linear_reduction(
            layout,
            statement,
            polynomial_evaluations(&polynomials[0]),
            transcript,
        )?;
        let native = PCS::open(
            &polynomials[0],
            &reduction.opening_point,
            reduction.opening_eval,
            setup,
            Some(hint),
            transcript,
        );
        Ok(PackedLinearBatchProof {
            reduction: Some(reduction.proof),
            native,
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
        if !has_packed_linear_view(statement) {
            if proof.reduction.is_some() {
                return Err(OpeningsError::VerificationFailed);
            }
            return PCS::verify_batch(setup, transcript, statement, &proof.native);
        }

        let reduction_proof = proof
            .reduction
            .as_ref()
            .ok_or(OpeningsError::VerificationFailed)?;
        let layout = PCS::verifier_layout(setup)
            .ok_or_else(|| invalid_batch("packed linear opening requires setup layout"))?;
        let commitment = validate_packed_linear_statement(layout, statement)?;
        PCS::validate_packed_verifier_inputs(setup, layout, &commitment)?;
        PCS::bind_packed_verifier_setup(setup, transcript);
        let reduction =
            verify_packed_linear_reduction(layout, statement, reduction_proof, transcript)?;
        PCS::verify(
            &reduction.result.joint_commitment,
            &reduction.opening_point,
            reduction.opening_eval,
            &proof.native,
            setup,
            transcript,
        )?;
        Ok(reduction.result)
    }
}

impl<PCS> ZkOpeningScheme for PackedLinearBatch<PCS>
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
        let (native, hiding, blind) = PCS::open_zk(poly, point, eval, setup, hint, transcript);
        (
            PackedLinearBatchProof {
                reduction: None,
                native,
            },
            hiding,
            blind,
        )
    }

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<Self::HidingCommitment, OpeningsError> {
        if proof.reduction.is_some() {
            return Err(OpeningsError::VerificationFailed);
        }
        PCS::verify_zk(commitment, point, &proof.native, setup, transcript)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        PCS::bind_zk_opening_inputs(transcript, point, hiding_commitment);
    }
}

impl<PCS> ZkBatchOpeningScheme for PackedLinearBatch<PCS>
where
    PCS: PackedLinearBatchBackend + ZkBatchOpeningScheme,
    PCS::Output: AppendToTranscript,
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
        if has_packed_linear_view(statement) {
            return Err(invalid_batch(
                "packed linear batch openings do not support ZK mode yet",
            ));
        }
        let (native, hiding, blind) =
            PCS::prove_batch_zk(setup, transcript, statement, evals, polynomials, hints)?;
        Ok((
            PackedLinearBatchProof {
                reduction: None,
                native,
            },
            hiding,
            blind,
        ))
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
        if has_packed_linear_view(statement) {
            return Err(invalid_batch(
                "packed linear batch openings do not support ZK mode yet",
            ));
        }
        if proof.reduction.is_some() {
            return Err(OpeningsError::VerificationFailed);
        }
        PCS::verify_batch_zk(setup, transcript, statement, &proof.native)
    }
}
