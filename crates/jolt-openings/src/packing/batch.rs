use jolt_crypto::Commitment;
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement, CommitmentLayoutDigest,
    CommitmentScheme, OpeningsError, ZkBatchOpeningScheme, ZkOpeningScheme,
};

use super::{
    reduction::{
        has_packing_view, prove_packing_reduction, validate_packing_statement,
        verify_packing_reduction,
    },
    types::{
        PackingBatch, PackingBatchProof, PackingLayout, PackingProverSetup, PackingSetupParams,
        PackingVerifierSetup,
    },
    util::{invalid_batch, polynomial_evaluations},
};

impl<PCS, L> Commitment for PackingBatch<PCS, L>
where
    PCS: CommitmentScheme,
    L: 'static,
{
    type Output = PCS::Output;
}

impl<PCS, L> CommitmentScheme for PackingBatch<PCS, L>
where
    PCS: CommitmentScheme,
    L: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    type Field = PCS::Field;
    type Proof = PackingBatchProof<PCS::Proof>;
    type ProverSetup = PackingProverSetup<PCS::ProverSetup, L>;
    type VerifierSetup = PackingVerifierSetup<PCS::VerifierSetup, L>;
    type Polynomial = PCS::Polynomial;
    type OpeningHint = PCS::OpeningHint;
    type SetupParams = PackingSetupParams<PCS::SetupParams, L>;

    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        let (prover, verifier) = PCS::setup(params.pcs);
        (
            PackingProverSetup {
                pcs: prover,
                layout: params.layout.clone(),
            },
            PackingVerifierSetup {
                pcs: verifier,
                layout: params.layout,
            },
        )
    }

    fn verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        PackingVerifierSetup {
            pcs: PCS::verifier_setup(&prover_setup.pcs),
            layout: prover_setup.layout.clone(),
        }
    }

    fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        PCS::commit(poly, &setup.pcs)
    }

    fn open(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        PackingBatchProof {
            reduction: None,
            native: PCS::open(poly, point, eval, &setup.pcs, hint, transcript),
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
        PCS::verify(
            commitment,
            point,
            eval,
            &proof.native,
            &setup.pcs,
            transcript,
        )
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        PCS::bind_opening_inputs(transcript, point, eval);
    }
}

impl<PCS, L> BatchOpeningScheme for PackingBatch<PCS, L>
where
    PCS: BatchOpeningScheme,
    PCS::Output: AppendToTranscript + CommitmentLayoutDigest,
    L: PackingLayout + Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
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
        if !has_packing_view(statement) {
            let native = PCS::prove_batch(&setup.pcs, transcript, statement, polynomials, hints)?;
            return Ok(PackingBatchProof {
                reduction: None,
                native,
            });
        }

        let layout = &setup.layout;
        let commitment = validate_packing_statement(layout, statement)?;
        validate_packed_commitment_digest(layout, &commitment)?;
        validate_packed_prover_inputs::<PCS::Field, _, _, _>(layout, polynomials, &hints)?;
        let hint = hints
            .into_iter()
            .next()
            .ok_or_else(|| invalid_batch("packing proof requires one opening hint"))?;
        let reduction = prove_packing_reduction(
            layout,
            statement,
            polynomial_evaluations(&polynomials[0]),
            transcript,
        )?;
        let native = PCS::open(
            &polynomials[0],
            &reduction.opening_point,
            reduction.opening_eval,
            &setup.pcs,
            Some(hint),
            transcript,
        );
        Ok(PackingBatchProof {
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
        if !has_packing_view(statement) {
            if proof.reduction.is_some() {
                return Err(OpeningsError::VerificationFailed);
            }
            return PCS::verify_batch(&setup.pcs, transcript, statement, &proof.native);
        }

        let reduction_proof = proof
            .reduction
            .as_ref()
            .ok_or(OpeningsError::VerificationFailed)?;
        let layout = &setup.layout;
        let commitment = validate_packing_statement(layout, statement)?;
        validate_packed_commitment_digest(layout, &commitment)?;
        let reduction = verify_packing_reduction(layout, statement, reduction_proof, transcript)?;
        PCS::verify(
            &reduction.result.joint_commitment,
            &reduction.opening_point,
            reduction.opening_eval,
            &proof.native,
            &setup.pcs,
            transcript,
        )?;
        Ok(reduction.result)
    }
}

impl<PCS, L> ZkOpeningScheme for PackingBatch<PCS, L>
where
    PCS: ZkOpeningScheme,
    L: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    type HidingCommitment = PCS::HidingCommitment;
    type Blind = PCS::Blind;

    fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        PCS::commit_zk(poly, &setup.pcs)
    }

    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let (native, hiding, blind) = PCS::open_zk(poly, point, eval, &setup.pcs, hint, transcript);
        (
            PackingBatchProof {
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
        PCS::verify_zk(commitment, point, &proof.native, &setup.pcs, transcript)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        PCS::bind_zk_opening_inputs(transcript, point, hiding_commitment);
    }
}

impl<PCS, L> ZkBatchOpeningScheme for PackingBatch<PCS, L>
where
    PCS: ZkBatchOpeningScheme,
    PCS::Output: AppendToTranscript + CommitmentLayoutDigest,
    L: PackingLayout + Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
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
        if has_packing_view(statement) {
            return Err(invalid_batch(
                "packing batch openings do not support ZK mode yet",
            ));
        }
        let (native, hiding, blind) =
            PCS::prove_batch_zk(&setup.pcs, transcript, statement, evals, polynomials, hints)?;
        Ok((
            PackingBatchProof {
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
        if has_packing_view(statement) {
            return Err(invalid_batch(
                "packing batch openings do not support ZK mode yet",
            ));
        }
        if proof.reduction.is_some() {
            return Err(OpeningsError::VerificationFailed);
        }
        PCS::verify_batch_zk(&setup.pcs, transcript, statement, &proof.native)
    }
}

fn validate_packed_commitment_digest<C, L>(layout: &L, commitment: &C) -> Result<(), OpeningsError>
where
    C: CommitmentLayoutDigest,
    L: PackingLayout,
{
    if let Some(commitment_digest) = commitment.layout_digest() {
        if commitment_digest != layout.digest() {
            return Err(invalid_batch(
                "packing commitment layout digest does not match setup layout",
            ));
        }
    }
    Ok(())
}

fn validate_packed_prover_inputs<F, P, H, L>(
    layout: &L,
    polynomials: &[P],
    hints: &[H],
) -> Result<(), OpeningsError>
where
    F: jolt_field::Field,
    P: MultilinearPoly<F>,
    L: PackingLayout,
{
    if polynomials.len() != 1 {
        return Err(invalid_batch(format!(
            "packing proof expects one packed polynomial, got {}",
            polynomials.len()
        )));
    }
    if polynomials[0].num_vars() != layout.dimension() {
        return Err(invalid_batch(format!(
            "packing polynomial has {} variables but layout has {}",
            polynomials[0].num_vars(),
            layout.dimension()
        )));
    }
    if hints.len() != 1 {
        return Err(invalid_batch(format!(
            "packing proof expects one opening hint, got {}",
            hints.len()
        )));
    }
    Ok(())
}
