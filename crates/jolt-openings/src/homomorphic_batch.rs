use jolt_crypto::HomomorphicCommitment;
use jolt_field::{Field, FromPrimitiveInt};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::{
    AdditivelyHomomorphic, BatchOpeningResult, BatchOpeningScheme, BatchOpeningStatement,
    CommitmentScheme, OpeningsError, PhysicalView, ZkBatchOpeningScheme, ZkOpeningScheme,
};

impl<PCS> BatchOpeningScheme for PCS
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
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
        validate_batch_prover_inputs::<Self, _, _, _>(statement, polynomials, hints.len())?;
        let reduction = clear_homomorphic_batch_reduction::<Self, _, _, _>(statement, transcript)?;
        let joint_polynomial = combine_polynomials::<Self>(polynomials, &reduction.proof_scalars)?;
        let combined_hint = Self::combine_hints(hints, &reduction.proof_scalars);
        let proof = Self::open(
            &joint_polynomial,
            statement.pcs_point.as_slice(),
            reduction.joint_eval,
            setup,
            Some(combined_hint),
            transcript,
        );
        Self::bind_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &reduction.joint_eval,
        );
        Ok(proof)
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
        validate_batch_statement(statement)?;
        let reduction = clear_homomorphic_batch_reduction::<Self, _, _, _>(statement, transcript)?;
        let joint_commitment =
            combine_commitments::<Self, _, _, _>(statement, &reduction.proof_scalars);
        Self::verify(
            &joint_commitment,
            statement.pcs_point.as_slice(),
            reduction.joint_eval,
            proof,
            setup,
            transcript,
        )?;
        Self::bind_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &reduction.joint_eval,
        );
        Ok(BatchOpeningResult {
            coefficients: reduction.logical_coefficients,
            joint_commitment,
            reduced_opening: reduction.joint_eval,
        })
    }
}

impl<PCS> ZkBatchOpeningScheme for PCS
where
    PCS: AdditivelyHomomorphic + ZkOpeningScheme,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
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
        validate_batch_prover_inputs::<Self, _, _, _>(statement, polynomials, hints.len())?;
        if evals.len() != statement.claims.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "evaluation count {} does not match claim count {}",
                evals.len(),
                statement.claims.len()
            )));
        }

        let reduction =
            hidden_homomorphic_batch_reduction::<Self, _, _, _>(statement, evals, transcript)?;
        let joint_polynomial = combine_polynomials::<Self>(polynomials, &reduction.proof_scalars)?;
        let combined_hint = Self::combine_hints(hints, &reduction.proof_scalars);
        let (proof, hiding_commitment, blind) = Self::open_zk(
            &joint_polynomial,
            statement.pcs_point.as_slice(),
            reduction.joint_eval,
            setup,
            combined_hint,
            transcript,
        );
        Self::bind_zk_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &hiding_commitment,
        );
        Ok((proof, hiding_commitment, blind))
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
        validate_batch_statement(statement)?;
        let scalars = hidden_homomorphic_batch_scalars::<Self, _, _, _>(statement, transcript)?;
        let joint_commitment =
            combine_commitments::<Self, _, _, _>(statement, &scalars.proof_scalars);
        let hiding_commitment = Self::verify_zk(
            &joint_commitment,
            statement.pcs_point.as_slice(),
            proof,
            setup,
            transcript,
        )?;
        Self::bind_zk_opening_inputs(
            transcript,
            statement.logical_point.as_slice(),
            &hiding_commitment,
        );
        Ok(BatchOpeningResult {
            coefficients: scalars.logical_coefficients,
            joint_commitment,
            reduced_opening: hiding_commitment,
        })
    }
}

struct HomomorphicBatchReduction<F> {
    proof_scalars: Vec<F>,
    logical_coefficients: Vec<F>,
    joint_eval: F,
}

struct HomomorphicBatchScalars<F> {
    proof_scalars: Vec<F>,
    logical_coefficients: Vec<F>,
}

fn clear_homomorphic_batch_reduction<PCS, T, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId>,
    transcript: &mut T,
) -> Result<HomomorphicBatchReduction<PCS::Field>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    validate_batch_statement(statement)?;
    let scaled_claims = statement
        .claims
        .iter()
        .map(|claim| claim.claim * claim.scale)
        .collect::<Vec<_>>();
    transcript.append(&LabelWithCount(b"rlc_claims", scaled_claims.len() as u64));
    for claim in &scaled_claims {
        claim.append_to_transcript(transcript);
    }
    let gamma_powers = transcript.challenge_scalar_powers(scaled_claims.len());
    let logical_coefficients = statement
        .claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| *gamma * claim.scale)
        .collect::<Vec<_>>();
    let joint_eval = gamma_powers
        .iter()
        .zip(&scaled_claims)
        .fold(PCS::Field::from_u64(0), |acc, (gamma, claim)| {
            acc + *gamma * *claim
        });
    Ok(HomomorphicBatchReduction {
        proof_scalars: gamma_powers,
        logical_coefficients,
        joint_eval,
    })
}

fn hidden_homomorphic_batch_reduction<PCS, T, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, ()>,
    evals: &[PCS::Field],
    transcript: &mut T,
) -> Result<HomomorphicBatchReduction<PCS::Field>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let scalars = hidden_homomorphic_batch_scalars::<PCS, _, _, _>(statement, transcript)?;
    let joint_eval = scalars
        .proof_scalars
        .iter()
        .zip(evals)
        .fold(PCS::Field::from_u64(0), |acc, (scalar, eval)| {
            acc + *scalar * *eval
        });
    Ok(HomomorphicBatchReduction {
        proof_scalars: scalars.proof_scalars,
        logical_coefficients: scalars.logical_coefficients,
        joint_eval,
    })
}

fn hidden_homomorphic_batch_scalars<PCS, T, OpeningId, RelationId>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, ()>,
    transcript: &mut T,
) -> Result<HomomorphicBatchScalars<PCS::Field>, OpeningsError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    validate_batch_statement(statement)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let logical_coefficients = statement
        .claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| *gamma * claim.scale)
        .collect();
    Ok(HomomorphicBatchScalars {
        proof_scalars: gamma_powers,
        logical_coefficients,
    })
}

fn validate_batch_statement<F, C, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId, Claim>,
) -> Result<(), OpeningsError>
where
    F: Field,
{
    if statement.claims.is_empty() {
        return Err(OpeningsError::InvalidBatch(
            "batch opening requires at least one claim".to_owned(),
        ));
    }
    for claim in &statement.claims {
        if !matches!(claim.view, PhysicalView::Direct) {
            return Err(OpeningsError::InvalidBatch(
                "homomorphic batch openings only support direct physical views".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_batch_prover_inputs<PCS, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, Claim>,
    polynomials: &[PCS::Polynomial],
    hint_count: usize,
) -> Result<(), OpeningsError>
where
    PCS: CommitmentScheme,
{
    validate_batch_statement(statement)?;
    if polynomials.len() != statement.claims.len() {
        return Err(OpeningsError::InvalidBatch(format!(
            "polynomial count {} does not match claim count {}",
            polynomials.len(),
            statement.claims.len()
        )));
    }
    if hint_count != statement.claims.len() {
        return Err(OpeningsError::InvalidBatch(format!(
            "hint count {hint_count} does not match claim count {}",
            statement.claims.len()
        )));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != statement.pcs_point.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "polynomial has {} variables but opening point has {}",
                polynomial.num_vars(),
                statement.pcs_point.len()
            )));
        }
    }
    Ok(())
}

fn combine_commitments<PCS, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<PCS::Field, PCS::Output, OpeningId, RelationId, Claim>,
    scalars: &[PCS::Field],
) -> PCS::Output
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field>,
{
    let commitments = statement
        .claims
        .iter()
        .map(|claim| claim.commitment.clone())
        .collect::<Vec<_>>();
    PCS::combine(&commitments, scalars)
}

fn combine_polynomials<PCS>(
    polynomials: &[PCS::Polynomial],
    scalars: &[PCS::Field],
) -> Result<PCS::Polynomial, OpeningsError>
where
    PCS: CommitmentScheme,
{
    debug_assert_eq!(polynomials.len(), scalars.len());
    let mut combined = Vec::new();
    for (polynomial, scalar) in polynomials.iter().zip(scalars) {
        let evals = polynomial_evaluations(polynomial);
        if combined.is_empty() {
            combined = vec![PCS::Field::from_u64(0); evals.len()];
        } else if combined.len() != evals.len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "polynomial evaluation length {} does not match first length {}",
                evals.len(),
                combined.len()
            )));
        }
        for (acc, eval) in combined.iter_mut().zip(evals) {
            *acc += *scalar * eval;
        }
    }
    Ok(PCS::Polynomial::from(combined))
}

fn polynomial_evaluations<F, P>(polynomial: &P) -> Vec<F>
where
    F: Field,
    P: MultilinearPoly<F>,
{
    let mut evals = Vec::with_capacity(1usize << polynomial.num_vars());
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}
