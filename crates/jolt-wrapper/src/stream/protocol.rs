use jolt_crypto::Bn254;
use jolt_field::{Field, Fr, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::AdditivelyHomomorphic;
use jolt_poly::{EqPolynomial, MultilinearPoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_transcript::{AppendToTranscript, Blake3Transcript, Transcript};

use super::{
    absorb_output_claims, combine_evaluations, prove_stage, verify_stage_with,
    verify_stage_without_output, ClaimReduction, ColumnBatching, Commitment, PackedColumns,
    PackingLayout, ReductionClaim, StageMember, StageMemberSpec, StageProof, StageResult,
    StreamError, TensorStreamStatement, TensorTerm, WrapperProof, STREAM_LABEL,
};

pub fn prove_stream(
    packed: &PackedColumns,
    statement: &TensorStreamStatement,
    row_prover: &mut dyn ProveRounds<Fr>,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, StreamError> {
    validate_statement(packed.layout, statement)?;
    if row_prover.num_rounds() != packed.layout.row_vars() {
        return Err(StreamError::PointDimension {
            expected: packed.layout.row_vars(),
            actual: row_prover.num_rounds(),
        });
    }
    let mut transcript = new_stream_transcript(
        &statement.key_digest,
        &[statement.row_input_claim],
        &packed.commitments,
    );
    let mut row_members = [StageMember {
        prover: row_prover,
        input_claim: statement.row_input_claim,
        degree: statement.row_degree,
        offset: 0,
    }];
    let (row_stage, row_result) = prove_stage(&mut row_members, &mut transcript)?;
    let row_output = single_member_output(&row_result)?;
    let column_values = packed.column_evaluations(&row_result.point)?;
    let mut column_batch = ColumnBatching::new(column_values, statement.terms.clone())?;
    let column_input = column_batch.input_claim();
    if column_input != row_output {
        return Err(StreamError::StageLink);
    }
    let mut column_members = [StageMember {
        prover: &mut column_batch,
        input_claim: column_input,
        degree: 2,
        offset: 0,
    }];
    let (column_stage, column_result) = prove_stage(&mut column_members, &mut transcript)?;
    let reduced_claims = column_batch.final_evaluations();
    let expected_column = ColumnBatching::expected_final(
        packed.layout.padded_column_count,
        &statement.terms,
        &column_result.point,
        &reduced_claims,
    )?;
    if column_result.output_claims != [expected_column] {
        return Err(StreamError::StageOutputClaim);
    }
    let claims = canonical_reduction_claims(
        packed.layout,
        &row_result.point,
        &column_result.point,
        &reduced_claims,
    )?;
    prove_reduced_opening(
        packed,
        vec![row_stage, column_stage],
        vec![row_result.output_claims, column_result.output_claims],
        claims,
        setup,
        &mut transcript,
    )
}

fn prove_reduced_opening(
    packed: &PackedColumns,
    mut stages: Vec<StageProof>,
    mut stage_claims: Vec<Vec<Fr>>,
    claims: Vec<ReductionClaim>,
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut Blake3Transcript<Fr>,
) -> Result<WrapperProof, StreamError> {
    for claim in &claims {
        transcript.append(&claim.value);
    }
    let coefficients: Vec<Fr> = claims.iter().map(|_| transcript.challenge()).collect();
    let mut reduction = ClaimReduction::new(&packed.evaluations, &claims, &coefficients)?;
    let input_claim = reduction.input_claim();
    let mut members = [StageMember {
        prover: &mut reduction,
        input_claim,
        degree: 2,
        offset: 0,
    }];
    let (stage, result) = prove_stage(&mut members, transcript)?;
    stages.push(stage);
    stage_claims.push(result.output_claims.clone());
    let weights = reduction.final_weights();
    let combined_evaluations = combine_evaluations(&packed.evaluations, &weights);
    let claimed_eval = combined_evaluations.as_slice().evaluate(&result.point);
    if result.output_claims != [claimed_eval] {
        return Err(StreamError::OpeningClaim);
    }
    let opening =
        HyperKZGScheme::<Bn254>::open(setup, &combined_evaluations, &result.point, transcript)
            .map_err(StreamError::HyperKzg)?;
    Ok(WrapperProof {
        commitments: packed.commitments.clone(),
        stages,
        stage_claims,
        reduced_claims: claims.into_iter().map(|claim| claim.value).collect(),
        opening,
    })
}

pub fn verify_stream(
    proof: &WrapperProof,
    statement: &TensorStreamStatement,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<Vec<StageResult>, StreamError> {
    let layout = PackingLayout::new(statement.rows, statement.column_count, statement.k)?;
    validate_statement(layout, statement)?;
    let arity = tensor_arity(&statement.terms)?;
    if proof.stages.len() != 3
        || proof.stage_claims.len() != 3
        || proof.commitments.len() != layout.group_count
        || proof.reduced_claims.len() != arity
        || proof.opening.com.len() + 1 != layout.packed_vars()
    {
        return Err(StreamError::StageCount);
    }
    let mut transcript = new_stream_transcript(
        &statement.key_digest,
        &[statement.row_input_claim],
        &proof.commitments,
    );
    let row_stage = proof.stages.first().ok_or(StreamError::StageCount)?;
    let row_shape = [StageMemberSpec {
        rounds: layout.row_vars(),
        degree: statement.row_degree,
        offset: 0,
    }];
    let row_result = verify_stage_with(
        row_stage,
        &row_shape,
        &[statement.row_input_claim],
        &mut transcript,
        |result| checked_stage_claims(proof, 0, vec![single_member_output(result)?]),
    )?;
    let row_output = single_member_output(&row_result)?;
    let column_stage = proof.stages.get(1).ok_or(StreamError::StageCount)?;
    let column_shape = [StageMemberSpec {
        rounds: arity * layout.column_vars(),
        degree: 2,
        offset: 0,
    }];
    let column_result = verify_stage_with(
        column_stage,
        &column_shape,
        &[row_output],
        &mut transcript,
        |result| {
            checked_stage_claims(
                proof,
                1,
                vec![ColumnBatching::expected_final(
                    layout.padded_column_count,
                    &statement.terms,
                    &result.point,
                    &proof.reduced_claims,
                )?],
            )
        },
    )?;
    let claims = canonical_reduction_claims(
        layout,
        &row_result.point,
        &column_result.point,
        &proof.reduced_claims,
    )?;
    let mut results = vec![row_result, column_result];
    results.push(verify_reduced_opening(
        proof,
        layout,
        &claims,
        setup,
        &mut transcript,
    )?);
    Ok(results)
}

fn verify_reduced_opening(
    proof: &WrapperProof,
    layout: PackingLayout,
    claims: &[ReductionClaim],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut Blake3Transcript<Fr>,
) -> Result<StageResult, StreamError> {
    for value in &proof.reduced_claims {
        transcript.append(value);
    }
    let coefficients: Vec<Fr> = claims.iter().map(|_| transcript.challenge()).collect();
    let final_shape = [StageMemberSpec {
        rounds: layout.packed_vars(),
        degree: 2,
        offset: 0,
    }];
    let final_stage = proof.stages.get(2).ok_or(StreamError::StageCount)?;
    let expected_input: Fr = proof
        .reduced_claims
        .iter()
        .zip(&coefficients)
        .map(|(&value, &coefficient)| value * coefficient)
        .sum();
    let mut final_result =
        verify_stage_without_output(final_stage, &final_shape, &[expected_input], transcript)?;
    let weights = reduction_weights(
        proof.commitments.len(),
        claims,
        &coefficients,
        &final_result.point,
    )?;
    let commitment = HyperKZGScheme::<Bn254>::combine(&proof.commitments, &weights);
    let coefficient_inverse = final_result
        .coefficients
        .first()
        .copied()
        .ok_or(StreamError::StageMemberCount)?
        .inverse()
        .ok_or(StreamError::StageOutputClaim)?;
    let claimed_eval = final_result.final_claim * coefficient_inverse;
    let output_claims = checked_stage_claims(proof, 2, vec![claimed_eval])?;
    absorb_output_claims(&output_claims, transcript);
    final_result.output_claims = vec![claimed_eval];
    HyperKZGScheme::<Bn254>::verify(
        setup,
        &commitment,
        &final_result.point,
        &claimed_eval,
        &proof.opening,
        transcript,
    )
    .map_err(StreamError::HyperKzg)?;
    Ok(final_result)
}

fn checked_stage_claims(
    proof: &WrapperProof,
    stage: usize,
    expected: Vec<Fr>,
) -> Result<Vec<Fr>, StreamError> {
    if proof.stage_claims.get(stage) != Some(&expected) {
        return Err(StreamError::StageOutputClaim);
    }
    Ok(expected)
}

pub fn new_stream_transcript(
    key_digest: &[u8; 32],
    public_statement: &[Fr],
    commitments: &[Commitment],
) -> Blake3Transcript<Fr> {
    let mut transcript = Blake3Transcript::<Fr>::new(STREAM_LABEL);
    transcript.append_bytes(key_digest);
    for value in public_statement {
        transcript.append(value);
    }
    absorb_commitments(commitments, &mut transcript);
    transcript
}

pub fn absorb_commitments<T: Transcript<Challenge = Fr>>(
    commitments: &[Commitment],
    transcript: &mut T,
) {
    for commitment in commitments {
        commitment.append_to_transcript(transcript);
    }
}

fn reduction_weights(
    polynomial_count: usize,
    claims: &[ReductionClaim],
    coefficients: &[Fr],
    point: &[Fr],
) -> Result<Vec<Fr>, StreamError> {
    let mut weights = vec![Fr::zero(); polynomial_count];
    for (claim_index, (claim, &coefficient)) in claims.iter().zip(coefficients).enumerate() {
        if claim.polynomial_weights.len() != polynomial_count {
            return Err(StreamError::PolynomialWeightCount {
                claim: claim_index,
                expected: polynomial_count,
                actual: claim.polynomial_weights.len(),
            });
        }
        if claim.point.len() != point.len() {
            return Err(StreamError::PointDimension {
                expected: point.len(),
                actual: claim.point.len(),
            });
        }
        let eq = EqPolynomial::<Fr>::mle(&claim.point, point);
        for (weight, &polynomial_coefficient) in weights.iter_mut().zip(&claim.polynomial_weights) {
            *weight += coefficient * polynomial_coefficient * eq;
        }
    }
    Ok(weights)
}

fn canonical_reduction_claims(
    layout: PackingLayout,
    row_point: &[Fr],
    column_batch_point: &[Fr],
    values: &[Fr],
) -> Result<Vec<ReductionClaim>, StreamError> {
    if column_batch_point.len() != values.len() * layout.column_vars() {
        return Err(StreamError::PointDimension {
            expected: values.len() * layout.column_vars(),
            actual: column_batch_point.len(),
        });
    }
    column_batch_point
        .chunks_exact(layout.column_vars())
        .zip(values)
        .map(|(column_point, &value)| {
            Ok(ReductionClaim {
                polynomial_weights: layout.group_weights(column_point)?,
                point: layout.packed_point(row_point, column_point)?,
                value,
            })
        })
        .collect()
}

fn tensor_arity(terms: &[TensorTerm]) -> Result<usize, StreamError> {
    terms
        .first()
        .map(|term| term.columns.len())
        .filter(|arity| *arity > 0)
        .ok_or(StreamError::EmptyTensor)
}

fn single_member_output(result: &StageResult) -> Result<Fr, StreamError> {
    let coefficient = result
        .coefficients
        .first()
        .copied()
        .ok_or(StreamError::StageMemberCount)?;
    if result.coefficients.len() != 1 {
        return Err(StreamError::StageMemberCount);
    }
    Ok(result.final_claim * coefficient.inverse().ok_or(StreamError::StageOutputClaim)?)
}

fn validate_statement(
    layout: PackingLayout,
    statement: &TensorStreamStatement,
) -> Result<(), StreamError> {
    if layout.rows != statement.rows
        || layout.column_count != statement.column_count
        || layout.k != statement.k
    {
        return Err(StreamError::StageLink);
    }
    if statement.row_degree == 0 {
        return Err(StreamError::StageOutputClaim);
    }
    for (term_index, term) in statement.terms.iter().enumerate() {
        if term.columns.len() != tensor_arity(&statement.terms)? {
            return Err(StreamError::TensorArity {
                term: term_index,
                expected: tensor_arity(&statement.terms)?,
                actual: term.columns.len(),
            });
        }
        if let Some(&column) = term
            .columns
            .iter()
            .find(|&&column| column >= layout.padded_column_count)
        {
            return Err(StreamError::ColumnOutOfRange {
                column,
                columns: layout.padded_column_count,
            });
        }
    }
    Ok(())
}
