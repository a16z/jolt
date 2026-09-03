use jolt_crypto::Bn254;
use jolt_field::{Field, Fr, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::AdditivelyHomomorphic;
use jolt_poly::{EqPolynomial, MultilinearPoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_transcript::{AppendToTranscript, Blake3Transcript, Transcript};

use super::{
    absorb_output_claims, combine_evaluations, prove_kzg_stage, prove_stage, verify_kzg_stage,
    verify_stage_with, verify_stage_without_output, ClaimReduction, ColumnReduction, Commitment,
    PackedColumns, PackingLayout, ReductionClaim, StageAEncoding, StageMember, StageMemberSpec,
    StageProof, StageResult, StreamError, TensorStreamStatement, TensorTerm, WrapperProof,
    STREAM_LABEL,
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
    absorb_stage_a_encoding(statement.stage_a_encoding, &mut transcript);
    let (row_stage, row_result) = match statement.stage_a_encoding {
        StageAEncoding::Compressed => {
            let mut row_members = [StageMember {
                prover: row_prover,
                input_claim: statement.row_input_claim,
                degree: statement.row_degree,
                offset: 0,
            }];
            prove_stage(&mut row_members, &mut transcript)?
        }
        StageAEncoding::KzgCommitted => prove_kzg_stage(
            row_prover,
            statement.row_input_claim,
            statement.row_degree,
            setup,
            &mut transcript,
        )?,
    };
    let row_output = single_member_output(&row_result)?;
    let column_values = packed.column_evaluations(&row_result.point)?;
    let factor_columns = tensor_factor_columns(&statement.terms);
    let factor_claims = factor_columns
        .iter()
        .map(|&column| {
            column_values
                .get(column)
                .copied()
                .ok_or(StreamError::ColumnOutOfRange {
                    column,
                    columns: column_values.len(),
                })
        })
        .collect::<Result<Vec<_>, StreamError>>()?;
    if tensor_value(&statement.terms, &factor_claims)? != row_output {
        return Err(StreamError::StageLink);
    }
    let mut reductions = factor_columns
        .iter()
        .map(|&column| ColumnReduction::new(column_values.clone(), column))
        .collect::<Result<Vec<_>, StreamError>>()?;
    let mut column_members = reductions
        .iter_mut()
        .zip(&factor_claims)
        .map(|(reduction, &input_claim)| StageMember {
            prover: reduction as &mut dyn ProveRounds<Fr>,
            input_claim,
            degree: 2,
            offset: 0,
        })
        .collect::<Vec<_>>();
    let (column_stage, column_result) = prove_stage(&mut column_members, &mut transcript)?;
    drop(column_members);
    let reduced_claim = reductions
        .first()
        .map(ColumnReduction::final_evaluation)
        .ok_or(StreamError::EmptyTensor)?;
    if reductions
        .iter()
        .any(|reduction| reduction.final_evaluation() != reduced_claim)
    {
        return Err(StreamError::OpeningClaim);
    }
    let expected_column = factor_columns
        .iter()
        .map(|&column| {
            ColumnReduction::expected_final(
                packed.layout.padded_column_count,
                column,
                &column_result.point,
                reduced_claim,
            )
        })
        .collect::<Result<Vec<_>, StreamError>>()?;
    if column_result.output_claims != expected_column {
        return Err(StreamError::StageOutputClaim);
    }
    let claims = canonical_reduction_claims(
        packed.layout,
        &row_result.point,
        &column_result.point,
        reduced_claim,
    )?;
    prove_reduced_opening(
        packed,
        vec![row_stage, column_stage],
        vec![factor_claims],
        claims,
        setup,
        &mut transcript,
    )
}

fn prove_reduced_opening(
    packed: &PackedColumns,
    mut stages: Vec<StageProof>,
    stage_claims: Vec<Vec<Fr>>,
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
    let factor_columns = tensor_factor_columns(&statement.terms);
    if proof.stages.len() != 3
        || proof.stage_claims.len() != 1
        || proof.stage_claims.first().map(Vec::len) != Some(factor_columns.len())
        || proof.commitments.len() != layout.group_count
        || proof.reduced_claims.len() != 1
        || proof.opening.com.len() + 1 != layout.packed_vars()
    {
        return Err(StreamError::StageCount);
    }
    let mut transcript = new_stream_transcript(
        &statement.key_digest,
        &[statement.row_input_claim],
        &proof.commitments,
    );
    absorb_stage_a_encoding(statement.stage_a_encoding, &mut transcript);
    let row_stage = proof.stages.first().ok_or(StreamError::StageCount)?;
    let row_shape = [StageMemberSpec {
        rounds: layout.row_vars(),
        degree: statement.row_degree,
        offset: 0,
    }];
    let row_result = match statement.stage_a_encoding {
        StageAEncoding::Compressed => verify_stage_with(
            row_stage,
            &row_shape,
            &[statement.row_input_claim],
            &mut transcript,
            |result| Ok(vec![single_member_output(result)?]),
        )?,
        StageAEncoding::KzgCommitted => verify_kzg_stage(
            row_stage,
            statement.row_input_claim,
            layout.row_vars(),
            statement.row_degree,
            setup,
            &mut transcript,
        )?,
    };
    let row_output = single_member_output(&row_result)?;
    let factor_claims = proof.stage_claims.first().ok_or(StreamError::StageCount)?;
    if tensor_value(&statement.terms, factor_claims)? != row_output {
        return Err(StreamError::StageLink);
    }
    let reduced_claim = proof
        .reduced_claims
        .first()
        .copied()
        .ok_or(StreamError::StageCount)?;
    let column_stage = proof.stages.get(1).ok_or(StreamError::StageCount)?;
    let column_shape = vec![
        StageMemberSpec {
            rounds: layout.column_vars(),
            degree: 2,
            offset: 0,
        };
        factor_columns.len()
    ];
    let column_result = verify_stage_with(
        column_stage,
        &column_shape,
        factor_claims,
        &mut transcript,
        |result| {
            factor_columns
                .iter()
                .map(|&column| {
                    ColumnReduction::expected_final(
                        layout.padded_column_count,
                        column,
                        &result.point,
                        reduced_claim,
                    )
                })
                .collect()
        },
    )?;
    let claims = canonical_reduction_claims(
        layout,
        &row_result.point,
        &column_result.point,
        reduced_claim,
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
    let output_claims = vec![claimed_eval];
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

fn absorb_stage_a_encoding<T: Transcript>(encoding: StageAEncoding, transcript: &mut T) {
    let tag = match encoding {
        StageAEncoding::Compressed => 0,
        StageAEncoding::KzgCommitted => 1,
    };
    transcript.append_bytes(&[tag]);
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
    value: Fr,
) -> Result<Vec<ReductionClaim>, StreamError> {
    if column_batch_point.len() != layout.column_vars() {
        return Err(StreamError::PointDimension {
            expected: layout.column_vars(),
            actual: column_batch_point.len(),
        });
    }
    Ok(vec![ReductionClaim {
        polynomial_weights: layout.group_weights(column_batch_point)?,
        point: layout.packed_point(row_point, column_batch_point)?,
        value,
    }])
}

fn tensor_factor_columns(terms: &[TensorTerm]) -> Vec<usize> {
    terms
        .iter()
        .flat_map(|term| term.columns.iter().copied())
        .collect()
}

fn tensor_value(terms: &[TensorTerm], factors: &[Fr]) -> Result<Fr, StreamError> {
    let arity = tensor_arity(terms)?;
    if factors.len() != terms.len() * arity {
        return Err(StreamError::StageMemberCount);
    }
    Ok(terms
        .iter()
        .zip(factors.chunks_exact(arity))
        .map(|(term, values)| {
            values
                .iter()
                .fold(term.coefficient, |product, value| product * *value)
        })
        .sum())
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
