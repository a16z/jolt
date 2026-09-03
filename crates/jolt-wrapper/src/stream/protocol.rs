use jolt_crypto::Bn254;
use jolt_field::{Fr, Zero};
use jolt_hyperkzg::{
    HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup, NoopVerifierObserver,
    VerifierObserver,
};
use jolt_openings::AdditivelyHomomorphic;
use jolt_poly::MultilinearPoly;
use jolt_sumcheck::prover::ProveRounds;
use jolt_transcript::{AppendToTranscript, Keccak256Transcript, Transcript};

use super::{
    combine_evaluations, prove_kzg_batch_stage, prove_kzg_stage, prove_stage,
    verify_kzg_batch_stage_observed, verify_kzg_stage_observed, verify_stage_with_observed,
    AssemblyStatement, ColumnReduction, Commitment, PackedColumns, PackingLayout, ReductionClaim,
    StageAEncoding, StageMember, StageMemberSpec, StageProof, StageResult, StreamError,
    TensorStreamStatement, TensorTerm, VerifierCost, WrapperProof, STREAM_LABEL,
};

struct CountingKeccakTranscript {
    inner: Keccak256Transcript<Fr>,
    hashes: usize,
}

impl Default for CountingKeccakTranscript {
    fn default() -> Self {
        Self::new(b"")
    }
}

impl Transcript for CountingKeccakTranscript {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Keccak256Transcript::new(label),
            hashes: 1,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.hashes += 1;
        self.inner.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Fr {
        self.hashes += 1;
        self.inner.challenge()
    }

    fn challenge_scalar(&mut self) -> Fr {
        self.hashes += 1;
        self.inner.challenge_scalar()
    }

    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

pub fn prove_assembly<F>(
    packed: &PackedColumns,
    statement: &AssemblyStatement,
    members: &mut [StageMember<'_>],
    setup: &HyperKZGProverSetup<Bn254>,
    final_claims: F,
) -> Result<WrapperProof, StreamError>
where
    F: FnOnce(&StageResult, &[Fr]) -> Result<Vec<Fr>, StreamError>,
{
    validate_assembly(packed.layout, statement, members)?;
    let mut transcript = new_stream_transcript(
        &statement.key_digest,
        &statement.public_inputs,
        &packed.commitments,
    );
    let (row_stage, row_result) = prove_kzg_batch_stage(members, setup, &mut transcript)?;
    let column_values = packed.column_evaluations(&row_result.point)?;
    let factor_claims = statement
        .factor_columns
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
        .collect::<Result<Vec<_>, _>>()?;
    if final_claims(&row_result, &factor_claims)? != row_result.output_claims {
        return Err(StreamError::StageOutputClaim);
    }
    let mut reductions = statement
        .factor_columns
        .iter()
        .map(|&column| ColumnReduction::new(column_values.clone(), column))
        .collect::<Result<Vec<_>, _>>()?;
    let mut column_members = reductions
        .iter_mut()
        .zip(&factor_claims)
        .map(|(reduction, &input_claim)| StageMember {
            prover: reduction,
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
    let expected_column = statement
        .factor_columns
        .iter()
        .map(|&column| {
            ColumnReduction::expected_final(
                packed.layout.padded_column_count,
                column,
                &column_result.point,
                reduced_claim,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if column_result.output_claims != expected_column {
        return Err(StreamError::StageOutputClaim);
    }
    let claim = canonical_reduction_claim(
        packed.layout,
        &row_result.point,
        &column_result.point,
        reduced_claim,
    )?;
    prove_direct_opening(
        packed,
        vec![row_stage, column_stage],
        vec![factor_claims],
        &claim,
        setup,
        &mut transcript,
    )
}

pub fn verify_assembly<F>(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    setup: &HyperKZGVerifierSetup<Bn254>,
    final_claims: F,
) -> Result<Vec<StageResult>, StreamError>
where
    F: Fn(&StageResult, &[Fr], &mut VerifierCost) -> Result<Vec<Fr>, StreamError>,
{
    verify_assembly_with_cost(proof, statement, setup, final_claims).map(|(results, _)| results)
}

pub fn verify_assembly_with_cost<F>(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    setup: &HyperKZGVerifierSetup<Bn254>,
    final_claims: F,
) -> Result<(Vec<StageResult>, VerifierCost), StreamError>
where
    F: Fn(&StageResult, &[Fr], &mut VerifierCost) -> Result<Vec<Fr>, StreamError>,
{
    let mut transcript = seed_stream_transcript::<CountingKeccakTranscript>(
        &statement.key_digest,
        &statement.public_inputs,
        &proof.commitments,
    );
    let mut cost = VerifierCost::default();
    let results = verify_assembly_observed(
        proof,
        statement,
        setup,
        &mut transcript,
        &mut cost,
        final_claims,
    )?;
    cost.keccak = transcript.hashes;
    Ok((results, cost))
}

fn verify_assembly_observed<T, F>(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut VerifierCost,
    final_claims: F,
) -> Result<Vec<StageResult>, StreamError>
where
    T: Transcript<Challenge = Fr>,
    F: Fn(&StageResult, &[Fr], &mut VerifierCost) -> Result<Vec<Fr>, StreamError>,
{
    let layout = PackingLayout::new(statement.rows, statement.column_count, statement.k)?;
    validate_assembly_proof(layout, statement, proof)?;
    let factor_claims = proof.stage_claims.first().ok_or(StreamError::StageCount)?;
    let row_proof = proof.stages.first().ok_or(StreamError::StageCount)?;
    let specs: Vec<StageMemberSpec> = statement.members.iter().map(|member| member.spec).collect();
    let input_claims: Vec<Fr> = statement
        .members
        .iter()
        .map(|member| member.input_claim)
        .collect();
    let row_result = verify_kzg_batch_stage_observed(
        row_proof,
        &specs,
        &input_claims,
        setup,
        transcript,
        observer,
        |result, observer| final_claims(result, factor_claims, observer),
    )?;
    let reduced_claim = proof
        .reduced_claims
        .first()
        .copied()
        .ok_or(StreamError::StageCount)?;
    let column_proof = proof.stages.get(1).ok_or(StreamError::StageCount)?;
    let column_shape = vec![
        StageMemberSpec {
            rounds: layout.column_vars(),
            degree: 2,
            offset: 0,
        };
        statement.factor_columns.len()
    ];
    let column_result = verify_stage_with_observed(
        column_proof,
        &column_shape,
        factor_claims,
        transcript,
        observer,
        |result, observer| {
            statement
                .factor_columns
                .iter()
                .map(|&column| {
                    ColumnReduction::expected_final_observed(
                        layout.padded_column_count,
                        column,
                        &result.point,
                        reduced_claim,
                        observer,
                    )
                })
                .collect()
        },
    )?;
    let claim = canonical_reduction_claim_observed(
        layout,
        &row_result.point,
        &column_result.point,
        reduced_claim,
        observer,
    )?;
    verify_direct_opening(proof, &claim, setup, transcript, observer)?;
    Ok(vec![row_result, column_result])
}

fn validate_assembly(
    layout: PackingLayout,
    statement: &AssemblyStatement,
    members: &[StageMember<'_>],
) -> Result<(), StreamError> {
    if layout.rows != statement.rows
        || layout.column_count != statement.column_count
        || layout.k != statement.k
        || members.len() != statement.members.len()
    {
        return Err(StreamError::StageMemberCount);
    }
    for (member, expected) in members.iter().zip(&statement.members) {
        if member.prover.num_rounds() != expected.spec.rounds
            || member.degree != expected.spec.degree
            || member.offset != expected.spec.offset
            || member.input_claim != expected.input_claim
        {
            return Err(StreamError::StageMemberCount);
        }
    }
    validate_factor_columns(layout, &statement.factor_columns)
}

fn validate_assembly_proof(
    layout: PackingLayout,
    statement: &AssemblyStatement,
    proof: &WrapperProof,
) -> Result<(), StreamError> {
    if !proof.public_challenges.is_empty()
        || proof.stages.len() != 2
        || proof.stage_claims.len() != 1
        || proof.stage_claims.first().map(Vec::len) != Some(statement.factor_columns.len())
        || proof.commitments.len() != layout.group_count
        || proof.reduced_claims.len() != 1
        || proof.opening.com.len() + 1 != layout.packed_vars()
    {
        return Err(StreamError::StageCount);
    }
    validate_factor_columns(layout, &statement.factor_columns)
}

fn validate_factor_columns(
    layout: PackingLayout,
    factor_columns: &[usize],
) -> Result<(), StreamError> {
    if factor_columns.is_empty() {
        return Err(StreamError::EmptyTensor);
    }
    if let Some(&column) = factor_columns
        .iter()
        .find(|&&column| column >= layout.column_count)
    {
        return Err(StreamError::ColumnOutOfRange {
            column,
            columns: layout.column_count,
        });
    }
    Ok(())
}

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
    let claim = canonical_reduction_claim(
        packed.layout,
        &row_result.point,
        &column_result.point,
        reduced_claim,
    )?;
    prove_direct_opening(
        packed,
        vec![row_stage, column_stage],
        vec![factor_claims],
        &claim,
        setup,
        &mut transcript,
    )
}

fn prove_direct_opening(
    packed: &PackedColumns,
    stages: Vec<StageProof>,
    stage_claims: Vec<Vec<Fr>>,
    claim: &ReductionClaim,
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut Keccak256Transcript<Fr>,
) -> Result<WrapperProof, StreamError> {
    transcript.append(&claim.value);
    let combined_evaluations = combine_evaluations(&packed.evaluations, &claim.polynomial_weights);
    if combined_evaluations.as_slice().evaluate(&claim.point) != claim.value {
        return Err(StreamError::OpeningClaim);
    }
    let opening =
        HyperKZGScheme::<Bn254>::open(setup, &combined_evaluations, &claim.point, transcript)
            .map_err(StreamError::HyperKzg)?;
    Ok(WrapperProof {
        public_challenges: Vec::new(),
        commitments: packed.commitments.clone(),
        stages,
        stage_claims,
        reduced_claims: vec![claim.value],
        opening,
    })
}

pub fn verify_stream(
    proof: &WrapperProof,
    statement: &TensorStreamStatement,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<Vec<StageResult>, StreamError> {
    let mut transcript = new_stream_transcript(
        &statement.key_digest,
        &[statement.row_input_claim],
        &proof.commitments,
    );
    verify_stream_observed(
        proof,
        statement,
        setup,
        &mut transcript,
        &mut VerifierCost::default(),
    )
}

pub fn verify_stream_with_cost(
    proof: &WrapperProof,
    statement: &TensorStreamStatement,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<(Vec<StageResult>, VerifierCost), StreamError> {
    let mut transcript = seed_stream_transcript::<CountingKeccakTranscript>(
        &statement.key_digest,
        &[statement.row_input_claim],
        &proof.commitments,
    );
    let mut cost = VerifierCost::default();
    let result = verify_stream_observed(proof, statement, setup, &mut transcript, &mut cost)?;
    cost.keccak = transcript.hashes;
    Ok((result, cost))
}

fn verify_stream_observed<T: Transcript<Challenge = Fr>>(
    proof: &WrapperProof,
    statement: &TensorStreamStatement,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut VerifierCost,
) -> Result<Vec<StageResult>, StreamError> {
    let layout = PackingLayout::new(statement.rows, statement.column_count, statement.k)?;
    validate_statement(layout, statement)?;
    let factor_columns = tensor_factor_columns(&statement.terms);
    if !proof.public_challenges.is_empty()
        || proof.stages.len() != 2
        || proof.stage_claims.len() != 1
        || proof.stage_claims.first().map(Vec::len) != Some(factor_columns.len())
        || proof.commitments.len() != layout.group_count
        || proof.reduced_claims.len() != 1
        || proof.opening.com.len() + 1 != layout.packed_vars()
    {
        return Err(StreamError::StageCount);
    }
    absorb_stage_a_encoding(statement.stage_a_encoding, transcript);
    let row_stage = proof.stages.first().ok_or(StreamError::StageCount)?;
    let row_shape = [StageMemberSpec {
        rounds: layout.row_vars(),
        degree: statement.row_degree,
        offset: 0,
    }];
    let row_result = match statement.stage_a_encoding {
        StageAEncoding::Compressed => verify_stage_with_observed(
            row_stage,
            &row_shape,
            &[statement.row_input_claim],
            transcript,
            observer,
            |result, observer| Ok(vec![single_member_output_observed(result, observer)?]),
        )?,
        StageAEncoding::KzgCommitted => verify_kzg_stage_observed(
            row_stage,
            statement.row_input_claim,
            layout.row_vars(),
            statement.row_degree,
            setup,
            transcript,
            observer,
        )?,
    };
    let row_output = single_output_claim(&row_result)?;
    let factor_claims = proof.stage_claims.first().ok_or(StreamError::StageCount)?;
    if tensor_value_observed(&statement.terms, factor_claims, observer)? != row_output {
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
    let column_result = verify_stage_with_observed(
        column_stage,
        &column_shape,
        factor_claims,
        transcript,
        observer,
        |result, observer| {
            factor_columns
                .iter()
                .map(|&column| {
                    ColumnReduction::expected_final_observed(
                        layout.padded_column_count,
                        column,
                        &result.point,
                        reduced_claim,
                        observer,
                    )
                })
                .collect()
        },
    )?;
    let claim = canonical_reduction_claim_observed(
        layout,
        &row_result.point,
        &column_result.point,
        reduced_claim,
        observer,
    )?;
    verify_direct_opening(proof, &claim, setup, transcript, observer)?;
    Ok(vec![row_result, column_result])
}

fn verify_direct_opening<T: Transcript<Challenge = Fr>>(
    proof: &WrapperProof,
    claim: &ReductionClaim,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut VerifierCost,
) -> Result<(), StreamError> {
    transcript.append(&claim.value);
    let commitment =
        HyperKZGScheme::<Bn254>::combine(&proof.commitments, &claim.polynomial_weights);
    observer.ec_mul(proof.commitments.len());
    observer.ec_add(proof.commitments.len());
    HyperKZGScheme::<Bn254>::verify_observed(
        setup,
        &commitment,
        &claim.point,
        &claim.value,
        &proof.opening,
        transcript,
        observer,
    )
    .map_err(StreamError::HyperKzg)?;
    Ok(())
}

pub fn new_stream_transcript(
    key_digest: &[u8; 32],
    public_statement: &[Fr],
    commitments: &[Commitment],
) -> Keccak256Transcript<Fr> {
    seed_stream_transcript::<Keccak256Transcript<Fr>>(key_digest, public_statement, commitments)
}

fn seed_stream_transcript<T: Transcript<Challenge = Fr>>(
    key_digest: &[u8; 32],
    public_statement: &[Fr],
    commitments: &[Commitment],
) -> T {
    let mut transcript = T::new(STREAM_LABEL);
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

fn canonical_reduction_claim(
    layout: PackingLayout,
    row_point: &[Fr],
    column_batch_point: &[Fr],
    value: Fr,
) -> Result<ReductionClaim, StreamError> {
    canonical_reduction_claim_observed(
        layout,
        row_point,
        column_batch_point,
        value,
        &mut NoopVerifierObserver,
    )
}

fn canonical_reduction_claim_observed<O: VerifierObserver>(
    layout: PackingLayout,
    row_point: &[Fr],
    column_batch_point: &[Fr],
    value: Fr,
    observer: &mut O,
) -> Result<ReductionClaim, StreamError> {
    if column_batch_point.len() != layout.column_vars() {
        return Err(StreamError::PointDimension {
            expected: layout.column_vars(),
            actual: column_batch_point.len(),
        });
    }
    Ok(ReductionClaim {
        polynomial_weights: layout.group_weights_observed(column_batch_point, observer)?,
        point: layout.packed_point(row_point, column_batch_point)?,
        value,
    })
}

fn tensor_factor_columns(terms: &[TensorTerm]) -> Vec<usize> {
    terms
        .iter()
        .flat_map(|term| term.columns.iter().copied())
        .collect()
}

fn tensor_value(terms: &[TensorTerm], factors: &[Fr]) -> Result<Fr, StreamError> {
    tensor_value_observed(terms, factors, &mut NoopVerifierObserver)
}

fn tensor_value_observed<O: VerifierObserver>(
    terms: &[TensorTerm],
    factors: &[Fr],
    observer: &mut O,
) -> Result<Fr, StreamError> {
    let arity = tensor_arity(terms)?;
    if factors.len() != terms.len() * arity {
        return Err(StreamError::StageMemberCount);
    }
    let mut sum = Fr::zero();
    for (term, values) in terms.iter().zip(factors.chunks_exact(arity)) {
        let mut product = term.coefficient;
        for &value in values {
            product = observer.fr_mul(product, value);
        }
        sum += product;
    }
    Ok(sum)
}

fn tensor_arity(terms: &[TensorTerm]) -> Result<usize, StreamError> {
    terms
        .first()
        .map(|term| term.columns.len())
        .filter(|arity| *arity > 0)
        .ok_or(StreamError::EmptyTensor)
}

fn single_member_output(result: &StageResult) -> Result<Fr, StreamError> {
    single_member_output_observed(result, &mut NoopVerifierObserver)
}

fn single_member_output_observed<O: VerifierObserver>(
    result: &StageResult,
    observer: &mut O,
) -> Result<Fr, StreamError> {
    let coefficient = result
        .coefficients
        .first()
        .copied()
        .ok_or(StreamError::StageMemberCount)?;
    if result.coefficients.len() != 1 {
        return Err(StreamError::StageMemberCount);
    }
    let inverse = observer
        .fr_inv(coefficient)
        .ok_or(StreamError::StageOutputClaim)?;
    Ok(observer.fr_mul(result.final_claim, inverse))
}

fn single_output_claim(result: &StageResult) -> Result<Fr, StreamError> {
    if result.output_claims.len() != 1 {
        return Err(StreamError::StageMemberCount);
    }
    result
        .output_claims
        .first()
        .copied()
        .ok_or(StreamError::StageMemberCount)
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
