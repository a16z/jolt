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
    coefficient_evaluation_observed, prove_kzg_batch_stage_deferred, prove_kzg_stage, prove_stage,
    term_reduction, term_reduction_observed, verify_kzg_batch_stage_deferred_observed,
    verify_kzg_stage_observed, verify_stage_with_observed, AssemblyStatement, ColumnReduction,
    Commitment, PackedColumns, PackingLayout, ReductionClaim, StageAEncoding, StageMember,
    StageMemberSpec, StageProof, StageResult, StreamError, TensorStreamStatement, TensorTerm,
    TermContext, TermExporter, TermStageProver, VerifierCost, WeightedColumnReduction,
    WrapperProof, STREAM_LABEL,
};

const TERM_EVALUATION_LABEL: &[u8] = b"term_evaluation";

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

pub fn prove_assembly(
    packed: &PackedColumns,
    statement: &AssemblyStatement,
    members: &mut [StageMember<'_>],
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, StreamError> {
    validate_assembly(packed.layout, statement, members)?;
    let (mut transcript, phase_challenges) = assembly_transcript::<Keccak256Transcript<Fr>>(
        &statement.key_digest,
        &statement.public_inputs,
        &packed.commitments,
        statement,
    )?;
    let (row_stage, row_result) = prove_kzg_batch_stage_deferred(members, setup, &mut transcript)?;
    let column_values = packed.column_evaluations(&row_result.point)?;
    let context = TermContext {
        row_point: &row_result.point,
        batching_coefficients: &row_result.coefficients,
        challenges: &phase_challenges,
    };
    let terms = exporters
        .iter()
        .flat_map(|exporter| exporter.terms(&context))
        .collect::<Vec<_>>();
    let mut term_prover = TermStageProver::new(&terms, &column_values, statement.k)?;
    if term_prover.input_claim() != row_result.final_claim {
        return Err(StreamError::StageLink);
    }
    let (term_stage, term_result) = prove_kzg_stage(
        &mut term_prover,
        row_result.final_claim,
        6,
        setup,
        &mut transcript,
    )?;
    let term_evaluations = term_prover.factor_evaluations()?;
    let coefficient = term_prover
        .coefficient_evaluation()
        .ok_or(StreamError::StageCount)?;
    if term_evaluations
        .iter()
        .fold(coefficient, |product, &value| product * value)
        != term_result.final_claim
    {
        return Err(StreamError::StageOutputClaim);
    }
    absorb_term_evaluations(&term_evaluations, &mut transcript);
    let lambdas = term_evaluations
        .iter()
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    let reduction = term_reduction(
        &terms,
        &term_result.point,
        &lambdas,
        packed.layout.padded_column_count,
        statement.k,
    )?;
    let column_claim = reduction.claim(&term_evaluations, &lambdas)?;
    let mut column_prover = WeightedColumnReduction::new(column_values, reduction.weights)?;
    if column_prover.input_claim() != column_claim {
        return Err(StreamError::OpeningClaim);
    }
    let (column_stage, column_result) = {
        let mut column_members = [StageMember {
            prover: &mut column_prover,
            input_claim: column_claim,
            degree: 2,
            offset: 0,
        }];
        prove_stage(&mut column_members, &mut transcript)?
    };
    let reduced_claim = column_prover
        .value_evaluation()
        .ok_or(StreamError::StageCount)?;
    let weight_evaluation = column_prover
        .weight_evaluation()
        .ok_or(StreamError::StageCount)?;
    if column_result.output_claims != [weight_evaluation * reduced_claim] {
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
        vec![row_stage, term_stage, column_stage],
        Vec::new(),
        term_evaluations,
        &claim,
        setup,
        &mut transcript,
    )
}

pub fn verify_assembly(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<Vec<StageResult>, StreamError> {
    verify_assembly_with_cost(proof, statement, exporters, setup).map(|(results, _)| results)
}

pub fn verify_assembly_with_cost(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<(Vec<StageResult>, VerifierCost), StreamError> {
    let (mut transcript, phase_challenges) = assembly_transcript::<CountingKeccakTranscript>(
        &statement.key_digest,
        &statement.public_inputs,
        &proof.commitments,
        statement,
    )?;
    let mut cost = VerifierCost::default();
    let results = verify_assembly_observed(
        proof,
        statement,
        exporters,
        setup,
        &mut transcript,
        &phase_challenges,
        &mut cost,
    )?;
    cost.keccak = transcript.hashes;
    Ok((results, cost))
}

fn verify_assembly_observed<T>(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    phase_challenges: &[Fr],
    observer: &mut VerifierCost,
) -> Result<Vec<StageResult>, StreamError>
where
    T: Transcript<Challenge = Fr>,
{
    let layout = PackingLayout::new(statement.rows, statement.column_count, statement.k)?;
    validate_assembly_proof(layout, statement, proof)?;
    let row_proof = proof.stages.first().ok_or(StreamError::StageCount)?;
    let specs: Vec<StageMemberSpec> = statement.members.iter().map(|member| member.spec).collect();
    let input_claims: Vec<Fr> = statement
        .members
        .iter()
        .map(|member| member.input_claim)
        .collect();
    let row_result = verify_kzg_batch_stage_deferred_observed(
        row_proof,
        &specs,
        &input_claims,
        setup,
        transcript,
        observer,
    )?;
    let context = TermContext {
        row_point: &row_result.point,
        batching_coefficients: &row_result.coefficients,
        challenges: phase_challenges,
    };
    let terms = exporters
        .iter()
        .flat_map(|exporter| exporter.terms(&context))
        .collect::<Vec<_>>();
    let term_rounds = terms.len().next_power_of_two().max(2).trailing_zeros() as usize;
    let term_proof = proof.stages.get(1).ok_or(StreamError::StageCount)?;
    let term_result = verify_kzg_stage_observed(
        term_proof,
        row_result.final_claim,
        term_rounds,
        6,
        setup,
        transcript,
        observer,
    )?;
    let factor_count = terms
        .iter()
        .map(|term| term.factors.len())
        .max()
        .ok_or(StreamError::EmptyTensor)?;
    if proof.term_evaluations.len() != factor_count {
        return Err(StreamError::StageMemberCount);
    }
    absorb_term_evaluations(&proof.term_evaluations, transcript);
    let coefficient = coefficient_evaluation_observed(&terms, &term_result.point, observer)?;
    let mut expected = coefficient;
    for &evaluation in &proof.term_evaluations {
        expected = observer.fr_mul(expected, evaluation);
    }
    if expected != term_result.final_claim {
        return Err(StreamError::StageOutputClaim);
    }
    let lambdas = proof
        .term_evaluations
        .iter()
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    let reduction = term_reduction_observed(
        &terms,
        &term_result.point,
        &lambdas,
        layout.padded_column_count,
        statement.k,
        observer,
    )?;
    let column_claim = reduction.claim_observed(&proof.term_evaluations, &lambdas, observer)?;
    let reduced_claim = proof
        .reduced_claims
        .first()
        .copied()
        .ok_or(StreamError::StageCount)?;
    let column_proof = proof.stages.get(2).ok_or(StreamError::StageCount)?;
    let column_shape = [StageMemberSpec {
        rounds: layout.column_vars(),
        degree: 2,
        offset: 0,
    }];
    let column_result = verify_stage_with_observed(
        column_proof,
        &column_shape,
        &[column_claim],
        transcript,
        observer,
        |result, observer| {
            let weight = super::multilinear_evaluation_observed(
                &reduction.weights,
                &result.point,
                observer,
            )?;
            Ok(vec![observer.fr_mul(weight, reduced_claim)])
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
    Ok(vec![row_result, term_result, column_result])
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
    validate_commitment_phases(layout, statement)
}

fn validate_assembly_proof(
    layout: PackingLayout,
    statement: &AssemblyStatement,
    proof: &WrapperProof,
) -> Result<(), StreamError> {
    if !proof.public_challenges.is_empty()
        || proof.stages.len() != 3
        || !proof.stage_claims.is_empty()
        || proof.term_evaluations.is_empty()
        || proof.commitments.len() != layout.group_count
        || proof.reduced_claims.len() != 1
        || proof.opening.com.len() + 1 != layout.packed_vars()
    {
        return Err(StreamError::StageCount);
    }
    validate_commitment_phases(layout, statement)
}

fn validate_commitment_phases(
    layout: PackingLayout,
    statement: &AssemblyStatement,
) -> Result<(), StreamError> {
    if statement.commitment_phases.is_empty()
        || statement
            .commitment_phases
            .iter()
            .map(|phase| phase.group_count)
            .sum::<usize>()
            != layout.group_count
    {
        return Err(StreamError::StageCount);
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
        Vec::new(),
        &claim,
        setup,
        &mut transcript,
    )
}

fn prove_direct_opening(
    packed: &PackedColumns,
    stages: Vec<StageProof>,
    stage_claims: Vec<Vec<Fr>>,
    term_evaluations: Vec<Fr>,
    claim: &ReductionClaim,
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut Keccak256Transcript<Fr>,
) -> Result<WrapperProof, StreamError> {
    transcript.append(&claim.value);
    let combined_evaluations = packed.rlc_evaluations(&claim.polynomial_weights)?;
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
        term_evaluations,
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

pub fn commitment_prefix_challenges(
    key_digest: &[u8; 32],
    public_statement: &[Fr],
    phases: &[(&[Commitment], usize)],
) -> Vec<Fr> {
    let mut transcript = Keccak256Transcript::<Fr>::new(STREAM_LABEL);
    transcript.append_bytes(key_digest);
    for value in public_statement {
        transcript.append(value);
    }
    let mut challenges = Vec::new();
    for &(commitments, challenge_count) in phases {
        absorb_commitments(commitments, &mut transcript);
        challenges.extend((0..challenge_count).map(|_| transcript.challenge()));
    }
    challenges
}

fn assembly_transcript<T: Transcript<Challenge = Fr>>(
    key_digest: &[u8; 32],
    public_statement: &[Fr],
    commitments: &[Commitment],
    statement: &AssemblyStatement,
) -> Result<(T, Vec<Fr>), StreamError> {
    let mut transcript = T::new(STREAM_LABEL);
    transcript.append_bytes(key_digest);
    for value in public_statement {
        transcript.append(value);
    }
    let mut challenges = Vec::new();
    let mut start = 0usize;
    for phase in &statement.commitment_phases {
        let end = start
            .checked_add(phase.group_count)
            .ok_or(StreamError::StageCount)?;
        let phase_commitments = commitments.get(start..end).ok_or(StreamError::StageCount)?;
        absorb_commitments(phase_commitments, &mut transcript);
        challenges.extend((0..phase.challenge_count).map(|_| transcript.challenge()));
        start = end;
    }
    if start != commitments.len() {
        return Err(StreamError::StageCount);
    }
    Ok((transcript, challenges))
}

fn absorb_term_evaluations<T: Transcript<Challenge = Fr>>(evaluations: &[Fr], transcript: &mut T) {
    for evaluation in evaluations {
        transcript.append_labeled(TERM_EVALUATION_LABEL, evaluation);
    }
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
