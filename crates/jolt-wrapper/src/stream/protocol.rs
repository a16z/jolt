use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Zero};
use jolt_hyperkzg::{
    HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup, NoopVerifierObserver,
    VariableBatchKzgProof, VerifierObserver,
};
use jolt_openings::AdditivelyHomomorphic;
use jolt_poly::{EqPolynomial, MultilinearPoly};
use jolt_r1cs::ConstraintMatrices;
use jolt_transcript::{Keccak256Transcript, Transcript};

use crate::carry::CarryProver;
use crate::spartan::{
    assignment, draw_matrix_weights, draw_point, project_witness_columns, public_contributions,
    public_contributions_observed, validate_dimensions, witness_linear_eval_observed,
    CarryTermExporter, InnerSumcheck, OuterSumcheck, SharedWitnessColumn, SpartanError,
    INNER_DEGREE, OUTER_DEGREE,
};

use super::{
    assembly_transcript, coefficient_evaluation_with_weights_observed, eq_evaluations_observed,
    prove_batch_rounds, prove_rounds, prove_shared_opening, prove_stage, term_reduction,
    term_reduction_with_weights_observed, verify_batch_rounds, verify_rounds,
    verify_shared_opening, verify_stage_with_observed, AssemblyStatement, ColumnId, Commitment,
    CountingKeccakTranscript, PackedColumns, PackingLayout, PendingRoundStage, ReductionClaim,
    StageMember, StageMemberSpec, StageProof, StageResult, StreamError, TermContext, TermExporter,
    TermStageProver, VerifiedRoundStage, VerifierCost, WeightedColumnReduction, WrapperProof,
};

const MIN_COMMITTED_DEGREE: usize = 5;

const TERM_EVALUATION_LABEL: &[u8] = b"term_evaluation";

pub fn prove_assembly(
    packed: &PackedColumns,
    statement: &AssemblyStatement,
    members: &mut [StageMember<'_>],
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, StreamError> {
    validate_assembly(packed.layout, statement, members)?;
    validate_pinned_values(&packed.commitments, statement)?;
    let (mut transcript, phase_challenges) = assembly_transcript::<Keccak256Transcript<Fr>>(
        &statement.key_digest,
        &statement.public_inputs,
        &packed.commitments,
        statement,
    )?;
    prove_assembly_tail(
        packed,
        statement,
        members,
        exporters,
        setup,
        &mut transcript,
        &phase_challenges,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
}

pub struct SpartanAssembly<'a> {
    pub matrices: &'a ConstraintMatrices<Fr>,
    pub public_inputs: &'a [Fr],
    pub witness: &'a [Fr],
    pub witness_column: ColumnId,
    pub carry_member: usize,
}

pub(crate) struct SpartanVerifierAssembly<'a> {
    pub matrices: &'a ConstraintMatrices<Fr>,
    pub public_inputs: &'a [Fr],
    pub witness_column: ColumnId,
    pub carry_member: usize,
}

pub fn prove_spartan_assembly(
    packed: &PackedColumns,
    statement: &AssemblyStatement,
    mut members: Vec<StageMember<'_>>,
    exporters: &[&dyn TermExporter],
    spartan: SpartanAssembly<'_>,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, SpartanError> {
    validate_pinned_values(&packed.commitments, statement)?;
    validate_dimensions(
        spartan.matrices,
        spartan.public_inputs.len(),
        spartan.witness.len(),
    )?;
    if spartan.carry_member != members.len() || statement.members.len() != members.len() + 1 {
        return Err(StreamError::StageMemberCount.into());
    }
    let z = assignment(spartan.public_inputs, spartan.witness);
    spartan
        .matrices
        .check_witness(&z)
        .map_err(SpartanError::Unsatisfied)?;
    let shared_witness = SharedWitnessColumn::new(spartan.witness, statement.rows)?;
    let (mut transcript, phase_challenges) = assembly_transcript::<Keccak256Transcript<Fr>>(
        &statement.key_digest,
        &statement.public_inputs,
        &packed.commitments,
        statement,
    )?;

    let tau = draw_point(
        spartan
            .matrices
            .num_constraints
            .next_power_of_two()
            .trailing_zeros() as usize,
        &mut transcript,
    );
    let mut outer = OuterSumcheck::new(spartan.matrices, &z, &tau);
    let (outer_rounds, outer_result) =
        prove_rounds(&mut outer, Fr::zero(), OUTER_DEGREE, setup, &mut transcript)?;
    let [az, bz, cz] = outer.finals();
    let expected_outer = EqPolynomial::<Fr>::mle(&tau, &outer_result.point) * (az * bz - cz);
    if outer_result.final_claim != expected_outer {
        return Err(SpartanError::OuterFinalClaim);
    }
    for value in [az, bz, cz] {
        transcript.append(&value);
    }
    let matrix_weights = draw_matrix_weights(&mut transcript);
    let row_weights = EqPolynomial::<Fr>::evals(&outer_result.point, None);
    let contributions =
        public_contributions(spartan.matrices, &row_weights, spartan.public_inputs)?;
    let inner_claim = matrix_weights[0] * (az - contributions.a)
        + matrix_weights[1] * (bz - contributions.b)
        + matrix_weights[2] * (cz - contributions.c);
    let padded_witness = shared_witness.inner_evaluations().to_vec();
    let linear_form = project_witness_columns(
        spartan.matrices,
        &row_weights,
        1 + spartan.public_inputs.len(),
        padded_witness.len(),
        matrix_weights,
    );
    let mut inner = InnerSumcheck::new(linear_form, padded_witness, inner_claim)?;
    let mut inner_members = [StageMember {
        prover: &mut inner,
        input_claim: inner_claim,
        degree: INNER_DEGREE,
        offset: 0,
    }];
    let (inner_proof, inner_result) = prove_stage(&mut inner_members, &mut transcript)?;
    let [linear_eval, witness_eval] = inner.finals();
    if inner_result.output_claims != [linear_eval * witness_eval] {
        return Err(SpartanError::InnerFinalClaim);
    }
    transcript.append(&witness_eval);

    let source_point = shared_witness.source_point(&inner_result.point)?;
    let mut carry = CarryProver::new(shared_witness.evaluations(), &source_point, witness_eval)
        .map_err(|_| StreamError::StageLink)?;
    let carry_degree = carry.degree();
    let mut stage_members = members
        .iter_mut()
        .map(|member| StageMember {
            prover: &mut *member.prover,
            input_claim: member.input_claim,
            degree: member.degree,
            offset: member.offset,
        })
        .collect::<Vec<_>>();
    stage_members.push(StageMember {
        prover: &mut carry,
        input_claim: witness_eval,
        degree: carry_degree,
        offset: 0,
    });
    let mut statement = statement.clone();
    statement.members[spartan.carry_member].input_claim = witness_eval;
    let carry_exporter = CarryTermExporter {
        source_point: &source_point,
        wire: spartan.witness_column,
        member_index: spartan.carry_member,
    };
    let mut all_exporters = exporters.to_vec();
    all_exporters.push(&carry_exporter);
    Ok(prove_assembly_tail(
        packed,
        &statement,
        &mut stage_members,
        &all_exporters,
        setup,
        &mut transcript,
        &phase_challenges,
        vec![inner_proof],
        vec![outer_rounds],
        vec![az, bz, cz, witness_eval],
    )?)
}

#[expect(
    clippy::too_many_arguments,
    reason = "prefix stages and shared opening state join the common assembly tail"
)]
fn prove_assembly_tail(
    packed: &PackedColumns,
    statement: &AssemblyStatement,
    members: &mut [StageMember<'_>],
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut Keccak256Transcript<Fr>,
    phase_challenges: &[Fr],
    mut stages: Vec<StageProof>,
    mut shared_stages: Vec<PendingRoundStage>,
    reduced_claims: Vec<Fr>,
) -> Result<WrapperProof, StreamError> {
    validate_assembly(packed.layout, statement, members)?;
    let (row_rounds, row_result) = prove_batch_rounds(members, setup, transcript)?;
    let column_values = packed.column_evaluations(&row_result.point)?;
    let context = TermContext {
        row_point: &row_result.point,
        batching_coefficients: &row_result.coefficients,
        challenges: phase_challenges,
    };
    let terms = exporters
        .iter()
        .flat_map(|exporter| exporter.terms(&context))
        .collect::<Vec<_>>();
    let mut term_prover = TermStageProver::new(&terms, &column_values, statement.k)?;
    let term_degree = exporters
        .iter()
        .map(|exporter| exporter.max_factors())
        .max()
        .ok_or(StreamError::EmptyTensor)?
        .saturating_add(1)
        .max(MIN_COMMITTED_DEGREE);
    if term_prover.degree() > term_degree {
        return Err(StreamError::StageEncoding);
    }
    if term_prover.input_claim() != row_result.final_claim {
        return Err(StreamError::StageLink);
    }
    let (term_rounds, term_result) = prove_rounds(
        &mut term_prover,
        row_result.final_claim,
        term_degree,
        setup,
        transcript,
    )?;
    let (mut committed_stages, round_opening) = {
        shared_stages.extend([row_rounds, term_rounds]);
        prove_shared_opening(shared_stages, setup, transcript)?
    };
    let term_stage = committed_stages.pop().ok_or(StreamError::StageCount)?;
    let row_stage = committed_stages.pop().ok_or(StreamError::StageCount)?;
    committed_stages.append(&mut stages);
    stages = committed_stages;
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
    absorb_term_evaluations(&term_evaluations, transcript);
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
        prove_stage(&mut column_members, transcript)?
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
    stages.extend([row_stage, term_stage, column_stage]);
    let mut proof = prove_direct_opening(
        packed,
        stages,
        Some(round_opening),
        Vec::new(),
        term_evaluations,
        &claim,
        setup,
        transcript,
    )?;
    proof.reduced_claims.extend(reduced_claims);
    proof.commitments = wire_commitments(&packed.commitments, statement)?;
    Ok(proof)
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
    let layout = PackingLayout::new(statement.rows, statement.column_count, statement.k)?;
    let commitments = full_commitments(&proof.commitments, statement, layout.group_count)?;
    let (mut transcript, phase_challenges) = assembly_transcript::<CountingKeccakTranscript>(
        &statement.key_digest,
        &statement.public_inputs,
        &commitments,
        statement,
    )?;
    verify_assembly_from_transcript(
        proof,
        statement,
        exporters,
        setup,
        (&mut transcript, &phase_challenges, &commitments),
        VerifierCost::default(),
    )
}

pub(crate) fn verify_assembly_from_transcript(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
    prefix: (&mut CountingKeccakTranscript, &[Fr], &[Commitment]),
    mut cost: VerifierCost,
) -> Result<(Vec<StageResult>, VerifierCost), StreamError> {
    let (transcript, phase_challenges, commitments) = prefix;
    let results = verify_assembly_observed(
        proof,
        statement,
        exporters,
        setup,
        transcript,
        (phase_challenges, commitments),
        &mut cost,
    )?;
    cost.keccak = transcript.hashes;
    Ok((results, cost))
}

pub(crate) fn verify_spartan_assembly_from_transcript(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    exporters: &[&dyn TermExporter],
    spartan: SpartanVerifierAssembly<'_>,
    setup: &HyperKZGVerifierSetup<Bn254>,
    prefix: (&mut CountingKeccakTranscript, &[Fr], &[Commitment]),
    mut cost: VerifierCost,
) -> Result<(Vec<StageResult>, VerifierCost), SpartanError> {
    let witness_len = spartan
        .matrices
        .num_vars
        .checked_sub(1 + spartan.public_inputs.len())
        .ok_or(StreamError::StageEncoding)?;
    validate_dimensions(spartan.matrices, spartan.public_inputs.len(), witness_len)?;
    if spartan.carry_member + 1 != statement.members.len() {
        return Err(StreamError::StageMemberCount.into());
    }
    let claims = proof
        .reduced_claims
        .get(1..5)
        .ok_or(StreamError::StageCount)?;
    let [az, bz, cz, witness_eval] =
        <[Fr; 4]>::try_from(claims).map_err(|_| StreamError::StageCount)?;
    let (transcript, phase_challenges, commitments) = prefix;
    let tau = draw_point(
        spartan
            .matrices
            .num_constraints
            .next_power_of_two()
            .trailing_zeros() as usize,
        transcript,
    );
    let outer_proof = proof.stages.first().ok_or(StreamError::StageCount)?;
    let (outer, outer_rounds) =
        verify_rounds(outer_proof, Fr::zero(), tau.len(), OUTER_DEGREE, transcript)?;
    let eq = eq_mle_observed(&tau, &outer.point, &mut cost)?;
    let product = cost.fr_mul(az, bz) - cz;
    if outer.final_claim != cost.fr_mul(eq, product) {
        return Err(SpartanError::OuterFinalClaim);
    }
    for value in [az, bz, cz] {
        transcript.append(&value);
    }
    let matrix_weights = draw_matrix_weights(transcript);
    let row_weights = eq_evaluations_observed(&outer.point, &mut cost);
    let contributions = public_contributions_observed(
        spartan.matrices,
        &row_weights,
        spartan.public_inputs,
        &mut cost,
    )?;
    let mut inner_claim = Fr::zero();
    for (weight, value) in matrix_weights.into_iter().zip([
        az - contributions.a,
        bz - contributions.b,
        cz - contributions.c,
    ]) {
        inner_claim += cost.fr_mul(weight, value);
    }
    let inner_rounds = witness_len
        .checked_next_power_of_two()
        .ok_or(StreamError::StageEncoding)?
        .trailing_zeros() as usize;
    let inner_proof = proof.stages.get(1).ok_or(StreamError::StageCount)?;
    let inner_shape = [StageMemberSpec {
        rounds: inner_rounds,
        degree: INNER_DEGREE,
        offset: 0,
    }];
    let inner = verify_stage_with_observed(
        inner_proof,
        &inner_shape,
        &[inner_claim],
        transcript,
        &mut cost,
        |result, observer| {
            let column_weights = eq_evaluations_observed(&result.point, observer);
            let linear_eval = witness_linear_eval_observed(
                spartan.matrices,
                &row_weights,
                &column_weights,
                1 + spartan.public_inputs.len(),
                matrix_weights,
                observer,
            )?;
            Ok(vec![observer.fr_mul(linear_eval, witness_eval)])
        },
    )?;
    transcript.append(&witness_eval);
    let common_rounds = statement.rows.trailing_zeros() as usize;
    let prefix_rounds = common_rounds
        .checked_sub(inner.point.len())
        .ok_or(StreamError::StageEncoding)?;
    let mut source_point = vec![Fr::zero(); prefix_rounds];
    source_point.extend_from_slice(&inner.point);
    let mut statement = statement.clone();
    statement
        .members
        .get_mut(spartan.carry_member)
        .ok_or(StreamError::StageMemberCount)?
        .input_claim = witness_eval;
    let carry_exporter = CarryTermExporter {
        source_point: &source_point,
        wire: spartan.witness_column,
        member_index: spartan.carry_member,
    };
    let mut all_exporters = exporters.to_vec();
    all_exporters.push(&carry_exporter);
    let mut results = verify_assembly_tail(
        proof,
        &statement,
        &all_exporters,
        setup,
        transcript,
        (phase_challenges, commitments),
        &mut cost,
        2,
        vec![outer_rounds],
        5,
    )?;
    results.insert(0, inner);
    results.insert(0, outer);
    cost.keccak = transcript.hashes;
    Ok((results, cost))
}

fn eq_mle_observed<O: VerifierObserver>(
    left: &[Fr],
    right: &[Fr],
    observer: &mut O,
) -> Result<Fr, StreamError> {
    if left.len() != right.len() {
        return Err(StreamError::PointDimension {
            expected: left.len(),
            actual: right.len(),
        });
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(Fr::one(), |value, (&left, &right)| {
            let both = observer.fr_mul(left, right);
            let neither = observer.fr_mul(Fr::one() - left, Fr::one() - right);
            observer.fr_mul(value, both + neither)
        }))
}

fn verify_assembly_observed<T>(
    proof: &WrapperProof,
    statement: &AssemblyStatement,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    assembly: (&[Fr], &[Commitment]),
    observer: &mut VerifierCost,
) -> Result<Vec<StageResult>, StreamError>
where
    T: Transcript<Challenge = Fr>,
{
    verify_assembly_tail(
        proof,
        statement,
        exporters,
        setup,
        transcript,
        assembly,
        observer,
        0,
        Vec::new(),
        1,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "prefix stages and shared opening state join the common assembly tail"
)]
fn verify_assembly_tail<'a, T>(
    proof: &'a WrapperProof,
    statement: &AssemblyStatement,
    exporters: &[&dyn TermExporter],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    assembly: (&[Fr], &[Commitment]),
    observer: &mut VerifierCost,
    stage_offset: usize,
    mut shared_stages: Vec<VerifiedRoundStage<'a>>,
    reduced_claim_count: usize,
) -> Result<Vec<StageResult>, StreamError>
where
    T: Transcript<Challenge = Fr>,
{
    let (phase_challenges, commitments) = assembly;
    let layout = PackingLayout::new(statement.rows, statement.column_count, statement.k)?;
    validate_assembly_proof(
        layout,
        statement,
        proof,
        stage_offset + 3,
        reduced_claim_count,
    )?;
    let row_proof = proof
        .stages
        .get(stage_offset)
        .ok_or(StreamError::StageCount)?;
    let specs: Vec<StageMemberSpec> = statement.members.iter().map(|member| member.spec).collect();
    let input_claims: Vec<Fr> = statement
        .members
        .iter()
        .map(|member| member.input_claim)
        .collect();
    let (row_result, row_rounds) =
        verify_batch_rounds(row_proof, &specs, &input_claims, transcript, observer)?;
    let context = TermContext {
        row_point: &row_result.point,
        batching_coefficients: &row_result.coefficients,
        challenges: phase_challenges,
    };
    let terms = exporters
        .iter()
        .flat_map(|exporter| exporter.terms_observed(&context, observer))
        .collect::<Vec<_>>();
    let term_degree = exporters
        .iter()
        .map(|exporter| exporter.max_factors())
        .max()
        .ok_or(StreamError::EmptyTensor)?
        .saturating_add(1)
        .max(MIN_COMMITTED_DEGREE);
    let term_rounds = terms.len().next_power_of_two().max(2).trailing_zeros() as usize;
    let term_proof = proof
        .stages
        .get(stage_offset + 1)
        .ok_or(StreamError::StageCount)?;
    let (term_result, term_rounds) = verify_rounds(
        term_proof,
        row_result.final_claim,
        term_rounds,
        term_degree,
        transcript,
    )?;
    shared_stages.extend([row_rounds, term_rounds]);
    verify_shared_opening(
        &shared_stages,
        proof
            .round_opening
            .as_ref()
            .ok_or(StreamError::StageEncoding)?,
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
    let term_weights = eq_evaluations_observed(&term_result.point, observer);
    let coefficient =
        coefficient_evaluation_with_weights_observed(&terms, &term_weights, observer)?;
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
    let reduction = term_reduction_with_weights_observed(
        &terms,
        &term_weights,
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
    let column_proof = proof
        .stages
        .get(stage_offset + 2)
        .ok_or(StreamError::StageCount)?;
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
    verify_direct_opening(proof, commitments, &claim, setup, transcript, observer)?;
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
    validate_commitment_phases(layout, statement)?;
    validate_pinned_commitments(layout.group_count, statement)
}

fn validate_assembly_proof(
    layout: PackingLayout,
    statement: &AssemblyStatement,
    proof: &WrapperProof,
    stage_count: usize,
    reduced_claim_count: usize,
) -> Result<(), StreamError> {
    if !proof.public_challenges.is_empty()
        || proof.stages.len() != stage_count
        || !proof.stage_claims.is_empty()
        || proof.term_evaluations.is_empty()
        || proof.round_opening.is_none()
        || proof.commitments.len()
            != layout
                .group_count
                .checked_sub(statement.pinned_commitments.len())
                .ok_or(StreamError::StageCount)?
        || proof.reduced_claims.len() != reduced_claim_count
        || proof.opening.com.len() + 1 != layout.packed_vars()
    {
        return Err(StreamError::StageCount);
    }
    validate_commitment_phases(layout, statement)?;
    validate_pinned_commitments(layout.group_count, statement)
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

fn validate_pinned_commitments(
    group_count: usize,
    statement: &AssemblyStatement,
) -> Result<(), StreamError> {
    for (position, &(group, _)) in statement.pinned_commitments.iter().enumerate() {
        if group >= group_count
            || statement.pinned_commitments[..position]
                .iter()
                .any(|(existing, _)| *existing == group)
        {
            return Err(StreamError::PinnedCommitment(group));
        }
    }
    Ok(())
}

fn validate_pinned_values(
    commitments: &[Commitment],
    statement: &AssemblyStatement,
) -> Result<(), StreamError> {
    validate_pinned_commitments(commitments.len(), statement)?;
    for &(group, expected) in &statement.pinned_commitments {
        if commitments.get(group) != Some(&expected) {
            return Err(StreamError::PinnedCommitment(group));
        }
    }
    Ok(())
}

fn wire_commitments(
    commitments: &[Commitment],
    statement: &AssemblyStatement,
) -> Result<Vec<Commitment>, StreamError> {
    validate_pinned_values(commitments, statement)?;
    Ok(commitments
        .iter()
        .enumerate()
        .filter(|(group, _)| {
            !statement
                .pinned_commitments
                .iter()
                .any(|(pinned, _)| pinned == group)
        })
        .map(|(_, commitment)| *commitment)
        .collect())
}

fn full_commitments(
    wire: &[Commitment],
    statement: &AssemblyStatement,
    group_count: usize,
) -> Result<Vec<Commitment>, StreamError> {
    validate_pinned_commitments(group_count, statement)?;
    let mut wire = wire.iter();
    let mut commitments = Vec::with_capacity(group_count);
    for group in 0..group_count {
        let commitment = statement
            .pinned_commitments
            .iter()
            .find_map(|&(pinned, commitment)| (pinned == group).then_some(commitment))
            .or_else(|| wire.next().copied())
            .ok_or(StreamError::StageCount)?;
        commitments.push(commitment);
    }
    if wire.next().is_some() {
        return Err(StreamError::StageCount);
    }
    Ok(commitments)
}

#[expect(
    clippy::too_many_arguments,
    reason = "proof sections and the opening claim are serialized separately"
)]
fn prove_direct_opening(
    packed: &PackedColumns,
    stages: Vec<StageProof>,
    round_opening: Option<VariableBatchKzgProof<Bn254>>,
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
        round_opening,
        stage_claims,
        term_evaluations,
        reduced_claims: vec![claim.value],
        opening,
    })
}

fn verify_direct_opening<T: Transcript<Challenge = Fr>>(
    proof: &WrapperProof,
    commitments: &[Commitment],
    claim: &ReductionClaim,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut VerifierCost,
) -> Result<(), StreamError> {
    transcript.append(&claim.value);
    let commitment = HyperKZGScheme::<Bn254>::combine(commitments, &claim.polynomial_weights);
    observer.ec_mul(commitments.len());
    observer.ec_add(commitments.len());
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

fn absorb_term_evaluations<T: Transcript<Challenge = Fr>>(evaluations: &[Fr], transcript: &mut T) {
    for evaluation in evaluations {
        transcript.append_labeled(TERM_EVALUATION_LABEL, evaluation);
    }
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
