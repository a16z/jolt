use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::{program_image, registers as registers_claim_reduction},
        dimensions::{ReadWriteDimensions, TraceDimensions, REGISTER_ADDRESS_BITS},
        instruction, ram,
        ram::{RamValCheckInit, RamValCheckInitContribution as FormulaInitContribution},
        registers,
    },
    JoltAdviceKind, JoltRelationId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{block_selector_mle_msb, sparse_segments_mle_msb, try_eq_mle, LtPolynomial};
use jolt_program::preprocess::PublicInitialRam;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::{LabelWithCount, Transcript};
use num_traits::One;

use super::{
    inputs::{Deps, Stage4Claims},
    outputs::{
        RamValCheckInitialEvaluation, Stage4ClearOutput, Stage4Output, Stage4PublicOutput,
        Stage4ZkOutput, VerifiedRamValCheckAdviceContribution,
        VerifiedRamValCheckProgramImageContribution, VerifiedStage4Batch, VerifiedStage4Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage2::outputs::Stage2ClearOutput, stage3::outputs::Stage3ClearOutput, zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4InputClaims<F: Field> {
    pub registers_read_write: F,
    pub ram_val_check: F,
}

pub struct Stage4InputClaimRequest<'a, F: Field> {
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage3: &'a Stage3ClearOutput<F>,
    pub ram_val_check_initial_eval: F,
    pub registers_gamma: F,
    pub ram_val_check_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ExpectedOutputClaims<F: Field> {
    pub registers_read_write: F,
    pub ram_val_check: F,
}

pub struct Stage4ExpectedOutputRequest<'a, F: Field> {
    pub fixed_register_cycle_point: &'a [F],
    pub registers_read_write_opening_point: &'a [F],
    pub fixed_ram_cycle_point: &'a [F],
    pub ram_val_check_cycle_point: &'a [F],
    pub registers_gamma: F,
    pub ram_val_check_gamma: F,
    pub claims: &'a Stage4Claims<F>,
}

pub struct Stage4OpeningPointRequest<'a, F: Field> {
    pub register_dimensions: ReadWriteDimensions,
    pub registers_read_write_sumcheck_point: &'a [F],
    pub ram_val_check_sumcheck_point: &'a [F],
    pub fixed_ram_address_point: &'a [F],
    pub fixed_ram_cycle_point: &'a [F],
}

pub struct Stage4OpeningPoints<F: Field> {
    pub registers_read_write_sumcheck_point: Vec<F>,
    pub registers_read_write_opening_point: Vec<F>,
    pub ram_val_check_sumcheck_point: Vec<F>,
    pub ram_val_check_cycle_point: Vec<F>,
    pub ram_val_check_opening_point: Vec<F>,
}

const STAGE4_BATCH_BASE_OUTPUT_CLAIMS: usize = 7;

pub fn stage4_input_claims<F: Field>(
    request: Stage4InputClaimRequest<'_, F>,
) -> Stage4InputClaims<F> {
    let registers_gamma2 = request.registers_gamma * request.registers_gamma;

    Stage4InputClaims {
        registers_read_write: request
            .stage3
            .output_claims
            .registers_claim_reduction
            .rd_write_value
            + request.registers_gamma
                * request
                    .stage3
                    .output_claims
                    .registers_claim_reduction
                    .rs1_value
            + registers_gamma2
                * request
                    .stage3
                    .output_claims
                    .registers_claim_reduction
                    .rs2_value,
        ram_val_check: request.stage2.output_claims.ram_read_write.val
            + request.ram_val_check_gamma * request.stage2.output_claims.ram_output_check
            - (F::one() + request.ram_val_check_gamma) * request.ram_val_check_initial_eval,
    }
}

pub fn stage4_expected_output_claims<F: Field>(
    request: Stage4ExpectedOutputRequest<'_, F>,
) -> Result<Stage4ExpectedOutputClaims<F>, VerifierError> {
    if request.registers_read_write_opening_point.len() < REGISTER_ADDRESS_BITS {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: format!(
                "register read-write opening point has {} variables, expected at least {REGISTER_ADDRESS_BITS}",
                request.registers_read_write_opening_point.len()
            ),
        });
    }
    let (_, registers_cycle_point) = request
        .registers_read_write_opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let eq_cycle =
        try_eq_mle(request.fixed_register_cycle_point, registers_cycle_point).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            }
        })?;
    let registers = &request.claims.registers_read_write;
    let registers_read_write = eq_cycle
        * (registers.rd_wa * (registers.rd_inc + registers.registers_val)
            + request.registers_gamma * registers.rs1_ra * registers.registers_val
            + request.registers_gamma
                * request.registers_gamma
                * registers.rs2_ra
                * registers.registers_val);

    let ram_lt = LtPolynomial::evaluate(
        request.ram_val_check_cycle_point,
        request.fixed_ram_cycle_point,
    );
    let ram_val_check = (ram_lt + request.ram_val_check_gamma)
        * request.claims.ram_val_check.ram_inc
        * request.claims.ram_val_check.ram_ra;

    Ok(Stage4ExpectedOutputClaims {
        registers_read_write,
        ram_val_check,
    })
}

pub fn stage4_expected_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage4ExpectedOutputClaims<F>,
) -> Result<F, VerifierError> {
    let [registers_coefficient, ram_val_coefficient] = coefficients else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: "Stage 4 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    Ok(
        *registers_coefficient * expected_outputs.registers_read_write
            + *ram_val_coefficient * expected_outputs.ram_val_check,
    )
}

pub fn stage4_opening_points<F: Field>(
    request: Stage4OpeningPointRequest<'_, F>,
) -> Result<Stage4OpeningPoints<F>, VerifierError> {
    let registers_read_write_opening = request
        .register_dimensions
        .read_write_opening_point(request.registers_read_write_sumcheck_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;

    let ram_val_check_cycle_point = request
        .ram_val_check_sumcheck_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    if ram_val_check_cycle_point.len() != request.fixed_ram_cycle_point.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!(
                "RAM value cycle point length mismatch: expected {}, got {}",
                request.fixed_ram_cycle_point.len(),
                ram_val_check_cycle_point.len()
            ),
        });
    }
    let ram_val_check_opening_point = [
        request.fixed_ram_address_point,
        ram_val_check_cycle_point.as_slice(),
    ]
    .concat();

    Ok(Stage4OpeningPoints {
        registers_read_write_sumcheck_point: request.registers_read_write_sumcheck_point.to_vec(),
        registers_read_write_opening_point: registers_read_write_opening.opening_point,
        ram_val_check_sumcheck_point: request.ram_val_check_sumcheck_point.to_vec(),
        ram_val_check_cycle_point,
        ram_val_check_opening_point,
    })
}

/// Canonical order in which Stage 4 batched-sumcheck input claims are absorbed
/// into the transcript. The prover's `absorb_input_claims` and the verifier's
/// batched-sumcheck claim list both derive their order from this single
/// function, so the Fiat-Shamir batching coefficients cannot drift between
/// prover and verifier. Mirrors [`stage4_output_claim_values`].
pub fn stage4_input_claim_values<F: Field>(claims: &Stage4InputClaims<F>) -> Vec<F> {
    let mut values = vec![claims.registers_read_write];
    values.push(claims.ram_val_check);
    values
}

/// Canonical order in which Stage 4 output opening claims are absorbed into the
/// transcript. Single-sources [`append_stage4_opening_claims`] so the prover's
/// `output_claim_values` and the verifier's transcript appends cannot drift.
pub fn stage4_output_claim_values<F: Field>(claims: &Stage4Claims<F>) -> Vec<F> {
    let mut values = Vec::with_capacity(
        STAGE4_BATCH_BASE_OUTPUT_CLAIMS
            + usize::from(claims.advice.untrusted.is_some())
            + usize::from(claims.advice.trusted.is_some())
            + usize::from(claims.program_image_contribution.is_some()),
    );
    if let Some(opening_claim) = claims.advice.untrusted {
        values.push(opening_claim);
    }
    if let Some(opening_claim) = claims.advice.trusted {
        values.push(opening_claim);
    }
    if let Some(opening_claim) = claims.program_image_contribution {
        values.push(opening_claim);
    }
    values.push(claims.registers_read_write.registers_val);
    values.push(claims.registers_read_write.rs1_ra);
    values.push(claims.registers_read_write.rs2_ra);
    values.push(claims.registers_read_write.rd_wa);
    values.push(claims.registers_read_write.rd_inc);
    values.push(claims.ram_val_check.ram_ra);
    values.push(claims.ram_val_check.ram_inc);
    values
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage4Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let register_dimensions = proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);

    let registers_claims = registers::read_write_checking::<PCS::Field>(register_dimensions);
    check_boolean_hypercube(
        registers_claims.id,
        registers_claims.sumcheck.degree,
        &registers_claims.sumcheck.domain,
    )?;
    let registers_gamma = transcript.challenge_scalar();

    let (ram_read_write_opening_point, ram_output_check_opening_point) = match deps {
        Deps::Clear { stage2, .. } => (
            &stage2.batch.ram_read_write.opening_point,
            &stage2.batch.ram_output_check.opening_point,
        ),
        Deps::Zk { stage2, .. } => (
            &stage2.ram_val_check_inputs.ram_read_write_opening_point,
            &stage2.ram_val_check_inputs.ram_output_check_opening_point,
        ),
    };
    if ram_read_write_opening_point.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!(
                "RAM read-write opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_read_write_opening_point.len()
            ),
        });
    }
    let (r_address, r_cycle) = ram_read_write_opening_point.split_at(log_k);
    if ram_output_check_opening_point != r_address {
        let [ram_val, ram_val_final] = ram::val_check_input_openings();
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamValCheck,
            left: ram_val,
            right: ram_val_final,
        });
    }

    let ram_val_check_public_eval =
        public_initial_ram_evaluation(checked, preprocessing, r_address)?;

    append_ram_val_check_gamma_domain_separator(transcript);
    let ram_val_check_gamma = transcript.challenge_scalar();

    let ram_val_check_sumcheck = ram::val_check_sumcheck(trace_dimensions);
    check_boolean_hypercube(
        JoltRelationId::RamValCheck,
        ram_val_check_sumcheck.degree,
        &ram_val_check_sumcheck.domain,
    )?;

    let public =
        |challenges: Vec<PCS::Field>, batching_coefficients: Vec<PCS::Field>| Stage4PublicOutput {
            challenges,
            batching_coefficients,
            registers_gamma,
            ram_val_check_gamma,
        };

    if checked.zk {
        let Deps::Zk { .. } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage3" });
        };
        let mut statements = vec![SumcheckStatement::new(
            registers_claims.sumcheck.rounds,
            registers_claims.sumcheck.degree,
        )];
        statements.push(SumcheckStatement::new(
            ram_val_check_sumcheck.rounds,
            ram_val_check_sumcheck.degree,
        ));
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage4_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage4_sumcheck_proof,
                proof_label: "stage4_sumcheck_proof",
                output_claim_count: stage4_committed_output_claims(checked, proof),
                stage: JoltRelationId::RegistersReadWriteChecking,
            })?;

        let registers_point = consistency
            .try_instance_point(registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;
        let ram_val_point = consistency
            .try_instance_point(ram_val_check_sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamValCheck,
                reason: error.to_string(),
            })?;
        let opening_points = stage4_opening_points(Stage4OpeningPointRequest {
            register_dimensions,
            registers_read_write_sumcheck_point: &registers_point,
            ram_val_check_sumcheck_point: &ram_val_point,
            fixed_ram_address_point: r_address,
            fixed_ram_cycle_point: r_cycle,
        })?;

        return Ok(Stage4Output::Zk(Stage4ZkOutput {
            public: public(
                consistency.challenges(),
                consistency.batching_coefficients.clone(),
            ),
            batch_consistency: consistency,
            batch_output_claims,
            ram_val_check_public_eval,
            registers_read_write_opening_point: opening_points.registers_read_write_opening_point,
            ram_val_check_opening_point: opening_points.ram_val_check_opening_point,
        }));
    }

    let Deps::Clear { stage2, stage3, .. } = deps else {
        return Err(VerifierError::ExpectedClearProof { field: "stage3" });
    };
    let claims = &proof.clear_claims()?.stage4;
    let ram_val_check_init = ram_val_check_initial_evaluation(
        checked,
        proof,
        claims,
        r_address,
        ram_val_check_public_eval,
    )?;

    // WARNING: contribution order and selectors must stay in lockstep with
    // `ram_val_check_init` in zk/blindfold/mod.rs — the BlindFold constraint
    // is built from the same decomposition.
    let mut init_contributions = Vec::new();
    if ram_val_check_init.program_image_contribution.is_some() {
        init_contributions.push(FormulaInitContribution::program_image(-PCS::Field::one()));
    }
    for contribution in &ram_val_check_init.advice_contributions {
        let neg_selector = -contribution.selector;
        init_contributions.push(match contribution.kind {
            JoltAdviceKind::Trusted => FormulaInitContribution::trusted(neg_selector),
            JoltAdviceKind::Untrusted => FormulaInitContribution::untrusted(neg_selector),
        });
    }
    let ram_val_check_claims = ram::val_check::<PCS::Field>(
        trace_dimensions,
        RamValCheckInit::decomposed(ram_val_check_init.public_eval, init_contributions),
    );

    let [_right_operand_is_rs2, rs2_value_instruction, _right_operand_is_imm, _imm, _left_operand_is_rs1, rs1_value_instruction, _left_operand_is_pc, _unexpanded_pc] =
        instruction::input_virtualization_output_openings();
    let [_rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced] =
        registers_claim_reduction::claim_reduction_output_openings();
    if stage3.output_claims.registers_claim_reduction.rs1_value
        != stage3.output_claims.instruction_input.rs1_value
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RegistersReadWriteChecking,
            left: rs1_value_reduced,
            right: rs1_value_instruction,
        });
    }
    if stage3.output_claims.registers_claim_reduction.rs2_value
        != stage3.output_claims.instruction_input.rs2_value
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RegistersReadWriteChecking,
            left: rs2_value_reduced,
            right: rs2_value_instruction,
        });
    }

    let input_claims = stage4_input_claims(Stage4InputClaimRequest {
        stage2,
        stage3,
        ram_val_check_initial_eval: ram_val_check_init.full_eval,
        registers_gamma,
        ram_val_check_gamma,
    });

    let mut sumcheck_dimensions = vec![(
        registers_claims.sumcheck.rounds,
        registers_claims.sumcheck.degree,
    )];
    sumcheck_dimensions.push((
        ram_val_check_claims.sumcheck.rounds,
        ram_val_check_claims.sumcheck.degree,
    ));
    let sumcheck_claims: Vec<_> = sumcheck_dimensions
        .into_iter()
        .zip(stage4_input_claim_values(&input_claims))
        .map(|((rounds, degree), input_claim)| SumcheckClaim::new(rounds, degree, input_claim))
        .collect();
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage4_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::RegistersReadWriteChecking,
        reason: error.to_string(),
    })?;

    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
    let ram_val_point = batch
        .try_instance_point(ram_val_check_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamValCheck,
            reason: error.to_string(),
        })?;
    let opening_points = stage4_opening_points(Stage4OpeningPointRequest {
        register_dimensions,
        registers_read_write_sumcheck_point: registers_point,
        ram_val_check_sumcheck_point: ram_val_point,
        fixed_ram_address_point: r_address,
        fixed_ram_cycle_point: r_cycle,
    })?;

    let expected_outputs = stage4_expected_output_claims(Stage4ExpectedOutputRequest {
        fixed_register_cycle_point: &stage3.batch.registers_claim_reduction.opening_point,
        registers_read_write_opening_point: &opening_points.registers_read_write_opening_point,
        fixed_ram_cycle_point: r_cycle,
        ram_val_check_cycle_point: &opening_points.ram_val_check_cycle_point,
        registers_gamma,
        ram_val_check_gamma,
        claims,
    })?;
    let expected_final_claim =
        stage4_expected_final_claim(&batch.batching_coefficients, &expected_outputs)?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::RegistersReadWriteChecking,
        });
    }

    append_stage4_opening_claims(
        transcript,
        proof.untrusted_advice_commitment.is_some(),
        checked.trusted_advice_commitment_present,
        checked.precommitted.program_image.is_some(),
        claims,
    )?;

    Ok(Stage4Output::Clear(Stage4ClearOutput {
        public: public(
            batch.reduction.point.as_slice().to_vec(),
            batch.batching_coefficients.clone(),
        ),
        output_claims: claims.clone(),
        ram_val_check_init,
        batch: VerifiedStage4Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: batch.reduction.point.clone(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            registers_read_write: VerifiedStage4Sumcheck {
                input_claim: input_claims.registers_read_write,
                sumcheck_point: opening_points.registers_read_write_sumcheck_point,
                opening_point: opening_points.registers_read_write_opening_point,
                expected_output_claim: expected_outputs.registers_read_write,
            },
            ram_val_check: VerifiedStage4Sumcheck {
                input_claim: input_claims.ram_val_check,
                sumcheck_point: opening_points.ram_val_check_sumcheck_point,
                opening_point: opening_points.ram_val_check_opening_point,
                expected_output_claim: expected_outputs.ram_val_check,
            },
        },
    }))
}

fn ram_val_check_initial_evaluation<PCS, VC, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    claims: &Stage4Claims<PCS::Field>,
    r_address: &[PCS::Field],
    public_eval: PCS::Field,
) -> Result<RamValCheckInitialEvaluation<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let mut full_eval = public_eval;
    let program_image_contribution = collect_program_image_contribution(
        checked.precommitted.program_image.is_some(),
        claims.program_image_contribution,
        r_address,
        &mut full_eval,
    )?;
    let mut advice_contributions = Vec::new();
    let untrusted_present = proof.untrusted_advice_commitment.is_some();
    collect_advice_contribution(
        JoltAdviceKind::Untrusted,
        untrusted_present,
        claims.advice.untrusted,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;
    collect_advice_contribution(
        JoltAdviceKind::Trusted,
        checked.trusted_advice_commitment_present,
        claims.advice.trusted,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;

    Ok(RamValCheckInitialEvaluation {
        public_eval,
        program_image_contribution,
        advice_contributions,
        full_eval,
    })
}

fn collect_program_image_contribution<F: Field>(
    committed_program: bool,
    opening_claim: Option<F>,
    r_address: &[F],
    full_eval: &mut F,
) -> Result<Option<VerifiedRamValCheckProgramImageContribution<F>>, VerifierError> {
    let opening = program_image::ram_val_check_contribution_opening();
    if !committed_program {
        if opening_claim.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaim { id: opening });
        }
        return Ok(None);
    }

    let opening_claim = opening_claim.ok_or(VerifierError::MissingOpeningClaim { id: opening })?;
    *full_eval += opening_claim;
    Ok(Some(VerifiedRamValCheckProgramImageContribution {
        opening_claim,
        opening_point: r_address.to_vec(),
    }))
}

fn public_initial_ram_evaluation<PCS, VC>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    r_address: &[PCS::Field],
) -> Result<PCS::Field, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    // In committed program mode the image words are bound via the staged
    // `ProgramImageInitContributionRw` opening, so only inputs are public here.
    let public_initial_ram = match preprocessing.program.as_full() {
        Some(full) => PublicInitialRam::new(&full.ram, &checked.public_io),
        None => PublicInitialRam::inputs_only(&checked.public_io),
    }
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamValCheck,
        reason: error.to_string(),
    })?;
    for segment in &public_initial_ram.segments {
        let end = segment.start_index + segment.words.len();
        if end > checked.ram_K {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamValCheck,
                reason: format!(
                    "public initial RAM segment [{}, {}) exceeds RAM domain {}",
                    segment.start_index, end, checked.ram_K
                ),
            });
        }
    }

    Ok(sparse_segments_mle_msb(
        public_initial_ram
            .segments
            .iter()
            .map(|segment| (segment.start_index, segment.words.as_slice())),
        r_address,
    ))
}

fn stage4_committed_output_claims<PCS, VC, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
) -> usize
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    STAGE4_BATCH_BASE_OUTPUT_CLAIMS
        + usize::from(proof.untrusted_advice_commitment.is_some())
        + usize::from(checked.trusted_advice_commitment_present)
        + usize::from(checked.precommitted.program_image.is_some())
}

fn collect_advice_contribution<F: Field>(
    kind: JoltAdviceKind,
    present: bool,
    opening_claim: Option<F>,
    checked: &CheckedInputs,
    r_address: &[F],
    full_eval: &mut F,
    contributions: &mut Vec<VerifiedRamValCheckAdviceContribution<F>>,
) -> Result<(), VerifierError> {
    let opening = ram::val_check_advice_opening(kind);
    if !present {
        if opening_claim.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaim { id: opening });
        }
        return Ok(());
    }

    let opening_claim = opening_claim.ok_or(VerifierError::MissingOpeningClaim { id: opening })?;
    let layout = &checked.public_io.memory_layout;
    let (start_address, max_size) = match kind {
        JoltAdviceKind::Trusted => (layout.trusted_advice_start, layout.max_trusted_advice_size),
        JoltAdviceKind::Untrusted => (
            layout.untrusted_advice_start,
            layout.max_untrusted_advice_size,
        ),
    };
    if max_size == 0 {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!("{kind:?} advice commitment is present but configured size is zero"),
        });
    }

    let start_index = layout
        .remapped_word_address(start_address)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: error.to_string(),
        })? as usize;
    let advice_num_vars = ((max_size as usize) / 8).next_power_of_two().ilog2() as usize;
    let selector =
        block_selector_mle_msb(start_index, advice_num_vars, r_address).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamValCheck,
                reason: error.to_string(),
            }
        })?;
    let opening_point = r_address[r_address.len() - advice_num_vars..].to_vec();
    *full_eval += selector * opening_claim;
    contributions.push(VerifiedRamValCheckAdviceContribution {
        kind,
        selector,
        opening_claim,
        opening_point,
    });
    Ok(())
}

fn check_boolean_hypercube(
    stage: JoltRelationId,
    degree: usize,
    domain: &JoltSumcheckDomain,
) -> Result<(), VerifierError> {
    if degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree { stage, degree });
    }
    if !matches!(domain, JoltSumcheckDomain::BooleanHypercube) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain { stage });
    }
    Ok(())
}

pub fn append_stage4_opening_claims<F, T>(
    transcript: &mut T,
    untrusted_advice_commitment_present: bool,
    trusted_advice_commitment_present: bool,
    committed_program: bool,
    claims: &Stage4Claims<F>,
) -> Result<(), VerifierError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if untrusted_advice_commitment_present {
        let id = ram::val_check_advice_opening(JoltAdviceKind::Untrusted);
        let opening_claim = claims
            .advice
            .untrusted
            .ok_or(VerifierError::MissingOpeningClaim { id })?;
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    if trusted_advice_commitment_present {
        let id = ram::val_check_advice_opening(JoltAdviceKind::Trusted);
        let opening_claim = claims
            .advice
            .trusted
            .ok_or(VerifierError::MissingOpeningClaim { id })?;
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    if committed_program {
        let id = program_image::ram_val_check_contribution_opening();
        let opening_claim = claims
            .program_image_contribution
            .ok_or(VerifierError::MissingOpeningClaim { id })?;
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.registers_val);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rs1_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rs2_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rd_wa);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rd_inc);
    transcript.append_labeled(b"opening_claim", &claims.ram_val_check.ram_ra);
    transcript.append_labeled(b"opening_claim", &claims.ram_val_check.ram_inc);
    Ok(())
}

pub fn append_ram_val_check_gamma_domain_separator<T: Transcript>(transcript: &mut T) {
    transcript.append(&LabelWithCount(b"ram_val_check_gamma", 0));
    transcript.append_bytes(&[]);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stages::stage4::inputs::{
        RamValCheckAdviceOpeningClaims, RamValCheckOutputOpeningClaims,
        RegistersReadWriteOutputOpeningClaims,
    };
    use jolt_field::{CanonicalBytes, FixedByteSize, Fr, FromPrimitiveInt};

    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
        state: [u8; 32],
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(0)
        }

        fn state(&self) -> [u8; 32] {
            self.state
        }
    }

    #[test]
    fn opening_claim_appends_follow_core_order_without_advice() {
        let claims = test_claims();
        let mut transcript = RecordingTranscript::new(b"stage4-openings");

        let result = append_stage4_opening_claims(&mut transcript, false, false, false, &claims);
        assert!(result.is_ok(), "stage 4 openings should append: {result:?}");

        let expected = [
            claims.registers_read_write.registers_val,
            claims.registers_read_write.rs1_ra,
            claims.registers_read_write.rs2_ra,
            claims.registers_read_write.rd_wa,
            claims.registers_read_write.rd_inc,
        ];
        let mut expected = expected.to_vec();
        expected.extend([claims.ram_val_check.ram_ra, claims.ram_val_check.ram_inc]);

        assert_opening_claim_payloads(&transcript, &expected);
    }

    #[test]
    fn ram_val_check_gamma_domain_separator_matches_core_empty_bytes_append() {
        let mut transcript = RecordingTranscript::new(b"stage4-gamma");

        append_ram_val_check_gamma_domain_separator(&mut transcript);

        assert_eq!(transcript.chunks.len(), 2);
        let mut packed = vec![0; 32];
        packed[..b"ram_val_check_gamma".len()].copy_from_slice(b"ram_val_check_gamma");
        assert_eq!(transcript.chunks[0], packed);
        assert!(transcript.chunks[1].is_empty());
    }

    fn test_claims() -> Stage4Claims<Fr> {
        Stage4Claims {
            advice: RamValCheckAdviceOpeningClaims {
                untrusted: Some(Fr::from_u64(1)),
                trusted: Some(Fr::from_u64(2)),
            },
            program_image_contribution: None,
            registers_read_write: RegistersReadWriteOutputOpeningClaims {
                registers_val: Fr::from_u64(3),
                rs1_ra: Fr::from_u64(4),
                rs2_ra: Fr::from_u64(5),
                rd_wa: Fr::from_u64(6),
                rd_inc: Fr::from_u64(7),
            },
            ram_val_check: RamValCheckOutputOpeningClaims {
                ram_ra: Fr::from_u64(8),
                ram_inc: Fr::from_u64(9),
            },
        }
    }

    fn assert_opening_claim_payloads(transcript: &RecordingTranscript, expected: &[Fr]) {
        assert_eq!(transcript.chunks.len(), expected.len() * 2);
        let label = opening_claim_label();
        for (index, expected_payload) in expected.iter().copied().enumerate() {
            assert_eq!(transcript.chunks[2 * index], label);
            assert_eq!(
                transcript.chunks[2 * index + 1],
                scalar_bytes(expected_payload)
            );
        }
    }

    fn opening_claim_label() -> Vec<u8> {
        let mut label = vec![0; 32];
        label[..b"opening_claim".len()].copy_from_slice(b"opening_claim");
        label
    }

    fn scalar_bytes(value: Fr) -> Vec<u8> {
        let mut bytes = vec![0; Fr::NUM_BYTES];
        value.to_bytes_le(&mut bytes);
        bytes.reverse();
        bytes
    }
}
