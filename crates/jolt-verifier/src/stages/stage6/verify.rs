#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{
        bytecode as field_bytecode, claim_reductions::increments as field_increments,
        dimensions::FieldRegistersTraceDimensions,
    },
    FieldInlineChallengeId, FieldInlinePublicId, FieldInlineRelationClaims,
    FieldInlineVirtualPolynomial, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity::{self, BooleanityDimensions},
        bytecode::{
            self, BytecodeReadRafCommittedEvaluationInputs, BytecodeReadRafDimensions,
            BytecodeReadRafEvaluationInputs,
        },
        claim_reductions::{advice, bytecode as bytecode_reduction, increments, program_image},
        dimensions::{
            committed_address_chunks, JoltFormulaDimensions, TraceDimensions, REGISTER_ADDRESS_BITS,
        },
        instruction::{self, InstructionRaVirtualizationDimensions},
        ram::{self, RamRaVirtualizationDimensions},
    },
    AdviceClaimReductionLayout, AdviceClaimReductionPublic, BooleanityChallenge, BooleanityPublic,
    BytecodeClaimReductionChallenge, BytecodeReadRafChallenge, IncClaimReductionChallenge,
    IncClaimReductionPublic, InstructionRaVirtualizationChallenge, JoltAdviceKind, JoltChallengeId,
    JoltPublicId, JoltRelationClaims, JoltRelationId, JoltSumcheckDomain, JoltVirtualPolynomial,
    PrecommittedReductionLayout, RamHammingBooleanityChallenge, RamRaVirtualizationChallenge,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::{try_eq_mle, Point};
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::Transcript;
use num_traits::{One, Zero};

use super::{
    inputs::{Deps, Stage6Claims},
    outputs::{
        AdviceCyclePhasePublicOutput, BooleanityPublicOutput, BytecodeReadRafPublicOutput,
        InstructionRaVirtualizationPublicOutput, RamRaVirtualizationPublicOutput,
        Stage6AddressPhasePublicOutput, Stage6ClearOutput, Stage6Output, Stage6PublicOutput,
        Stage6SumcheckPublicOutput, Stage6ZkOutput, VerifiedAdviceCyclePhaseSumcheck,
        VerifiedBooleanitySumcheck, VerifiedBytecodeReadRafSumcheck,
        VerifiedInstructionRaVirtualizationSumcheck, VerifiedRamRaVirtualizationSumcheck,
        VerifiedStage6AddressPhaseSumcheck, VerifiedStage6Batch, VerifiedStage6Sumcheck,
    },
    verify_a, verify_b,
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
        stage4::Stage4ClearOutput, stage5::Stage5ClearOutput, stage5::Stage5ZkOutput,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[cfg(feature = "field-inline")]
const fn field_inline_stage6_output_claim_count() -> usize {
    1
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_stage6_output_claim_count() -> usize {
    0
}

// WARNING: this is the verifier-internal batch claim record used by the
// canonical `verify()` implementation. It intentionally tracks the
// bytecode/program-image claim reductions that new 09 added to the Stage 6
// batch but that the public, prover-facing `Stage6BatchInputClaims` (defined
// below) does not surface. Keep it private so it cannot be confused with the
// exported type.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VerifyStage6BatchInputClaims<F: Field> {
    bytecode_read_raf_address: F,
    booleanity_address: F,
    bytecode_read_raf: F,
    booleanity: F,
    ram_hamming_booleanity: F,
    ram_ra_virtualization: F,
    instruction_ra_virtualization: F,
    inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction: F,
    trusted_advice_cycle_phase: Option<F>,
    untrusted_advice_cycle_phase: Option<F>,
    bytecode_claim_reduction: Option<F>,
    program_image_claim_reduction: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VerifyStage6BatchExpectedOutputClaims<F: Field> {
    bytecode_read_raf_address: F,
    booleanity_address: F,
    bytecode_read_raf: F,
    booleanity: F,
    ram_hamming_booleanity: F,
    ram_ra_virtualization: F,
    instruction_ra_virtualization: F,
    inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction: F,
    trusted_advice_cycle_phase: Option<F>,
    untrusted_advice_cycle_phase: Option<F>,
    bytecode_claim_reduction: Option<F>,
    program_image_claim_reduction: Option<F>,
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage6Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage5" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage5" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions =
        jolt_claims::protocols::jolt::formulas::dimensions::TraceDimensions::new(log_t);
    let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let bytecode_reduction_layout = checked.precommitted.bytecode.as_ref();
    let program_image_reduction_layout = checked.precommitted.program_image.as_ref();
    let committed_program = bytecode_reduction_layout.is_some();

    let bytecode_address_claims =
        bytecode::read_raf_address_phase::<PCS::Field>(formula_dimensions.bytecode_read_raf);
    let bytecode_claims = if committed_program {
        bytecode::read_raf_cycle_phase_committed::<PCS::Field>(formula_dimensions.bytecode_read_raf)
    } else {
        bytecode::read_raf_cycle_phase::<PCS::Field>(formula_dimensions.bytecode_read_raf)
    };
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_address_claims =
        booleanity::booleanity_address_phase::<PCS::Field>(booleanity_dimensions);
    let booleanity_claims = booleanity::booleanity_cycle_phase::<PCS::Field>(booleanity_dimensions);
    let ram_hamming_claims = ram::hamming_booleanity::<PCS::Field>(trace_dimensions);
    let ram_ra_claims =
        ram::ra_virtualization::<PCS::Field>(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = instruction::ra_virtualization::<PCS::Field>(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = increments::claim_reduction::<PCS::Field>(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_inc_claims =
        field_increments::claim_reduction::<PCS::Field>(FieldRegistersTraceDimensions::new(log_t));

    let trusted_advice_layout = checked.precommitted.trusted_advice.as_ref();
    let untrusted_advice_layout = checked.precommitted.untrusted_advice.as_ref();
    let trusted_advice_claims = trusted_advice_layout.map(|layout| {
        advice::cycle_phase::<PCS::Field>(JoltAdviceKind::Trusted, layout.dimensions())
    });
    let untrusted_advice_claims = untrusted_advice_layout.map(|layout| {
        advice::cycle_phase::<PCS::Field>(JoltAdviceKind::Untrusted, layout.dimensions())
    });
    let bytecode_reduction_claims = bytecode_reduction_layout.map(|layout| {
        bytecode_reduction::cycle_phase::<PCS::Field>(layout.dimensions(), layout.chunk_count())
    });
    let program_image_reduction_claims = program_image_reduction_layout
        .map(|layout| program_image::cycle_phase::<PCS::Field>(layout.dimensions()));

    for claim in [
        &bytecode_address_claims,
        &bytecode_claims,
        &booleanity_address_claims,
        &booleanity_claims,
        &ram_hamming_claims,
        &ram_ra_claims,
        &instruction_ra_claims,
        &inc_claims,
    ] {
        validate_compressed_stage_claim(claim)?;
    }
    #[cfg(feature = "field-inline")]
    validate_compressed_field_inline_stage_claim(&field_inc_claims)?;
    for claim in [
        &trusted_advice_claims,
        &untrusted_advice_claims,
        &bytecode_reduction_claims,
        &program_image_reduction_claims,
    ]
    .into_iter()
    .flatten()
    {
        validate_compressed_stage_claim(claim)?;
    }

    let bytecode_gamma_powers = transcript.challenge_scalar_powers(8);
    let bytecode_gamma = bytecode_gamma_powers[1];
    let stage1_gammas = transcript.challenge_scalar_powers(2 + NUM_CIRCUIT_FLAGS);
    let stage2_gammas = transcript.challenge_scalar_powers(4);
    let stage3_gammas = transcript.challenge_scalar_powers(9);
    let stage4_gammas = transcript.challenge_scalar_powers(3);
    let stage5_gammas =
        transcript.challenge_scalar_powers(2 + LookupTableKind::<RISCV_XLEN>::COUNT);

    let (stage5_instruction_address, stage5_instruction_cycle) = match deps {
        Deps::Clear { stage5, .. } => (
            &stage5.batch.instruction_read_raf.r_address,
            &stage5.batch.instruction_read_raf.r_cycle,
        ),
        Deps::Zk { stage5 } => (
            &stage5.instruction_read_raf.r_address,
            &stage5.instruction_read_raf.r_cycle,
        ),
    };
    let mut booleanity_reference_address = stage5_instruction_address.clone();
    booleanity_reference_address.reverse();
    if booleanity_reference_address.len() < proof.one_hot_config.committed_chunk_bits() {
        let missing =
            proof.one_hot_config.committed_chunk_bits() - booleanity_reference_address.len();
        booleanity_reference_address.extend(transcript.challenge_vector(missing));
    } else {
        booleanity_reference_address = booleanity_reference_address
            [booleanity_reference_address.len() - proof.one_hot_config.committed_chunk_bits()..]
            .to_vec();
    }
    let mut booleanity_reference_cycle = stage5_instruction_cycle.clone();
    booleanity_reference_cycle.reverse();
    let mut booleanity_gamma = transcript.challenge();
    if booleanity_gamma.is_zero() {
        booleanity_gamma = PCS::Field::one();
    }

    let public = |instruction_ra_gamma_powers: Vec<PCS::Field>,
                  inc_gamma: PCS::Field,
                  #[cfg(feature = "field-inline")] field_inc_gamma: PCS::Field,
                  eta: Option<PCS::Field>,
                  address_phase_challenges: Vec<PCS::Field>,
                  address_phase_batching_coefficients: Vec<PCS::Field>,
                  challenges: Vec<PCS::Field>,
                  batching_coefficients: Vec<PCS::Field>| Stage6PublicOutput {
        address_phase_challenges,
        address_phase_batching_coefficients,
        challenges,
        batching_coefficients,
        bytecode_gamma_powers: bytecode_gamma_powers.clone(),
        stage1_gammas: stage1_gammas.clone(),
        stage2_gammas: stage2_gammas.clone(),
        stage3_gammas: stage3_gammas.clone(),
        stage4_gammas: stage4_gammas.clone(),
        stage5_gammas: stage5_gammas.clone(),
        booleanity_reference_address: booleanity_reference_address.clone(),
        booleanity_reference_cycle: booleanity_reference_cycle.clone(),
        booleanity_gamma,
        instruction_ra_gamma_powers: instruction_ra_gamma_powers.clone(),
        inc_gamma,
        #[cfg(feature = "field-inline")]
        field_inline: super::outputs::FieldInlineStage6PublicOutput { field_inc_gamma },
        bytecode_reduction_eta: eta,
    };

    if checked.zk {
        let Deps::Zk { stage5 } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage5" });
        };
        let stage6a = verify_a::verify_zk(
            checked,
            proof,
            transcript,
            &bytecode_address_claims,
            &booleanity_address_claims,
        )?;

        let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
            formula_dimensions
                .instruction_ra_virtualization
                .num_virtual_ra_polys(),
        );
        let inc_gamma = transcript.challenge_scalar();
        #[cfg(feature = "field-inline")]
        let field_inc_gamma = transcript.challenge_scalar();
        let eta = committed_program.then(|| transcript.challenge_scalar());

        let mut statements = vec![
            SumcheckStatement::new(
                bytecode_claims.sumcheck.rounds,
                bytecode_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                booleanity_claims.sumcheck.rounds,
                booleanity_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                ram_hamming_claims.sumcheck.rounds,
                ram_hamming_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(ram_ra_claims.sumcheck.rounds, ram_ra_claims.sumcheck.degree),
            SumcheckStatement::new(
                instruction_ra_claims.sumcheck.rounds,
                instruction_ra_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(inc_claims.sumcheck.rounds, inc_claims.sumcheck.degree),
        ];
        #[cfg(feature = "field-inline")]
        statements.push(SumcheckStatement::new(
            field_inc_claims.sumcheck.rounds,
            field_inc_claims.sumcheck.degree,
        ));
        if let Some(claim) = &trusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &untrusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &bytecode_reduction_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &program_image_reduction_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage6b_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;

        let bytecode_point = consistency
            .try_instance_point(bytecode_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: error.to_string(),
            })?;
        let bytecode_r_cycle = bytecode_point.iter().rev().copied().collect::<Vec<_>>();
        let bytecode_opening_point = [
            stage6a.bytecode_r_address.as_slice(),
            bytecode_r_cycle.as_slice(),
        ]
        .concat();
        let bytecode_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(&stage6a.bytecode_r_address)
            .into_iter()
            .map(|r_address_chunk| {
                [r_address_chunk.as_slice(), bytecode_r_cycle.as_slice()].concat()
            })
            .collect::<Vec<_>>();

        let booleanity_point = consistency
            .try_instance_point(booleanity_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            })?;
        let booleanity_r_cycle = booleanity_point.iter().rev().copied().collect::<Vec<_>>();
        let booleanity_opening_point = [
            stage6a.booleanity_r_address.as_slice(),
            booleanity_r_cycle.as_slice(),
        ]
        .concat();

        let ram_hamming_point = consistency
            .try_instance_point(ram_hamming_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            })?;
        let ram_hamming_opening_point = trace_dimensions
            .cycle_opening_point(&ram_hamming_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            })?;

        let ram_ra_point = consistency
            .try_instance_point(ram_ra_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: error.to_string(),
            })?;
        let ram_ra_cycle = trace_dimensions
            .cycle_opening_point(&ram_ra_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: error.to_string(),
            })?;
        let ram_reduced_opening_point = &stage5.ram_ra_claim_reduction.opening_point;
        if ram_reduced_opening_point.len() != log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: format!(
                    "RAM RA reduction opening point length mismatch: expected {}, got {}",
                    log_k + log_t,
                    ram_reduced_opening_point.len()
                ),
            });
        }
        let (ram_reduced_address, _) = ram_reduced_opening_point.split_at(log_k);
        let ram_ra_opening_point = [ram_reduced_address, ram_ra_cycle.as_slice()].concat();
        let ram_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(ram_reduced_address)
            .into_iter()
            .map(|r_address_chunk| [r_address_chunk.as_slice(), ram_ra_cycle.as_slice()].concat())
            .collect::<Vec<_>>();

        let instruction_ra_point = consistency
            .try_instance_point(instruction_ra_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionRaVirtualization,
                reason: error.to_string(),
            })?;
        let instruction_ra_cycle = trace_dimensions
            .cycle_opening_point(&instruction_ra_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionRaVirtualization,
                reason: error.to_string(),
            })?;
        let instruction_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(&stage5.instruction_read_raf.r_address)
            .into_iter()
            .map(|r_address_chunk| {
                [r_address_chunk.as_slice(), instruction_ra_cycle.as_slice()].concat()
            })
            .collect::<Vec<_>>();
        let instruction_ra_opening_point = [
            stage5.instruction_read_raf.r_address.as_slice(),
            instruction_ra_cycle.as_slice(),
        ]
        .concat();

        let inc_point = consistency
            .try_instance_point(inc_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let inc_opening_point =
            trace_dimensions
                .cycle_opening_point(&inc_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                })?;
        #[cfg(feature = "field-inline")]
        let (field_inc_point, field_inc_opening_point) = {
            let field_inc_point = consistency
                .try_instance_point(field_inc_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                })?;
            let field_inc_opening_point = trace_dimensions
                .cycle_opening_point(&field_inc_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                })?;
            (field_inc_point, field_inc_opening_point)
        };

        let trusted_advice = if let (Some(layout), Some(claim)) =
            (trusted_advice_layout, trusted_advice_claims.as_ref())
        {
            Some(verify_b::advice_cycle_phase_public(
                &consistency,
                claim,
                layout,
                JoltAdviceKind::Trusted,
            )?)
        } else {
            None
        };
        let untrusted_advice = if let (Some(layout), Some(claim)) =
            (untrusted_advice_layout, untrusted_advice_claims.as_ref())
        {
            Some(verify_b::advice_cycle_phase_public(
                &consistency,
                claim,
                layout,
                JoltAdviceKind::Untrusted,
            )?)
        } else {
            None
        };
        let bytecode_cycle_phase = if let (Some(layout), Some(claim)) = (
            bytecode_reduction_layout,
            bytecode_reduction_claims.as_ref(),
        ) {
            Some(verify_b::committed_reduction_cycle_phase_public(
                &consistency,
                claim,
                layout.precommitted(),
                JoltRelationId::BytecodeClaimReductionCyclePhase,
            )?)
        } else {
            None
        };
        let program_image_cycle_phase = if let (Some(layout), Some(claim)) = (
            program_image_reduction_layout,
            program_image_reduction_claims.as_ref(),
        ) {
            Some(verify_b::committed_reduction_cycle_phase_public(
                &consistency,
                claim,
                layout.precommitted(),
                JoltRelationId::ProgramImageClaimReductionCyclePhase,
            )?)
        } else {
            None
        };

        let bytecode_output_openings =
            bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
        let booleanity_output_openings =
            booleanity::booleanity_output_openings(formula_dimensions.ra_layout);
        let ram_ra_output_openings =
            ram::ra_virtualization_output_openings(formula_dimensions.ram_ra_virtualization);
        let instruction_ra_output_openings = instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        );
        let flat_instruction_ra_output_openings = instruction_ra_output_openings.all();
        let aliased_bytecode_ra_openings = verify_b::aliased_booleanity_bytecode_openings(
            &bytecode_ra_opening_points,
            &booleanity_opening_point,
        );
        let bytecode_reduction_output_claims = bytecode_reduction_layout.map_or(0, |layout| {
            bytecode_reduction::cycle_phase_output_openings(
                layout.dimensions(),
                layout.chunk_count(),
            )
            .len()
        });
        let program_image_reduction_output_claims = program_image_reduction_layout
            .map_or(0, |layout| {
                program_image::cycle_phase_output_openings(layout.dimensions()).len()
            });
        let committed_output_claims = bytecode_output_openings.bytecode_ra.len()
            + booleanity_output_openings.len()
            - aliased_bytecode_ra_openings
            + 1
            + ram_ra_output_openings.len()
            + flat_instruction_ra_output_openings.len()
            + 2
            + field_inline_stage6_output_claim_count()
            + usize::from(trusted_advice_claims.is_some())
            + usize::from(untrusted_advice_claims.is_some())
            + bytecode_reduction_output_claims
            + program_image_reduction_output_claims;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage6b_sumcheck_proof,
                proof_label: "stage6b_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::BytecodeReadRaf,
            })?;

        return Ok(Stage6Output::Zk(Stage6ZkOutput {
            public: public(
                instruction_ra_gamma_powers,
                inc_gamma,
                #[cfg(feature = "field-inline")]
                field_inc_gamma,
                eta,
                stage6a.address_phase_consistency.challenges(),
                stage6a
                    .address_phase_consistency
                    .batching_coefficients
                    .clone(),
                consistency.challenges(),
                consistency.batching_coefficients.clone(),
            ),
            batch_consistency: consistency,
            batch_output_claims,
            address_phase_consistency: stage6a.address_phase_consistency,
            address_phase_output_claims: stage6a.address_phase_output_claims,
            bytecode_read_raf_address: Stage6AddressPhasePublicOutput {
                sumcheck_point: stage6a.bytecode_address_point,
                opening_point: stage6a.bytecode_r_address.clone(),
            },
            booleanity_address: Stage6AddressPhasePublicOutput {
                sumcheck_point: stage6a.booleanity_address_point,
                opening_point: stage6a.booleanity_r_address.clone(),
            },
            bytecode_read_raf: BytecodeReadRafPublicOutput {
                sumcheck_point: bytecode_point,
                r_address: stage6a.bytecode_r_address,
                r_cycle: bytecode_r_cycle,
                full_opening_point: bytecode_opening_point,
                bytecode_ra_opening_points,
            },
            booleanity: BooleanityPublicOutput {
                sumcheck_point: booleanity_point,
                r_address: stage6a.booleanity_r_address,
                r_cycle: booleanity_r_cycle,
                opening_point: booleanity_opening_point,
                reference_address: booleanity_reference_address,
                reference_cycle: booleanity_reference_cycle,
            },
            ram_hamming_booleanity: Stage6SumcheckPublicOutput {
                sumcheck_point: ram_hamming_point,
                opening_point: ram_hamming_opening_point,
            },
            ram_ra_virtualization: RamRaVirtualizationPublicOutput {
                sumcheck_point: ram_ra_point,
                opening_point: ram_ra_opening_point,
                ram_ra_opening_points,
            },
            instruction_ra_virtualization: InstructionRaVirtualizationPublicOutput {
                sumcheck_point: instruction_ra_point,
                opening_point: instruction_ra_opening_point,
                instruction_ra_opening_points,
            },
            inc_claim_reduction: Stage6SumcheckPublicOutput {
                sumcheck_point: inc_point,
                opening_point: inc_opening_point,
            },
            #[cfg(feature = "field-inline")]
            field_inline: super::outputs::FieldInlineStage6ZkOutput {
                field_registers_inc_claim_reduction: Stage6SumcheckPublicOutput {
                    sumcheck_point: field_inc_point,
                    opening_point: field_inc_opening_point,
                },
            },
            trusted_advice_cycle_phase: trusted_advice,
            untrusted_advice_cycle_phase: untrusted_advice,
            bytecode_cycle_phase,
            program_image_cycle_phase,
        }));
    }

    let Deps::Clear {
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
    } = deps
    else {
        return Err(VerifierError::ExpectedClearProof { field: "stage5" });
    };
    let claims = &proof.clear_claims()?.stage6;
    if committed_program != claims.address_phase.bytecode_val_stages.is_some() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode Val-stage claims presence ({}) does not match committed program mode ({committed_program})",
                claims.address_phase.bytecode_val_stages.is_some()
            ),
        });
    }

    let bytecode_input_openings = bytecode::read_raf_input_openings();
    let [(spartan_shift_unexpanded_pc, instruction_input_unexpanded_pc)] =
        bytecode::read_raf_consistency_openings();
    if stage3.output_claims.shift.unexpanded_pc
        != stage3.output_claims.instruction_input.unexpanded_pc
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
            left: spartan_shift_unexpanded_pc,
            right: instruction_input_unexpanded_pc,
        });
    }

    let [ram_ra_reduced] = ram::ra_virtualization_input_openings();
    let instruction_ra_input_openings = instruction::ra_virtualization_input_openings(
        formula_dimensions.instruction_ra_virtualization,
    );
    let [ram_inc_read_write, ram_inc_val_check, rd_inc_read_write, rd_inc_val_evaluation] =
        increments::claim_reduction_input_openings();
    #[cfg(feature = "field-inline")]
    let [field_rd_inc_read_write, field_rd_inc_val_evaluation] =
        field_increments::claim_reduction_input_openings();

    let bytecode_read_raf_address_input = bytecode_address_claims.input.expression().try_evaluate(
        |id| {
            if *id == bytecode_input_openings.spartan_outer.unexpanded_pc {
                return Ok(stage1.outer.unexpanded_pc);
            }
            if *id == bytecode_input_openings.spartan_outer.imm {
                return Ok(stage1.outer.imm);
            }
            for (flag, opening) in &bytecode_input_openings.spartan_outer.op_flags {
                if *id == *opening {
                    return stage1
                        .outer
                        .claim(JoltVirtualPolynomial::OpFlags(*flag))
                        .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                }
            }
            if *id == bytecode_input_openings.spartan_product.jump {
                return Ok(stage2.output_claims.product_remainder.jump_flag);
            }
            if *id == bytecode_input_openings.spartan_product.branch {
                return Ok(stage2.output_claims.product_remainder.branch_flag);
            }
            if *id
                == bytecode_input_openings
                    .spartan_product
                    .write_lookup_output_to_rd
            {
                return Ok(stage2
                    .output_claims
                    .product_remainder
                    .write_lookup_output_to_rd);
            }
            if *id == bytecode_input_openings.spartan_product.virtual_instruction {
                return Ok(stage2.output_claims.product_remainder.virtual_instruction);
            }
            if *id == bytecode_input_openings.instruction_input.imm {
                return Ok(stage3.output_claims.instruction_input.imm);
            }
            if *id
                == bytecode_input_openings
                    .instruction_input
                    .unexpanded_pc_from_shift
            {
                return Ok(stage3.output_claims.shift.unexpanded_pc);
            }
            if *id
                == bytecode_input_openings
                    .instruction_input
                    .left_operand_is_rs1_value
            {
                return Ok(stage3.output_claims.instruction_input.left_operand_is_rs1);
            }
            if *id == bytecode_input_openings.instruction_input.left_operand_is_pc {
                return Ok(stage3.output_claims.instruction_input.left_operand_is_pc);
            }
            if *id
                == bytecode_input_openings
                    .instruction_input
                    .right_operand_is_rs2_value
            {
                return Ok(stage3.output_claims.instruction_input.right_operand_is_rs2);
            }
            if *id
                == bytecode_input_openings
                    .instruction_input
                    .right_operand_is_imm
            {
                return Ok(stage3.output_claims.instruction_input.right_operand_is_imm);
            }
            if *id == bytecode_input_openings.instruction_input.is_noop_from_shift {
                return Ok(stage3.output_claims.shift.is_noop);
            }
            if *id
                == bytecode_input_openings
                    .instruction_input
                    .virtual_instruction_from_shift
            {
                return Ok(stage3.output_claims.shift.is_virtual);
            }
            if *id
                == bytecode_input_openings
                    .instruction_input
                    .is_first_in_sequence_from_shift
            {
                return Ok(stage3.output_claims.shift.is_first_in_sequence);
            }
            if *id == bytecode_input_openings.registers_read_write.rd_wa {
                return Ok(stage4.output_claims.registers_read_write.rd_wa);
            }
            if *id == bytecode_input_openings.registers_read_write.rs1_ra {
                return Ok(stage4.output_claims.registers_read_write.rs1_ra);
            }
            if *id == bytecode_input_openings.registers_read_write.rs2_ra {
                return Ok(stage4.output_claims.registers_read_write.rs2_ra);
            }
            if *id == bytecode_input_openings.registers_val_evaluation.rd_wa {
                return Ok(stage5.output_claims.registers_val_evaluation.rd_wa);
            }
            if *id
                == bytecode_input_openings
                    .registers_val_evaluation
                    .instruction_raf_flag
            {
                return Ok(stage5
                    .output_claims
                    .instruction_read_raf
                    .instruction_raf_flag);
            }
            for (table, opening) in &bytecode_input_openings
                .registers_val_evaluation
                .lookup_table_flags
            {
                if *id == *opening {
                    return stage5
                        .output_claims
                        .instruction_read_raf
                        .lookup_table_flags
                        .get(table.index())
                        .copied()
                        .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                }
            }
            if *id == bytecode_input_openings.spartan_outer_pc {
                return Ok(stage1.outer.pc);
            }
            if *id == bytecode_input_openings.spartan_shift_pc {
                return Ok(stage3.output_claims.shift.pc);
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => Ok(bytecode_gamma),
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage1Gamma) => {
                Ok(stage1_gammas[1])
            }
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage2Gamma) => {
                Ok(stage2_gammas[1])
            }
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage3Gamma) => {
                Ok(stage3_gammas[1])
            }
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage4Gamma) => {
                Ok(stage4_gammas[1])
            }
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage5Gamma) => {
                Ok(stage5_gammas[1])
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let booleanity_address_input = PCS::Field::zero();

    if trusted_advice_claims.is_none() && claims.advice_cycle_phase.trusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if untrusted_advice_claims.is_none() && claims.advice_cycle_phase.untrusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted),
        });
    }
    if bytecode_reduction_claims.is_none() && claims.bytecode_claim_reduction.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        });
    }
    if program_image_reduction_claims.is_none() && claims.program_image_claim_reduction.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        });
    }

    let stage6a = verify_a::verify_clear(
        proof,
        transcript,
        claims,
        &bytecode_address_claims,
        &booleanity_address_claims,
        bytecode_read_raf_address_input,
        booleanity_address_input,
    )?;
    let address_batch = &stage6a.address_batch;
    let bytecode_address_point = stage6a.bytecode_address_point.clone();
    let bytecode_r_address = stage6a.bytecode_r_address.clone();
    let booleanity_address_point = stage6a.booleanity_address_point.clone();
    let booleanity_r_address = stage6a.booleanity_r_address.clone();

    let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
        formula_dimensions
            .instruction_ra_virtualization
            .num_virtual_ra_polys(),
    );
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(PCS::Field::one);
    let inc_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_inc_gamma = transcript.challenge_scalar();
    let eta = committed_program.then(|| transcript.challenge_scalar());

    let input_claims = VerifyStage6BatchInputClaims {
        bytecode_read_raf_address: stage6a.bytecode_read_raf_input,
        booleanity_address: stage6a.booleanity_input,
        bytecode_read_raf: claims.address_phase.bytecode_read_raf,
        booleanity: claims.address_phase.booleanity,
        ram_hamming_booleanity: PCS::Field::zero(),
        ram_ra_virtualization: ram_ra_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_ra_reduced => {
                    Ok(stage5.output_claims.ram_ra_claim_reduction.ram_ra)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        instruction_ra_virtualization: instruction_ra_claims.input.expression().try_evaluate(
            |id| {
                for (index, opening) in instruction_ra_input_openings.iter().enumerate() {
                    if *id == *opening {
                        return stage5
                            .output_claims
                            .instruction_read_raf
                            .instruction_ra
                            .get(index)
                            .copied()
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                    }
                }
                Err(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| match id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => Ok(instruction_ra_gamma),
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        inc_claim_reduction: inc_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_inc_read_write => Ok(stage2.output_claims.ram_read_write.inc),
                id if id == ram_inc_val_check => Ok(stage4.output_claims.ram_val_check.ram_inc),
                id if id == rd_inc_read_write => {
                    Ok(stage4.output_claims.registers_read_write.rd_inc)
                }
                id if id == rd_inc_val_evaluation => {
                    Ok(stage5.output_claims.registers_val_evaluation.rd_inc)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => {
                    Ok(inc_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction: field_inc_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == field_rd_inc_read_write => Ok(stage4
                    .output_claims
                    .field_inline
                    .field_registers_read_write
                    .field_rd_inc),
                id if id == field_rd_inc_val_evaluation => Ok(stage5
                    .output_claims
                    .field_inline
                    .field_registers_val_evaluation
                    .field_rd_inc),
                id => Err(VerifierError::MissingFieldInlineOpeningClaim { id }),
            },
            |id| match id {
                FieldInlineChallengeId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionChallenge::Gamma,
                ) => Ok(field_inc_gamma),
                _ => Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: format!("missing field-inline challenge input {id:?}"),
                }),
            },
            |id| {
                Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: format!("missing field-inline public input {id:?}"),
                })
            },
        )?,
        trusted_advice_cycle_phase: trusted_advice_claims
            .as_ref()
            .map(|claim| {
                verify_b::advice_cycle_phase_input::<PCS::Field>(
                    claim,
                    stage4,
                    JoltAdviceKind::Trusted,
                )
            })
            .transpose()?,
        untrusted_advice_cycle_phase: untrusted_advice_claims
            .as_ref()
            .map(|claim| {
                verify_b::advice_cycle_phase_input::<PCS::Field>(
                    claim,
                    stage4,
                    JoltAdviceKind::Untrusted,
                )
            })
            .transpose()?,
        bytecode_claim_reduction: bytecode_reduction_claims
            .as_ref()
            .map(|claim| {
                let stage_claims = claims.address_phase.bytecode_val_stages.as_ref().ok_or(
                    VerifierError::MissingOpeningClaim {
                        id: bytecode_reduction::bytecode_val_stage_opening(0),
                    },
                )?;
                claim.input.expression().try_evaluate(
                    |id| {
                        for (stage, stage_claim) in stage_claims.iter().enumerate() {
                            if *id == bytecode_reduction::bytecode_val_stage_opening(stage) {
                                return Ok(*stage_claim);
                            }
                        }
                        Err(VerifierError::MissingOpeningClaim { id: *id })
                    },
                    |id| match id {
                        JoltChallengeId::BytecodeClaimReduction(
                            BytecodeClaimReductionChallenge::Eta,
                        ) => eta.ok_or(VerifierError::MissingStageClaimChallenge { id: *id }),
                        _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    },
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )
            })
            .transpose()?,
        program_image_claim_reduction: program_image_reduction_claims
            .as_ref()
            .map(|claim| {
                let contribution = stage4
                    .ram_val_check_init
                    .program_image_contribution
                    .as_ref()
                    .ok_or(VerifierError::MissingOpeningClaim {
                        id: program_image::ram_val_check_contribution_opening(),
                    })?;
                claim.input.expression().try_evaluate(
                    |id| {
                        if *id == program_image::ram_val_check_contribution_opening() {
                            Ok(contribution.opening_claim)
                        } else {
                            Err(VerifierError::MissingOpeningClaim { id: *id })
                        }
                    },
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )
            })
            .transpose()?,
    };

    let mut sumcheck_claims = vec![
        SumcheckClaim::new(
            bytecode_claims.sumcheck.rounds,
            bytecode_claims.sumcheck.degree,
            input_claims.bytecode_read_raf,
        ),
        SumcheckClaim::new(
            booleanity_claims.sumcheck.rounds,
            booleanity_claims.sumcheck.degree,
            input_claims.booleanity,
        ),
        SumcheckClaim::new(
            ram_hamming_claims.sumcheck.rounds,
            ram_hamming_claims.sumcheck.degree,
            input_claims.ram_hamming_booleanity,
        ),
        SumcheckClaim::new(
            ram_ra_claims.sumcheck.rounds,
            ram_ra_claims.sumcheck.degree,
            input_claims.ram_ra_virtualization,
        ),
        SumcheckClaim::new(
            instruction_ra_claims.sumcheck.rounds,
            instruction_ra_claims.sumcheck.degree,
            input_claims.instruction_ra_virtualization,
        ),
        SumcheckClaim::new(
            inc_claims.sumcheck.rounds,
            inc_claims.sumcheck.degree,
            input_claims.inc_claim_reduction,
        ),
    ];
    #[cfg(feature = "field-inline")]
    sumcheck_claims.push(SumcheckClaim::new(
        field_inc_claims.sumcheck.rounds,
        field_inc_claims.sumcheck.degree,
        input_claims.field_registers_inc_claim_reduction,
    ));
    if let (Some(claim), Some(input_claim)) = (
        &trusted_advice_claims,
        input_claims.trusted_advice_cycle_phase,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }
    if let (Some(claim), Some(input_claim)) = (
        &untrusted_advice_claims,
        input_claims.untrusted_advice_cycle_phase,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }
    if let (Some(claim), Some(input_claim)) = (
        &bytecode_reduction_claims,
        input_claims.bytecode_claim_reduction,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }
    if let (Some(claim), Some(input_claim)) = (
        &program_image_reduction_claims,
        input_claims.program_image_claim_reduction,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }

    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage6b_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let bytecode_point = batch
        .try_instance_point(bytecode_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
    let bytecode_r_cycle = bytecode_point.iter().rev().copied().collect::<Vec<_>>();
    let bytecode_opening_point =
        [bytecode_r_address.as_slice(), bytecode_r_cycle.as_slice()].concat();
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    if claims.bytecode_read_raf.bytecode_ra.len() != bytecode_output_openings.bytecode_ra.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                bytecode_output_openings.bytecode_ra.len(),
                claims.bytecode_read_raf.bytecode_ra.len()
            ),
        });
    }

    let stage1_point = stage1.remainder.sumcheck_point.as_slice();
    let (_, stage1_cycle_binding) =
        stage1_point
            .split_first()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "Stage 1 remainder point is empty".to_string(),
            })?;
    let stage1_cycle = stage1_cycle_binding
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage2_cycle = stage2.batch.product_remainder.opening_point.clone();
    let stage3_cycle = stage3.batch.shift.opening_point.clone();
    let (stage4_register_address, stage4_cycle) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let (stage5_register_address, stage5_cycle) = stage5
        .batch
        .registers_val_evaluation
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let entry_bytecode_index = preprocessing
        .program
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    let evaluate_bytecode_ra_claim = |id: &jolt_claims::protocols::jolt::JoltOpeningId| {
        for (index, opening) in bytecode_output_openings.bytecode_ra.iter().enumerate() {
            if *id == *opening {
                return Ok(claims.bytecode_read_raf.bytecode_ra[index]);
            }
        }
        Err(VerifierError::MissingOpeningClaim { id: *id })
    };
    let bytecode_output = if committed_program {
        let bytecode_public_values = bytecode::read_raf_committed_public_values::<PCS::Field>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &bytecode_r_address,
                r_cycle: &bytecode_r_cycle,
                stage_cycle_points: [
                    &stage1_cycle,
                    &stage2_cycle,
                    &stage3_cycle,
                    stage4_cycle,
                    stage5_cycle,
                ],
                entry_bytecode_index,
            },
        );
        let stage_claims = claims.address_phase.bytecode_val_stages.as_ref().ok_or(
            VerifierError::MissingOpeningClaim {
                id: bytecode_reduction::bytecode_val_stage_opening(0),
            },
        )?;
        bytecode_claims.output.expression().try_evaluate(
            |id| {
                for (stage, stage_claim) in stage_claims.iter().enumerate() {
                    if *id == bytecode_reduction::bytecode_val_stage_opening(stage) {
                        return Ok(*stage_claim);
                    }
                }
                evaluate_bytecode_ra_claim(id)
            },
            |id| match id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                    Ok(bytecode_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| match id {
                JoltPublicId::BytecodeReadRaf(public_id) => bytecode_public_values
                    .value(*public_id)
                    .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
                _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
            },
        )?
    } else {
        let full_program = preprocessing.program.as_full().ok_or_else(|| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "full bytecode table is unavailable".to_string(),
            }
        })?;
        let bytecode_public_values =
            bytecode::read_raf_public_values::<PCS::Field>(BytecodeReadRafEvaluationInputs {
                bytecode: &full_program.bytecode.bytecode,
                r_address: &bytecode_r_address,
                r_cycle: &bytecode_r_cycle,
                stage_cycle_points: [
                    &stage1_cycle,
                    &stage2_cycle,
                    &stage3_cycle,
                    stage4_cycle,
                    stage5_cycle,
                ],
                register_read_write_point: stage4_register_address,
                register_val_evaluation_point: stage5_register_address,
                entry_bytecode_index,
                stage1_gammas: &stage1_gammas,
                stage2_gammas: &stage2_gammas,
                stage3_gammas: &stage3_gammas,
                stage4_gammas: &stage4_gammas,
                stage5_gammas: &stage5_gammas,
            })
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: error.to_string(),
            })?;
        bytecode_claims.output.expression().try_evaluate(
            evaluate_bytecode_ra_claim,
            |id| match id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                    Ok(bytecode_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| match id {
                JoltPublicId::BytecodeReadRaf(public_id) => bytecode_public_values
                    .value(*public_id)
                    .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
                _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
            },
        )?
    };
    let bytecode_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(&bytecode_r_address)
        .into_iter()
        .map(|r_address_chunk| [r_address_chunk.as_slice(), bytecode_r_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

    let booleanity_point = batch
        .try_instance_point(booleanity_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;
    let booleanity_r_cycle = booleanity_point.iter().rev().copied().collect::<Vec<_>>();
    let booleanity_opening_point = [
        booleanity_r_address.as_slice(),
        booleanity_r_cycle.as_slice(),
    ]
    .concat();
    if claims.booleanity.instruction_ra.len() != formula_dimensions.ra_layout.instruction()
        || claims.booleanity.bytecode_ra.len() != formula_dimensions.ra_layout.bytecode()
        || claims.booleanity.ram_ra.len() != formula_dimensions.ra_layout.ram()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: format!(
                "booleanity RA claim count mismatch: expected ({}, {}, {}), got ({}, {}, {})",
                formula_dimensions.ra_layout.instruction(),
                formula_dimensions.ra_layout.bytecode(),
                formula_dimensions.ra_layout.ram(),
                claims.booleanity.instruction_ra.len(),
                claims.booleanity.bytecode_ra.len(),
                claims.booleanity.ram_ra.len()
            ),
        });
    }
    let booleanity_reference_eq_point = booleanity_reference_address
        .iter()
        .rev()
        .chain(booleanity_reference_cycle.iter().rev())
        .copied()
        .collect::<Vec<_>>();
    let booleanity_full_point = [booleanity_address_point.as_slice(), booleanity_point].concat();
    let eq_address_cycle = try_eq_mle(&booleanity_full_point, &booleanity_reference_eq_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;
    let booleanity_output_openings =
        booleanity::booleanity_output_openings(formula_dimensions.ra_layout);
    let booleanity_output = booleanity_claims.output.expression().try_evaluate(
        |id| {
            for (index, opening) in booleanity_output_openings.iter().enumerate() {
                if *id == *opening {
                    if index < formula_dimensions.ra_layout.instruction() {
                        return Ok(claims.booleanity.instruction_ra[index]);
                    }
                    let bytecode_index = index - formula_dimensions.ra_layout.instruction();
                    if bytecode_index < formula_dimensions.ra_layout.bytecode() {
                        return Ok(claims.booleanity.bytecode_ra[bytecode_index]);
                    }
                    let ram_index = bytecode_index - formula_dimensions.ra_layout.bytecode();
                    return Ok(claims.booleanity.ram_ra[ram_index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => Ok(booleanity_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::Booleanity(BooleanityPublic::EqAddressCycle) => Ok(eq_address_cycle),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;
    let ram_hamming_point = batch
        .try_instance_point(ram_hamming_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamHammingBooleanity,
            reason: error.to_string(),
        })?;
    let ram_hamming_opening_point = trace_dimensions
        .cycle_opening_point(ram_hamming_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamHammingBooleanity,
            reason: error.to_string(),
        })?;
    let eq_spartan_outer_cycle =
        try_eq_mle(ram_hamming_point, stage1_cycle_binding).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            }
        })?;
    let [ram_hamming_weight] = ram::hamming_booleanity_output_openings();
    let ram_hamming_output = ram_hamming_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_hamming_weight => Ok(claims.ram_hamming_booleanity.ram_hamming_weight),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match *id {
            JoltChallengeId::RamHammingBooleanity(RamHammingBooleanityChallenge::EqCycle) => {
                Ok(eq_spartan_outer_cycle)
            }
            id => Err(VerifierError::MissingStageClaimChallenge { id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    let ram_ra_point = batch
        .try_instance_point(ram_ra_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        })?;
    let ram_ra_cycle = trace_dimensions
        .cycle_opening_point(ram_ra_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        })?;
    let ram_ra_output_openings =
        ram::ra_virtualization_output_openings(formula_dimensions.ram_ra_virtualization);
    if claims.ram_ra_virtualization.ram_ra.len() != ram_ra_output_openings.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: format!(
                "RAM RA virtualization claim count mismatch: expected {}, got {}",
                ram_ra_output_openings.len(),
                claims.ram_ra_virtualization.ram_ra.len()
            ),
        });
    }
    let ram_reduced_opening_point = &stage5.batch.ram_ra_claim_reduction.opening_point;
    if ram_reduced_opening_point.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: format!(
                "RAM RA reduction opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_reduced_opening_point.len()
            ),
        });
    }
    let (ram_reduced_address, ram_reduced_cycle) = ram_reduced_opening_point.split_at(log_k);
    let eq_ram_ra_cycle = try_eq_mle(ram_reduced_cycle, &ram_ra_cycle).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        }
    })?;
    let ram_ra_output = ram_ra_claims.output.expression().try_evaluate(
        |id| {
            for (index, opening) in ram_ra_output_openings.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims.ram_ra_virtualization.ram_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match *id {
            JoltChallengeId::RamRaVirtualization(RamRaVirtualizationChallenge::EqCycle) => {
                Ok(eq_ram_ra_cycle)
            }
            id => Err(VerifierError::MissingStageClaimChallenge { id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let ram_ra_opening_point = [ram_reduced_address, ram_ra_cycle.as_slice()].concat();
    let ram_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(ram_reduced_address)
        .into_iter()
        .map(|r_address_chunk| [r_address_chunk.as_slice(), ram_ra_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

    let instruction_ra_point = batch
        .try_instance_point(instruction_ra_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_ra_cycle = trace_dimensions
        .cycle_opening_point(instruction_ra_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_ra_output_openings = instruction::ra_virtualization_output_openings(
        formula_dimensions.instruction_ra_virtualization,
    );
    let flat_instruction_ra_output_openings = instruction_ra_output_openings.all();
    if claims
        .instruction_ra_virtualization
        .committed_instruction_ra
        .len()
        != flat_instruction_ra_output_openings.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: format!(
                "instruction RA virtualization claim count mismatch: expected {}, got {}",
                flat_instruction_ra_output_openings.len(),
                claims
                    .instruction_ra_virtualization
                    .committed_instruction_ra
                    .len()
            ),
        });
    }
    let eq_instruction_ra_cycle = try_eq_mle(
        &stage5.batch.instruction_read_raf.r_cycle,
        &instruction_ra_cycle,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionRaVirtualization,
        reason: error.to_string(),
    })?;
    let instruction_ra_output = instruction_ra_claims.output.expression().try_evaluate(
        |id| {
            for (index, opening) in flat_instruction_ra_output_openings.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims
                        .instruction_ra_virtualization
                        .committed_instruction_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::Gamma,
            ) => Ok(instruction_ra_gamma),
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::EqCycle,
            ) => Ok(eq_instruction_ra_cycle),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let instruction_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(&stage5.batch.instruction_read_raf.r_address)
        .into_iter()
        .map(|r_address_chunk| {
            [r_address_chunk.as_slice(), instruction_ra_cycle.as_slice()].concat()
        })
        .collect::<Vec<_>>();
    let instruction_ra_opening_point = [
        stage5.batch.instruction_read_raf.r_address.as_slice(),
        instruction_ra_cycle.as_slice(),
    ]
    .concat();

    let inc_point = batch
        .try_instance_point(inc_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let inc_opening_point = trace_dimensions
        .cycle_opening_point(inc_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let (_, ram_read_write_cycle) = stage2.batch.ram_read_write.opening_point.split_at(log_k);
    let (_, ram_val_check_cycle) = stage4.batch.ram_val_check.opening_point.split_at(log_k);
    let (_, registers_read_write_cycle) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let (_, registers_val_evaluation_cycle) = stage5
        .batch
        .registers_val_evaluation
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let eq_ram_read_write =
        try_eq_mle(&inc_opening_point, ram_read_write_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let eq_ram_val_check =
        try_eq_mle(&inc_opening_point, ram_val_check_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let eq_registers_read_write = try_eq_mle(&inc_opening_point, registers_read_write_cycle)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let eq_registers_val_evaluation =
        try_eq_mle(&inc_opening_point, registers_val_evaluation_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let [ram_inc, rd_inc] = increments::claim_reduction_output_openings();
    let inc_output = inc_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_inc => Ok(claims.inc_claim_reduction.ram_inc),
            id if id == rd_inc => Ok(claims.inc_claim_reduction.rd_inc),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => Ok(inc_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamReadWrite) => {
                Ok(eq_ram_read_write)
            }
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamValCheck) => {
                Ok(eq_ram_val_check)
            }
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRegistersReadWrite) => {
                Ok(eq_registers_read_write)
            }
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRegistersValEvaluation) => {
                Ok(eq_registers_val_evaluation)
            }
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;
    #[cfg(feature = "field-inline")]
    let (field_inc_point, field_inc_opening_point, field_inc_output) = {
        let field_inc_point = batch
            .try_instance_point(field_inc_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let field_inc_opening_point = trace_dimensions
            .cycle_opening_point(field_inc_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let field_log_k = proof.protocol.field_inline.field_register_log_k;
        if stage4.batch.field_registers_read_write.opening_point.len() < field_log_k {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: format!(
                    "field-register read-write opening point is shorter than the field-register address: expected at least {field_log_k}, got {}",
                    stage4.batch.field_registers_read_write.opening_point.len()
                ),
            });
        }
        if stage5
            .batch
            .field_registers_val_evaluation
            .opening_point
            .len()
            < field_log_k
        {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: format!(
                    "field-register val-evaluation opening point is shorter than the field-register address: expected at least {field_log_k}, got {}",
                    stage5.batch.field_registers_val_evaluation.opening_point.len()
                ),
            });
        }
        let (_, field_read_write_cycle) = stage4
            .batch
            .field_registers_read_write
            .opening_point
            .split_at(field_log_k);
        let (_, field_val_evaluation_cycle) = stage5
            .batch
            .field_registers_val_evaluation
            .opening_point
            .split_at(field_log_k);
        let eq_field_read_write = try_eq_mle(&field_inc_opening_point, field_read_write_cycle)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let eq_field_val_evaluation =
            try_eq_mle(&field_inc_opening_point, field_val_evaluation_cycle).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                }
            })?;
        let [field_rd_inc] = field_increments::claim_reduction_output_openings();
        let output = field_inc_claims.output.expression().try_evaluate(
            |id| match *id {
                id if id == field_rd_inc => Ok(claims
                    .field_inline
                    .field_registers_inc_claim_reduction
                    .field_rd_inc),
                id => Err(VerifierError::MissingFieldInlineOpeningClaim { id }),
            },
            |id| match id {
                FieldInlineChallengeId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionChallenge::Gamma,
                ) => Ok(field_inc_gamma),
                _ => Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: format!("missing field-inline challenge input {id:?}"),
                }),
            },
            |id| match id {
                FieldInlinePublicId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionPublic::EqReadWrite,
                ) => Ok(eq_field_read_write),
                FieldInlinePublicId::FieldRegistersIncClaimReduction(
                    FieldRegistersIncClaimReductionPublic::EqValEvaluation,
                ) => Ok(eq_field_val_evaluation),
            },
        )?;
        (field_inc_point.to_vec(), field_inc_opening_point, output)
    };

    let trusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        trusted_advice_layout,
        trusted_advice_claims.as_ref(),
        claims.advice_cycle_phase.trusted.as_ref(),
    ) {
        Some(verify_b::verify_advice_cycle_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Trusted,
            opening_claim,
            stage4,
        )?)
    } else {
        None
    };
    let untrusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        untrusted_advice_layout,
        untrusted_advice_claims.as_ref(),
        claims.advice_cycle_phase.untrusted.as_ref(),
    ) {
        Some(verify_b::verify_advice_cycle_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Untrusted,
            opening_claim,
            stage4,
        )?)
    } else {
        None
    };

    if let (Some(layout), Some(_)) = (trusted_advice_layout, trusted_advice_claims.as_ref()) {
        if trusted_advice.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_output_openings(
                    JoltAdviceKind::Trusted,
                    layout.dimensions(),
                )[0],
            });
        }
    }
    if let (Some(layout), Some(_)) = (untrusted_advice_layout, untrusted_advice_claims.as_ref()) {
        if untrusted_advice.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_output_openings(
                    JoltAdviceKind::Untrusted,
                    layout.dimensions(),
                )[0],
            });
        }
    }
    let bytecode_cycle_phase =
        if let (Some(layout), Some(claim)) = (
            bytecode_reduction_layout,
            bytecode_reduction_claims.as_ref(),
        ) {
            let output_claims = claims.bytecode_claim_reduction.as_ref().ok_or(
                VerifierError::MissingOpeningClaim {
                    id: bytecode_reduction::cycle_phase_output_openings(
                        layout.dimensions(),
                        layout.chunk_count(),
                    )[0],
                },
            )?;
            let eta = eta.ok_or(VerifierError::MissingStageClaimChallenge {
                id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
            })?;
            let weights = verify_b::bytecode_reduction_weights(
                layout,
                verify_b::BytecodeReductionWeightInputs {
                    eta,
                    stage1_gammas: &stage1_gammas,
                    stage2_gammas: &stage2_gammas,
                    stage3_gammas: &stage3_gammas,
                    stage4_gammas: &stage4_gammas,
                    stage5_gammas: &stage5_gammas,
                    register_read_write_point: stage4_register_address,
                    register_val_evaluation_point: stage5_register_address,
                    bytecode_r_address: &bytecode_r_address,
                },
            )?;
            let input_claim = input_claims.bytecode_claim_reduction.ok_or(
                VerifierError::MissingOpeningClaim {
                    id: bytecode_reduction::bytecode_val_stage_opening(0),
                },
            )?;
            Some(verify_b::verify_bytecode_cycle_phase(
                &batch,
                claim,
                layout,
                output_claims,
                weights,
                input_claim,
            )?)
        } else {
            None
        };
    let program_image_cycle_phase = if let (Some(layout), Some(claim)) = (
        program_image_reduction_layout,
        program_image_reduction_claims.as_ref(),
    ) {
        let output_claim = claims.program_image_claim_reduction.as_ref().ok_or(
            VerifierError::MissingOpeningClaim {
                id: program_image::cycle_phase_output_openings(layout.dimensions())[0],
            },
        )?;
        let r_addr_rw = &stage4.batch.ram_val_check.opening_point[..log_k];
        let input_claim = input_claims.program_image_claim_reduction.ok_or(
            VerifierError::MissingOpeningClaim {
                id: program_image::ram_val_check_contribution_opening(),
            },
        )?;
        Some(verify_b::verify_program_image_cycle_phase(
            &batch,
            claim,
            layout,
            output_claim,
            r_addr_rw,
            input_claim,
        )?)
    } else {
        None
    };

    let expected_outputs = VerifyStage6BatchExpectedOutputClaims {
        bytecode_read_raf_address: claims.address_phase.bytecode_read_raf,
        booleanity_address: claims.address_phase.booleanity,
        bytecode_read_raf: bytecode_output,
        booleanity: booleanity_output,
        ram_hamming_booleanity: ram_hamming_output,
        ram_ra_virtualization: ram_ra_output,
        instruction_ra_virtualization: instruction_ra_output,
        inc_claim_reduction: inc_output,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction: field_inc_output,
        trusted_advice_cycle_phase: trusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        untrusted_advice_cycle_phase: untrusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        bytecode_claim_reduction: bytecode_cycle_phase
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        program_image_claim_reduction: program_image_cycle_phase
            .as_ref()
            .map(|verified| verified.expected_output_claim),
    };
    let mut expected_outputs_in_order = vec![
        expected_outputs.bytecode_read_raf,
        expected_outputs.booleanity,
        expected_outputs.ram_hamming_booleanity,
        expected_outputs.ram_ra_virtualization,
        expected_outputs.instruction_ra_virtualization,
        expected_outputs.inc_claim_reduction,
    ];
    #[cfg(feature = "field-inline")]
    expected_outputs_in_order.push(expected_outputs.field_registers_inc_claim_reduction);
    if let Some(output_claim) = expected_outputs.trusted_advice_cycle_phase {
        expected_outputs_in_order.push(output_claim);
    }
    if let Some(output_claim) = expected_outputs.untrusted_advice_cycle_phase {
        expected_outputs_in_order.push(output_claim);
    }
    if let Some(output_claim) = expected_outputs.bytecode_claim_reduction {
        expected_outputs_in_order.push(output_claim);
    }
    if let Some(output_claim) = expected_outputs.program_image_claim_reduction {
        expected_outputs_in_order.push(output_claim);
    }
    if batch.batching_coefficients.len() != expected_outputs_in_order.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "Stage 6 batch verifier returned {} coefficients for {} instances",
                batch.batching_coefficients.len(),
                expected_outputs_in_order.len()
            ),
        });
    }
    let expected_final_claim = batch
        .batching_coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum();
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
        });
    }

    verify_b::append_opening_claims(
        transcript,
        claims,
        &bytecode_ra_opening_points,
        &booleanity_opening_point,
    );

    Ok(Stage6Output::Clear(Stage6ClearOutput {
        public: public(
            instruction_ra_gamma_powers,
            inc_gamma,
            #[cfg(feature = "field-inline")]
            field_inc_gamma,
            eta,
            address_batch.reduction.point.as_slice().to_vec(),
            address_batch.batching_coefficients.clone(),
            batch.reduction.point.as_slice().to_vec(),
            batch.batching_coefficients.clone(),
        ),
        output_claims: claims.clone(),
        batch: VerifiedStage6Batch {
            address_phase_batching_coefficients: address_batch.batching_coefficients.clone(),
            address_phase_sumcheck_point: address_batch.reduction.point.clone(),
            address_phase_sumcheck_final_claim: address_batch.reduction.value,
            address_phase_expected_final_claim: stage6a.expected_final_claim,
            bytecode_read_raf_address: VerifiedStage6AddressPhaseSumcheck {
                input_claim: input_claims.bytecode_read_raf_address,
                sumcheck_point: bytecode_address_point.clone(),
                opening_point: bytecode_r_address.clone(),
                expected_output_claim: expected_outputs.bytecode_read_raf_address,
            },
            booleanity_address: VerifiedStage6AddressPhaseSumcheck {
                input_claim: input_claims.booleanity_address,
                sumcheck_point: booleanity_address_point.clone(),
                opening_point: booleanity_r_address.clone(),
                expected_output_claim: expected_outputs.booleanity_address,
            },
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: batch.reduction.point.clone(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            bytecode_read_raf: VerifiedBytecodeReadRafSumcheck {
                input_claim: input_claims.bytecode_read_raf,
                sumcheck_point: bytecode_point.to_vec(),
                r_address: bytecode_r_address,
                r_cycle: bytecode_r_cycle,
                full_opening_point: bytecode_opening_point,
                bytecode_ra_opening_points,
                expected_output_claim: expected_outputs.bytecode_read_raf,
            },
            booleanity: VerifiedBooleanitySumcheck {
                input_claim: input_claims.booleanity,
                sumcheck_point: booleanity_point.to_vec(),
                r_address: booleanity_r_address,
                r_cycle: booleanity_r_cycle,
                opening_point: booleanity_opening_point,
                reference_address: booleanity_reference_address,
                reference_cycle: booleanity_reference_cycle,
                expected_output_claim: expected_outputs.booleanity,
            },
            ram_hamming_booleanity: VerifiedStage6Sumcheck {
                input_claim: input_claims.ram_hamming_booleanity,
                sumcheck_point: ram_hamming_point.to_vec(),
                opening_point: ram_hamming_opening_point,
                expected_output_claim: expected_outputs.ram_hamming_booleanity,
            },
            ram_ra_virtualization: VerifiedRamRaVirtualizationSumcheck {
                input_claim: input_claims.ram_ra_virtualization,
                sumcheck_point: ram_ra_point.to_vec(),
                opening_point: ram_ra_opening_point,
                ram_ra_opening_points,
                expected_output_claim: expected_outputs.ram_ra_virtualization,
            },
            instruction_ra_virtualization: VerifiedInstructionRaVirtualizationSumcheck {
                input_claim: input_claims.instruction_ra_virtualization,
                sumcheck_point: instruction_ra_point.to_vec(),
                opening_point: instruction_ra_opening_point,
                instruction_ra_opening_points,
                expected_output_claim: expected_outputs.instruction_ra_virtualization,
            },
            inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: input_claims.inc_claim_reduction,
                sumcheck_point: inc_point.to_vec(),
                opening_point: inc_opening_point,
                expected_output_claim: expected_outputs.inc_claim_reduction,
            },
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: input_claims.field_registers_inc_claim_reduction,
                sumcheck_point: field_inc_point,
                opening_point: field_inc_opening_point,
                expected_output_claim: expected_outputs.field_registers_inc_claim_reduction,
            },
            trusted_advice_cycle_phase: trusted_advice,
            untrusted_advice_cycle_phase: untrusted_advice,
            bytecode_cycle_phase,
            program_image_cycle_phase,
        },
    }))
}

fn validate_compressed_stage_claim<F: Field>(
    claim: &JoltRelationClaims<F>,
) -> Result<(), VerifierError> {
    if claim.sumcheck.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage: claim.id,
            degree: claim.sumcheck.degree,
        });
    }
    if !matches!(claim.sumcheck.domain, JoltSumcheckDomain::BooleanHypercube) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain { stage: claim.id });
    }
    Ok(())
}

#[cfg(feature = "field-inline")]
fn validate_compressed_field_inline_stage_claim<F: Field>(
    claim: &FieldInlineRelationClaims<F>,
) -> Result<(), VerifierError> {
    if claim.sumcheck.degree == 0 {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: format!(
                "field-inline stage {:?} sumcheck degree is invalid: {}",
                claim.id, claim.sumcheck.degree
            ),
        });
    }
    Ok(())
}

// ============================================================================
// Stage 6 prover/verifier shared helpers.
//
// The functions and types below are the public, prover-facing Stage 6 API
// re-exported by `mod.rs`. They are single-sourced extractions of the value
// derivations performed inline by the canonical `verify()` above: each helper
// computes exactly the same quantity (same formula, same Fiat-Shamir gamma
// counts, same instance/absorption order) so the prover's Stage 6 batch cannot
// drift from the verifier.
//
// NOTE on claim reductions: new 09 added bytecode and program-image claim
// reductions to the verifier's Stage 6 batch (handled inline by `verify()`
// using `VerifyStage6BatchInputClaims`/`VerifyStage6BatchExpectedOutputClaims`).
// The public structs below do NOT carry those reduction fields; the prover-side
// Stage 6 batch does not (yet) materialize them. The `stage6_input_claim_values`
// / `stage6_expected_output_claim_values` ordering therefore covers only the
// non-reduction instances that the public batch shares with `verify()`; the
// reductions are appended last by `verify()` itself, after every shared
// instance, preserving the shared prefix order.
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BatchInputClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    pub ram_hamming_booleanity: F,
    pub ram_ra_virtualization: F,
    pub instruction_ra_virtualization: F,
    pub inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: F,
    pub trusted_advice_cycle_phase: Option<F>,
    pub untrusted_advice_cycle_phase: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BooleanityReference<F: Field> {
    pub address: Vec<F>,
    pub cycle: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptChallenges<F: Field> {
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference: Stage6BooleanityReference<F>,
    pub booleanity_gamma: F,
    pub instruction_ra_gamma_powers: Vec<F>,
    pub instruction_ra_gamma: F,
    pub inc_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_inc_gamma: F,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6InputClaimChallengeValues<'a, F: Field> {
    pub bytecode_gamma_powers: &'a [F],
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
    pub instruction_ra_gamma: F,
    pub inc_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_inc_gamma: F,
}

impl<F: Field> Stage6TranscriptChallenges<F> {
    pub fn input_claim_challenge_values(&self) -> Stage6InputClaimChallengeValues<'_, F> {
        Stage6InputClaimChallengeValues {
            bytecode_gamma_powers: &self.bytecode_gamma_powers,
            stage1_gammas: &self.stage1_gammas,
            stage2_gammas: &self.stage2_gammas,
            stage3_gammas: &self.stage3_gammas,
            stage4_gammas: &self.stage4_gammas,
            stage5_gammas: &self.stage5_gammas,
            instruction_ra_gamma: self.instruction_ra_gamma,
            inc_gamma: self.inc_gamma,
            #[cfg(feature = "field-inline")]
            field_inc_gamma: self.field_inc_gamma,
        }
    }
}

pub fn stage6_stage1_cycle_binding<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> Result<&[F], VerifierError> {
    let (_, cycle) = stage1
        .remainder
        .sumcheck_point
        .as_slice()
        .split_first()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "Stage 1 remainder point is empty".to_string(),
        })?;
    Ok(cycle)
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeRegisterPoints<'a, F: Field> {
    pub register_read_write_address: &'a [F],
    pub register_read_write_cycle: &'a [F],
    pub register_val_evaluation_address: &'a [F],
    pub register_val_evaluation_cycle: &'a [F],
}

pub fn stage6_bytecode_register_points<'a, F: Field>(
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
) -> Result<Stage6BytecodeRegisterPoints<'a, F>, VerifierError> {
    let (register_read_write_address, register_read_write_cycle) = stage6_checked_split(
        "Stage 6 stage4 register read-write opening",
        &stage4.batch.registers_read_write.opening_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let (register_val_evaluation_address, register_val_evaluation_cycle) = stage6_checked_split(
        "Stage 6 stage5 register value-evaluation opening",
        &stage5.batch.registers_val_evaluation.opening_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    Ok(Stage6BytecodeRegisterPoints {
        register_read_write_address,
        register_read_write_cycle,
        register_val_evaluation_address,
        register_val_evaluation_cycle,
    })
}

pub fn stage6_bytecode_cycle_points<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<[Vec<F>; 5], VerifierError> {
    let stage1_cycle = stage6_stage1_cycle_binding(stage1)?
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage2_cycle = stage2.batch.product_remainder.opening_point.clone();
    let stage3_cycle = stage3.batch.shift.opening_point.clone();
    let register_points = stage6_bytecode_register_points(stage4, stage5)?;
    Ok([
        stage1_cycle,
        stage2_cycle,
        stage3_cycle,
        register_points.register_read_write_cycle.to_vec(),
        register_points.register_val_evaluation_cycle.to_vec(),
    ])
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeRaPoint<'a, F: Field> {
    pub r_address: &'a [F],
    pub r_cycle: &'a [F],
}

impl<F: Field> Stage6BytecodeRaPoint<'_, F> {
    pub fn committed_address_chunks(self, committed_chunk_bits: usize) -> Vec<Vec<F>> {
        committed_address_chunks(self.r_address, committed_chunk_bits)
    }

    pub fn committed_opening_points(self, committed_chunk_bits: usize) -> Vec<Vec<F>> {
        self.committed_address_chunks(committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), self.r_cycle].concat())
            .collect()
    }
}

pub fn stage6_bytecode_ra_point<'a, F: Field>(
    r_address: &'a [F],
    r_cycle: &'a [F],
) -> Stage6BytecodeRaPoint<'a, F> {
    Stage6BytecodeRaPoint { r_address, r_cycle }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BytecodeReadRafPoint<F: Field> {
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
}

impl<F: Field> Stage6BytecodeReadRafPoint<F> {
    pub fn ra_point(&self) -> Stage6BytecodeRaPoint<'_, F> {
        stage6_bytecode_ra_point(&self.r_address, &self.r_cycle)
    }

    pub fn committed_opening_points(&self, committed_chunk_bits: usize) -> Vec<Vec<F>> {
        self.ra_point()
            .committed_opening_points(committed_chunk_bits)
    }
}

pub fn stage6_bytecode_read_raf_point<F: Field>(
    dimensions: BytecodeReadRafDimensions,
    sumcheck_point: &[F],
) -> Result<Stage6BytecodeReadRafPoint<F>, VerifierError> {
    let opening = dimensions.opening_point(sumcheck_point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        }
    })?;
    Ok(Stage6BytecodeReadRafPoint {
        r_address: opening.r_address,
        r_cycle: opening.r_cycle,
        opening_point: opening.opening_point,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6RamReducedOpeningPoint<'a, F: Field> {
    pub full: &'a [F],
    pub address: &'a [F],
    pub cycle: &'a [F],
}

impl<F: Field> Stage6RamReducedOpeningPoint<'_, F> {
    pub fn committed_address_chunks(self, committed_chunk_bits: usize) -> Vec<Vec<F>> {
        committed_address_chunks(self.address, committed_chunk_bits)
    }

    pub fn opening_point(self, cycle: &[F]) -> Vec<F> {
        [self.address, cycle].concat()
    }

    pub fn committed_opening_points(self, cycle: &[F], committed_chunk_bits: usize) -> Vec<Vec<F>> {
        self.committed_address_chunks(committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), cycle].concat())
            .collect()
    }
}

pub fn stage6_ram_reduced_opening_point<F: Field>(
    point: &[F],
    log_k: usize,
    log_t: usize,
) -> Result<Stage6RamReducedOpeningPoint<'_, F>, VerifierError> {
    let (address, cycle) = stage6_checked_exact_split(
        "Stage 6 RAM RA reduction opening point",
        point,
        log_k,
        log_k + log_t,
        JoltRelationId::RamRaVirtualization,
    )?;
    Ok(Stage6RamReducedOpeningPoint {
        full: point,
        address,
        cycle,
    })
}

pub fn stage6_stage5_ram_reduced_opening_point<F: Field>(
    stage5: &Stage5ClearOutput<F>,
    log_k: usize,
    log_t: usize,
) -> Result<Stage6RamReducedOpeningPoint<'_, F>, VerifierError> {
    stage6_ram_reduced_opening_point(
        &stage5.batch.ram_ra_claim_reduction.opening_point,
        log_k,
        log_t,
    )
}

pub fn stage6_zk_stage5_ram_reduced_opening_point<F: Field, C>(
    stage5: &Stage5ZkOutput<F, C>,
    log_k: usize,
    log_t: usize,
) -> Result<Stage6RamReducedOpeningPoint<'_, F>, VerifierError> {
    stage6_ram_reduced_opening_point(&stage5.ram_ra_claim_reduction.opening_point, log_k, log_t)
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6InstructionReadRafPoint<'a, F: Field> {
    pub address: &'a [F],
    pub cycle: &'a [F],
}

impl<F: Field> Stage6InstructionReadRafPoint<'_, F> {
    pub fn committed_address_chunks(self, committed_chunk_bits: usize) -> Vec<Vec<F>> {
        committed_address_chunks(self.address, committed_chunk_bits)
    }

    pub fn opening_point(self, cycle: &[F]) -> Vec<F> {
        [self.address, cycle].concat()
    }

    pub fn committed_opening_points(self, cycle: &[F], committed_chunk_bits: usize) -> Vec<Vec<F>> {
        self.committed_address_chunks(committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), cycle].concat())
            .collect()
    }
}

pub fn stage6_instruction_read_raf_point<F: Field>(
    stage5: &Stage5ClearOutput<F>,
) -> Stage6InstructionReadRafPoint<'_, F> {
    Stage6InstructionReadRafPoint {
        address: &stage5.batch.instruction_read_raf.r_address,
        cycle: &stage5.batch.instruction_read_raf.r_cycle,
    }
}

pub fn stage6_zk_instruction_read_raf_point<F: Field, C>(
    stage5: &Stage5ZkOutput<F, C>,
) -> Stage6InstructionReadRafPoint<'_, F> {
    Stage6InstructionReadRafPoint {
        address: &stage5.instruction_read_raf.r_address,
        cycle: &stage5.instruction_read_raf.r_cycle,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6AdviceCyclePhaseReference<'a, F: Field> {
    pub opening_claim: F,
    pub opening_point: &'a [F],
}

pub fn stage6_advice_cycle_phase_reference<F: Field>(
    stage4: &Stage4ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<Stage6AdviceCyclePhaseReference<'_, F>, VerifierError> {
    let contribution = stage4
        .ram_val_check_init
        .advice_contributions
        .iter()
        .find(|contribution| contribution.kind == kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::ram_val_check_advice_opening(kind),
        })?;
    Ok(Stage6AdviceCyclePhaseReference {
        opening_claim: contribution.opening_claim,
        opening_point: &contribution.opening_point,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6IncClaimReductionCyclePoints<'a, F: Field> {
    pub ram_read_write_cycle: &'a [F],
    pub ram_val_check_cycle: &'a [F],
    pub registers_read_write_cycle: &'a [F],
    pub registers_val_evaluation_cycle: &'a [F],
}

impl<F: Field> Stage6IncClaimReductionCyclePoints<'_, F> {
    pub fn reversed_cycles(&self) -> [Vec<F>; 4] {
        [
            self.ram_read_write_cycle.iter().rev().copied().collect(),
            self.ram_val_check_cycle.iter().rev().copied().collect(),
            self.registers_read_write_cycle
                .iter()
                .rev()
                .copied()
                .collect(),
            self.registers_val_evaluation_cycle
                .iter()
                .rev()
                .copied()
                .collect(),
        ]
    }
}

pub fn stage6_inc_claim_reduction_cycle_points<'a, F: Field>(
    stage2: &'a Stage2ClearOutput<F>,
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
    log_k: usize,
) -> Result<Stage6IncClaimReductionCyclePoints<'a, F>, VerifierError> {
    let (_, ram_read_write_cycle) = stage6_checked_split(
        "Stage 6 RAM read-write opening",
        &stage2.batch.ram_read_write.opening_point,
        log_k,
        JoltRelationId::IncClaimReduction,
    )?;
    let (_, ram_val_check_cycle) = stage6_checked_split(
        "Stage 6 RAM value-check opening",
        &stage4.batch.ram_val_check.opening_point,
        log_k,
        JoltRelationId::IncClaimReduction,
    )?;
    let (_, registers_read_write_cycle) = stage6_checked_split(
        "Stage 6 register read-write opening",
        &stage4.batch.registers_read_write.opening_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::IncClaimReduction,
    )?;
    let (_, registers_val_evaluation_cycle) = stage6_checked_split(
        "Stage 6 register value-evaluation opening",
        &stage5.batch.registers_val_evaluation.opening_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::IncClaimReduction,
    )?;
    Ok(Stage6IncClaimReductionCyclePoints {
        ram_read_write_cycle,
        ram_val_check_cycle,
        registers_read_write_cycle,
        registers_val_evaluation_cycle,
    })
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug)]
pub struct FieldInlineStage6IncClaimReductionCyclePoints<'a, F: Field> {
    pub read_write_cycle: &'a [F],
    pub val_evaluation_cycle: &'a [F],
}

#[cfg(feature = "field-inline")]
impl<F: Field> FieldInlineStage6IncClaimReductionCyclePoints<'_, F> {
    pub fn reversed_cycles(&self) -> [Vec<F>; 2] {
        [
            self.read_write_cycle.iter().rev().copied().collect(),
            self.val_evaluation_cycle.iter().rev().copied().collect(),
        ]
    }
}

#[cfg(feature = "field-inline")]
pub fn stage6_field_registers_inc_claim_reduction_cycle_points<'a, F: Field>(
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
    field_log_k: usize,
    log_t: usize,
) -> Result<FieldInlineStage6IncClaimReductionCyclePoints<'a, F>, VerifierError> {
    let expected_len = field_log_k + log_t;
    let (_, read_write_cycle) = stage6_checked_exact_split(
        "Stage 6 field-register read-write opening",
        &stage4.batch.field_registers_read_write.opening_point,
        field_log_k,
        expected_len,
        JoltRelationId::IncClaimReduction,
    )?;
    let (_, val_evaluation_cycle) = stage6_checked_exact_split(
        "Stage 6 field-register value-evaluation opening",
        &stage5.batch.field_registers_val_evaluation.opening_point,
        field_log_k,
        expected_len,
        JoltRelationId::IncClaimReduction,
    )?;
    Ok(FieldInlineStage6IncClaimReductionCyclePoints {
        read_write_cycle,
        val_evaluation_cycle,
    })
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug)]
pub struct FieldInlineStage6BytecodeRegisterPoints<'a, F: Field> {
    pub read_write_address: &'a [F],
    pub read_write_cycle: &'a [F],
    pub val_evaluation_address: &'a [F],
    pub val_evaluation_cycle: &'a [F],
}

#[cfg(feature = "field-inline")]
pub fn stage6_field_inline_bytecode_register_points<'a, F: Field>(
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
    field_log_k: usize,
    log_t: usize,
) -> Result<FieldInlineStage6BytecodeRegisterPoints<'a, F>, VerifierError> {
    let expected_len = field_log_k + log_t;
    let (read_write_address, read_write_cycle) = stage6_checked_exact_split(
        "Stage 6 field-register read-write opening",
        &stage4.batch.field_registers_read_write.opening_point,
        field_log_k,
        expected_len,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let (val_evaluation_address, val_evaluation_cycle) = stage6_checked_exact_split(
        "Stage 6 field-register value-evaluation opening",
        &stage5.batch.field_registers_val_evaluation.opening_point,
        field_log_k,
        expected_len,
        JoltRelationId::BytecodeReadRaf,
    )?;
    Ok(FieldInlineStage6BytecodeRegisterPoints {
        read_write_address,
        read_write_cycle,
        val_evaluation_address,
        val_evaluation_cycle,
    })
}

pub fn stage6_validate_dependencies<F: Field>(
    stage3: &Stage3ClearOutput<F>,
) -> Result<(), VerifierError> {
    let [(spartan_shift_unexpanded_pc, instruction_input_unexpanded_pc)] =
        bytecode::read_raf_consistency_openings();
    if stage3.output_claims.shift.unexpanded_pc
        != stage3.output_claims.instruction_input.unexpanded_pc
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
            left: spartan_shift_unexpanded_pc,
            right: instruction_input_unexpanded_pc,
        });
    }
    Ok(())
}

fn stage6_checked_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
    stage: JoltRelationId,
) -> Result<(&'a [F], &'a [F]), VerifierError> {
    if point.len() < split_at {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: format!(
                "{label} has {} variables, expected at least {split_at}",
                point.len()
            ),
        });
    }
    Ok(point.split_at(split_at))
}

fn stage6_checked_exact_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
    expected_len: usize,
    stage: JoltRelationId,
) -> Result<(&'a [F], &'a [F]), VerifierError> {
    if point.len() != expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: format!(
                "{label} length mismatch: expected {expected_len}, got {}",
                point.len()
            ),
        });
    }
    Ok(point.split_at(split_at))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 input claims bind every prior clear stage plus independent transcript challenges."
)]
pub fn stage6_batch_input_claims<F: Field>(
    trace_dimensions: TraceDimensions,
    bytecode_dimensions: BytecodeReadRafDimensions,
    ram_ra_dimensions: RamRaVirtualizationDimensions,
    instruction_ra_dimensions: InstructionRaVirtualizationDimensions,
    trusted_advice_layout: Option<&AdviceClaimReductionLayout>,
    untrusted_advice_layout: Option<&AdviceClaimReductionLayout>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    challenges: Stage6InputClaimChallengeValues<'_, F>,
) -> Result<Stage6BatchInputClaims<F>, VerifierError> {
    let ram_ra_claims = ram::ra_virtualization::<F>(ram_ra_dimensions);
    let instruction_ra_claims = instruction::ra_virtualization::<F>(instruction_ra_dimensions);
    let inc_claims = increments::claim_reduction::<F>(trace_dimensions);
    let trusted_advice_claims = trusted_advice_layout
        .map(|layout| advice::cycle_phase::<F>(JoltAdviceKind::Trusted, layout.dimensions()));
    let untrusted_advice_claims = untrusted_advice_layout
        .map(|layout| advice::cycle_phase::<F>(JoltAdviceKind::Untrusted, layout.dimensions()));

    stage6_validate_dependencies(stage3)?;

    let [ram_ra_reduced] = ram::ra_virtualization_input_openings();
    let instruction_ra_input_openings =
        instruction::ra_virtualization_input_openings(instruction_ra_dimensions);
    let [ram_inc_read_write, ram_inc_val_check, rd_inc_read_write, rd_inc_val_evaluation] =
        increments::claim_reduction_input_openings();
    Ok(Stage6BatchInputClaims {
        bytecode_read_raf: stage6_bytecode_read_raf_address_input(
            bytecode_dimensions,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            challenges,
        )?,
        booleanity: F::zero(),
        ram_hamming_booleanity: F::zero(),
        ram_ra_virtualization: ram_ra_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_ra_reduced => {
                    Ok(stage5.output_claims.ram_ra_claim_reduction.ram_ra)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        instruction_ra_virtualization: instruction_ra_claims.input.expression().try_evaluate(
            |id| {
                for (index, opening) in instruction_ra_input_openings.iter().enumerate() {
                    if *id == *opening {
                        return stage5
                            .output_claims
                            .instruction_read_raf
                            .instruction_ra
                            .get(index)
                            .copied()
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                    }
                }
                Err(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| match id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => Ok(challenges.instruction_ra_gamma),
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        inc_claim_reduction: inc_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_inc_read_write => Ok(stage2.output_claims.ram_read_write.inc),
                id if id == ram_inc_val_check => Ok(stage4.output_claims.ram_val_check.ram_inc),
                id if id == rd_inc_read_write => {
                    Ok(stage4.output_claims.registers_read_write.rd_inc)
                }
                id if id == rd_inc_val_evaluation => {
                    Ok(stage5.output_claims.registers_val_evaluation.rd_inc)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => {
                    Ok(challenges.inc_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction: {
            let read_write_inc = stage4
                .output_claims
                .field_inline
                .field_registers_read_write
                .field_rd_inc;
            let val_evaluation_inc = stage5
                .output_claims
                .field_inline
                .field_registers_val_evaluation
                .field_rd_inc;
            read_write_inc + challenges.field_inc_gamma * val_evaluation_inc
        },
        trusted_advice_cycle_phase: trusted_advice_claims
            .as_ref()
            .map(|claim| {
                verify_b::advice_cycle_phase_input::<F>(claim, stage4, JoltAdviceKind::Trusted)
            })
            .transpose()?,
        untrusted_advice_cycle_phase: untrusted_advice_claims
            .as_ref()
            .map(|claim| {
                verify_b::advice_cycle_phase_input::<F>(claim, stage4, JoltAdviceKind::Untrusted)
            })
            .transpose()?,
    })
}

pub fn stage6_bytecode_read_raf_address_input<F: Field>(
    bytecode_dimensions: BytecodeReadRafDimensions,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    challenges: Stage6InputClaimChallengeValues<'_, F>,
) -> Result<F, VerifierError> {
    let bytecode_claims = bytecode::read_raf_address_phase::<F>(bytecode_dimensions);
    let bytecode_gamma = challenges.bytecode_gamma_powers[1];
    let bytecode_input_openings = bytecode::read_raf_input_openings();
    stage6_validate_dependencies(stage3)?;
    {
        let input_claim = bytecode_claims.input.expression().try_evaluate(
            |id| {
                if *id == bytecode_input_openings.spartan_outer.unexpanded_pc {
                    return Ok(stage1.outer.unexpanded_pc);
                }
                if *id == bytecode_input_openings.spartan_outer.imm {
                    return Ok(stage1.outer.imm);
                }
                for (flag, opening) in &bytecode_input_openings.spartan_outer.op_flags {
                    if *id == *opening {
                        return stage1
                            .outer
                            .claim(JoltVirtualPolynomial::OpFlags(*flag))
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                    }
                }
                if *id == bytecode_input_openings.spartan_product.jump {
                    return Ok(stage2.output_claims.product_remainder.jump_flag);
                }
                if *id == bytecode_input_openings.spartan_product.branch {
                    return Ok(stage2.output_claims.product_remainder.branch_flag);
                }
                if *id
                    == bytecode_input_openings
                        .spartan_product
                        .write_lookup_output_to_rd
                {
                    return Ok(stage2
                        .output_claims
                        .product_remainder
                        .write_lookup_output_to_rd);
                }
                if *id == bytecode_input_openings.spartan_product.virtual_instruction {
                    return Ok(stage2.output_claims.product_remainder.virtual_instruction);
                }
                if *id == bytecode_input_openings.instruction_input.imm {
                    return Ok(stage3.output_claims.instruction_input.imm);
                }
                if *id
                    == bytecode_input_openings
                        .instruction_input
                        .unexpanded_pc_from_shift
                {
                    return Ok(stage3.output_claims.shift.unexpanded_pc);
                }
                if *id
                    == bytecode_input_openings
                        .instruction_input
                        .left_operand_is_rs1_value
                {
                    return Ok(stage3.output_claims.instruction_input.left_operand_is_rs1);
                }
                if *id == bytecode_input_openings.instruction_input.left_operand_is_pc {
                    return Ok(stage3.output_claims.instruction_input.left_operand_is_pc);
                }
                if *id
                    == bytecode_input_openings
                        .instruction_input
                        .right_operand_is_rs2_value
                {
                    return Ok(stage3.output_claims.instruction_input.right_operand_is_rs2);
                }
                if *id
                    == bytecode_input_openings
                        .instruction_input
                        .right_operand_is_imm
                {
                    return Ok(stage3.output_claims.instruction_input.right_operand_is_imm);
                }
                if *id == bytecode_input_openings.instruction_input.is_noop_from_shift {
                    return Ok(stage3.output_claims.shift.is_noop);
                }
                if *id
                    == bytecode_input_openings
                        .instruction_input
                        .virtual_instruction_from_shift
                {
                    return Ok(stage3.output_claims.shift.is_virtual);
                }
                if *id
                    == bytecode_input_openings
                        .instruction_input
                        .is_first_in_sequence_from_shift
                {
                    return Ok(stage3.output_claims.shift.is_first_in_sequence);
                }
                if *id == bytecode_input_openings.registers_read_write.rd_wa {
                    return Ok(stage4.output_claims.registers_read_write.rd_wa);
                }
                if *id == bytecode_input_openings.registers_read_write.rs1_ra {
                    return Ok(stage4.output_claims.registers_read_write.rs1_ra);
                }
                if *id == bytecode_input_openings.registers_read_write.rs2_ra {
                    return Ok(stage4.output_claims.registers_read_write.rs2_ra);
                }
                if *id == bytecode_input_openings.registers_val_evaluation.rd_wa {
                    return Ok(stage5.output_claims.registers_val_evaluation.rd_wa);
                }
                if *id
                    == bytecode_input_openings
                        .registers_val_evaluation
                        .instruction_raf_flag
                {
                    return Ok(stage5
                        .output_claims
                        .instruction_read_raf
                        .instruction_raf_flag);
                }
                for (table, opening) in &bytecode_input_openings
                    .registers_val_evaluation
                    .lookup_table_flags
                {
                    if *id == *opening {
                        return stage5
                            .output_claims
                            .instruction_read_raf
                            .lookup_table_flags
                            .get(table.index())
                            .copied()
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                    }
                }
                if *id == bytecode_input_openings.spartan_outer_pc {
                    return Ok(stage1.outer.pc);
                }
                if *id == bytecode_input_openings.spartan_shift_pc {
                    return Ok(stage3.output_claims.shift.pc);
                }
                Err(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| match id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                    Ok(bytecode_gamma)
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage1Gamma) => {
                    Ok(challenges.stage1_gammas[1])
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage2Gamma) => {
                    Ok(challenges.stage2_gammas[1])
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage3Gamma) => {
                    Ok(challenges.stage3_gammas[1])
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage4Gamma) => {
                    Ok(challenges.stage4_gammas[1])
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage5Gamma) => {
                    Ok(challenges.stage5_gammas[1])
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?;

        #[cfg(feature = "field-inline")]
        let input_claim = {
            let mut input_claim = input_claim;
            let field_openings = field_bytecode::read_raf_input_openings();
            input_claim += field_bytecode::read_raf_input_extension::<F>().try_evaluate(
                |id| {
                    for (index, flag) in field_bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS
                        .into_iter()
                        .enumerate()
                    {
                        if *id == field_openings[index] {
                            return stage1
                                .field_inline
                                .claim(FieldInlineVirtualPolynomial::FieldOpFlag(flag))
                                .ok_or(VerifierError::MissingFieldInlineOpeningClaim { id: *id });
                        }
                    }
                    if *id == field_openings[8] {
                        return Ok(stage4
                            .output_claims
                            .field_inline
                            .field_registers_read_write
                            .field_rd_wa);
                    }
                    if *id == field_openings[9] {
                        return Ok(stage4
                            .output_claims
                            .field_inline
                            .field_registers_read_write
                            .field_rs1_ra);
                    }
                    if *id == field_openings[10] {
                        return Ok(stage4
                            .output_claims
                            .field_inline
                            .field_registers_read_write
                            .field_rs2_ra);
                    }
                    if *id == field_openings[11] {
                        return Ok(stage5
                            .output_claims
                            .field_inline
                            .field_registers_val_evaluation
                            .field_rd_wa);
                    }
                    Err(VerifierError::MissingFieldInlineOpeningClaim { id: *id })
                },
                |id| match id {
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                        Ok(bytecode_gamma)
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage1Gamma) => {
                        Ok(challenges.stage1_gammas[1])
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage4Gamma) => {
                        Ok(challenges.stage4_gammas[1])
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage5Gamma) => {
                        Ok(challenges.stage5_gammas[1])
                    }
                    _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                },
                |()| Ok(F::zero()),
            )?;
            input_claim
        };

        Ok(input_claim)
    }
}

pub fn stage6_booleanity_reference<F, T>(
    instruction_address: &[F],
    instruction_cycle: &[F],
    committed_chunk_bits: usize,
    transcript: &mut T,
) -> Stage6BooleanityReference<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let mut address = instruction_address.to_vec();
    address.reverse();
    if address.len() < committed_chunk_bits {
        address.extend(transcript.challenge_vector(committed_chunk_bits - address.len()));
    } else {
        address = address[address.len() - committed_chunk_bits..].to_vec();
    }
    let mut cycle = instruction_cycle.to_vec();
    cycle.reverse();
    Stage6BooleanityReference { address, cycle }
}

/// The Stage 6 transcript challenges drawn BEFORE the stage 6a address-phase
/// sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PreAddressChallenges<F: Field> {
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference: Stage6BooleanityReference<F>,
    pub booleanity_gamma: F,
}

/// The Stage 6 transcript challenges drawn AFTER the stage 6a address-phase
/// sumcheck (and after its output openings are appended to the transcript).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PostAddressChallenges<F: Field> {
    pub instruction_ra_gamma_powers: Vec<F>,
    pub instruction_ra_gamma: F,
    pub inc_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_inc_gamma: F,
}

pub fn stage6_pre_address_transcript_challenges<F, T>(
    instruction_address: &[F],
    instruction_cycle: &[F],
    committed_chunk_bits: usize,
    lookup_table_flag_count: usize,
    transcript: &mut T,
) -> Stage6PreAddressChallenges<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let bytecode_gamma_powers = transcript.challenge_scalar_powers(stage6_bytecode_gamma_count());
    let stage1_gammas = transcript.challenge_scalar_powers(stage6_stage1_gamma_count());
    let stage2_gammas = transcript.challenge_scalar_powers(stage6_stage2_gamma_count());
    let stage3_gammas = transcript.challenge_scalar_powers(stage6_stage3_gamma_count());
    let stage4_gammas = transcript.challenge_scalar_powers(stage6_stage4_gamma_count());
    let stage5_gammas =
        transcript.challenge_scalar_powers(stage6_stage5_gamma_count(lookup_table_flag_count));
    let booleanity_reference = stage6_booleanity_reference(
        instruction_address,
        instruction_cycle,
        committed_chunk_bits,
        transcript,
    );
    let mut booleanity_gamma = transcript.challenge();
    if booleanity_gamma == F::zero() {
        booleanity_gamma = F::one();
    }
    Stage6PreAddressChallenges {
        bytecode_gamma_powers,
        stage1_gammas,
        stage2_gammas,
        stage3_gammas,
        stage4_gammas,
        stage5_gammas,
        booleanity_reference,
        booleanity_gamma,
    }
}

pub fn stage6_post_address_transcript_challenges<F, T>(
    instruction_ra_dimensions: InstructionRaVirtualizationDimensions,
    transcript: &mut T,
) -> Stage6PostAddressChallenges<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let instruction_ra_gamma_powers =
        transcript.challenge_scalar_powers(instruction_ra_dimensions.num_virtual_ra_polys());
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(F::one);
    let inc_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_inc_gamma = transcript.challenge_scalar();
    Stage6PostAddressChallenges {
        instruction_ra_gamma_powers,
        instruction_ra_gamma,
        inc_gamma,
        #[cfg(feature = "field-inline")]
        field_inc_gamma,
    }
}

impl<F: Field> Stage6TranscriptChallenges<F> {
    /// Reassembles the full transcript-challenge record from the pre/post address
    /// halves, preserving the canonical field order.
    pub fn from_address_phases(
        pre: Stage6PreAddressChallenges<F>,
        post: Stage6PostAddressChallenges<F>,
    ) -> Self {
        Self {
            bytecode_gamma_powers: pre.bytecode_gamma_powers,
            stage1_gammas: pre.stage1_gammas,
            stage2_gammas: pre.stage2_gammas,
            stage3_gammas: pre.stage3_gammas,
            stage4_gammas: pre.stage4_gammas,
            stage5_gammas: pre.stage5_gammas,
            booleanity_reference: pre.booleanity_reference,
            booleanity_gamma: pre.booleanity_gamma,
            instruction_ra_gamma_powers: post.instruction_ra_gamma_powers,
            instruction_ra_gamma: post.instruction_ra_gamma,
            inc_gamma: post.inc_gamma,
            #[cfg(feature = "field-inline")]
            field_inc_gamma: post.field_inc_gamma,
        }
    }
}

pub fn stage6_public_output<F: Field>(
    transcript_challenges: &Stage6TranscriptChallenges<F>,
    address_phase_challenges: Vec<F>,
    address_phase_batching_coefficients: Vec<F>,
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    bytecode_reduction_eta: Option<F>,
) -> Stage6PublicOutput<F> {
    Stage6PublicOutput {
        address_phase_challenges,
        address_phase_batching_coefficients,
        challenges,
        batching_coefficients,
        bytecode_gamma_powers: transcript_challenges.bytecode_gamma_powers.clone(),
        stage1_gammas: transcript_challenges.stage1_gammas.clone(),
        stage2_gammas: transcript_challenges.stage2_gammas.clone(),
        stage3_gammas: transcript_challenges.stage3_gammas.clone(),
        stage4_gammas: transcript_challenges.stage4_gammas.clone(),
        stage5_gammas: transcript_challenges.stage5_gammas.clone(),
        booleanity_reference_address: transcript_challenges.booleanity_reference.address.clone(),
        booleanity_reference_cycle: transcript_challenges.booleanity_reference.cycle.clone(),
        booleanity_gamma: transcript_challenges.booleanity_gamma,
        instruction_ra_gamma_powers: transcript_challenges.instruction_ra_gamma_powers.clone(),
        inc_gamma: transcript_challenges.inc_gamma,
        bytecode_reduction_eta,
        #[cfg(feature = "field-inline")]
        field_inline: super::outputs::FieldInlineStage6PublicOutput {
            field_inc_gamma: transcript_challenges.field_inc_gamma,
        },
    }
}

pub struct Stage6ClearOutputRequest<'a, F: Field> {
    pub transcript_challenges: &'a Stage6TranscriptChallenges<F>,
    pub output_claims: Stage6Claims<F>,
    pub input_claims: &'a Stage6BatchInputClaims<F>,
    pub expected_outputs: &'a Stage6BatchExpectedOutputClaims<F>,
    pub batching_coefficients: &'a [F],
    pub sumcheck_point: &'a [F],
    pub sumcheck_final_claim: F,
    pub points: &'a Stage6BatchPoints<F>,
}

pub fn stage6_clear_output<F: Field>(
    request: Stage6ClearOutputRequest<'_, F>,
) -> Result<Stage6ClearOutput<F>, VerifierError> {
    let expected_final_claim =
        stage6_expected_final_claim(request.batching_coefficients, request.expected_outputs)?;
    if request.sumcheck_final_claim != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
        });
    }
    let trusted_advice_cycle_phase = stage6_advice_cycle_phase_verified(
        JoltAdviceKind::Trusted,
        request.points.trusted_advice_cycle_phase.as_ref(),
        request.input_claims.trusted_advice_cycle_phase,
        request.expected_outputs.trusted_advice_cycle_phase,
    );
    let untrusted_advice_cycle_phase = stage6_advice_cycle_phase_verified(
        JoltAdviceKind::Untrusted,
        request.points.untrusted_advice_cycle_phase.as_ref(),
        request.input_claims.untrusted_advice_cycle_phase,
        request.expected_outputs.untrusted_advice_cycle_phase,
    );

    let empty_address_sumcheck = VerifiedStage6AddressPhaseSumcheck {
        input_claim: F::zero(),
        sumcheck_point: Vec::new(),
        opening_point: Vec::new(),
        expected_output_claim: F::zero(),
    };

    Ok(Stage6ClearOutput {
        public: stage6_public_output(
            request.transcript_challenges,
            Vec::new(),
            Vec::new(),
            request.sumcheck_point.to_vec(),
            request.batching_coefficients.to_vec(),
            None,
        ),
        output_claims: request.output_claims,
        batch: VerifiedStage6Batch {
            address_phase_batching_coefficients: Vec::new(),
            address_phase_sumcheck_point: Point::high_to_low(Vec::new()),
            address_phase_sumcheck_final_claim: F::zero(),
            address_phase_expected_final_claim: F::zero(),
            bytecode_read_raf_address: empty_address_sumcheck.clone(),
            booleanity_address: empty_address_sumcheck,
            batching_coefficients: request.batching_coefficients.to_vec(),
            sumcheck_point: Point::high_to_low(request.sumcheck_point.to_vec()),
            sumcheck_final_claim: request.sumcheck_final_claim,
            expected_final_claim,
            bytecode_read_raf: VerifiedBytecodeReadRafSumcheck {
                input_claim: request.input_claims.bytecode_read_raf,
                sumcheck_point: request.points.bytecode_read_raf_sumcheck_point.clone(),
                r_address: request.points.bytecode_read_raf_r_address.clone(),
                r_cycle: request.points.bytecode_read_raf_r_cycle.clone(),
                full_opening_point: request.points.bytecode_read_raf_full_opening_point.clone(),
                bytecode_ra_opening_points: request.points.bytecode_ra_opening_points.clone(),
                expected_output_claim: request.expected_outputs.bytecode_read_raf,
            },
            booleanity: VerifiedBooleanitySumcheck {
                input_claim: request.input_claims.booleanity,
                sumcheck_point: request.points.booleanity_sumcheck_point.clone(),
                r_address: request.points.booleanity_r_address.clone(),
                r_cycle: request.points.booleanity_r_cycle.clone(),
                opening_point: request.points.booleanity_opening_point.clone(),
                reference_address: request
                    .transcript_challenges
                    .booleanity_reference
                    .address
                    .clone(),
                reference_cycle: request
                    .transcript_challenges
                    .booleanity_reference
                    .cycle
                    .clone(),
                expected_output_claim: request.expected_outputs.booleanity,
            },
            ram_hamming_booleanity: VerifiedStage6Sumcheck {
                input_claim: request.input_claims.ram_hamming_booleanity,
                sumcheck_point: request.points.ram_hamming_booleanity_sumcheck_point.clone(),
                opening_point: request.points.ram_hamming_booleanity_opening_point.clone(),
                expected_output_claim: request.expected_outputs.ram_hamming_booleanity,
            },
            ram_ra_virtualization: VerifiedRamRaVirtualizationSumcheck {
                input_claim: request.input_claims.ram_ra_virtualization,
                sumcheck_point: request.points.ram_ra_virtualization_sumcheck_point.clone(),
                opening_point: request.points.ram_ra_virtualization_opening_point.clone(),
                ram_ra_opening_points: request.points.ram_ra_opening_points.clone(),
                expected_output_claim: request.expected_outputs.ram_ra_virtualization,
            },
            instruction_ra_virtualization: VerifiedInstructionRaVirtualizationSumcheck {
                input_claim: request.input_claims.instruction_ra_virtualization,
                sumcheck_point: request
                    .points
                    .instruction_ra_virtualization_sumcheck_point
                    .clone(),
                opening_point: request
                    .points
                    .instruction_ra_virtualization_opening_point
                    .clone(),
                instruction_ra_opening_points: request.points.instruction_ra_opening_points.clone(),
                expected_output_claim: request.expected_outputs.instruction_ra_virtualization,
            },
            inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: request.input_claims.inc_claim_reduction,
                sumcheck_point: request.points.inc_claim_reduction_sumcheck_point.clone(),
                opening_point: request.points.inc_claim_reduction_opening_point.clone(),
                expected_output_claim: request.expected_outputs.inc_claim_reduction,
            },
            // Field-inline placeholder: this request-based Stage 6 reconstruction
            // does not thread field-register inc-claim-reduction data. Field-inline
            // is not exercised by the muldiv gates.
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: F::zero(),
                sumcheck_point: Vec::new(),
                opening_point: Vec::new(),
                expected_output_claim: F::zero(),
            },
            trusted_advice_cycle_phase,
            untrusted_advice_cycle_phase,
            bytecode_cycle_phase: None,
            program_image_cycle_phase: None,
        },
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeReadRafExpectedOutputInputs<'a, F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub public_values: &'a bytecode::BytecodeReadRafPublicValues<F>,
    pub bytecode_ra: &'a [F],
    pub gamma: F,
}

pub fn stage6_bytecode_read_raf_expected_output<F: Field>(
    inputs: Stage6BytecodeReadRafExpectedOutputInputs<'_, F>,
) -> Result<F, VerifierError> {
    let output_openings = bytecode::read_raf_output_openings(inputs.dimensions);
    if inputs.bytecode_ra.len() != output_openings.bytecode_ra.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                output_openings.bytecode_ra.len(),
                inputs.bytecode_ra.len()
            ),
        });
    }
    let claim = bytecode::read_raf::<F>(inputs.dimensions);
    claim.output.expression().try_evaluate(
        |id| {
            for (index, opening) in output_openings.bytecode_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(inputs.bytecode_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => Ok(inputs.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::BytecodeReadRaf(public_id) => inputs
                .public_values
                .value(*public_id)
                .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeReadRafOutputCoefficientInputs<'a, F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub public_values: &'a bytecode::BytecodeReadRafPublicValues<F>,
    pub gamma: F,
}

pub fn stage6_bytecode_read_raf_output_coefficient<F: Field>(
    inputs: Stage6BytecodeReadRafOutputCoefficientInputs<'_, F>,
) -> Result<F, VerifierError> {
    let output_openings = bytecode::read_raf_output_openings(inputs.dimensions);
    let bytecode_ra = vec![F::one(); output_openings.bytecode_ra.len()];
    stage6_bytecode_read_raf_expected_output(Stage6BytecodeReadRafExpectedOutputInputs {
        dimensions: inputs.dimensions,
        public_values: inputs.public_values,
        bytecode_ra: &bytecode_ra,
        gamma: inputs.gamma,
    })
}

pub fn stage6_advice_cycle_phase_expected_output<F: Field>(
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    reference_opening_point: &[F],
    sumcheck_point: &[F],
    opening_claim: F,
) -> Result<F, VerifierError> {
    let claim = advice::cycle_phase::<F>(kind, layout.dimensions());
    let output_openings = advice::cycle_phase_output_openings(kind, layout.dimensions());
    claim.output.expression().try_evaluate(
        |id| {
            if output_openings.contains(id) {
                Ok(opening_claim)
            } else {
                Err(VerifierError::MissingOpeningClaim { id: *id })
            }
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| match id {
            JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                public_kind,
            )) if *public_kind == kind => layout
                .cycle_phase_final_output_scale(reference_opening_point, sumcheck_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::AdviceClaimReductionCyclePhase,
                    reason: error.to_string(),
                }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6IncClaimReductionExpectedOutputInputs<'a, F: Field> {
    pub opening_point: &'a [F],
    pub ram_read_write_cycle: &'a [F],
    pub ram_val_check_cycle: &'a [F],
    pub registers_read_write_cycle: &'a [F],
    pub registers_val_evaluation_cycle: &'a [F],
    pub ram_inc: F,
    pub rd_inc: F,
    pub gamma: F,
}

pub fn stage6_inc_claim_reduction_expected_output<F: Field>(
    inputs: Stage6IncClaimReductionExpectedOutputInputs<'_, F>,
) -> Result<F, VerifierError> {
    increments::claim_reduction_output_claim(increments::ClaimReductionOutputClaimInputs {
        coefficients: increments::ClaimReductionOutputCoefficientInputs {
            opening_point: inputs.opening_point,
            ram_read_write_cycle: inputs.ram_read_write_cycle,
            ram_val_check_cycle: inputs.ram_val_check_cycle,
            registers_read_write_cycle: inputs.registers_read_write_cycle,
            registers_val_evaluation_cycle: inputs.registers_val_evaluation_cycle,
            gamma: inputs.gamma,
        },
        ram_inc: inputs.ram_inc,
        rd_inc: inputs.rd_inc,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::IncClaimReduction,
        reason: error.to_string(),
    })
}

pub fn stage6_ram_hamming_booleanity_expected_output<F: Field>(
    sumcheck_point: &[F],
    stage1_cycle_binding: &[F],
    ram_hamming_weight: F,
) -> Result<F, VerifierError> {
    let eq_spartan_outer_cycle =
        try_eq_mle(sumcheck_point, stage1_cycle_binding).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            }
        })?;
    let claim = ram::hamming_booleanity::<F>(
        jolt_claims::protocols::jolt::formulas::dimensions::TraceDimensions::new(
            sumcheck_point.len(),
        ),
    );
    let [ram_hamming_weight_opening] = ram::hamming_booleanity_output_openings();
    claim.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_hamming_weight_opening => Ok(ram_hamming_weight),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match *id {
            JoltChallengeId::RamHammingBooleanity(RamHammingBooleanityChallenge::EqCycle) => {
                Ok(eq_spartan_outer_cycle)
            }
            id => Err(VerifierError::MissingStageClaimChallenge { id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6RamRaVirtualizationExpectedOutputInputs<'a, F: Field> {
    pub dimensions: RamRaVirtualizationDimensions,
    pub r_cycle: &'a [F],
    pub ram_reduced_cycle: &'a [F],
    pub ram_ra: &'a [F],
}

pub fn stage6_ram_ra_virtualization_expected_output<F: Field>(
    inputs: Stage6RamRaVirtualizationExpectedOutputInputs<'_, F>,
) -> Result<F, VerifierError> {
    let output_openings = ram::ra_virtualization_output_openings(inputs.dimensions);
    if inputs.ram_ra.len() != output_openings.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: format!(
                "RAM RA virtualization claim count mismatch: expected {}, got {}",
                output_openings.len(),
                inputs.ram_ra.len()
            ),
        });
    }
    let eq_cycle = try_eq_mle(inputs.ram_reduced_cycle, inputs.r_cycle).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        }
    })?;
    let claim = ram::ra_virtualization::<F>(inputs.dimensions);
    claim.output.expression().try_evaluate(
        |id| {
            for (index, opening) in output_openings.iter().enumerate() {
                if *id == *opening {
                    return Ok(inputs.ram_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match *id {
            JoltChallengeId::RamRaVirtualization(RamRaVirtualizationChallenge::EqCycle) => {
                Ok(eq_cycle)
            }
            id => Err(VerifierError::MissingStageClaimChallenge { id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6InstructionRaVirtualizationExpectedOutputInputs<'a, F: Field> {
    pub dimensions: InstructionRaVirtualizationDimensions,
    pub instruction_read_raf_cycle: &'a [F],
    pub r_cycle: &'a [F],
    pub committed_instruction_ra: &'a [F],
    pub gamma: F,
}

pub fn stage6_instruction_ra_virtualization_expected_output<F: Field>(
    inputs: Stage6InstructionRaVirtualizationExpectedOutputInputs<'_, F>,
) -> Result<F, VerifierError> {
    let output_openings = instruction::ra_virtualization_output_openings(inputs.dimensions);
    let flat_output_openings = output_openings.all();
    if inputs.committed_instruction_ra.len() != flat_output_openings.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: format!(
                "instruction RA virtualization claim count mismatch: expected {}, got {}",
                flat_output_openings.len(),
                inputs.committed_instruction_ra.len()
            ),
        });
    }
    let eq_cycle =
        try_eq_mle(inputs.instruction_read_raf_cycle, inputs.r_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionRaVirtualization,
                reason: error.to_string(),
            }
        })?;
    let claim = instruction::ra_virtualization::<F>(inputs.dimensions);
    claim.output.expression().try_evaluate(
        |id| {
            for (index, opening) in flat_output_openings.iter().enumerate() {
                if *id == *opening {
                    return Ok(inputs.committed_instruction_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::Gamma,
            ) => Ok(inputs.gamma),
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::EqCycle,
            ) => Ok(eq_cycle),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BooleanityExpectedOutputInputs<'a, F: Field> {
    pub dimensions: BooleanityDimensions,
    pub sumcheck_point: &'a [F],
    pub reference: &'a Stage6BooleanityReference<F>,
    pub instruction_ra: &'a [F],
    pub bytecode_ra: &'a [F],
    pub ram_ra: &'a [F],
    pub gamma: F,
}

pub fn stage6_booleanity_expected_output<F: Field>(
    inputs: Stage6BooleanityExpectedOutputInputs<'_, F>,
) -> Result<F, VerifierError> {
    let output_openings = booleanity::booleanity_output_opening_groups(inputs.dimensions.layout);
    if inputs.instruction_ra.len() != output_openings.instruction_ra.len()
        || inputs.bytecode_ra.len() != output_openings.bytecode_ra.len()
        || inputs.ram_ra.len() != output_openings.ram_ra.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: format!(
                "booleanity RA claim count mismatch: expected ({}, {}, {}), got ({}, {}, {})",
                output_openings.instruction_ra.len(),
                output_openings.bytecode_ra.len(),
                output_openings.ram_ra.len(),
                inputs.instruction_ra.len(),
                inputs.bytecode_ra.len(),
                inputs.ram_ra.len()
            ),
        });
    }
    let reference_eq_point = inputs
        .reference
        .address
        .iter()
        .rev()
        .chain(inputs.reference.cycle.iter().rev())
        .copied()
        .collect::<Vec<_>>();
    let eq_address_cycle =
        try_eq_mle(inputs.sumcheck_point, &reference_eq_point).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            }
        })?;
    let claim = booleanity::booleanity::<F>(inputs.dimensions);
    claim.output.expression().try_evaluate(
        |id| {
            for (index, opening) in output_openings.instruction_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(inputs.instruction_ra[index]);
                }
            }
            for (index, opening) in output_openings.bytecode_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(inputs.bytecode_ra[index]);
                }
            }
            for (index, opening) in output_openings.ram_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(inputs.ram_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => Ok(inputs.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::Booleanity(BooleanityPublic::EqAddressCycle) => Ok(eq_address_cycle),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Copy, Debug)]
pub struct FieldInlineStage6IncClaimReductionExpectedOutputInputs<'a, F: Field> {
    pub opening_point: &'a [F],
    pub read_write_cycle: &'a [F],
    pub val_evaluation_cycle: &'a [F],
    pub field_rd_inc: F,
    pub gamma: F,
}

#[cfg(feature = "field-inline")]
pub fn stage6_field_registers_inc_claim_reduction_expected_output<F: Field>(
    inputs: FieldInlineStage6IncClaimReductionExpectedOutputInputs<'_, F>,
) -> Result<F, VerifierError> {
    let eq_read_write =
        try_eq_mle(inputs.opening_point, inputs.read_write_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let eq_val_evaluation =
        try_eq_mle(inputs.opening_point, inputs.val_evaluation_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    Ok((eq_read_write + inputs.gamma * eq_val_evaluation) * inputs.field_rd_inc)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BatchExpectedOutputClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    pub ram_hamming_booleanity: F,
    pub ram_ra_virtualization: F,
    pub instruction_ra_virtualization: F,
    pub inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: F,
    pub trusted_advice_cycle_phase: Option<F>,
    pub untrusted_advice_cycle_phase: Option<F>,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BatchPointInputs<'a, F: Field> {
    pub bytecode_read_raf: &'a [F],
    pub booleanity: &'a [F],
    pub ram_hamming_booleanity: &'a [F],
    pub ram_ra_virtualization: &'a [F],
    pub instruction_ra_virtualization: &'a [F],
    pub inc_claim_reduction: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: &'a [F],
    pub trusted_advice_cycle_phase: Option<&'a [F]>,
    pub untrusted_advice_cycle_phase: Option<&'a [F]>,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BatchPointContext<'a, F: Field> {
    pub trace_dimensions: TraceDimensions,
    pub bytecode_read_raf_dimensions: BytecodeReadRafDimensions,
    pub booleanity_dimensions: BooleanityDimensions,
    pub committed_chunk_bits: usize,
    pub ram_reduced_opening_point: Stage6RamReducedOpeningPoint<'a, F>,
    pub instruction_read_raf: Stage6InstructionReadRafPoint<'a, F>,
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BatchPoints<F: Field> {
    pub bytecode_read_raf_sumcheck_point: Vec<F>,
    pub bytecode_read_raf_r_address: Vec<F>,
    pub bytecode_read_raf_r_cycle: Vec<F>,
    pub bytecode_read_raf_full_opening_point: Vec<F>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
    pub booleanity_sumcheck_point: Vec<F>,
    pub booleanity_r_address: Vec<F>,
    pub booleanity_r_cycle: Vec<F>,
    pub booleanity_opening_point: Vec<F>,
    pub ram_hamming_booleanity_sumcheck_point: Vec<F>,
    pub ram_hamming_booleanity_opening_point: Vec<F>,
    pub ram_ra_virtualization_sumcheck_point: Vec<F>,
    pub ram_ra_virtualization_opening_point: Vec<F>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
    pub instruction_ra_virtualization_sumcheck_point: Vec<F>,
    pub instruction_ra_virtualization_opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub inc_claim_reduction_sumcheck_point: Vec<F>,
    pub inc_claim_reduction_opening_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction_sumcheck_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction_opening_point: Vec<F>,
    pub trusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
    pub untrusted_advice_cycle_phase: Option<AdviceCyclePhasePublicOutput<F>>,
}

pub fn stage6_batch_points<F: Field>(
    inputs: Stage6BatchPointInputs<'_, F>,
    context: Stage6BatchPointContext<'_, F>,
) -> Result<Stage6BatchPoints<F>, VerifierError> {
    let bytecode_opening = stage6_bytecode_read_raf_point(
        context.bytecode_read_raf_dimensions,
        inputs.bytecode_read_raf,
    )?;
    let bytecode_ra_opening_points =
        bytecode_opening.committed_opening_points(context.committed_chunk_bits);

    let booleanity_opening = context
        .booleanity_dimensions
        .opening_point(inputs.booleanity)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;

    let ram_hamming_opening = context
        .trace_dimensions
        .cycle_opening_point(inputs.ram_hamming_booleanity)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamHammingBooleanity,
            reason: error.to_string(),
        })?;

    let ram_ra_cycle = context
        .trace_dimensions
        .cycle_opening_point(inputs.ram_ra_virtualization)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        })?;
    let ram_ra_opening_point = context
        .ram_reduced_opening_point
        .opening_point(&ram_ra_cycle);
    let ram_ra_opening_points = context
        .ram_reduced_opening_point
        .committed_opening_points(&ram_ra_cycle, context.committed_chunk_bits);

    let instruction_ra_cycle = context
        .trace_dimensions
        .cycle_opening_point(inputs.instruction_ra_virtualization)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_ra_opening_point = context
        .instruction_read_raf
        .opening_point(&instruction_ra_cycle);
    let instruction_ra_opening_points = context
        .instruction_read_raf
        .committed_opening_points(&instruction_ra_cycle, context.committed_chunk_bits);

    let inc_opening = context
        .trace_dimensions
        .cycle_opening_point(inputs.inc_claim_reduction)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;

    #[cfg(feature = "field-inline")]
    let field_registers_inc_claim_reduction_opening_point = context
        .trace_dimensions
        .cycle_opening_point(inputs.field_registers_inc_claim_reduction)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;

    Ok(Stage6BatchPoints {
        bytecode_read_raf_sumcheck_point: inputs.bytecode_read_raf.to_vec(),
        bytecode_read_raf_r_address: bytecode_opening.r_address,
        bytecode_read_raf_r_cycle: bytecode_opening.r_cycle,
        bytecode_read_raf_full_opening_point: bytecode_opening.opening_point,
        bytecode_ra_opening_points,
        booleanity_sumcheck_point: inputs.booleanity.to_vec(),
        booleanity_r_address: booleanity_opening.r_address,
        booleanity_r_cycle: booleanity_opening.r_cycle,
        booleanity_opening_point: booleanity_opening.opening_point,
        ram_hamming_booleanity_sumcheck_point: inputs.ram_hamming_booleanity.to_vec(),
        ram_hamming_booleanity_opening_point: ram_hamming_opening,
        ram_ra_virtualization_sumcheck_point: inputs.ram_ra_virtualization.to_vec(),
        ram_ra_virtualization_opening_point: ram_ra_opening_point,
        ram_ra_opening_points,
        instruction_ra_virtualization_sumcheck_point: inputs.instruction_ra_virtualization.to_vec(),
        instruction_ra_virtualization_opening_point: instruction_ra_opening_point,
        instruction_ra_opening_points,
        inc_claim_reduction_sumcheck_point: inputs.inc_claim_reduction.to_vec(),
        inc_claim_reduction_opening_point: inc_opening,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction_sumcheck_point: inputs
            .field_registers_inc_claim_reduction
            .to_vec(),
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction_opening_point,
        trusted_advice_cycle_phase: stage6_advice_cycle_phase_points(
            JoltAdviceKind::Trusted,
            context.trusted_advice_layout,
            inputs.trusted_advice_cycle_phase,
        )?,
        untrusted_advice_cycle_phase: stage6_advice_cycle_phase_points(
            JoltAdviceKind::Untrusted,
            context.untrusted_advice_layout,
            inputs.untrusted_advice_cycle_phase,
        )?,
    })
}

fn stage6_advice_cycle_phase_points<F: Field>(
    kind: JoltAdviceKind,
    layout: Option<&AdviceClaimReductionLayout>,
    point: Option<&[F]>,
) -> Result<Option<AdviceCyclePhasePublicOutput<F>>, VerifierError> {
    match (layout, point) {
        (Some(layout), Some(point)) => Ok(Some(AdviceCyclePhasePublicOutput {
            kind,
            sumcheck_point: point.to_vec(),
            opening_point: layout.cycle_phase_opening_point(point).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::AdviceClaimReductionCyclePhase,
                    reason: error.to_string(),
                }
            })?,
            cycle_phase_variables: layout.cycle_phase_variable_challenges(point).map_err(
                |error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::AdviceClaimReductionCyclePhase,
                    reason: error.to_string(),
                },
            )?,
        })),
        (None, None) => Ok(None),
        (Some(_), None) => Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: format!("Stage 6 {kind:?} advice cycle-phase point is missing"),
        }),
        (None, Some(_)) => Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: format!("Stage 6 {kind:?} advice cycle-phase point is unexpected"),
        }),
    }
}

pub fn stage6_expected_output_claim_values<F: Field>(
    expected_outputs: &Stage6BatchExpectedOutputClaims<F>,
) -> Vec<F> {
    let mut values = vec![
        expected_outputs.bytecode_read_raf,
        expected_outputs.booleanity,
        expected_outputs.ram_hamming_booleanity,
        expected_outputs.ram_ra_virtualization,
        expected_outputs.instruction_ra_virtualization,
        expected_outputs.inc_claim_reduction,
    ];
    #[cfg(feature = "field-inline")]
    values.push(expected_outputs.field_registers_inc_claim_reduction);
    if let Some(output_claim) = expected_outputs.trusted_advice_cycle_phase {
        values.push(output_claim);
    }
    if let Some(output_claim) = expected_outputs.untrusted_advice_cycle_phase {
        values.push(output_claim);
    }
    values
}

pub fn stage6_expected_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage6BatchExpectedOutputClaims<F>,
) -> Result<F, VerifierError> {
    let expected_outputs_in_order = stage6_expected_output_claim_values(expected_outputs);
    if coefficients.len() != expected_outputs_in_order.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "Stage 6 batch verifier returned {} coefficients for {} instances",
                coefficients.len(),
                expected_outputs_in_order.len()
            ),
        });
    }
    Ok(coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum())
}

pub const fn stage6_bytecode_gamma_count() -> usize {
    8
}

#[cfg(feature = "field-inline")]
pub const fn stage6_stage1_gamma_count() -> usize {
    field_bytecode::FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT
}

#[cfg(not(feature = "field-inline"))]
pub const fn stage6_stage1_gamma_count() -> usize {
    2 + NUM_CIRCUIT_FLAGS
}

pub const fn stage6_stage2_gamma_count() -> usize {
    4
}

pub const fn stage6_stage3_gamma_count() -> usize {
    9
}

#[cfg(feature = "field-inline")]
pub const fn stage6_stage4_gamma_count() -> usize {
    field_bytecode::FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT
}

#[cfg(not(feature = "field-inline"))]
pub const fn stage6_stage4_gamma_count() -> usize {
    3
}

#[cfg(feature = "field-inline")]
pub const fn stage6_stage5_gamma_count(lookup_table_flag_count: usize) -> usize {
    2 + lookup_table_flag_count + field_bytecode::FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS
}

#[cfg(not(feature = "field-inline"))]
pub const fn stage6_stage5_gamma_count(lookup_table_flag_count: usize) -> usize {
    2 + lookup_table_flag_count
}

pub fn stage6_advice_cycle_phase_verified<F: Field>(
    kind: JoltAdviceKind,
    proof: Option<&AdviceCyclePhasePublicOutput<F>>,
    input_claim: Option<F>,
    expected_output_claim: Option<F>,
) -> Option<VerifiedAdviceCyclePhaseSumcheck<F>> {
    match (proof, input_claim, expected_output_claim) {
        (Some(proof), Some(input_claim), Some(expected_output_claim)) => {
            Some(VerifiedAdviceCyclePhaseSumcheck {
                kind,
                input_claim,
                sumcheck_point: proof.sumcheck_point.clone(),
                opening_point: proof.opening_point.clone(),
                cycle_phase_variables: proof.cycle_phase_variables.clone(),
                expected_output_claim,
            })
        }
        _ => None,
    }
}

pub fn append_stage6_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6Claims<F>,
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in
        stage6_output_claim_values(claims, bytecode_ra_opening_points, booleanity_opening_point)
    {
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
}

pub fn stage6_output_claim_values<F: Field>(
    claims: &Stage6Claims<F>,
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> Vec<F> {
    let mut values = Vec::new();
    values.extend(claims.bytecode_read_raf.bytecode_ra.iter().copied());
    values.extend(claims.booleanity.instruction_ra.iter().copied());
    for (index, &claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_ra_opening_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_opening_point)
        {
            continue;
        }
        values.push(claim);
    }
    values.extend(claims.booleanity.ram_ra.iter().copied());
    values.push(claims.ram_hamming_booleanity.ram_hamming_weight);
    values.extend(claims.ram_ra_virtualization.ram_ra.iter().copied());
    values.extend(
        claims
            .instruction_ra_virtualization
            .committed_instruction_ra
            .iter()
            .copied(),
    );
    values.push(claims.inc_claim_reduction.ram_inc);
    values.push(claims.inc_claim_reduction.rd_inc);
    #[cfg(feature = "field-inline")]
    values.push(
        claims
            .field_inline
            .field_registers_inc_claim_reduction
            .field_rd_inc,
    );
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        values.push(opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        values.push(opening_claim.opening_claim);
    }
    values
}

pub fn stage6_input_claim_values<F: Field>(claims: &Stage6BatchInputClaims<F>) -> Vec<F> {
    let mut values = vec![
        claims.bytecode_read_raf,
        claims.booleanity,
        claims.ram_hamming_booleanity,
        claims.ram_ra_virtualization,
        claims.instruction_ra_virtualization,
        claims.inc_claim_reduction,
    ];
    #[cfg(feature = "field-inline")]
    values.push(claims.field_registers_inc_claim_reduction);
    if let Some(input_claim) = claims.trusted_advice_cycle_phase {
        values.push(input_claim);
    }
    if let Some(input_claim) = claims.untrusted_advice_cycle_phase {
        values.push(input_claim);
    }
    values
}
