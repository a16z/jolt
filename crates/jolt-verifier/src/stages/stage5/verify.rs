use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::JoltFormulaDimensions, instruction, ram, registers},
    JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::{FsNargRead, FsTranscript};

use super::{
    instruction_read_raf::{InstructionReadRaf, InstructionReadRafInputClaims},
    outputs::{
        Stage5Challenges, Stage5ClearOutput, Stage5Output, Stage5OutputClaims, Stage5ZkOutput,
    },
    ram_ra_claim_reduction::{RamRaClaimReduction, RamRaClaimReductionInputClaims},
    registers_val_evaluation::{RegistersValEvaluation, RegistersValEvaluationInputClaims},
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::{
            check_relation_boolean_hypercube, zip_openings, OpeningClaim, SumcheckInstance,
        },
        stage2::Stage2Output,
        stage4::Stage4Output,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// Combine the three stage 5 expected output claims with the batch's coefficients,
/// in canonical batch order (instruction read-RAF, RAM-RA reduction, register
/// value-evaluation). Shared by the verifier and the prover so the combination
/// cannot drift.
pub fn stage5_expected_final_claim<F: Field>(
    coefficients: &[F],
    instruction_read_raf: F,
    ram_ra_claim_reduction: F,
    registers_val_evaluation: F,
) -> Result<F, VerifierError> {
    let [instruction_coefficient, ram_coefficient, registers_coefficient] = coefficients else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: "Stage 5 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    Ok(*instruction_coefficient * instruction_read_raf
        + *ram_coefficient * ram_ra_claim_reduction
        + *registers_coefficient * registers_val_evaluation)
}

/// Pair the produced stage-5 openings with their derived points (point + value
/// together) from the wire claim values and each relation's opening points. Shared
/// by the verifier and the prover so these opening claims are built once.
pub fn stage5_output_claims_with_points<F: Field>(
    claims: &Stage5OutputClaims<F>,
    points: &Stage5OutputClaims<Vec<F>>,
) -> Stage5OutputClaims<OpeningClaim<F>> {
    Stage5OutputClaims {
        instruction_read_raf: zip_openings(
            &claims.instruction_read_raf,
            &points.instruction_read_raf,
        ),
        ram_ra_claim_reduction: zip_openings(
            &claims.ram_ra_claim_reduction,
            &points.ram_ra_claim_reduction,
        ),
        registers_val_evaluation: zip_openings(
            &claims.registers_val_evaluation,
            &points.registers_val_evaluation,
        ),
    }
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
) -> Result<Stage5Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FsNargRead<PCS::Field>,
    VC::Output: Clone + CanonicalSerialize + CanonicalDeserialize,
{
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = formula_dimensions.trace;

    let instruction_claims =
        instruction::read_raf::<PCS::Field>(formula_dimensions.instruction_read_raf);
    let ram_claims = ram::ra_claim_reduction::<PCS::Field>(trace_dimensions);
    let registers_claims = registers::val_evaluation::<PCS::Field>(trace_dimensions);

    for claim in [&instruction_claims, &ram_claims, &registers_claims] {
        check_relation_boolean_hypercube(claim)?;
    }
    let instruction_gamma = transcript.challenge_scalar();
    let ram_gamma = transcript.challenge_scalar();
    let challenges = Stage5Challenges {
        instruction_gamma,
        ram_gamma,
    };

    let instruction_relation =
        InstructionReadRaf::new(formula_dimensions.instruction_read_raf, instruction_gamma);
    let ram_relation = RamRaClaimReduction::new(trace_dimensions, log_k, ram_gamma);
    let registers_relation = RegistersValEvaluation::new(trace_dimensions);

    let instruction_output_openings =
        instruction::read_raf_output_openings(formula_dimensions.instruction_read_raf);
    let committed_output_claims = instruction_output_openings.opening_count()
        + ram::ra_claim_reduction_output_openings().len()
        + registers::val_evaluation_output_openings().len();

    if checked.zk {
        let stage2 = stage2.zk()?;
        let stage4 = stage4.zk()?;
        let statements = [
            SumcheckStatement::new(
                instruction_claims.sumcheck.rounds,
                instruction_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(ram_claims.sumcheck.rounds, ram_claims.sumcheck.degree),
            SumcheckStatement::new(
                registers_claims.sumcheck.rounds,
                registers_claims.sumcheck.degree,
            ),
        ];
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency_from_narg(
            &statements,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                output_claims: &consistency.consistency.output_claims,
                proof_label: "stage5_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::InstructionReadRaf,
            })?;

        let instruction_point = consistency
            .try_instance_point(instruction_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: error.to_string(),
            })?;
        let ram_point = consistency
            .try_instance_point(ram_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            })?;
        let registers_point = consistency
            .try_instance_point(registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;

        // Map each relation's committed sumcheck point to its produced opening
        // points, the point-only counterpart of the clear `output_claims`. The
        // instruction relation ignores its inputs (empty point cells suffice); RAM
        // and registers splice the fixed address/cycle prefixes from upstream points.
        let empty = Vec::<PCS::Field>::new;
        let instruction_inputs = InstructionReadRafInputClaims {
            lookup_output: empty(),
            left_lookup_operand: empty(),
            right_lookup_operand: empty(),
        };
        let ram_inputs = RamRaClaimReductionInputClaims {
            raf: stage2.output_points.ram_raf_evaluation_point().to_vec(),
            read_write: stage2.output_points.ram_read_write_point().to_vec(),
            val_check: stage4.output_points.ram_val_check_point().to_vec(),
        };
        let registers_inputs = RegistersValEvaluationInputClaims {
            registers_val: stage4.output_points.registers_read_write_point().to_vec(),
        };
        let output_points = Stage5OutputClaims {
            instruction_read_raf: instruction_relation
                .derive_opening_points(&instruction_point, &instruction_inputs)?,
            ram_ra_claim_reduction: ram_relation.derive_opening_points(&ram_point, &ram_inputs)?,
            registers_val_evaluation: registers_relation
                .derive_opening_points(&registers_point, &registers_inputs)?,
        };
        let instruction_r_address = output_points.instruction_r_address();

        return Ok(Stage5Output::Zk(Stage5ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
            instruction_r_address,
        }));
    }

    let stage2 = stage2.clear()?;
    let stage4 = stage4.clear()?;
    let claims = &proof.clear_claims()?.stage5;
    if claims.instruction_read_raf.lookup_table_flags.len()
        != instruction_output_openings.lookup_table_flags.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "lookup table flag claim count mismatch: expected {}, got {}",
                instruction_output_openings.lookup_table_flags.len(),
                claims.instruction_read_raf.lookup_table_flags.len()
            ),
        });
    }
    if claims.instruction_read_raf.instruction_ra.len()
        != instruction_output_openings.instruction_ra.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "instruction RA claim count mismatch: expected {}, got {}",
                instruction_output_openings.instruction_ra.len(),
                claims.instruction_read_raf.instruction_ra.len()
            ),
        });
    }

    // The reduced lookup output aliases the product remainder's lookup output
    // (same opening point and value); stage 2 validates that alias, which the
    // instruction read-RAF input wiring relies on when it falls back to the
    // product remainder.
    let instruction_inputs = InstructionReadRafInputClaims::from_upstream(stage2);
    let ram_inputs = RamRaClaimReductionInputClaims::from_upstream(stage2, stage4);
    let registers_inputs = RegistersValEvaluationInputClaims::from_upstream(stage4);

    let sumcheck_claims = [
        SumcheckClaim::new(
            instruction_claims.sumcheck.rounds,
            instruction_claims.sumcheck.degree,
            instruction_relation.input_claim(&instruction_inputs)?,
        ),
        SumcheckClaim::new(
            ram_claims.sumcheck.rounds,
            ram_claims.sumcheck.degree,
            ram_relation.input_claim(&ram_inputs)?,
        ),
        SumcheckClaim::new(
            registers_claims.sumcheck.rounds,
            registers_claims.sumcheck.degree,
            registers_relation.input_claim(&registers_inputs)?,
        ),
    ];
    let batch =
        BatchedSumcheckVerifier::verify_compressed_boolean_from_narg(&sumcheck_claims, transcript)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: error.to_string(),
            })?;

    let instruction_point = batch
        .try_instance_point(instruction_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        })?;
    let ram_point = batch
        .try_instance_point(ram_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;

    let points = Stage5OutputClaims {
        instruction_read_raf: instruction_relation
            .derive_opening_points(instruction_point, &instruction_inputs)?,
        ram_ra_claim_reduction: ram_relation.derive_opening_points(ram_point, &ram_inputs)?,
        registers_val_evaluation: registers_relation
            .derive_opening_points(registers_point, &registers_inputs)?,
    };
    let output_claims = stage5_output_claims_with_points(claims, &points);

    let instruction_output = instruction_relation
        .expected_output(&instruction_inputs, &output_claims.instruction_read_raf)?;
    let ram_output =
        ram_relation.expected_output(&ram_inputs, &output_claims.ram_ra_claim_reduction)?;
    let registers_output = registers_relation
        .expected_output(&registers_inputs, &output_claims.registers_val_evaluation)?;

    let expected_final_claim = stage5_expected_final_claim(
        batch.batching_coefficients.as_slice(),
        instruction_output,
        ram_output,
        registers_output,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::InstructionReadRaf,
        });
    }

    append_stage5_opening_claims(transcript, claims);

    let instruction_r_address = output_claims.instruction_r_address();
    Ok(Stage5Output::Clear(Stage5ClearOutput {
        challenges,
        output_claims,
        instruction_r_address,
    }))
}

fn append_stage5_opening_claims<F, T>(transcript: &mut T, claims: &Stage5OutputClaims<F>)
where
    F: Field,
    T: FsTranscript<F>,
{
    for opening_claim in claims.opening_values() {
        transcript.absorb_field(&opening_claim);
    }
}
