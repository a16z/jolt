//! Stage 3 verifier: Spartan shift, instruction input, and register reduction.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::registers as registers_claim_reduction, dimensions::TraceDimensions,
        instruction, spartan,
    },
    JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::FsNargRead;

use super::{
    instruction_input::{InstructionInput, InstructionInputInputClaims},
    outputs::{
        Stage3Challenges, Stage3ClearOutput, Stage3Output, Stage3OutputClaims, Stage3ZkOutput,
    },
    registers_claim_reduction::{RegistersClaimReduction, RegistersClaimReductionInputClaims},
    spartan_shift::{SpartanShift, SpartanShiftInputClaims},
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::{
            check_relation_boolean_hypercube, zip_openings, OpeningClaim, SumcheckInstance,
        },
        stage1::Stage1Output,
        stage2::Stage2Output,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

const STAGE3_BATCH_OUTPUT_CLAIMS: usize = 13;

/// Combine the three stage 3 expected output claims with the batch's coefficients,
/// in canonical batch order (shift, instruction input, register claim reduction).
/// Shared by the verifier and the prover so the combination cannot drift.
pub fn stage3_expected_final_claim<F: Field>(
    coefficients: &[F],
    shift: F,
    instruction_input: F,
    registers_claim_reduction: F,
) -> Result<F, VerifierError> {
    let [shift_coefficient, instruction_coefficient, registers_coefficient] = coefficients else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanShift,
            reason: "Stage 3 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    Ok(*shift_coefficient * shift
        + *instruction_coefficient * instruction_input
        + *registers_coefficient * registers_claim_reduction)
}

/// Pair the produced stage-3 openings with their derived points (point + value
/// together) from the wire claim values and each relation's shared opening point.
/// Shared by the verifier and the prover so this located form is built once.
pub fn stage3_output_claims_with_points<F: Field>(
    claims: &Stage3OutputClaims<F>,
    points: &Stage3OutputClaims<Vec<F>>,
) -> Stage3OutputClaims<OpeningClaim<F>> {
    Stage3OutputClaims {
        shift: zip_openings(&claims.shift, &points.shift),
        instruction_input: zip_openings(&claims.instruction_input, &points.instruction_input),
        registers_claim_reduction: zip_openings(
            &claims.registers_claim_reduction,
            &points.registers_claim_reduction,
        ),
    }
}

pub fn verify<PCS, VC, T>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
) -> Result<Stage3Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FsNargRead<PCS::Field>,
    VC::Output: Clone + CanonicalSerialize + CanonicalDeserialize,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = TraceDimensions::new(log_t);

    let shift_spec = spartan::shift::<PCS::Field>(dimensions);
    let instruction_spec = instruction::input_virtualization::<PCS::Field>(dimensions);
    let registers_spec = registers_claim_reduction::claim_reduction::<PCS::Field>(dimensions);

    for claim in [&shift_spec, &instruction_spec, &registers_spec] {
        check_relation_boolean_hypercube(claim)?;
    }

    let shift_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    let registers_gamma = transcript.challenge_scalar();

    let challenges = Stage3Challenges {
        shift_gamma,
        instruction_gamma,
        registers_gamma,
    };

    if checked.zk {
        let statements = [
            SumcheckStatement::new(shift_spec.sumcheck.rounds, shift_spec.sumcheck.degree),
            SumcheckStatement::new(
                instruction_spec.sumcheck.rounds,
                instruction_spec.sumcheck.degree,
            ),
            SumcheckStatement::new(
                registers_spec.sumcheck.rounds,
                registers_spec.sumcheck.degree,
            ),
        ];
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency_from_narg(
            &statements,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanShift,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                output_claims: &consistency.consistency.output_claims,
                proof_label: "stage3_sumcheck_proof",
                output_claim_count: STAGE3_BATCH_OUTPUT_CLAIMS,
                stage: JoltRelationId::SpartanShift,
            })?;

        return Ok(Stage3Output::Zk(Stage3ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
        }));
    }

    let stage1 = stage1.clear()?;
    let stage2 = stage2.clear()?;
    let claims = &proof.clear_claims()?.stage3;

    let shift_relation = SpartanShift::new(
        dimensions,
        shift_gamma,
        stage2.product_uniskip.tau_low.clone(),
        stage2.output_claims.product_remainder_point().to_vec(),
    );
    let instruction_relation = InstructionInput::new(
        dimensions,
        instruction_gamma,
        stage2.output_claims.product_remainder_point().to_vec(),
    );
    let registers_relation = RegistersClaimReduction::new(
        dimensions,
        registers_gamma,
        stage2.product_uniskip.tau_low.clone(),
    );

    let shift_inputs = SpartanShiftInputClaims::from_upstream(stage1, stage2);
    let instruction_inputs = InstructionInputInputClaims::from_upstream(stage2);
    let registers_inputs = RegistersClaimReductionInputClaims::from_upstream(stage1);

    let sumcheck_claims = [
        SumcheckClaim::new(
            shift_spec.sumcheck.rounds,
            shift_spec.sumcheck.degree,
            shift_relation.input_claim(&shift_inputs)?,
        ),
        SumcheckClaim::new(
            instruction_spec.sumcheck.rounds,
            instruction_spec.sumcheck.degree,
            instruction_relation.input_claim(&instruction_inputs)?,
        ),
        SumcheckClaim::new(
            registers_spec.sumcheck.rounds,
            registers_spec.sumcheck.degree,
            registers_relation.input_claim(&registers_inputs)?,
        ),
    ];
    let batch =
        BatchedSumcheckVerifier::verify_compressed_boolean_from_narg(&sumcheck_claims, transcript)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::SpartanShift,
                reason: error.to_string(),
            })?;

    let shift_point = batch
        .try_instance_point(shift_spec.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanShift,
            reason: error.to_string(),
        })?;
    let instruction_point = batch
        .try_instance_point(instruction_spec.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionInputVirtualization,
            reason: error.to_string(),
        })?;
    let registers_point = batch
        .try_instance_point(registers_spec.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersClaimReduction,
            reason: error.to_string(),
        })?;

    let points = Stage3OutputClaims {
        shift: shift_relation.derive_opening_points(shift_point, &shift_inputs)?,
        instruction_input: instruction_relation
            .derive_opening_points(instruction_point, &instruction_inputs)?,
        registers_claim_reduction: registers_relation
            .derive_opening_points(registers_point, &registers_inputs)?,
    };
    let output_claims = stage3_output_claims_with_points(claims, &points);

    let shift_output = shift_relation.expected_output(&shift_inputs, &output_claims.shift)?;
    let instruction_output = instruction_relation
        .expected_output(&instruction_inputs, &output_claims.instruction_input)?;
    let registers_output = registers_relation
        .expected_output(&registers_inputs, &output_claims.registers_claim_reduction)?;

    let expected_final_claim = stage3_expected_final_claim(
        &batch.batching_coefficients,
        shift_output,
        instruction_output,
        registers_output,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::SpartanShift,
        });
    }

    // After the per-relation output checks (which catch any single-claim offset),
    // enforce the cross-relation opening aliases the downstream stages relied on.
    claims.validate()?;

    claims.append_to_transcript(transcript);

    Ok(Stage3Output::Clear(Stage3ClearOutput {
        challenges,
        output_claims,
    }))
}

#[cfg(test)]
mod tests {
    use crate::stages::stage3::instruction_input::InstructionInputOutputClaims;
    use crate::stages::stage3::outputs::Stage3OutputClaims;
    use crate::stages::stage3::registers_claim_reduction::RegistersClaimReductionOutputClaims;
    use crate::stages::stage3::spartan_shift::SpartanShiftOutputClaims;
    use crate::stages::test_support::RecordingTranscript;
    use jolt_field::{CanonicalBytes, Fr, FromPrimitiveInt};

    #[test]
    fn opening_claim_appends_follow_core_alias_order() {
        let claims = Stage3OutputClaims {
            shift: SpartanShiftOutputClaims {
                unexpanded_pc: Fr::from_u64(1),
                pc: Fr::from_u64(2),
                is_virtual: Fr::from_u64(3),
                is_first_in_sequence: Fr::from_u64(4),
                is_noop: Fr::from_u64(5),
            },
            instruction_input: InstructionInputOutputClaims {
                left_operand_is_rs1: Fr::from_u64(6),
                rs1_value: Fr::from_u64(7),
                left_operand_is_pc: Fr::from_u64(8),
                unexpanded_pc: Fr::from_u64(9),
                right_operand_is_rs2: Fr::from_u64(10),
                rs2_value: Fr::from_u64(11),
                right_operand_is_imm: Fr::from_u64(12),
                imm: Fr::from_u64(13),
            },
            registers_claim_reduction: RegistersClaimReductionOutputClaims {
                rd_write_value: Fr::from_u64(14),
                rs1_value: Fr::from_u64(15),
                rs2_value: Fr::from_u64(16),
            },
        };
        let mut transcript = RecordingTranscript::default();

        claims.append_to_transcript(&mut transcript);

        // Canonical order: the five shift openings, then the instruction-input
        // openings minus its aliased `unexpanded_pc`, then only `rd_write_value`
        // from the register reduction (its `rs1`/`rs2` alias the instruction ones).
        let expected_payloads = [
            claims.shift.unexpanded_pc,
            claims.shift.pc,
            claims.shift.is_virtual,
            claims.shift.is_first_in_sequence,
            claims.shift.is_noop,
            claims.instruction_input.left_operand_is_rs1,
            claims.instruction_input.rs1_value,
            claims.instruction_input.left_operand_is_pc,
            claims.instruction_input.right_operand_is_rs2,
            claims.instruction_input.rs2_value,
            claims.instruction_input.right_operand_is_imm,
            claims.instruction_input.imm,
            claims.registers_claim_reduction.rd_write_value,
        ];
        assert_eq!(claims.opening_values().len(), expected_payloads.len());
        // Like jolt-core: each opening claim is absorbed as just its value (no label).
        assert_eq!(transcript.chunks.len(), expected_payloads.len());

        for (index, expected_payload) in expected_payloads.into_iter().enumerate() {
            assert_eq!(transcript.chunks[index], scalar_bytes(expected_payload));
        }
    }

    fn scalar_bytes(value: Fr) -> Vec<u8> {
        value.to_bytes_le_vec()
    }
}
