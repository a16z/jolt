use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::registers as registers_claim_reduction,
        instruction,
        spartan::{self, shift_input_openings, shift_output_openings},
    },
    InstructionInputChallenge, JoltChallengeId, JoltPublicId, JoltRelationId, JoltSumcheckDomain,
    RegistersClaimReductionChallenge, SpartanShiftChallenge, SpartanShiftPublic, TraceDimensions,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{try_eq_mle, EqPlusOnePolynomial};
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::Transcript;

use super::{
    inputs::{Deps, Stage3Claims},
    outputs::{
        Stage3ClearOutput, Stage3Output, Stage3PublicOutput, Stage3ZkOutput, VerifiedStage3Batch,
        VerifiedStage3Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, stages::zk::committed,
    verifier::CheckedInputs, VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage3BatchInputClaims<F: Field> {
    shift: F,
    instruction_input: F,
    registers_claim_reduction: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage3BatchExpectedOutputClaims<F: Field> {
    shift: F,
    instruction_input: F,
    registers_claim_reduction: F,
}

const STAGE3_BATCH_OUTPUT_CLAIMS: usize = 13;

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage3Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage2" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage2" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = TraceDimensions::new(log_t);

    let shift_claims = spartan::shift::<PCS::Field>(dimensions);
    let instruction_claims = instruction::input_virtualization::<PCS::Field>(dimensions);
    let registers_claims = registers_claim_reduction::claim_reduction::<PCS::Field>(dimensions);

    for claim in [&shift_claims, &instruction_claims, &registers_claims] {
        if claim.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: claim.id,
                degree: claim.sumcheck.degree,
            });
        }
        if !matches!(claim.sumcheck.domain, JoltSumcheckDomain::BooleanHypercube) {
            return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
                stage: claim.id,
            });
        }
    }

    let shift_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    let registers_gamma = transcript.challenge_scalar();

    let public =
        |challenges: Vec<PCS::Field>, batching_coefficients: Vec<PCS::Field>| Stage3PublicOutput {
            challenges,
            batching_coefficients,
            shift_gamma,
            instruction_gamma,
            registers_gamma,
        };

    if checked.zk {
        let statements = [
            SumcheckStatement::new(shift_claims.sumcheck.rounds, shift_claims.sumcheck.degree),
            SumcheckStatement::new(
                instruction_claims.sumcheck.rounds,
                instruction_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                registers_claims.sumcheck.rounds,
                registers_claims.sumcheck.degree,
            ),
        ];
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage3_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanShift,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage3_sumcheck_proof,
                proof_label: "stage3_sumcheck_proof",
                output_claim_count: STAGE3_BATCH_OUTPUT_CLAIMS,
                stage: JoltRelationId::SpartanShift,
            })?;

        return Ok(Stage3Output::Zk(Stage3ZkOutput {
            public: public(
                consistency.challenges(),
                consistency.batching_coefficients.clone(),
            ),
            batch_consistency: consistency,
            batch_output_claims,
        }));
    }

    let Deps::Clear { stage1, stage2 } = deps else {
        return Err(VerifierError::ExpectedClearProof { field: "stage2" });
    };
    let claims = &proof.clear_claims()?.stage3;

    let [(left_reduced, left_product), (right_reduced, right_product)] =
        instruction::input_virtualization_consistency_openings();
    if stage2.batch.product_remainder.opening_point
        != stage2.batch.instruction_claim_reduction.opening_point
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionInputVirtualization,
            left: left_reduced,
            right: left_product,
        });
    }
    let product_left = stage2
        .output_claims
        .product_remainder
        .left_instruction_input;
    let product_right = stage2
        .output_claims
        .product_remainder
        .right_instruction_input;
    let reduced_left = stage2
        .output_claims
        .instruction_claim_reduction
        .left_instruction_input
        .unwrap_or(product_left);
    let reduced_right = stage2
        .output_claims
        .instruction_claim_reduction
        .right_instruction_input
        .unwrap_or(product_right);
    if reduced_left != product_left {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionInputVirtualization,
            left: left_reduced,
            right: left_product,
        });
    }
    if reduced_right != product_right {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionInputVirtualization,
            left: right_reduced,
            right: right_product,
        });
    }

    let [next_unexpanded_pc, next_pc, next_is_virtual, next_is_first_in_sequence, next_is_noop] =
        shift_input_openings();
    let [right_instruction_input_product, left_instruction_input_product] =
        instruction::input_virtualization_input_openings();
    let [rd_write_value_spartan, rs1_value_spartan, rs2_value_spartan] =
        registers_claim_reduction::claim_reduction_input_openings();

    let input_claims = Stage3BatchInputClaims {
        shift: shift_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == next_unexpanded_pc => Ok(stage1.outer.next_unexpanded_pc),
                id if id == next_pc => Ok(stage1.outer.next_pc),
                id if id == next_is_virtual => Ok(stage1.outer.next_is_virtual),
                id if id == next_is_first_in_sequence => Ok(stage1.outer.next_is_first_in_sequence),
                id if id == next_is_noop => Ok(stage2.output_claims.product_remainder.next_is_noop),
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => Ok(shift_gamma),
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        instruction_input: instruction_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == right_instruction_input_product => Ok(product_right),
                id if id == left_instruction_input_product => Ok(product_left),
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => {
                    Ok(instruction_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        registers_claim_reduction: registers_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == rd_write_value_spartan => Ok(stage1.outer.rd_write_value),
                id if id == rs1_value_spartan => Ok(stage1.outer.rs1_value),
                id if id == rs2_value_spartan => Ok(stage1.outer.rs2_value),
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::RegistersClaimReduction(
                    RegistersClaimReductionChallenge::Gamma,
                ) => Ok(registers_gamma),
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
    };

    let sumcheck_claims = [
        SumcheckClaim::new(
            shift_claims.sumcheck.rounds,
            shift_claims.sumcheck.degree,
            input_claims.shift,
        ),
        SumcheckClaim::new(
            instruction_claims.sumcheck.rounds,
            instruction_claims.sumcheck.degree,
            input_claims.instruction_input,
        ),
        SumcheckClaim::new(
            registers_claims.sumcheck.rounds,
            registers_claims.sumcheck.degree,
            input_claims.registers_claim_reduction,
        ),
    ];
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage3_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::SpartanShift,
        reason: error.to_string(),
    })?;

    let shift_point = batch
        .try_instance_point(shift_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanShift,
            reason: error.to_string(),
        })?;
    let shift_opening_point = shift_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_plus_one_outer = EqPlusOnePolynomial::new(stage2.product_uniskip.tau_low.clone())
        .evaluate(&shift_opening_point);
    let eq_plus_one_product =
        EqPlusOnePolynomial::new(stage2.batch.product_remainder.opening_point.clone())
            .evaluate(&shift_opening_point);
    let [unexpanded_pc_shift, pc_shift, is_virtual_shift, is_first_in_sequence_shift, is_noop_shift] =
        shift_output_openings();
    let shift_output = shift_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == unexpanded_pc_shift => Ok(claims.shift.unexpanded_pc),
            id if id == pc_shift => Ok(claims.shift.pc),
            id if id == is_virtual_shift => Ok(claims.shift.is_virtual),
            id if id == is_first_in_sequence_shift => Ok(claims.shift.is_first_in_sequence),
            id if id == is_noop_shift => Ok(claims.shift.is_noop),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => Ok(shift_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::SpartanShift(SpartanShiftPublic::EqPlusOneOuter) => Ok(eq_plus_one_outer),
            JoltPublicId::SpartanShift(SpartanShiftPublic::EqPlusOneProduct) => {
                Ok(eq_plus_one_product)
            }
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    let instruction_point = batch
        .try_instance_point(instruction_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionInputVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_product = try_eq_mle(
        &instruction_opening_point,
        &stage2.batch.product_remainder.opening_point,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionInputVirtualization,
        reason: error.to_string(),
    })?;
    let [right_operand_is_rs2, rs2_value_input, right_operand_is_imm, imm_input, left_operand_is_rs1, rs1_value_input, left_operand_is_pc, unexpanded_pc_input] =
        instruction::input_virtualization_output_openings();
    let instruction_output = instruction_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == left_operand_is_rs1 => Ok(claims.instruction_input.left_operand_is_rs1),
            id if id == rs1_value_input => Ok(claims.instruction_input.rs1_value),
            id if id == left_operand_is_pc => Ok(claims.instruction_input.left_operand_is_pc),
            id if id == unexpanded_pc_input => Ok(claims.instruction_input.unexpanded_pc),
            id if id == right_operand_is_rs2 => Ok(claims.instruction_input.right_operand_is_rs2),
            id if id == rs2_value_input => Ok(claims.instruction_input.rs2_value),
            id if id == right_operand_is_imm => Ok(claims.instruction_input.right_operand_is_imm),
            id if id == imm_input => Ok(claims.instruction_input.imm),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => {
                Ok(instruction_gamma)
            }
            JoltChallengeId::InstructionInput(InstructionInputChallenge::EqProduct) => {
                Ok(eq_product)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersClaimReduction,
            reason: error.to_string(),
        })?;
    let registers_opening_point = registers_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_spartan = try_eq_mle(&registers_opening_point, &stage2.product_uniskip.tau_low)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersClaimReduction,
            reason: error.to_string(),
        })?;
    let [rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced] =
        registers_claim_reduction::claim_reduction_output_openings();
    let registers_output = registers_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == rd_write_value_reduced => {
                Ok(claims.registers_claim_reduction.rd_write_value)
            }
            id if id == rs1_value_reduced => Ok(claims.registers_claim_reduction.rs1_value),
            id if id == rs2_value_reduced => Ok(claims.registers_claim_reduction.rs2_value),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::RegistersClaimReduction(RegistersClaimReductionChallenge::Gamma) => {
                Ok(registers_gamma)
            }
            JoltChallengeId::RegistersClaimReduction(
                RegistersClaimReductionChallenge::EqSpartan,
            ) => Ok(eq_spartan),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    let expected_outputs = Stage3BatchExpectedOutputClaims {
        shift: shift_output,
        instruction_input: instruction_output,
        registers_claim_reduction: registers_output,
    };
    let [shift_coefficient, instruction_coefficient, registers_coefficient] =
        batch.batching_coefficients.as_slice()
    else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanShift,
            reason: "Stage 3 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    let expected_final_claim = *shift_coefficient * expected_outputs.shift
        + *instruction_coefficient * expected_outputs.instruction_input
        + *registers_coefficient * expected_outputs.registers_claim_reduction;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::SpartanShift,
        });
    }

    append_stage3_opening_claims(transcript, claims);

    Ok(Stage3Output::Clear(Stage3ClearOutput {
        public: public(
            batch.reduction.point.as_slice().to_vec(),
            batch.batching_coefficients.clone(),
        ),
        output_claims: claims.clone(),
        batch: VerifiedStage3Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: batch.reduction.point.as_slice().to_vec(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            shift: VerifiedStage3Sumcheck {
                input_claim: input_claims.shift,
                sumcheck_point: shift_point.to_vec(),
                opening_point: shift_opening_point,
                expected_output_claim: expected_outputs.shift,
            },
            instruction_input: VerifiedStage3Sumcheck {
                input_claim: input_claims.instruction_input,
                sumcheck_point: instruction_point.to_vec(),
                opening_point: instruction_opening_point,
                expected_output_claim: expected_outputs.instruction_input,
            },
            registers_claim_reduction: VerifiedStage3Sumcheck {
                input_claim: input_claims.registers_claim_reduction,
                sumcheck_point: registers_point.to_vec(),
                opening_point: registers_opening_point,
                expected_output_claim: expected_outputs.registers_claim_reduction,
            },
        },
    }))
}

fn append_stage3_opening_claims<F, T>(transcript: &mut T, claims: &Stage3Claims<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append_labeled(b"opening_claim", &claims.shift.unexpanded_pc);
    transcript.append_labeled(b"opening_claim", &claims.shift.pc);
    transcript.append_labeled(b"opening_claim", &claims.shift.is_virtual);
    transcript.append_labeled(b"opening_claim", &claims.shift.is_first_in_sequence);
    transcript.append_labeled(b"opening_claim", &claims.shift.is_noop);
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.left_operand_is_rs1,
    );
    transcript.append_labeled(b"opening_claim", &claims.instruction_input.rs1_value);
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.left_operand_is_pc,
    );
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.right_operand_is_rs2,
    );
    transcript.append_labeled(b"opening_claim", &claims.instruction_input.rs2_value);
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.right_operand_is_imm,
    );
    transcript.append_labeled(b"opening_claim", &claims.instruction_input.imm);
    transcript.append_labeled(
        b"opening_claim",
        &claims.registers_claim_reduction.rd_write_value,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stages::stage3::inputs::{
        InstructionInputOutputOpeningClaims, RegistersClaimReductionOutputOpeningClaims,
        SpartanShiftOutputOpeningClaims,
    };
    use jolt_field::{CanonicalBytes, FixedByteSize, Fr, FromPrimitiveInt};
    use jolt_transcript::Transcript;

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

        fn state(&self) -> &[u8; 32] {
            &self.state
        }
    }

    #[test]
    fn opening_claim_appends_follow_core_alias_order() {
        let claims = Stage3Claims {
            shift: SpartanShiftOutputOpeningClaims {
                unexpanded_pc: Fr::from_u64(1),
                pc: Fr::from_u64(2),
                is_virtual: Fr::from_u64(3),
                is_first_in_sequence: Fr::from_u64(4),
                is_noop: Fr::from_u64(5),
            },
            instruction_input: InstructionInputOutputOpeningClaims {
                left_operand_is_rs1: Fr::from_u64(6),
                rs1_value: Fr::from_u64(7),
                left_operand_is_pc: Fr::from_u64(8),
                unexpanded_pc: Fr::from_u64(9),
                right_operand_is_rs2: Fr::from_u64(10),
                rs2_value: Fr::from_u64(11),
                right_operand_is_imm: Fr::from_u64(12),
                imm: Fr::from_u64(13),
            },
            registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims {
                rd_write_value: Fr::from_u64(14),
                rs1_value: Fr::from_u64(15),
                rs2_value: Fr::from_u64(16),
            },
        };
        let mut transcript = RecordingTranscript::new(b"stage3-openings");

        append_stage3_opening_claims(&mut transcript, &claims);

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
        assert_eq!(transcript.chunks.len(), expected_payloads.len() * 2);

        let label = opening_claim_label();
        for (index, expected_payload) in expected_payloads.into_iter().enumerate() {
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
