use jolt_claims::protocols::jolt::{
    geometry::{dimensions::JoltFormulaDimensions, instruction},
    relations, JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;
use num_traits::Zero;

use super::{
    instruction_read_raf::{
        instruction_read_raf_input_points_from_upstream,
        instruction_read_raf_input_values_from_upstream, InstructionReadRaf,
    },
    outputs::{
        Stage5ClearOutput, Stage5InputClaims, Stage5InputPoints, Stage5Output, Stage5Sumchecks,
        Stage5ZkOutput,
    },
    ram_ra_claim_reduction::{
        ram_ra_claim_reduction_input_points_from_upstream,
        ram_ra_claim_reduction_input_values_from_upstream, RamRaClaimReduction,
    },
    registers_val_evaluation::{
        registers_val_evaluation_input_points_from_upstream,
        registers_val_evaluation_input_values_from_upstream, RegistersValEvaluation,
    },
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::{ConcreteSumcheck, OutputClaims},
        stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output},
        stage4::{Stage4Output, Stage4OutputClaims, Stage4OutputPoints},
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// Assemble the stage-5 consumed openings from the upstream clear outputs into the
/// generated `Stage5InputClaims` aggregate. This is the single place the stage's
/// Outputs→Inputs dataflow is expressed: each per-relation `*_from_upstream` helper
/// wires which upstream opening feeds which downstream input.
fn stage5_input_values_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputClaims<F>,
    stage4: &Stage4OutputClaims<F>,
) -> Stage5InputClaims<F> {
    Stage5InputClaims {
        instruction_read_raf: instruction_read_raf_input_values_from_upstream(stage2),
        ram_ra_claim_reduction: ram_ra_claim_reduction_input_values_from_upstream(stage2, stage4),
        registers_val_evaluation: registers_val_evaluation_input_values_from_upstream(stage4),
    }
}

/// Assemble the stage-5 consumed opening *points* from the upstream output-points
/// aggregates. ZK-agnostic: both the clear and ZK stage-2/stage-4 outputs expose
/// these, so the same wiring builds the input points in either mode.
fn stage5_input_points_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputPoints<F>,
    stage4: &Stage4OutputPoints<F>,
) -> Stage5InputPoints<F> {
    Stage5InputPoints {
        instruction_read_raf: instruction_read_raf_input_points_from_upstream(stage2),
        ram_ra_claim_reduction: ram_ra_claim_reduction_input_points_from_upstream(stage2, stage4),
        registers_val_evaluation: registers_val_evaluation_input_points_from_upstream(stage4),
    }
}

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
    T: Transcript<Challenge = PCS::Field>,
{
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = formula_dimensions.trace;

    let sumchecks = Stage5Sumchecks {
        instruction_read_raf: InstructionReadRaf::new(formula_dimensions.instruction_read_raf),
        ram_ra_claim_reduction: RamRaClaimReduction::new(trace_dimensions, log_k),
        registers_val_evaluation: RegistersValEvaluation::new(trace_dimensions),
    };

    // Draw each relation's batching gamma in declaration order (instruction, then
    // RAM); registers draws nothing. The drawn challenges feed the input/output
    // claims and populate the stage aggregate carried downstream.
    let challenges = sumchecks.draw_challenges(transcript)?;

    let instruction_output_openings =
        instruction::read_raf_output_openings(formula_dimensions.instruction_read_raf);
    let committed_output_claims = instruction_output_openings.opening_count()
        + relations::ram::RamRaClaimReductionOutputClaims::<PCS::Field> {
            ram_ra: PCS::Field::zero(),
        }
        .canonical_order()
        .len()
        + relations::registers::RegistersValEvaluationOutputClaims::<PCS::Field> {
            rd_inc: PCS::Field::zero(),
            rd_wa: PCS::Field::zero(),
        }
        .canonical_order()
        .len();

    if checked.zk {
        let stage2 = stage2.zk()?;
        let stage4 = stage4.zk()?;
        let consistency = sumchecks.verify_zk(&proof.stages.stage5_sumcheck_proof, transcript)?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage5_sumcheck_proof,
                proof_label: "stage5_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::InstructionReadRaf,
            })?;

        // Built via the same wiring as the clear path, off the ZK-agnostic upstream
        // output points; `derive_opening_points` ignores the instruction relation's
        // input points, so carrying the real claim-reduction point here is harmless.
        let input_points =
            stage5_input_points_from_upstream(&stage2.output_points, &stage4.output_points);
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;
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
    let input_values =
        stage5_input_values_from_upstream(&stage2.output_values, &stage4.output_values);
    let input_points =
        stage5_input_points_from_upstream(&stage2.output_points, &stage4.output_points);

    let batch = sumchecks.verify_clear(
        &input_values,
        &challenges,
        &proof.stages.stage5_sumcheck_proof,
        transcript,
    )?;

    let output_points =
        sumchecks.derive_opening_points(batch.reduction.point.as_slice(), &input_points)?;

    let instruction_output = sumchecks.instruction_read_raf.expected_output(
        &input_points.instruction_read_raf,
        &claims.instruction_read_raf,
        &output_points.instruction_read_raf,
        &challenges.instruction_read_raf,
    )?;
    let ram_output = sumchecks.ram_ra_claim_reduction.expected_output(
        &input_points.ram_ra_claim_reduction,
        &claims.ram_ra_claim_reduction,
        &output_points.ram_ra_claim_reduction,
        &challenges.ram_ra_claim_reduction,
    )?;
    let registers_output = sumchecks.registers_val_evaluation.expected_output(
        &input_points.registers_val_evaluation,
        &claims.registers_val_evaluation,
        &output_points.registers_val_evaluation,
        &challenges.registers_val_evaluation,
    )?;

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

    claims.append_to_transcript(transcript);

    let instruction_r_address = output_points.instruction_r_address();
    Ok(Stage5Output::Clear(Stage5ClearOutput {
        challenges,
        output_values: claims.clone(),
        output_points,
        instruction_r_address,
    }))
}
