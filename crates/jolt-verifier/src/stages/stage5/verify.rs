#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
use jolt_claims::protocols::jolt::{geometry::dimensions::JoltFormulaDimensions, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

#[cfg(feature = "field-inline")]
use super::field_registers_val_evaluation::{
    field_registers_val_evaluation_input_points_from_upstream,
    field_registers_val_evaluation_input_values_from_upstream, FieldRegistersValEvaluation,
};
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
        stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output},
        stage4::{Stage4Output, Stage4OutputClaims, Stage4OutputPoints},
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

pub fn stage5_input_values_from_upstream<F: JoltField>(
    stage2: &Stage2BatchOutputClaims<F>,
    stage4: &Stage4OutputClaims<F>,
) -> Stage5InputClaims<F> {
    Stage5InputClaims {
        instruction_read_raf: instruction_read_raf_input_values_from_upstream(stage2),
        ram_ra_claim_reduction: ram_ra_claim_reduction_input_values_from_upstream(stage2, stage4),
        registers_val_evaluation: registers_val_evaluation_input_values_from_upstream(stage4),
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation: field_registers_val_evaluation_input_values_from_upstream(
            stage4,
        ),
    }
}

pub fn stage5_input_points_from_upstream<F: JoltField>(
    stage2: &Stage2BatchOutputPoints<F>,
    stage4: &Stage4OutputPoints<F>,
) -> Stage5InputPoints<F> {
    Stage5InputPoints {
        instruction_read_raf: instruction_read_raf_input_points_from_upstream(stage2),
        ram_ra_claim_reduction: ram_ra_claim_reduction_input_points_from_upstream(stage2, stage4),
        registers_val_evaluation: registers_val_evaluation_input_points_from_upstream(stage4),
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation: field_registers_val_evaluation_input_points_from_upstream(
            stage4,
        ),
    }
}

#[jolt_verifier_derive::fs_scope(Stage5)]
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
    let log_k = crate::num::ilog2(checked.ram_K);
    let trace_dimensions = formula_dimensions.trace;

    let sumchecks = Stage5Sumchecks {
        instruction_read_raf: InstructionReadRaf::new(formula_dimensions.instruction_read_raf),
        ram_ra_claim_reduction: RamRaClaimReduction::new(trace_dimensions, log_k),
        registers_val_evaluation: RegistersValEvaluation::new(trace_dimensions),
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation: FieldRegistersValEvaluation::new(
            FieldRegistersTraceDimensions::new(trace_dimensions.log_t()),
        ),
    };

    let challenges = sumchecks.draw_challenges(transcript)?;

    if !checked.zk {
        let claims = &proof.clear_claims()?.stage5;
        let stage2 = stage2.clear()?;
        let stage4 = stage4.clear()?;
        sumchecks.validate_output_claims(claims)?;

        let input_values =
            stage5_input_values_from_upstream(&stage2.output_values, &stage4.output_values);
        let input_points =
            stage5_input_points_from_upstream(&stage2.output_points, &stage4.output_points);

        let output_points = sumchecks.verify_clear(
            &input_values,
            &input_points,
            &challenges,
            claims,
            &proof.stages.stage5_sumcheck_proof,
            transcript,
            5,
        )?;

        sumchecks.append_output_claims(transcript, claims);

        let instruction_r_address = output_points.instruction_r_address();
        return Ok(Stage5Output::Clear(Stage5ClearOutput {
            challenges,
            output_values: claims.clone(),
            output_points,
            instruction_r_address,
        }));
    }

    {
        let stage2 = stage2.zk()?;
        let stage4 = stage4.zk()?;
        let consistency = sumchecks.verify_zk(&proof.stages.stage5_sumcheck_proof, transcript)?;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage5_sumcheck_proof,
            "stage5_sumcheck_proof",
            sumchecks.output_claim_count(),
            JoltRelationId::InstructionReadRaf,
        )?;

        let input_points =
            stage5_input_points_from_upstream(&stage2.output_points, &stage4.output_points);
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;
        let instruction_r_address = output_points.instruction_r_address();

        Ok(Stage5Output::Zk(Stage5ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
            instruction_r_address,
        }))
    }
}
