use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
        ram,
    },
    JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::sparse_segments_mle_msb;
use jolt_program::preprocess::PublicInitialRam;
use jolt_transcript::Transcript;

use super::{
    outputs::{
        Stage4ClearOutput, Stage4InputClaims, Stage4InputPoints, Stage4Output, Stage4Sumchecks,
        Stage4ZkOutput,
    },
    ram_val_check::{
        ram_val_check_init_structure, ram_val_check_initial_evaluation,
        ram_val_check_input_points_from_upstream, ram_val_check_input_values_from_upstream,
        RamValCheck, RamValCheckInitStructure, RamValCheckInitialEvaluation,
    },
    registers_read_write_checking::{
        registers_read_write_input_points_from_upstream,
        registers_read_write_input_values_from_upstream, RegistersReadWriteChecking,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output},
        stage3::{Stage3Output, Stage3OutputClaims, Stage3OutputPoints},
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// Assemble the stage-4 consumed opening *values* from the upstream outputs into
/// the generated `Stage4InputClaims` aggregate. This is the single place the
/// stage's Outputs→Inputs dataflow is expressed: the register read-write inputs
/// come from stage 3's registers claim-reduction, and the RAM value-check inputs
/// come from stage 2's RAM `val`/`val_final` plus the reconstructed `Val_init`
/// decomposition (advice / program-image contributions).
fn stage4_input_values_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputClaims<F>,
    stage3: &Stage3OutputClaims<F>,
    ram_val_check_init: &RamValCheckInitialEvaluation<F>,
) -> Stage4InputClaims<F> {
    Stage4InputClaims {
        registers_read_write: registers_read_write_input_values_from_upstream(stage3),
        ram_val_check: ram_val_check_input_values_from_upstream(stage2, ram_val_check_init),
    }
}

/// Assemble the stage-4 consumed opening *points* from the upstream output-points
/// aggregates and the pre-branch init structure. ZK-agnostic: both the clear and
/// ZK upstream outputs expose these, so the same wiring builds the input points in
/// either mode.
fn stage4_input_points_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputPoints<F>,
    stage3: &Stage3OutputPoints<F>,
    structure: &RamValCheckInitStructure<F>,
) -> Stage4InputPoints<F> {
    Stage4InputPoints {
        registers_read_write: registers_read_write_input_points_from_upstream(stage3),
        ram_val_check: ram_val_check_input_points_from_upstream(stage2, structure),
    }
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
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

    let ram_read_write_opening_point = stage2.batch_output_points().ram_read_write_point();
    let ram_output_check_opening_point = stage2.batch_output_points().ram_output_check_point();
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
    let (r_address, _r_cycle) = ram_read_write_opening_point.split_at(log_k);
    if ram_output_check_opening_point != r_address {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamValCheck,
            left: ram::ram_val(),
            right: ram::ram_val_final(),
        });
    }

    let ram_val_check_public_eval =
        public_initial_ram_evaluation(checked, preprocessing, r_address)?;
    // The mode-agnostic init structure (public eval + contribution selectors and
    // staged points); the clear arm attaches the claimed opening values below. Its
    // decomposition must stay in lockstep with the prover's and BlindFold's.
    let init_structure = ram_val_check_init_structure(
        checked,
        proof.untrusted_advice_commitment.is_some(),
        r_address,
        ram_val_check_public_eval,
    )?;

    let sumchecks = Stage4Sumchecks {
        registers_read_write: RegistersReadWriteChecking::new(register_dimensions),
        ram_val_check: RamValCheck::new(trace_dimensions, log_k, init_structure.decomposition()),
    };

    // Draw the batching gammas in declaration order: the registers gamma (a single
    // `challenge_scalar`), then the RAM value-check gamma behind its
    // `b"ram_val_check_gamma"` domain separator (the relation's `draw_challenges`
    // override replays the separator at its exact transcript position).
    let challenges = sumchecks.draw_challenges(transcript)?;

    if checked.zk {
        let consistency = sumchecks.verify_zk(&proof.stages.stage4_sumcheck_proof, transcript)?;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage4_sumcheck_proof,
            "stage4_sumcheck_proof",
            sumchecks.output_claim_count(),
            JoltRelationId::RegistersReadWriteChecking,
        )?;

        // Built via the same wiring as the clear path, off the ZK-agnostic upstream
        // output points and init structure. Advice / program-image openings live in
        // BlindFold for ZK proofs, so `derive_opening_points` leaves those leaves
        // absent in the produced points.
        let input_points = stage4_input_points_from_upstream(
            stage2.batch_output_points(),
            stage3.output_points(),
            &init_structure,
        );
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

        return Ok(Stage4Output::Zk(Stage4ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            ram_val_check_public_eval,
            output_points,
        }));
    }

    let stage2 = stage2.clear()?;
    let stage3 = stage3.clear()?;
    let claims = &proof.clear_claims()?.stage4;
    sumchecks.validate_output_claims(claims)?;
    // Attaches the claimed advice / program-image opening values (consumed by the
    // input wiring and carried downstream for the stage-6/7 address-phase
    // reductions); presence against the init structure is validated by the
    // generated `validate_output_claims` above and re-checked here.
    let ram_val_check_init = ram_val_check_initial_evaluation(&init_structure, claims)?;

    let input_values = stage4_input_values_from_upstream(
        &stage2.output_values,
        &stage3.output_values,
        &ram_val_check_init,
    );
    let input_points = stage4_input_points_from_upstream(
        &stage2.output_points,
        &stage3.output_points,
        &init_structure,
    );

    let batch = sumchecks.verify_clear(
        &input_values,
        &challenges,
        &proof.stages.stage4_sumcheck_proof,
        transcript,
    )?;

    let output_points =
        sumchecks.derive_opening_points(batch.reduction.point.as_slice(), &input_points)?;

    let expected_final_claim = sumchecks.expected_final_claim(
        &batch.coefficients,
        &input_points,
        claims,
        &output_points,
        &challenges,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 4 });
    }

    claims.append_to_transcript(transcript);

    Ok(Stage4Output::Clear(Stage4ClearOutput {
        output_values: claims.clone(),
        output_points,
        ram_val_check_init,
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
        let end = segment.start_index + segment.words.len() as u128;
        if end > checked.ram_K as u128 {
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
