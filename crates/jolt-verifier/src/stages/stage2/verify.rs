use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::TraceDimensions, ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    },
    JoltRelationId,
};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::PublicIoMemory;
use jolt_transcript::Transcript;

use super::{
    instruction_claim_reduction::{
        instruction_claim_reduction_input_values_from_upstream, InstructionClaimReduction,
    },
    outputs::{
        Stage2BatchInputClaims, Stage2BatchSumchecks, Stage2ClearOutput, Stage2Output,
        Stage2ZkOutput,
    },
    product_remainder::{product_remainder_input_values_from_uniskip_output, ProductRemainder},
    product_uniskip::{product_uniskip_input_values_from_stage1, ProductUniskip},
    ram_output_check::{RamOutputCheck, RamOutputCheckInputClaims},
    ram_raf_evaluation::{ram_raf_evaluation_input_values_from_upstream, RamRafEvaluation},
    ram_read_write_checking::{ram_read_write_input_values_from_upstream, RamReadWriteChecking},
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::ConcreteSumcheck,
        stage1::{Stage1ClearOutput, Stage1Output},
        uniskip,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// Assemble the stage-2 batch consumed opening *values* from the upstream clear
/// outputs into the generated `Stage2BatchInputClaims` aggregate. Each per-relation
/// `*_from_upstream` helper wires which upstream opening feeds which downstream
/// input. The product-remainder input is the product uni-skip's output claim (a
/// separate stage-2 sub-sumcheck), not an upstream stage's opening.
fn stage2_batch_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    product_uniskip_output_claim: F,
) -> Stage2BatchInputClaims<F> {
    Stage2BatchInputClaims {
        ram_read_write: ram_read_write_input_values_from_upstream(stage1),
        product_remainder: product_remainder_input_values_from_uniskip_output(
            product_uniskip_output_claim,
        ),
        instruction_claim_reduction: instruction_claim_reduction_input_values_from_upstream(stage1),
        ram_raf_evaluation: ram_raf_evaluation_input_values_from_upstream(stage1),
        ram_output_check: RamOutputCheckInputClaims::default(),
    }
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
) -> Result<Stage2Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, stage1) {
        (true, Stage1Output::Clear(_)) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage1" });
        }
        (false, Stage1Output::Zk(_)) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage1" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = proof.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;
    let lowest_address = checked.public_io.memory_layout.get_lowest_address();
    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamOutputCheck,
            reason: error.to_string(),
        }
    })?;

    // The stage-1 remainder tail, reversed, feeds the low tau bindings — read
    // through the mode-agnostic accessor so one derivation serves both paths.
    let stage1_remainder_point = stage1.remainder_point();
    let mut tau_low = stage1_remainder_point
        .get(1..)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanProductVirtualization,
            reason: "Stage 1 remainder challenge vector is empty".to_string(),
        })?
        .to_vec();
    if tau_low.len() != log_t {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::SpartanProductVirtualization,
            reason: format!(
                "Stage 1 remainder challenge tail length mismatch: expected {log_t}, got {}",
                tau_low.len()
            ),
        });
    }
    tau_low.reverse();
    let tau_high = transcript.challenge();

    if !checked.zk {
        let claims = &proof.clear_claims()?.stage2;
        let stage1 = stage1.clear()?;
        let uniskip_params = uniskip::UniskipParams::spartan_product();
        let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
        let uniskip_input_values = product_uniskip_input_values_from_stage1(stage1);
        let uniskip_input_claim =
            uniskip_relation.input_claim(&uniskip_input_values, &NoChallenges::default())?;
        let uniskip_challenge = uniskip::verify_clear(
            &proof.stages.stage2_uni_skip_first_round_proof,
            &uniskip_params,
            uniskip_input_claim,
            claims.product_uniskip_output_claim,
            transcript,
        )?;

        // Build the five batch relations; each owns its input/output claim algebra
        // (single-sourced with its jolt-claims formula and the BlindFold constraint).
        // The product uni-skip stays hand-coded above. The output-check address
        // reference point is drawn AFTER the batch gammas, so its instance starts
        // with a placeholder and is completed right after the draws below (see
        // `set_output_address_challenges`).
        let mut sumchecks = Stage2BatchSumchecks {
            ram_read_write: RamReadWriteChecking::new(
                read_write_dimensions,
                log_k,
                tau_low.clone(),
            ),
            product_remainder: ProductRemainder::new(
                product_dimensions,
                uniskip_challenge,
                tau_high,
                tau_low.clone(),
            ),
            instruction_claim_reduction: InstructionClaimReduction::new(
                trace_dimensions,
                tau_low.clone(),
            ),
            ram_raf_evaluation: RamRafEvaluation::new(
                read_write_dimensions,
                raf_dimensions,
                log_k,
                lowest_address,
                tau_low.clone(),
            ),
            ram_output_check: RamOutputCheck::new(
                read_write_dimensions,
                Vec::new(),
                public_memory.clone(),
            ),
        };
        let challenges = sumchecks.draw_challenges(transcript)?;

        // The RAM output-check address reference point, drawn after both gammas. MUST
        // stay `challenge()` (not `challenge_scalar()`): both decode the same 16-byte
        // squeeze, but differently, so switching would silently change the address
        // point values without changing the transcript bytes.
        let output_address_challenges = (0..log_k)
            .map(|_| transcript.challenge())
            .collect::<Vec<_>>();
        sumchecks
            .ram_output_check
            .set_output_address_challenges(output_address_challenges);

        let input_points = sumchecks.empty_input_points();
        sumchecks.validate_output_claims(&claims.batch_outputs)?;

        let input_values =
            stage2_batch_input_values_from_upstream(stage1, claims.product_uniskip_output_claim);

        let output_points = sumchecks.run_clear(
            &input_values,
            &input_points,
            &challenges,
            &claims.batch_outputs,
            &proof.stages.stage2_sumcheck_proof,
            transcript,
            2,
        )?;

        sumchecks.append_output_claims(transcript, &claims.batch_outputs);

        return Ok(Stage2Output::Clear(Stage2ClearOutput {
            output_values: claims.batch_outputs.clone(),
            output_points,
            product_tau_low: tau_low,
        }));
    }

    let product_uniskip = uniskip::verify_zk(
        checked,
        &proof.stages.stage2_uni_skip_first_round_proof,
        &uniskip::UniskipParams::spartan_product(),
        transcript,
    )?;

    let mut sumchecks = Stage2BatchSumchecks {
        ram_read_write: RamReadWriteChecking::new(read_write_dimensions, log_k, tau_low.clone()),
        product_remainder: ProductRemainder::new(
            product_dimensions,
            product_uniskip.challenge,
            tau_high,
            tau_low.clone(),
        ),
        instruction_claim_reduction: InstructionClaimReduction::new(
            trace_dimensions,
            tau_low.clone(),
        ),
        ram_raf_evaluation: RamRafEvaluation::new(
            read_write_dimensions,
            raf_dimensions,
            log_k,
            lowest_address,
            tau_low.clone(),
        ),
        ram_output_check: RamOutputCheck::new(read_write_dimensions, Vec::new(), public_memory),
    };
    // Draw each relation's batching gamma in declaration order: the RAM read-write
    // gamma, then the instruction claim-reduction gamma (each a single
    // `challenge_scalar`; the other three batch relations draw nothing —
    // `NoChallenges`).
    let challenges = sumchecks.draw_challenges(transcript)?;

    // The RAM output-check address reference point, drawn after both gammas. MUST
    // stay `challenge()` (not `challenge_scalar()`).
    let output_address_challenges = (0..log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    sumchecks
        .ram_output_check
        .set_output_address_challenges(output_address_challenges.clone());

    let input_points = sumchecks.empty_input_points();

    let consistency = sumchecks.verify_zk(&proof.stages.stage2_sumcheck_proof, transcript)?;
    let batch_output_claims = committed::verify_output_claim_commitments(
        checked,
        &proof.stages.stage2_sumcheck_proof,
        "stage2_sumcheck_proof",
        sumchecks.output_claim_count(),
        JoltRelationId::RamReadWriteChecking,
    )?;
    let output_points =
        sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

    Ok(Stage2Output::Zk(Stage2ZkOutput {
        challenges,
        product_uniskip_challenge: product_uniskip.challenge,
        product_tau_low: tau_low,
        product_tau_high: tau_high,
        output_address_challenges,
        product_uniskip_consistency: product_uniskip.consistency,
        product_uniskip_output_claims: product_uniskip.output_claims,
        batch_consistency: consistency,
        batch_output_claims,
        output_points,
    }))
}
