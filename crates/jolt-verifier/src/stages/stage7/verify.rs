use jolt_claims::protocols::jolt::{
    geometry::{
        claim_reductions::{
            advice,
            bytecode::{self as bytecode_reduction},
            hamming_weight, program_image,
        },
        dimensions::JoltFormulaDimensions,
    },
    JoltAdviceKind, JoltOpeningId, JoltRelationId, PrecommittedReductionLayout,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use super::advice_address_phase::{
    trusted_advice_input_values_from_upstream, untrusted_advice_input_values_from_upstream,
    TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
};
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, BytecodeReductionAddressPhaseInputClaims,
    ProgramImageReductionAddressPhase, ProgramImageReductionAddressPhaseInputClaims,
};
use super::hamming_weight_claim_reduction::{
    hamming_weight_input_values_from_upstream, stage7_hamming_virtualization_address_points,
    HammingWeightClaimReduction,
};
use super::outputs::{
    Stage7ClearOutput, Stage7InputClaims, Stage7Output, Stage7Sumchecks, Stage7ZkOutput,
};
use crate::{
    proof::JoltProof,
    stages::{
        stage4::{Stage4ClearOutput, Stage4Output},
        stage6b::{outputs::Stage6bOutputPoints, Stage6bClearOutput, Stage6bOutput},
        zk::committed,
        PrecommittedSchedule,
    },
    verifier::CheckedInputs,
    VerifierError,
};

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
    stage6: &Stage6bOutput<PCS::Field, VC::Output>,
) -> Result<Stage7Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let hamming_dimensions = hamming_weight::HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        proof.one_hot_config.committed_chunk_bits(),
    );

    // The clear-only reference geometry each address phase's expected-output term
    // reads (advice / program-image RAM address points, bytecode cycle-phase
    // weights) lives in the stage 4/6 clear outputs, absent in ZK where those
    // terms are proved by BlindFold and the relations' `derive_output_term` never
    // runs.
    let clear = if checked.zk {
        None
    } else {
        Some((stage4.clear()?, stage6.clear()?))
    };

    // One construction serves both paths: the hamming reduction from the stage-6
    // booleanity point split and the per-RA virtualization points, and each
    // address phase from its layout + `has_address_phase` presence flag + stage-6b
    // cycle-phase variables + clear-only reference aux. All point/challenge data is
    // read mode-agnostically off `stage6.output_points()`.
    let sumchecks = build_stage7_sumchecks(
        hamming_dimensions,
        &checked.precommitted,
        stage6.output_points(),
        clear,
    )?;

    // Draw the hamming-weight reduction's batching gamma (a single `challenge_scalar`,
    // matching the relation's default `draw_challenges`) path-agnostically before the
    // ZK/clear branch; the advice and committed-program address phases draw nothing
    // (`NoChallenges`). BlindFold sources the gamma from
    // `challenges.hamming_weight_claim_reduction.gamma`.
    let challenges = sumchecks.draw_challenges(transcript)?;

    if checked.zk {
        let consistency = sumchecks.verify_zk(&proof.stages.stage7_sumcheck_proof, transcript)?;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage7_sumcheck_proof,
            "stage7_sumcheck_proof",
            sumchecks.output_claim_count(),
            JoltRelationId::HammingWeightClaimReduction,
        )?;

        // The produced opening points, derived off the committed batch consistency;
        // stage 8 reads the hamming point and resolves the precommitted finals off
        // them. BlindFold recomputes each relation's sumcheck point and publics
        // independently from `batch_consistency`.
        let input_points = sumchecks.empty_input_points();
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

        return Ok(Stage7Output::Zk(Stage7ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
        }));
    }

    let stage6 = stage6.clear()?;
    let claims = &proof.clear_claims()?.stage7;

    // Also rejects claims supplied for phases that did not run, with the same
    // `UnexpectedOpeningClaim` ids the former hand-written guards used (the id is
    // derived from the supplied claims' canonical order).
    sumchecks.validate_output_claims(claims)?;

    let input_values = stage7_input_values_from_upstream(&sumchecks, stage6)?;
    let input_points = sumchecks.empty_input_points();

    let batch = sumchecks.verify_clear(
        &input_values,
        &challenges,
        &proof.stages.stage7_sumcheck_proof,
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
        return Err(VerifierError::StageClaimOutputMismatch { stage: 7 });
    }

    sumchecks.append_output_claims(transcript, claims);

    Ok(Stage7Output::Clear(Stage7ClearOutput {
        output_values: claims.clone(),
        output_points,
    }))
}

/// Build the stage-7 sumcheck batch once, for both proving paths. The hamming
/// reduction and every present address phase are constructed from the stage-6
/// output points (mode-agnostic) and the clear-only stage 4/6 references (`None`
/// in ZK, where the address phases' `FinalScale` term is proved by BlindFold and
/// `derive_output_term` never runs). An address phase is present exactly when its
/// precommitted layout is committed and its dimensions carry active address rounds
/// — the presence flag the input / challenge aggregates track in lockstep.
fn build_stage7_sumchecks<F: Field>(
    hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    schedule: &PrecommittedSchedule,
    stage6_points: &Stage6bOutputPoints<F>,
    clear: Option<(&Stage4ClearOutput<F>, &Stage6bClearOutput<F>)>,
) -> Result<Stage7Sumchecks<F>, VerifierError> {
    let booleanity_opening = stage6_points.booleanity_opening_point().ok_or(
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "Stage 6 booleanity produced no opening point".to_string(),
        },
    )?;
    let (booleanity_r_address, booleanity_r_cycle) =
        booleanity_opening.split_at(hamming_dimensions.log_k_chunk);
    let hamming = HammingWeightClaimReduction::new(
        hamming_dimensions,
        booleanity_r_cycle.to_vec(),
        booleanity_r_address.to_vec(),
        stage7_hamming_virtualization_address_points(hamming_dimensions, stage6_points)?,
    );

    // The staged advice RAM address point from stage 4's RAM value-check (`None`
    // in ZK), the clear-only reference the advice `FinalScale` term reads.
    let advice_reference = |kind| {
        clear.and_then(|(stage4, _)| {
            stage4
                .ram_val_check_init
                .advice_contribution(kind)
                .map(|contribution| contribution.opening_point.clone())
        })
    };

    Ok(Stage7Sumchecks {
        hamming_weight_claim_reduction: hamming,
        trusted_advice: address_phase_member(
            schedule.trusted_advice.as_ref(),
            stage6_points.advice_cycle_phase_variables(JoltAdviceKind::Trusted),
            advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted),
            |layout, cycle_phase_variables| {
                TrustedAdviceAddressPhase::new(
                    layout,
                    advice_reference(JoltAdviceKind::Trusted),
                    cycle_phase_variables,
                )
            },
        )?,
        untrusted_advice: address_phase_member(
            schedule.untrusted_advice.as_ref(),
            stage6_points.advice_cycle_phase_variables(JoltAdviceKind::Untrusted),
            advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted),
            |layout, cycle_phase_variables| {
                UntrustedAdviceAddressPhase::new(
                    layout,
                    advice_reference(JoltAdviceKind::Untrusted),
                    cycle_phase_variables,
                )
            },
        )?,
        bytecode_address_phase: address_phase_member(
            schedule.bytecode.as_ref(),
            stage6_points.bytecode_cycle_phase_variables(),
            bytecode_reduction::cycle_phase_intermediate_opening(),
            |layout, cycle_phase_variables| {
                BytecodeReductionAddressPhase::new(
                    layout,
                    clear.and_then(|(_, stage6)| stage6.bytecode_reduction_weights.clone()),
                    cycle_phase_variables,
                )
            },
        )?,
        program_image_address_phase: address_phase_member(
            schedule.program_image.as_ref(),
            stage6_points.program_image_cycle_phase_variables(),
            program_image::cycle_phase_program_image_opening(),
            |layout, cycle_phase_variables| {
                ProgramImageReductionAddressPhase::new(
                    layout,
                    clear.and_then(|(stage4, _)| {
                        stage4
                            .ram_val_check_init
                            .program_image_contribution
                            .as_ref()
                            .map(|(point, _)| point.clone())
                    }),
                    cycle_phase_variables,
                )
            },
        )?,
    })
}

/// Construct a present address-phase member: gate on the layout being committed
/// with active address rounds first (an absent layout yields `Ok(None)`, matching
/// the member's presence flag), then lift missing stage-6b cycle-phase variables
/// to `MissingOpeningClaim` before building the instance.
fn address_phase_member<F: Field, L: PrecommittedReductionLayout, M>(
    layout: Option<&L>,
    cycle_phase_variables: Option<Vec<F>>,
    missing_cycle_opening: JoltOpeningId,
    build: impl FnOnce(&L, Vec<F>) -> M,
) -> Result<Option<M>, VerifierError> {
    let Some(layout) = layout.filter(|layout| layout.dimensions().has_address_phase()) else {
        return Ok(None);
    };
    let cycle_phase_variables =
        cycle_phase_variables.ok_or(VerifierError::MissingOpeningClaim {
            id: missing_cycle_opening,
        })?;
    Ok(Some(build(layout, cycle_phase_variables)))
}

/// Assemble the stage-7 consumed opening *values* from the upstream stage-6 clear
/// output into the generated `Stage7InputClaims` aggregate. The two advice members
/// and the two committed-program members are `Some` exactly when their address
/// phase runs (tracking each `Stage7Sumchecks` member's presence), so a present
/// member always has its input cell populated. Public because the prover's
/// stage-7 recipe builds its batch inputs through the same wiring.
pub fn stage7_input_values_from_upstream<F: Field>(
    sumchecks: &Stage7Sumchecks<F>,
    stage6: &Stage6bClearOutput<F>,
) -> Result<Stage7InputClaims<F>, VerifierError> {
    let cycle_phase = &stage6.output_values;
    Ok(Stage7InputClaims {
        hamming_weight_claim_reduction: hamming_weight_input_values_from_upstream(cycle_phase),
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| trusted_advice_input_values_from_upstream(cycle_phase))
            .transpose()?,
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| untrusted_advice_input_values_from_upstream(cycle_phase))
            .transpose()?,
        bytecode_address_phase: sumchecks
            .bytecode_address_phase
            .as_ref()
            .map(|_| {
                cycle_phase
                    .bytecode_reduction
                    .as_ref()
                    .and_then(|reduction| reduction.intermediate)
                    .ok_or(VerifierError::MissingOpeningClaim {
                        id: bytecode_reduction::cycle_phase_intermediate_opening(),
                    })
                    .map(
                        |cycle_phase_intermediate| BytecodeReductionAddressPhaseInputClaims {
                            cycle_phase_intermediate,
                        },
                    )
            })
            .transpose()?,
        program_image_address_phase: sumchecks
            .program_image_address_phase
            .as_ref()
            .map(|_| {
                cycle_phase
                    .program_image_reduction
                    .as_ref()
                    .map(|claim| claim.program_image)
                    .ok_or(VerifierError::MissingOpeningClaim {
                        id: program_image::cycle_phase_program_image_opening(),
                    })
                    .map(|value| ProgramImageReductionAddressPhaseInputClaims {
                        cycle_phase: value,
                    })
            })
            .transpose()?,
    })
}
