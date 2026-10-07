use jolt_claims::protocols::jolt::{
    geometry::{
        claim_reductions::{
            bytecode::{self as bytecode_reduction},
            program_image,
        },
        dimensions::JoltFormulaDimensions,
    },
    JoltOpeningId, JoltRelationId, PrecommittedReductionLayout,
};
use jolt_field::{CanonicalDecode, JoltField};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::sites::STAGE7;

#[cfg(not(feature = "akita"))]
use super::advice_address_phase::{
    trusted_advice_input_values_from_upstream, untrusted_advice_input_values_from_upstream,
    TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
};
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, BytecodeReductionAddressPhaseInputClaims,
    ProgramImageReductionAddressPhase, ProgramImageReductionAddressPhaseInputClaims,
};
use super::hamming_weight_claim_reduction::{
    hamming_weight_claim_reduction_dimensions, hamming_weight_input_values_from_upstream,
    stage7_hamming_virtualization_address_points, HammingWeightClaimReduction,
    HammingWeightClaimReductionDimensions,
};
use super::outputs::{
    Stage7ClearOutput, Stage7InputClaims, Stage7Output, Stage7Sumchecks, Stage7ZkOutput,
};
#[cfg(not(feature = "akita"))]
use crate::stages::stage6b::committed_reduction_cycle_phase::advice_reference_point_from_upstream;
use crate::{
    stages::{
        stage4::{Stage4ClearOutput, Stage4Output},
        stage6b::{outputs::Stage6bOutputPoints, Stage6bClearOutput, Stage6bOutput},
        PrecommittedSchedule,
    },
    verifier::CheckedInputs,
    VerifierError,
};
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::geometry::claim_reductions::advice;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::JoltAdviceKind;

pub fn verify<F, C, H>(
    checked: &CheckedInputs,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut VerifierTranscript<'_, H>,
    stage4: &Stage4Output<F, C>,
    stage6: &Stage6bOutput<F, C>,
) -> Result<Stage7Output<F, C>, VerifierError>
where
    F: JoltField,
    C: CanonicalDecode,
    H: Sponge,
{
    transcript.site(STAGE7);
    let hamming_dimensions = hamming_weight_claim_reduction_dimensions(
        formula_dimensions.ra_layout,
        checked.one_hot_config.committed_chunk_bits(),
    )?;

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

    // Draw the hamming-weight reduction's batching gamma (a single `challenge`,
    // matching the relation's default `draw_challenges`) path-agnostically before the
    // ZK/clear branch; the advice and committed-program address phases draw nothing
    // (`NoChallenges`). BlindFold sources the gamma from
    // `challenges.hamming_weight_claim_reduction.gamma`.
    let challenges = sumchecks.draw_challenges(transcript)?;

    if checked.zk {
        // The produced opening points, derived off the committed batch consistency;
        // stage 8 reads the hamming point and resolves the precommitted finals off
        // them. BlindFold recomputes each relation's sumcheck point and publics
        // independently from `batch_consistency`.
        let input_points = sumchecks.empty_input_points();
        let batch = sumchecks.verify_zk(checked.committed_row_len()?, &input_points, transcript)?;

        return Ok(Stage7Output::Zk(Stage7ZkOutput {
            challenges,
            batch_consistency: batch.consistency,
            batch_output_claims: batch.output_claims,
            output_points: batch.output_points,
        }));
    }

    let stage6 = stage6.clear()?;
    let input_values = stage7_input_values_from_upstream(&sumchecks, stage6)?;
    let input_points = sumchecks.empty_input_points();

    let (output_points, output_values) =
        sumchecks.verify_clear(&input_values, &input_points, &challenges, transcript, 7)?;

    Ok(Stage7Output::Clear(Stage7ClearOutput {
        output_values,
        output_points,
    }))
}

/// Build the stage-7 sumcheck batch once, for both proving paths. The hamming
/// reduction and reduction-backed address phases are constructed from the stage-6
/// output points (mode-agnostic) and the clear-only stage 4/6 references (`None`
/// in ZK, where the address phases' `FinalScale` term is proved by BlindFold and
/// `derive_output_term` never runs). Advice reductions are skipped on Akita: the
/// final grouped opening checks their direct stage-4 claims.
pub fn build_stage7_sumchecks<F: JoltField>(
    hamming_dimensions: HammingWeightClaimReductionDimensions,
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
    #[cfg(feature = "akita")]
    if stage6_points.fused_inc_opening_point() != booleanity_r_cycle {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason:
                "the read-raf FusedInc opening and Booleanity do not share the Stage 6b cycle point"
                    .to_string(),
        });
    }
    let hamming = HammingWeightClaimReduction::new(
        hamming_dimensions,
        booleanity_r_cycle.to_vec(),
        booleanity_r_address.to_vec(),
        stage7_hamming_virtualization_address_points(hamming_dimensions, stage6_points)?,
    );

    // The staged advice RAM address point from stage 4's RAM value-check (`None`
    // in ZK), the clear-only reference the advice `FinalScale` term reads.
    #[cfg(not(feature = "akita"))]
    let advice_reference = |kind| {
        clear.and_then(|(stage4, _)| {
            advice_reference_point_from_upstream(&stage4.ram_val_check_init, kind)
        })
    };

    Ok(Stage7Sumchecks {
        hamming_weight_claim_reduction: hamming,
        #[cfg(not(feature = "akita"))]
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
        #[cfg(not(feature = "akita"))]
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
fn address_phase_member<F: JoltField, L: PrecommittedReductionLayout, M>(
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
            id: missing_cycle_opening.into(),
        })?;
    Ok(Some(build(layout, cycle_phase_variables)))
}

/// Assemble the stage-7 consumed opening *values* from the upstream stage-6 clear
/// output into the generated `Stage7InputClaims` aggregate. The two advice members
/// and the two committed-program members are `Some` exactly when their address
/// phase runs (tracking each `Stage7Sumchecks` member's presence), so a present
/// member always has its input cell populated. Public because the prover's
/// stage-7 recipe builds its batch inputs through the same wiring.
pub fn stage7_input_values_from_upstream<F: JoltField>(
    sumchecks: &Stage7Sumchecks<F>,
    stage6: &Stage6bClearOutput<F>,
) -> Result<Stage7InputClaims<F>, VerifierError> {
    let cycle_phase = &stage6.output_values;
    Ok(Stage7InputClaims {
        hamming_weight_claim_reduction: hamming_weight_input_values_from_upstream(cycle_phase),
        #[cfg(not(feature = "akita"))]
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| trusted_advice_input_values_from_upstream(cycle_phase))
            .transpose()?,
        #[cfg(not(feature = "akita"))]
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
                        id: bytecode_reduction::cycle_phase_intermediate_opening().into(),
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
                        id: program_image::cycle_phase_program_image_opening().into(),
                    })
                    .map(|value| ProgramImageReductionAddressPhaseInputClaims {
                        cycle_phase: value,
                    })
            })
            .transpose()?,
    })
}
