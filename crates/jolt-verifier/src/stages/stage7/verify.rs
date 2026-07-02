use jolt_claims::protocols::jolt::{
    geometry::{
        claim_reductions::{
            advice,
            bytecode::{self as bytecode_reduction, BytecodeOutputWeightInputs},
            hamming_weight, program_image,
        },
        dimensions::JoltFormulaDimensions,
    },
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind,
    JoltCommittedPolynomial, JoltRelationId, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use super::advice_address_phase::{AdviceAddressPhase, AdviceAddressPhaseInputClaims};
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, BytecodeReductionAddressPhaseInputClaims,
    ProgramImageReductionAddressPhase, ProgramImageReductionAddressPhaseInputClaims,
};
use super::hamming_weight_claim_reduction::{
    HammingWeightClaimReduction, HammingWeightClaimReductionInputClaims,
};
use super::outputs::{
    PrecommittedFinalOpening, Stage7ClearOutput, Stage7InputClaims, Stage7InputPoints,
    Stage7Output, Stage7OutputClaims, Stage7OutputPoints, Stage7Sumchecks, Stage7ZkOutput,
};
use crate::{
    proof::JoltProof,
    stages::{
        stage4::{Stage4ClearOutput, Stage4Output},
        stage6b::{
            outputs::{BytecodeReductionWeights, Stage6bOutputPoints},
            Stage6bClearOutput, Stage6bOutput,
        },
        zk::committed,
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

    let layouts = Stage7Layouts {
        trusted_advice: checked.precommitted.trusted_advice.as_ref(),
        untrusted_advice: checked.precommitted.untrusted_advice.as_ref(),
        bytecode: checked.precommitted.bytecode.as_ref(),
        program_image: checked.precommitted.program_image.as_ref(),
    };

    // The clear-only reference geometry each address phase's expected-output term
    // reads (advice / program-image RAM address points, bytecode cycle-phase
    // weights). Empty in ZK, where those terms are proved by BlindFold and the
    // relations' `derive_output_term` never runs.
    let references = if checked.zk {
        Stage7ClearReferences::empty()
    } else {
        Stage7ClearReferences::from_clear(stage4.clear()?, stage6.clear()?)
    };

    // One construction serves both paths: the hamming reduction from the stage-6
    // booleanity point split and the per-RA virtualization points, and each
    // address phase from its layout + `has_address_phase` presence flag + stage-6b
    // cycle-phase variables + clear-only reference aux. All point/challenge data is
    // read mode-agnostically off `stage6.output_points()`.
    let sumchecks = build_stage7_sumchecks(
        hamming_dimensions,
        &layouts,
        stage6.output_points(),
        &references,
    )?;

    // Draw the hamming-weight reduction's batching gamma (a single `challenge_scalar`,
    // matching the relation's default `draw_challenges`) path-agnostically before the
    // ZK/clear branch; the advice and committed-program address phases draw nothing
    // (`NoChallenges`). BlindFold sources the gamma from
    // `challenges.hamming_weight_claim_reduction.gamma`.
    let challenges = sumchecks.draw_challenges(transcript)?;

    if checked.zk {
        let consistency = sumchecks.verify_zk(&proof.stages.stage7_sumcheck_proof, transcript)?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage7_sumcheck_proof,
                proof_label: "stage7_sumcheck_proof",
                output_claim_count: sumchecks.output_claim_count(),
                stage: JoltRelationId::HammingWeightClaimReduction,
            })?;

        // Built via the same wiring as the clear path off the ZK-agnostic upstream
        // points; the address-phase opening points come off `output_points`, replacing
        // the ad-hoc per-relation ZK point recoveries. BlindFold recomputes each
        // relation's sumcheck point and publics independently from `batch_consistency`.
        let input_points = stage7_input_points_from_upstream(&sumchecks);
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;
        let hamming_weight_opening_point = hamming_weight_opening_point(&output_points)?;
        let precommitted_final_openings =
            zk_precommitted_final_openings(&layouts, &output_points, stage6.output_points())?;

        return Ok(Stage7Output::Zk(Stage7ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
            hamming_weight_opening_point,
            precommitted_final_openings,
        }));
    }

    let stage6 = stage6.clear()?;
    let claims = &proof.clear_claims()?.stage7;

    // Also rejects claims supplied for phases that did not run, with the same
    // `UnexpectedOpeningClaim` ids the former hand-written guards used (the id is
    // derived from the supplied claims' canonical order).
    sumchecks.validate_output_claims(claims)?;

    let input_values = stage7_input_values_from_upstream(&sumchecks, stage6)?;
    let input_points = stage7_input_points_from_upstream(&sumchecks);

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

    claims.append_to_transcript(transcript);

    let hamming_weight_opening_point = hamming_weight_opening_point(&output_points)?;
    let precommitted_final_openings =
        clear_precommitted_final_openings(&layouts, claims, &output_points, stage6)?;

    Ok(Stage7Output::Clear(Stage7ClearOutput {
        output_values: claims.clone(),
        output_points,
        hamming_weight_opening_point,
        precommitted_final_openings,
    }))
}

/// The committed-program claim-reduction layouts present in this proof
/// configuration (each `Some` only when the corresponding precommitted poly is
/// committed).
#[derive(Clone, Copy)]
struct Stage7Layouts<'a> {
    trusted_advice: Option<&'a AdviceClaimReductionLayout>,
    untrusted_advice: Option<&'a AdviceClaimReductionLayout>,
    bytecode: Option<&'a BytecodeClaimReductionLayout>,
    program_image: Option<&'a ProgramImageClaimReductionLayout>,
}

/// The clear-only reference geometry consumed by the address phases'
/// expected-output terms. Every field is `None` in ZK (stage 4/6 clear outputs are
/// absent), where those terms are proved by BlindFold and the relations'
/// `derive_output_term` is unreachable.
struct Stage7ClearReferences<F: Field> {
    /// The trusted / untrusted advice RAM address points (staged from stage 4's RAM
    /// value-check), each `Some` only when that advice is committed.
    trusted_advice: Option<Vec<F>>,
    untrusted_advice: Option<Vec<F>>,
    /// The program-image RAM address point (staged from stage 4), `Some` only in
    /// committed-program mode.
    program_image: Option<Vec<F>>,
    /// The stage-6b bytecode cycle-phase output weights, `Some` only in
    /// committed-program mode.
    bytecode_weights: Option<BytecodeReductionWeights<F>>,
}

impl<F: Field> Stage7ClearReferences<F> {
    fn empty() -> Self {
        Self {
            trusted_advice: None,
            untrusted_advice: None,
            program_image: None,
            bytecode_weights: None,
        }
    }

    fn from_clear(stage4: &Stage4ClearOutput<F>, stage6: &Stage6bClearOutput<F>) -> Self {
        let advice = |kind| {
            stage4
                .ram_val_check_init
                .advice_contribution(kind)
                .map(|contribution| contribution.opening_point.clone())
        };
        Self {
            trusted_advice: advice(JoltAdviceKind::Trusted),
            untrusted_advice: advice(JoltAdviceKind::Untrusted),
            program_image: stage4
                .ram_val_check_init
                .program_image_contribution_point
                .clone(),
            bytecode_weights: stage6.bytecode_reduction_weights.clone(),
        }
    }
}

/// Build the stage-7 sumcheck batch once, for both proving paths. The hamming
/// reduction and every present address phase are constructed from the stage-6
/// output points (mode-agnostic) and the clear-only reference aux (`None` in ZK).
/// An address phase is present exactly when its precommitted layout is committed
/// and its dimensions carry active address rounds — the presence flag the input /
/// challenge aggregates track in lockstep.
fn build_stage7_sumchecks<F: Field>(
    hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    layouts: &Stage7Layouts<'_>,
    stage6_points: &Stage6bOutputPoints<F>,
    references: &Stage7ClearReferences<F>,
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

    Ok(Stage7Sumchecks {
        hamming_weight_claim_reduction: hamming,
        trusted_advice: build_advice_member(
            JoltAdviceKind::Trusted,
            layouts.trusted_advice,
            stage6_points,
            references.trusted_advice.clone(),
        )?,
        untrusted_advice: build_advice_member(
            JoltAdviceKind::Untrusted,
            layouts.untrusted_advice,
            stage6_points,
            references.untrusted_advice.clone(),
        )?,
        bytecode_address_phase: build_bytecode_member(
            layouts.bytecode,
            stage6_points,
            references.bytecode_weights.as_ref(),
        )?,
        program_image_address_phase: build_program_image_member(
            layouts.program_image,
            stage6_points,
            references.program_image.clone(),
        )?,
    })
}

fn build_advice_member<F: Field>(
    kind: JoltAdviceKind,
    layout: Option<&AdviceClaimReductionLayout>,
    stage6_points: &Stage6bOutputPoints<F>,
    reference_opening_point: Option<Vec<F>>,
) -> Result<Option<AdviceAddressPhase<F>>, VerifierError> {
    let Some(layout) = layout.filter(|layout| layout.dimensions().has_address_phase()) else {
        return Ok(None);
    };
    let cycle_phase_variables = stage6_points.advice_cycle_phase_variables(kind).ok_or(
        VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind),
        },
    )?;
    Ok(Some(AdviceAddressPhase::new(
        kind,
        layout,
        reference_opening_point,
        cycle_phase_variables,
    )))
}

fn build_bytecode_member<F: Field>(
    layout: Option<&BytecodeClaimReductionLayout>,
    stage6_points: &Stage6bOutputPoints<F>,
    weights: Option<&BytecodeReductionWeights<F>>,
) -> Result<Option<BytecodeReductionAddressPhase<F>>, VerifierError> {
    let Some(layout) = layout.filter(|layout| layout.dimensions().has_address_phase()) else {
        return Ok(None);
    };
    let cycle_phase_variables = stage6_points.bytecode_cycle_phase_variables().ok_or(
        VerifierError::MissingOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        },
    )?;
    let weight_inputs = weights.map(|weights| BytecodeOutputWeightInputs {
        r_bc: &weights.r_bc,
        chunk_rbc_weights: &weights.chunk_rbc_weights,
        lane_weights: &weights.lane_weights,
    });
    Ok(Some(BytecodeReductionAddressPhase::new(
        layout,
        weight_inputs,
        cycle_phase_variables,
    )))
}

fn build_program_image_member<F: Field>(
    layout: Option<&ProgramImageClaimReductionLayout>,
    stage6_points: &Stage6bOutputPoints<F>,
    reference_opening_point: Option<Vec<F>>,
) -> Result<Option<ProgramImageReductionAddressPhase<F>>, VerifierError> {
    let Some(layout) = layout.filter(|layout| layout.dimensions().has_address_phase()) else {
        return Ok(None);
    };
    let cycle_phase_variables = stage6_points.program_image_cycle_phase_variables().ok_or(
        VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        },
    )?;
    Ok(Some(ProgramImageReductionAddressPhase::new(
        layout,
        reference_opening_point,
        cycle_phase_variables,
    )))
}

/// Assemble the stage-7 consumed opening *values* from the upstream stage-6 clear
/// output into the generated `Stage7InputClaims` aggregate. The two advice members
/// and the two committed-program members are `Some` exactly when their address
/// phase runs (tracking each `Stage7Sumchecks` member's presence), so a present
/// member always has its input cell populated.
fn stage7_input_values_from_upstream<F: Field>(
    sumchecks: &Stage7Sumchecks<F>,
    stage6: &Stage6bClearOutput<F>,
) -> Result<Stage7InputClaims<F>, VerifierError> {
    Ok(Stage7InputClaims {
        hamming_weight_claim_reduction: hamming_input_values(stage6),
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| advice_input_values(stage6, JoltAdviceKind::Trusted)),
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| advice_input_values(stage6, JoltAdviceKind::Untrusted)),
        bytecode_address_phase: sumchecks
            .bytecode_address_phase
            .as_ref()
            .map(|_| clear_bytecode_input_values(stage6))
            .transpose()?,
        program_image_address_phase: sumchecks
            .program_image_address_phase
            .as_ref()
            .map(|_| clear_program_image_input_values(stage6))
            .transpose()?,
    })
}

/// Assemble the stage-7 consumed opening *points*. Every stage-7 relation derives
/// its produced points from its own sumcheck point and reads no input point, so all
/// input point cells are empty; presence tracks each `Stage7Sumchecks` member,
/// serving the clear and ZK paths alike.
fn stage7_input_points_from_upstream<F: Field>(
    sumchecks: &Stage7Sumchecks<F>,
) -> Stage7InputPoints<F> {
    Stage7InputPoints {
        hamming_weight_claim_reduction: hamming_input_points(),
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| advice_input_points()),
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| advice_input_points()),
        bytecode_address_phase: sumchecks
            .bytecode_address_phase
            .as_ref()
            .map(|_| bytecode_input_points()),
        program_image_address_phase: sumchecks
            .program_image_address_phase
            .as_ref()
            .map(|_| program_image_input_points()),
    }
}

/// The hamming reduction's consumed opening *values*, wired from stage 6. The
/// relation reads only their values (its produced points are derived from its own
/// sumcheck point), so no input points are needed.
fn hamming_input_values<F: Field>(
    stage6: &Stage6bClearOutput<F>,
) -> HammingWeightClaimReductionInputClaims<F> {
    let cycle_phase = &stage6.output_values;
    HammingWeightClaimReductionInputClaims {
        ram_hamming_weight: cycle_phase.ram_hamming_booleanity.ram_hamming_weight,
        instruction_booleanity: cycle_phase.booleanity.instruction_ra.clone(),
        bytecode_booleanity: cycle_phase.booleanity.bytecode_ra.clone(),
        ram_booleanity: cycle_phase.booleanity.ram_ra.clone(),
        instruction_virtualization: cycle_phase
            .instruction_ra_virtualization
            .committed_instruction_ra
            .clone(),
        bytecode_virtualization: cycle_phase.bytecode_read_raf.bytecode_ra.clone(),
        ram_virtualization: cycle_phase.ram_ra_virtualization.ram_ra.clone(),
    }
}

fn hamming_input_points<F: Field>() -> HammingWeightClaimReductionInputClaims<Vec<F>> {
    HammingWeightClaimReductionInputClaims {
        ram_hamming_weight: Vec::new(),
        instruction_booleanity: Vec::new(),
        bytecode_booleanity: Vec::new(),
        ram_booleanity: Vec::new(),
        instruction_virtualization: Vec::new(),
        bytecode_virtualization: Vec::new(),
        ram_virtualization: Vec::new(),
    }
}

/// The consumed cycle-phase advice opening *value* for `kind`, in the shared
/// `AdviceAddressPhaseInputClaims` with only that kind's slot filled (the relation
/// reads only its own kind's field).
fn advice_input_values<F: Field>(
    stage6: &Stage6bClearOutput<F>,
    kind: JoltAdviceKind,
) -> AdviceAddressPhaseInputClaims<F> {
    let claim = stage6_advice_cycle_phase_claim(stage6, kind).ok();
    match kind {
        JoltAdviceKind::Trusted => AdviceAddressPhaseInputClaims {
            trusted: claim,
            untrusted: None,
        },
        JoltAdviceKind::Untrusted => AdviceAddressPhaseInputClaims {
            trusted: None,
            untrusted: claim,
        },
    }
}

fn advice_input_points<F: Field>() -> AdviceAddressPhaseInputClaims<Vec<F>> {
    AdviceAddressPhaseInputClaims {
        trusted: None,
        untrusted: None,
    }
}

fn clear_bytecode_input_values<F: Field>(
    stage6: &Stage6bClearOutput<F>,
) -> Result<BytecodeReductionAddressPhaseInputClaims<F>, VerifierError> {
    let value = stage6
        .output_values
        .bytecode_reduction
        .as_ref()
        .and_then(|reduction| reduction.intermediate)
        .ok_or(VerifierError::MissingOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        })?;
    Ok(BytecodeReductionAddressPhaseInputClaims {
        cycle_phase_intermediate: value,
    })
}

fn bytecode_input_points<F: Field>() -> BytecodeReductionAddressPhaseInputClaims<Vec<F>> {
    BytecodeReductionAddressPhaseInputClaims {
        cycle_phase_intermediate: Vec::new(),
    }
}

fn clear_program_image_input_values<F: Field>(
    stage6: &Stage6bClearOutput<F>,
) -> Result<ProgramImageReductionAddressPhaseInputClaims<F>, VerifierError> {
    let value = stage6
        .output_values
        .program_image_reduction
        .as_ref()
        .map(|claim| claim.program_image)
        .ok_or(VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        })?;
    Ok(ProgramImageReductionAddressPhaseInputClaims { cycle_phase: value })
}

fn program_image_input_points<F: Field>() -> ProgramImageReductionAddressPhaseInputClaims<Vec<F>> {
    ProgramImageReductionAddressPhaseInputClaims {
        cycle_phase: Vec::new(),
    }
}

fn stage6_advice_cycle_phase_claim<F: Field>(
    stage6: &Stage6bClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, VerifierError> {
    let claim = match kind {
        JoltAdviceKind::Trusted => stage6
            .output_values
            .trusted_advice
            .as_ref()
            .and_then(|claim| claim.trusted),
        JoltAdviceKind::Untrusted => stage6
            .output_values
            .untrusted_advice
            .as_ref()
            .and_then(|claim| claim.untrusted),
    };
    claim.ok_or_else(|| VerifierError::MissingOpeningClaim {
        id: advice::cycle_phase_advice_opening(kind),
    })
}

/// The hamming-weight reduction's shared opening point off the produced points, or
/// an error when the reduction produced no openings.
fn hamming_weight_opening_point<F: Field>(
    output_points: &Stage7OutputPoints<F>,
) -> Result<Vec<F>, VerifierError> {
    output_points
        .hamming_weight_opening_point()
        .map(<[F]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "stage 7 produced no hamming-weight openings".to_string(),
        })
}

fn stage7_hamming_virtualization_address_points<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6_points: &Stage6bOutputPoints<F>,
) -> Result<Vec<Vec<F>>, VerifierError> {
    let instruction_ra_points = stage6_points
        .instruction_ra_virtualization
        .committed_instruction_ra();
    let bytecode_ra_points = stage6_points.bytecode_read_raf.bytecode_ra();
    let ram_ra_points = stage6_points.ram_ra_virtualization.ram_ra();
    if instruction_ra_points.len() != dimensions.layout.instruction()
        || bytecode_ra_points.len() != dimensions.layout.bytecode()
        || ram_ra_points.len() != dimensions.layout.ram()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "Stage 6 RA opening point count mismatch for Stage 7".to_string(),
        });
    }

    let mut points = Vec::with_capacity(dimensions.layout.total());
    for point in instruction_ra_points
        .iter()
        .chain(bytecode_ra_points)
        .chain(ram_ra_points)
    {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    Ok(points)
}

fn hamming_virtualization_address_point<F: Field>(
    log_k_chunk: usize,
    point: &[F],
) -> Result<Vec<F>, VerifierError> {
    point
        .get(..log_k_chunk)
        .map(<[F]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 6 RA opening point is too short for HammingWeight address chunk: expected at least {log_k_chunk}, got {}",
                point.len()
            ),
        })
}

/// Opening point and (clear-mode) claim payload recorded by the stage that
/// completed a precommitted claim reduction. `T` is a single claim for advice and
/// the program image, and the per-chunk claim slice for the committed bytecode.
struct PrecommittedFinalSource<'a, F, T = F> {
    point: &'a [F],
    opening_claim: Option<T>,
}

impl<'a, F, T> PrecommittedFinalSource<'a, F, T> {
    fn zk(point: &'a [F]) -> Self {
        Self {
            point,
            opening_claim: None,
        }
    }

    fn clear(point: &'a [F], opening_claim: T) -> Self {
        Self {
            point,
            opening_claim: Some(opening_claim),
        }
    }
}

impl<'a, F> PrecommittedFinalSource<'a, F, Vec<F>> {
    fn clear_chunks(point: &'a [F], chunks: Vec<F>) -> Self {
        Self {
            point,
            opening_claim: Some(chunks),
        }
    }
}

/// Resolve the final openings of the precommitted polynomials from whichever phase
/// completed each reduction: this stage's address phase (read off the produced
/// `output_points`/`output_values`) or the stage 6b cycle phase.
fn clear_precommitted_final_openings<F: Field>(
    layouts: &Stage7Layouts<'_>,
    output_values: &Stage7OutputClaims<F>,
    output_points: &Stage7OutputPoints<F>,
    stage6: &Stage6bClearOutput<F>,
) -> Result<Vec<PrecommittedFinalOpening<F>>, VerifierError> {
    let mut openings = Vec::new();
    for (kind, layout, address_value) in [
        (
            JoltAdviceKind::Trusted,
            layouts.trusted_advice,
            output_values
                .trusted_advice
                .as_ref()
                .and_then(|claims| claims.trusted),
        ),
        (
            JoltAdviceKind::Untrusted,
            layouts.untrusted_advice,
            output_values
                .untrusted_advice
                .as_ref()
                .and_then(|claims| claims.untrusted),
        ),
    ] {
        if let Some(layout) = layout {
            let address_phase = output_points
                .advice_point(kind)
                .zip(address_value)
                .map(|(point, value)| PrecommittedFinalSource::clear(point, value));
            let cycle_phase = stage6
                .output_points
                .advice_cycle_phase_opening_point(kind)
                .zip(stage6_advice_cycle_phase_claim(stage6, kind).ok())
                .map(|(point, value)| PrecommittedFinalSource::clear(point, value));
            openings.push(advice_final_opening(
                kind,
                layout,
                address_phase,
                cycle_phase,
            )?);
        }
    }
    if let Some(layout) = layouts.bytecode {
        let address_phase = output_points
            .bytecode_point()
            .zip(output_values.bytecode_address_phase.as_ref())
            .map(|(point, values)| {
                PrecommittedFinalSource::clear_chunks(point, values.chunks.clone())
            });
        let cycle_phase = match (
            stage6.output_points.bytecode_reduction_opening_point(),
            &stage6.output_values.bytecode_reduction,
        ) {
            (Some(opening_point), Some(reduction))
                if reduction.intermediate.is_none() && !reduction.chunks.is_empty() =>
            {
                Some(PrecommittedFinalSource::clear_chunks(
                    opening_point,
                    reduction.chunks.clone(),
                ))
            }
            _ => None,
        };
        openings.extend(bytecode_final_openings(layout, address_phase, cycle_phase)?);
    }
    if let Some(layout) = layouts.program_image {
        let address_phase = output_points
            .program_image_point()
            .zip(output_values.program_image_address_phase.as_ref())
            .map(|(point, values)| PrecommittedFinalSource::clear(point, values.program_image));
        let cycle_phase = stage6
            .output_points
            .program_image_opening_point()
            .zip(stage6.output_values.program_image_reduction.as_ref())
            .map(|(opening_point, claim)| {
                PrecommittedFinalSource::clear(opening_point, claim.program_image)
            });
        openings.push(program_image_final_opening(
            layout,
            address_phase,
            cycle_phase,
        )?);
    }
    Ok(openings)
}

/// The ZK counterpart of [`clear_precommitted_final_openings`]: the address-phase
/// opening points come off the produced `output_points`, the cycle-phase points off
/// the stage-6 points, and every opening claim stays committed (`None`).
fn zk_precommitted_final_openings<F: Field>(
    layouts: &Stage7Layouts<'_>,
    output_points: &Stage7OutputPoints<F>,
    stage6_points: &Stage6bOutputPoints<F>,
) -> Result<Vec<PrecommittedFinalOpening<F>>, VerifierError> {
    let mut openings = Vec::new();
    for (kind, layout) in [
        (JoltAdviceKind::Trusted, layouts.trusted_advice),
        (JoltAdviceKind::Untrusted, layouts.untrusted_advice),
    ] {
        if let Some(layout) = layout {
            let address_phase = output_points
                .advice_point(kind)
                .map(PrecommittedFinalSource::zk);
            let cycle_phase = stage6_points
                .advice_cycle_phase_opening_point(kind)
                .map(PrecommittedFinalSource::zk);
            openings.push(advice_final_opening(
                kind,
                layout,
                address_phase,
                cycle_phase,
            )?);
        }
    }
    if let Some(layout) = layouts.bytecode {
        let address_phase = output_points
            .bytecode_point()
            .map(PrecommittedFinalSource::zk);
        let cycle_phase = stage6_points
            .bytecode_reduction_opening_point()
            .map(PrecommittedFinalSource::zk);
        openings.extend(bytecode_final_openings(layout, address_phase, cycle_phase)?);
    }
    if let Some(layout) = layouts.program_image {
        let address_phase = output_points
            .program_image_point()
            .map(PrecommittedFinalSource::zk);
        let cycle_phase = stage6_points
            .program_image_opening_point()
            .map(PrecommittedFinalSource::zk);
        openings.push(program_image_final_opening(
            layout,
            address_phase,
            cycle_phase,
        )?);
    }
    Ok(openings)
}

/// Resolves the final opening of an advice polynomial from whichever phase
/// completed its reduction: this stage's address phase, or the stage 6b cycle
/// phase when no active address rounds remain.
fn advice_final_opening<F: Field>(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F>>,
) -> Result<PrecommittedFinalOpening<F>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: advice::final_advice_opening(kind),
    })?;
    let polynomial = match kind {
        JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
        JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
    };
    Ok(PrecommittedFinalOpening {
        polynomial,
        point: source.point.to_vec(),
        opening_claim: source.opening_claim,
    })
}

/// Resolves the final per-chunk openings of the committed bytecode from whichever
/// phase completed the reduction: this stage's address phase, or the stage 6b
/// cycle phase when no active address rounds remain.
fn bytecode_final_openings<F: Field>(
    layout: &BytecodeClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F, Vec<F>>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F, Vec<F>>>,
) -> Result<Vec<PrecommittedFinalOpening<F>>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: bytecode_reduction::final_bytecode_chunk_opening(0),
    })?;
    if let Some(chunk_claims) = &source.opening_claim {
        if chunk_claims.len() != layout.chunk_count() {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeClaimReduction,
                reason: format!(
                    "final bytecode chunk claim count mismatch: expected {}, got {}",
                    layout.chunk_count(),
                    chunk_claims.len()
                ),
            });
        }
    }
    Ok((0..layout.chunk_count())
        .map(|chunk_idx| PrecommittedFinalOpening {
            polynomial: JoltCommittedPolynomial::BytecodeChunk(chunk_idx),
            point: source.point.to_vec(),
            opening_claim: source
                .opening_claim
                .as_ref()
                .map(|chunk_claims| chunk_claims[chunk_idx]),
        })
        .collect())
}

/// Resolves the final opening of the committed program image from whichever phase
/// completed the reduction: this stage's address phase, or the stage 6b cycle
/// phase when no active address rounds remain.
fn program_image_final_opening<F: Field>(
    layout: &ProgramImageClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F>>,
) -> Result<PrecommittedFinalOpening<F>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: program_image::final_program_image_opening(),
    })?;
    Ok(PrecommittedFinalOpening {
        polynomial: JoltCommittedPolynomial::ProgramImageInit,
        point: source.point.to_vec(),
        opening_claim: source.opening_claim,
    })
}
