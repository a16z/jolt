use crate::stages::relations::OutputAppend;
use jolt_claims::protocols::jolt::{
    geometry::{bytecode, dimensions::JoltFormulaDimensions},
    BytecodeClaimReductionLayout, JoltRelationId, PrecommittedReductionLayout,
};
use jolt_claims::{NoChallenges, OutputClaims};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

#[cfg(feature = "akita")]
use super::fused_inc_claim_reduction::{
    input_points_from_upstream as fused_inc_input_points_from_upstream,
    input_values_from_upstream as fused_inc_input_values_from_upstream,
};
#[cfg(not(feature = "akita"))]
use super::inc_claim_reduction::{
    inc_claim_reduction_input_points_from_upstream, inc_claim_reduction_input_values_from_upstream,
    IncClaimReductionChallenges,
};
#[cfg(not(feature = "akita"))]
use super::outputs::{Stage6bCarriedChallenges, Stage6bZkOutput};
use super::{
    booleanity::{BooleanityCyclePhaseChallenges, BooleanityInputClaims},
    bytecode_read_raf::{BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims},
    committed_reduction_cycle_phase::{
        program_image_reduction_cycle_phase_input_values_from_upstream,
        trusted_advice_cycle_phase_input_values_from_upstream,
        untrusted_advice_cycle_phase_input_values_from_upstream,
        BytecodeReductionCyclePhaseChallenges, BytecodeReductionCyclePhaseInputClaims,
    },
    instruction_ra_virtualization::{
        instruction_ra_virtualization_input_points_from_upstream,
        instruction_ra_virtualization_input_values_from_upstream,
        InstructionRaVirtualizationChallenges,
    },
    outputs::{
        Stage6bChallenges, Stage6bClearOutput, Stage6bInputClaims, Stage6bInputPoints,
        Stage6bOutput, Stage6bOutputClaims, Stage6bSumchecks,
    },
    ram_hamming_booleanity::RamHammingBooleanityInputClaims,
    ram_ra_virtualization::{
        ram_ra_virtualization_input_points_from_upstream,
        ram_ra_virtualization_input_values_from_upstream,
    },
};
#[cfg(not(feature = "akita"))]
use crate::stages::zk::committed;
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::validate_member_presence,
        stage1::Stage1Output,
        stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output},
        stage3::Stage3Output,
        stage4::{Stage4ClearOutput, Stage4Output, Stage4OutputPoints},
        stage5::{Stage5Output, Stage5OutputClaims, Stage5OutputPoints},
        stage6a::{outputs::Stage6aOutputClaims, Stage6aOutput},
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6b consumes the stage-6a output plus all five prior stage outputs directly; bundling them would reintroduce the removed `Deps` indirection."
)]
pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
    stage5: &Stage5Output<PCS::Field, VC::Output>,
    stage6a: &Stage6aOutput<PCS::Field, VC::Output>,
    #[cfg(feature = "akita")]
    inc_virtualization: &crate::stages::inc_virtualization::IncVirtualizationOutput<PCS::Field>,
) -> Result<Stage6bOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    // The bytecode fold gamma shares stage 6a's squeeze; it and the booleanity
    // gamma ride on the stage-6a output as typed upstream values.
    let carried = stage6a.challenges();
    let bytecode_gamma = carried.bytecode_read_raf.gamma;
    let bytecode_reduction_layout = checked.precommitted.bytecode.as_ref();

    // Post-6a draws: the instruction-RA virtualization gamma, the increment gamma
    // (base only — the packed batch has no inc member), and (committed-program
    // only) the bytecode claim-reduction eta.
    let instruction_ra_gamma = transcript.challenge_scalar();
    #[cfg(not(feature = "akita"))]
    let inc_gamma = transcript.challenge_scalar();
    let eta = bytecode_reduction_layout
        .is_some()
        .then(|| transcript.challenge_scalar());

    // The batch is built after the post-6a draws, directly from the upstream stage
    // outputs; `build` derives every mode-agnostic constructor leg internally.
    #[cfg(not(feature = "akita"))]
    let sumchecks = Stage6bSumchecks::build(
        checked,
        preprocessing,
        proof,
        formula_dimensions,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6a,
        eta,
    )?;
    #[cfg(feature = "akita")]
    let sumchecks = Stage6bSumchecks::build(
        checked,
        preprocessing,
        proof,
        formula_dimensions,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6a,
        eta,
        inc_virtualization,
    )?;

    // Hand-assembled (the generated `draw_challenges` is suppressed): the bytecode
    // gamma shares stage 6a's squeeze and the booleanity gamma was drawn pre-6a
    // where the prover's booleanity subprotocol samples it, so a generated
    // per-member draw would squeeze for them at the wrong transcript position.
    let cycle_challenges = Stage6bChallenges {
        bytecode_read_raf: BytecodeReadRafCyclePhaseCommittedChallenges {
            gamma: bytecode_gamma,
        },
        booleanity: BooleanityCyclePhaseChallenges {
            gamma: carried.booleanity_gamma,
        },
        ram_hamming_booleanity: NoChallenges::default(),
        ram_ra_virtualization: NoChallenges::default(),
        instruction_ra_virtualization: InstructionRaVirtualizationChallenges {
            gamma: instruction_ra_gamma,
        },
        #[cfg(not(feature = "akita"))]
        inc_claim_reduction: IncClaimReductionChallenges { gamma: inc_gamma },
        #[cfg(feature = "akita")]
        fused_inc_claim_reduction: NoChallenges::default(),
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| NoChallenges::default()),
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| NoChallenges::default()),
        bytecode_reduction: sumchecks
            .bytecode_reduction
            .as_ref()
            .zip(eta)
            .map(|(_, eta)| BytecodeReductionCyclePhaseChallenges { eta }),
        program_image_reduction: sumchecks
            .program_image_reduction
            .as_ref()
            .map(|_| NoChallenges::default()),
    };

    let input_points = stage6b_input_points_from_upstream(
        &sumchecks,
        stage2.batch_output_points(),
        stage4.output_points(),
        stage5.output_points(),
        #[cfg(feature = "akita")]
        inc_virtualization,
    );

    // No zk protocol exists over the packed axis, so the committed arm (and its
    // runtime point-alias dedup arithmetic) is base-only.
    #[cfg(not(feature = "akita"))]
    if checked.zk {
        let consistency = sumchecks.verify_zk(&proof.stages.stage6b_sumcheck_proof, transcript)?;
        let cycle_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

        // The committed-claim count is the derived output-point-cell total minus the
        // runtime aliases between the booleanity bytecode-RA openings and the
        // bytecode read-RAF openings — a point-equality dedup that is not expressible
        // from the output Exprs, so it stays hand-written.
        let booleanity_opening_point =
            cycle_points.booleanity_opening_point().ok_or_else(|| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::Booleanity,
                    reason: "Stage 6 booleanity produced no opening point".to_string(),
                }
            })?;
        let aliased_bytecode_ra_openings = cycle_points
            .bytecode_read_raf
            .bytecode_ra
            .iter()
            .filter(|point| point.as_slice() == booleanity_opening_point)
            .count();
        let committed_output_claims = cycle_points.point_count() - aliased_bytecode_ra_openings;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage6b_sumcheck_proof,
            "stage6b_sumcheck_proof",
            committed_output_claims,
            JoltRelationId::BytecodeReadRaf,
        )?;

        return Ok(Stage6bOutput::Zk(Stage6bZkOutput {
            challenges: Stage6bCarriedChallenges {
                instruction_ra_gamma,
                inc_gamma,
                bytecode_reduction_eta: eta,
            },
            batch_consistency: consistency,
            batch_output_claims,
            output_points: cycle_points,
        }));
    }

    let stage2 = stage2.clear()?;
    let stage4 = stage4.clear()?;
    let stage5 = stage5.clear()?;
    let claims_6a = &stage6a.clear()?.output_values;
    let claims = &proof.clear_claims()?.stage6b;

    // Reject cycle-phase output claims whose presence disagrees with the member's
    // layout: a present reduction missing its claims, or claims supplied for a
    // reduction that did not run. Instance presence mirrors layout presence (see
    // `Stage6bSumchecks::build`). Hand-listed because 6b curates its own shape
    // checks (`no_output_shape`, so no generated validator runs these guards
    // itself); one call per `Option` member. Transcript-free (runs before the
    // batched verify); the tampering suite asserts generic rejection.
    validate_member_presence(
        sumchecks.trusted_advice.as_ref(),
        claims.trusted_advice.as_ref(),
    )?;
    validate_member_presence(
        sumchecks.untrusted_advice.as_ref(),
        claims.untrusted_advice.as_ref(),
    )?;
    validate_member_presence(
        sumchecks.bytecode_reduction.as_ref(),
        claims.bytecode_reduction.as_ref(),
    )?;
    validate_member_presence(
        sumchecks.program_image_reduction.as_ref(),
        claims.program_image_reduction.as_ref(),
    )?;

    #[cfg(not(feature = "akita"))]
    validate_cycle_phase_claim_shape(formula_dimensions, claims, bytecode_reduction_layout)?;
    #[cfg(feature = "akita")]
    validate_cycle_phase_claim_shape(
        formula_dimensions,
        claims,
        bytecode_reduction_layout,
        proof.one_hot_config.committed_chunk_bits(),
    )?;

    let input_values = stage6b_input_values_from_upstream(
        &sumchecks,
        claims_6a,
        &stage2.output_values,
        stage4,
        &stage5.output_values,
        #[cfg(feature = "akita")]
        inc_virtualization,
    )?;
    let cycle_points = sumchecks.verify_clear(
        &input_values,
        &input_points,
        &cycle_challenges,
        claims,
        &proof.stages.stage6b_sumcheck_proof,
        transcript,
        6,
    )?;

    let booleanity_opening_point = cycle_points
        .booleanity_opening_point()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: "Stage 6 booleanity produced no opening point".to_string(),
        })?
        .to_vec();
    append_opening_claims(
        transcript,
        claims,
        &cycle_points.bytecode_read_raf.bytecode_ra,
        &booleanity_opening_point,
    );

    Ok(Stage6bOutput::Clear(Stage6bClearOutput {
        output_values: claims.clone(),
        output_points: cycle_points,
        bytecode_reduction_weights: sumchecks
            .bytecode_reduction
            .as_ref()
            .map(|reduction| reduction.weights().clone()),
    }))
}

/// The wire-shape checks over the cycle-phase output claims that the generated
/// drivers cannot express: the bytecode RA claim count and the bytecode reduction's
/// intermediate-vs-chunks shape. Member presence is enforced separately by the
/// hand-listed `validate_member_presence` calls; a missing advice inner opening is caught by
/// `expected_final_claim` (the advice cycle phase's `expected_output`).
fn validate_cycle_phase_claim_shape<F: Field>(
    formula_dimensions: &JoltFormulaDimensions,
    claims: &Stage6bOutputClaims<F>,
    bytecode_reduction_layout: Option<&BytecodeClaimReductionLayout>,
    #[cfg(feature = "akita")] committed_chunk_bits: usize,
) -> Result<(), VerifierError> {
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    if claims.bytecode_read_raf.bytecode_ra.len() != bytecode_output_openings.bytecode_ra.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                bytecode_output_openings.bytecode_ra.len(),
                claims.bytecode_read_raf.bytecode_ra.len()
            ),
        });
    }

    // The packed unsigned-inc chunk claims: one per chunk of the shared
    // one-hot chunking.
    #[cfg(feature = "akita")]
    {
        let expected_chunks =
            jolt_claims::protocols::jolt::lattice::geometry::UnsignedIncChunking::new(
                committed_chunk_bits,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            })?
            .chunk_count();
        if claims.booleanity.unsigned_inc_chunks.len() != expected_chunks {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::Booleanity,
                reason: format!(
                    "unsigned-inc chunk claim count mismatch: expected {expected_chunks}, got {}",
                    claims.booleanity.unsigned_inc_chunks.len()
                ),
            });
        }
    }

    if let (Some(layout), Some(output_claims)) = (
        bytecode_reduction_layout,
        claims.bytecode_reduction.as_ref(),
    ) {
        let has_address_phase = layout.dimensions().has_address_phase();
        // The wire shape must match the reduction mode: an `intermediate` (no
        // chunks) when an address phase follows, else exactly `chunk_count`
        // chunks (no intermediate).
        let shape_ok = match (
            &output_claims.intermediate,
            output_claims.chunks.is_empty(),
            has_address_phase,
        ) {
            (Some(_), true, true) => true,
            (None, false, false) => output_claims.chunks.len() == layout.chunk_count(),
            _ => false,
        };
        if !shape_ok {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
                reason: format!(
                    "bytecode reduction cycle output shape mismatch (address phase: {has_address_phase})"
                ),
            });
        }
    }

    Ok(())
}

/// Assemble the stage-6b consumed opening *values* from the address-phase claims
/// and the upstream clear outputs into the generated `Stage6bInputClaims`
/// aggregate. The `Option` cells track member presence, so a present member always
/// has its input cell populated. Public because the prover's stage-6b recipe
/// builds its batch inputs through the same wiring.
pub fn stage6b_input_values_from_upstream<F: Field>(
    sumchecks: &Stage6bSumchecks<F>,
    address_claims: &Stage6aOutputClaims<F>,
    #[cfg_attr(feature = "akita", expect(unused_variables))] stage2: &Stage2BatchOutputClaims<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5OutputClaims<F>,
    #[cfg(feature = "akita")]
    inc_virtualization: &crate::stages::inc_virtualization::IncVirtualizationOutput<F>,
) -> Result<Stage6bInputClaims<F>, VerifierError> {
    Ok(Stage6bInputClaims {
        bytecode_read_raf: BytecodeReadRafInputClaims {
            address_phase: address_claims.bytecode_read_raf.intermediate,
        },
        booleanity: BooleanityInputClaims {
            address_phase: address_claims.booleanity.intermediate,
        },
        ram_hamming_booleanity: RamHammingBooleanityInputClaims::default(),
        ram_ra_virtualization: ram_ra_virtualization_input_values_from_upstream(stage5),
        instruction_ra_virtualization: instruction_ra_virtualization_input_values_from_upstream(
            stage5,
        ),
        #[cfg(not(feature = "akita"))]
        inc_claim_reduction: inc_claim_reduction_input_values_from_upstream(
            stage2,
            &stage4.output_values,
            stage5,
        ),
        #[cfg(feature = "akita")]
        fused_inc_claim_reduction: fused_inc_input_values_from_upstream(inc_virtualization),
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| {
                trusted_advice_cycle_phase_input_values_from_upstream(&stage4.ram_val_check_init)
            })
            .transpose()?,
        untrusted_advice: sumchecks
            .untrusted_advice
            .as_ref()
            .map(|_| {
                untrusted_advice_cycle_phase_input_values_from_upstream(&stage4.ram_val_check_init)
            })
            .transpose()?,
        bytecode_reduction: sumchecks.bytecode_reduction.as_ref().map(|_| {
            BytecodeReductionCyclePhaseInputClaims {
                val_stages: address_claims.bytecode_read_raf.val_stages.clone(),
            }
        }),
        program_image_reduction: sumchecks
            .program_image_reduction
            .as_ref()
            .map(|_| {
                program_image_reduction_cycle_phase_input_values_from_upstream(
                    &stage4.ram_val_check_init,
                )
            })
            .transpose()?,
    })
}

/// Assemble the stage-6b consumed opening *points*. ZK-agnostic: only the RA / inc
/// members read the upstream output-points aggregates (which both modes expose); the
/// remaining seven members derive their produced points from their own sumcheck point
/// and read no input point, so their cells come from the generated
/// `empty_input_points` (empty, and present for present `Option` members exactly as
/// the generated `derive_opening_points` requires).
pub fn stage6b_input_points_from_upstream<F: Field>(
    sumchecks: &Stage6bSumchecks<F>,
    #[cfg_attr(feature = "akita", expect(unused_variables))] stage2: &Stage2BatchOutputPoints<F>,
    #[cfg_attr(feature = "akita", expect(unused_variables))] stage4: &Stage4OutputPoints<F>,
    stage5: &Stage5OutputPoints<F>,
    #[cfg(feature = "akita")]
    inc_virtualization: &crate::stages::inc_virtualization::IncVirtualizationOutput<F>,
) -> Stage6bInputPoints<F> {
    Stage6bInputPoints {
        ram_ra_virtualization: ram_ra_virtualization_input_points_from_upstream(stage5),
        instruction_ra_virtualization: instruction_ra_virtualization_input_points_from_upstream(
            stage5,
        ),
        #[cfg(not(feature = "akita"))]
        inc_claim_reduction: inc_claim_reduction_input_points_from_upstream(stage2, stage4, stage5),
        #[cfg(feature = "akita")]
        fused_inc_claim_reduction: fused_inc_input_points_from_upstream(inc_virtualization),
        ..sumchecks.empty_input_points()
    }
}

/// The stage-6b Fiat-Shamir opening-claim values in canonical absorb order.
/// Full relations and the optional members single-source their per-field order
/// from the `OutputClaims` derive's `opening_values`; `booleanity` stays
/// explicit because its `bytecode_ra` openings are conditionally deduped
/// against the bytecode-read-RAF points (a runtime point-equality the output
/// `Expr`s cannot express). Public because the prover's recorder absorbs the
/// same curated sequence.
pub fn stage6b_opening_values<F: Field>(
    claims: &Stage6bOutputClaims<F>,
    bytecode_read_raf_points: &[Vec<F>],
    booleanity_point: &[F],
) -> Vec<F> {
    let mut values = claims.bytecode_read_raf.opening_values();
    values.extend(&claims.booleanity.instruction_ra);
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_read_raf_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_point)
        {
            continue;
        }
        values.push(*opening_claim);
    }
    values.extend(&claims.booleanity.ram_ra);
    #[cfg(feature = "akita")]
    {
        values.extend(&claims.booleanity.unsigned_inc_chunks);
        values.push(claims.booleanity.unsigned_inc_msb);
    }
    values.extend(claims.ram_hamming_booleanity.opening_values());
    values.extend(claims.ram_ra_virtualization.opening_values());
    values.extend(claims.instruction_ra_virtualization.opening_values());
    #[cfg(not(feature = "akita"))]
    values.extend(claims.inc_claim_reduction.opening_values());
    #[cfg(feature = "akita")]
    values.extend(claims.fused_inc_claim_reduction.opening_values());
    // Each advice member is a single-slot per-kind claims struct, so it
    // contributes exactly its own kind's opening.
    if let Some(advice) = &claims.trusted_advice {
        values.extend(advice.opening_values());
    }
    if let Some(advice) = &claims.untrusted_advice {
        values.extend(advice.opening_values());
    }
    if let Some(reduction) = &claims.bytecode_reduction {
        values.extend(reduction.opening_values());
    }
    if let Some(reduction) = &claims.program_image_reduction {
        values.extend(reduction.opening_values());
    }
    values
}

fn append_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6bOutputClaims<F>,
    bytecode_read_raf_points: &[Vec<F>],
    booleanity_point: &[F],
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    // Full relations and the optional members delegate to their derived
    // `append_openings`, single-sourcing the per-field Fiat-Shamir order from the
    // `OutputClaims` derive. `booleanity` stays explicit because its `bytecode_ra`
    // openings are conditionally deduped against the bytecode-read-RAF points.
    claims.bytecode_read_raf.append_openings(transcript);
    for opening_claim in &claims.booleanity.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_read_raf_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_point)
        {
            continue;
        }
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    #[cfg(feature = "akita")]
    {
        for opening_claim in &claims.booleanity.unsigned_inc_chunks {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
        transcript.append_labeled(b"opening_claim", &claims.booleanity.unsigned_inc_msb);
    }
    claims.ram_hamming_booleanity.append_openings(transcript);
    claims.ram_ra_virtualization.append_openings(transcript);
    claims
        .instruction_ra_virtualization
        .append_openings(transcript);
    #[cfg(not(feature = "akita"))]
    claims.inc_claim_reduction.append_openings(transcript);
    #[cfg(feature = "akita")]
    claims.fused_inc_claim_reduction.append_openings(transcript);
    // The optional members single-source their per-field Fiat-Shamir order from the
    // `OutputClaims` derive too. Each advice member is a single-slot per-kind claims
    // struct, so it absorbs exactly its own kind's opening.
    if let Some(advice) = &claims.trusted_advice {
        advice.append_openings(transcript);
    }
    if let Some(advice) = &claims.untrusted_advice {
        advice.append_openings(transcript);
    }
    if let Some(reduction) = &claims.bytecode_reduction {
        reduction.append_openings(transcript);
    }
    if let Some(reduction) = &claims.program_image_reduction {
        reduction.append_openings(transcript);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "akita"))]
    use super::super::booleanity::BooleanityOutputClaims;
    use super::super::bytecode_read_raf::BytecodeReadRafOutputClaims;
    #[cfg(feature = "akita")]
    use super::super::fused_inc_claim_reduction::FusedIncClaimReductionOutputClaims;
    #[cfg(not(feature = "akita"))]
    use super::super::inc_claim_reduction::IncClaimReductionOutputClaims;
    use super::super::instruction_ra_virtualization::InstructionRaVirtualizationOutputClaims;
    use super::super::ram_hamming_booleanity::RamHammingBooleanityOutputClaims;
    use super::super::ram_ra_virtualization::RamRaVirtualizationOutputClaims;
    use super::*;
    use crate::stages::relations::append_recording::RecordingTranscript;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// Per-mode sample claims with sentinel values in the canonical append
    /// order: base interleaves the inc member after the RA virtualizations;
    /// Akita replaces it with the lattice booleanity chunk/MSB cells and the
    /// separate fused-increment reduction output.
    fn sample_claims() -> (Stage6bOutputClaims<Fr>, u64) {
        #[cfg(not(feature = "akita"))]
        let (booleanity, last) = (
            BooleanityOutputClaims {
                instruction_ra: vec![fr(3)],
                bytecode_ra: vec![fr(4)],
                ram_ra: vec![fr(5)],
            },
            10,
        );
        #[cfg(feature = "akita")]
        let (booleanity, last) = (
            jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityOutputClaims {
                instruction_ra: vec![fr(3)],
                bytecode_ra: vec![fr(4)],
                ram_ra: vec![fr(5)],
                unsigned_inc_chunks: vec![fr(6)],
                unsigned_inc_msb: fr(7),
            },
            11,
        );
        #[cfg(not(feature = "akita"))]
        let (hamming, ram_ra_virt, instruction_ra_virt) = (fr(6), fr(7), fr(8));
        #[cfg(feature = "akita")]
        let (hamming, ram_ra_virt, instruction_ra_virt) = (fr(8), fr(9), fr(10));
        (
            Stage6bOutputClaims {
                bytecode_read_raf: BytecodeReadRafOutputClaims {
                    bytecode_ra: vec![fr(1), fr(2)],
                },
                booleanity,
                ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                    ram_hamming_weight: hamming,
                },
                ram_ra_virtualization: RamRaVirtualizationOutputClaims {
                    ram_ra: vec![ram_ra_virt],
                },
                instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                    committed_instruction_ra: vec![instruction_ra_virt],
                },
                #[cfg(feature = "akita")]
                fused_inc_claim_reduction: FusedIncClaimReductionOutputClaims { fused_inc: fr(11) },
                #[cfg(not(feature = "akita"))]
                inc_claim_reduction: IncClaimReductionOutputClaims {
                    ram_inc: fr(9),
                    rd_inc: fr(10),
                },
                trusted_advice: None,
                untrusted_advice: None,
                bytecode_reduction: None,
                program_image_reduction: None,
            },
            last,
        )
    }

    /// Locks the stage-6b cycle-phase Fiat-Shamir append order against silent drift.
    /// The full relations are single-sourced via their `OutputClaims` derive;
    /// `booleanity` (conditional `bytecode_ra` dedup) and the optional reductions
    /// stay explicit. Points are empty so no `bytecode_ra` element is deduped;
    /// the `None` reductions carry absent sentinels to prove they are not appended.
    #[test]
    fn append_opening_claims_follows_canonical_order() {
        let (claims, last) = sample_claims();

        let mut got = RecordingTranscript::default();
        append_opening_claims(&mut got, &claims, &[], &[]);

        let mut want = RecordingTranscript::default();
        for value in (1..=last).map(fr) {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }
}
