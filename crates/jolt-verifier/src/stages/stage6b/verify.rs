use jolt_claims::protocols::jolt::{
    geometry::{
        booleanity::{self, BooleanityDimensions},
        bytecode::{self},
        claim_reductions::{
            advice,
            bytecode::{self as bytecode_reduction, BytecodeLaneWeightInputs},
            program_image,
        },
        dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
        instruction,
    },
    BytecodeClaimReductionLayout, JoltAdviceKind, JoltRelationId, PrecommittedReductionLayout,
};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;
use num_traits::One;

use super::{
    batch::Stage6bParams,
    booleanity::{BooleanityCyclePhaseChallenges, BooleanityInputClaims},
    bytecode_read_raf::{
        BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
        BytecodeReadRafTableFoldInputs,
    },
    committed_reduction_cycle_phase::{
        advice_cycle_phase_input_values_from_upstream,
        program_image_reduction_cycle_phase_input_values_from_upstream,
        AdviceCyclePhaseInputClaims, BytecodeReductionCyclePhaseChallenges,
        BytecodeReductionCyclePhaseInputClaims, ProgramImageReductionCyclePhaseInputClaims,
    },
    inc_claim_reduction::{
        inc_claim_reduction_input_points_from_upstream,
        inc_claim_reduction_input_values_from_upstream, IncClaimReductionChallenges,
    },
    instruction_ra_virtualization::{
        instruction_ra_virtualization_input_points_from_upstream,
        instruction_ra_virtualization_input_values_from_upstream,
        InstructionRaVirtualizationChallenges,
    },
    outputs::{
        BytecodeReductionWeights, Stage6bCarriedChallenges, Stage6bChallenges, Stage6bClearOutput,
        Stage6bInputClaims, Stage6bInputPoints, Stage6bOutput, Stage6bOutputClaims,
        Stage6bSumchecks, Stage6bZkOutput,
    },
    ram_hamming_booleanity::RamHammingBooleanityInputClaims,
    ram_ra_virtualization::{
        ram_ra_virtualization_input_points_from_upstream,
        ram_ra_virtualization_input_values_from_upstream,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::OutputAppend,
        stage1::Stage1Output,
        stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output},
        stage3::Stage3Output,
        stage4::{Stage4ClearOutput, Stage4Output, Stage4OutputPoints},
        stage5::{Stage5Output, Stage5OutputClaims, Stage5OutputPoints},
        stage6a::{outputs::Stage6aOutputClaims, Stage6aOutput},
        zk::committed,
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
) -> Result<Stage6bOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = formula_dimensions.trace.log_t();
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = formula_dimensions.trace;

    let trusted_advice_layout = checked.precommitted.trusted_advice.as_ref();
    let untrusted_advice_layout = checked.precommitted.untrusted_advice.as_ref();
    let bytecode_reduction_layout = checked.precommitted.bytecode.as_ref();
    let program_image_reduction_layout = checked.precommitted.program_image.as_ref();
    let committed_program = bytecode_reduction_layout.is_some();

    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
    );

    // The pre-/around-6a draws consumed here: the bytecode fold gamma (shared with
    // stage 6a's squeeze), the per-stage folding gammas, and the booleanity gamma
    // and reference (drawn pre-6a where the prover's booleanity subprotocol samples
    // them). They ride on the stage-6a output as typed upstream values.
    let carried = stage6a.challenges();
    let bytecode_gamma = carried.bytecode_gamma_powers[1];
    let bytecode_r_address = stage6a
        .output_points()
        .bytecode_read_raf
        .intermediate
        .clone();
    let booleanity_r_address = stage6a.output_points().booleanity.intermediate.clone();

    // Post-6a draws: the instruction-RA virtualization gamma, the increment gamma,
    // and (committed-program only) the bytecode claim-reduction eta.
    let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
        formula_dimensions
            .instruction_ra_virtualization
            .num_virtual_ra_polys(),
    );
    // `powers(1)` keeps ONE as the folding gamma (not the squeezed scalar) — a
    // prover-matched edge the relation's default draw cannot reproduce.
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(PCS::Field::one);
    let inc_gamma = transcript.challenge_scalar();
    let eta = committed_program.then(|| transcript.challenge_scalar());

    // Cycle-phase constructor legs, wired mode-agnostically off the upstream
    // outputs; the post-batch opening points are derived against these same
    // values through the relation objects.
    let stage4_points = stage4.output_points();
    let stage5_points = stage5.output_points();
    let stage5_instruction_address = stage5.instruction_r_address();
    let stage5_instruction_cycle = stage5_points.instruction_r_cycle();
    let stage1_cycle_binding = stage6_stage1_cycle_binding(stage1)?;
    let stage2_points = stage2.batch_output_points();
    let stage3_points = stage3.output_points();
    let (register_read_write_address, register_read_write_cycle) = stage6_checked_split(
        "Stage 6 stage4 register read-write opening",
        stage4_points.registers_read_write_point(),
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let (register_val_evaluation_address, register_val_evaluation_cycle) = stage6_checked_split(
        "Stage 6 stage5 register value-evaluation opening",
        stage5_points.registers_opening_point(),
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let stage_cycle_points = [
        stage1_cycle_binding.iter().rev().copied().collect(),
        stage2_points.product_remainder_point().to_vec(),
        stage3_points.shift_opening_point().to_vec(),
        register_read_write_cycle.to_vec(),
        register_val_evaluation_cycle.to_vec(),
    ];
    let ram_reduced = stage5_points.ram_reduced_opening_point();
    if ram_reduced.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: format!(
                "Stage 6 RAM RA reduction opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_reduced.len()
            ),
        });
    }
    let (ram_reduced_address, ram_reduced_cycle) = ram_reduced.split_at(log_k);
    let (_, ram_read_write_cycle) = stage6_checked_split(
        "Stage 6 RAM read-write opening",
        stage2_points.ram_read_write_point(),
        log_k,
        JoltRelationId::IncClaimReduction,
    )?;
    let (ram_val_check_address, ram_val_check_cycle) = stage6_checked_split(
        "Stage 6 RAM value-check opening",
        stage4_points.ram_val_check_point(),
        log_k,
        JoltRelationId::IncClaimReduction,
    )?;
    let inc_cycle_points = [
        ram_read_write_cycle.to_vec(),
        ram_val_check_cycle.to_vec(),
        register_read_write_cycle.to_vec(),
        register_val_evaluation_cycle.to_vec(),
    ];
    let entry_bytecode_index = preprocessing
        .program
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    // The full-program table fold is expected_output-only, so ZK (which never
    // runs it) skips the aux entirely.
    let bytecode_table_fold = if checked.zk || committed_program {
        None
    } else {
        Some(BytecodeReadRafTableFoldInputs {
            bytecode: preprocessing
                .program
                .as_full()
                .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::BytecodeReadRaf,
                    reason: "full bytecode table is unavailable".to_string(),
                })?
                .bytecode
                .bytecode
                .as_slice(),
            register_read_write_point: register_read_write_address,
            register_val_evaluation_point: register_val_evaluation_address,
            stage_gammas: [
                &carried.stage1_gammas,
                &carried.stage2_gammas,
                &carried.stage3_gammas,
                &carried.stage4_gammas,
                &carried.stage5_gammas,
            ],
        })
    };
    let cycle_bytecode_reduction_weights = match (bytecode_reduction_layout, eta) {
        (Some(layout), Some(eta)) => Some(bytecode_reduction_weights(
            layout,
            BytecodeReductionWeightInputs {
                eta,
                stage1_gammas: &carried.stage1_gammas,
                stage2_gammas: &carried.stage2_gammas,
                stage3_gammas: &carried.stage3_gammas,
                stage4_gammas: &carried.stage4_gammas,
                stage5_gammas: &carried.stage5_gammas,
                register_read_write_point: register_read_write_address,
                register_val_evaluation_point: register_val_evaluation_address,
                bytecode_r_address: &bytecode_r_address,
            },
        )?),
        _ => None,
    };
    // Clear-only value legs: the staged Val openings and the advice reference
    // points feed only `input_claim` / `expected_output`, which never run in ZK.
    let (address_val_stages, trusted_advice_reference_point, untrusted_advice_reference_point) =
        if checked.zk {
            (Vec::new(), None, None)
        } else {
            let stage4 = stage4.clear()?;
            let claims_6a = &proof.clear_claims()?.stage6a;
            let reference = |kind| {
                stage4
                    .ram_val_check_init
                    .advice_contribution(kind)
                    .map(|contribution| contribution.opening_point.clone())
            };
            (
                claims_6a.bytecode_read_raf.val_stages.clone(),
                reference(JoltAdviceKind::Trusted),
                reference(JoltAdviceKind::Untrusted),
            )
        };

    let sumchecks = Stage6bSumchecks::build(Stage6bParams {
        bytecode_dimensions: formula_dimensions.bytecode_read_raf,
        booleanity_dimensions,
        trace_dimensions,
        ram_ra_dimensions: formula_dimensions.ram_ra_virtualization,
        instruction_ra_dimensions: formula_dimensions.instruction_ra_virtualization,
        committed_chunk_bits: proof.one_hot_config.committed_chunk_bits(),
        entry_bytecode_index,
        bytecode_table_fold,
        bytecode_r_address,
        booleanity_r_address,
        address_val_stages,
        stage_cycle_points,
        booleanity_reference_address: carried.booleanity_reference_address.clone(),
        booleanity_reference_cycle: carried.booleanity_reference_cycle.clone(),
        stage1_cycle_binding,
        ram_reduced_address: ram_reduced_address.to_vec(),
        ram_reduced_cycle: ram_reduced_cycle.to_vec(),
        instruction_r_address: stage5_instruction_address.to_vec(),
        instruction_r_cycle: stage5_instruction_cycle.to_vec(),
        inc_cycle_points,
        trusted_advice_layout,
        untrusted_advice_layout,
        bytecode_reduction_layout,
        program_image_reduction_layout,
        bytecode_reduction_weights: cycle_bytecode_reduction_weights.clone(),
        program_image_r_addr_rw: ram_val_check_address.to_vec(),
        trusted_advice_reference_point,
        untrusted_advice_reference_point,
    })?;

    // Hand-assembled (the generated `draw_challenges` is suppressed): the bytecode
    // gamma shares stage 6a's squeeze, the booleanity gamma was drawn pre-6a where
    // the prover's booleanity subprotocol samples it, and the instruction-RA gamma
    // keeps the `powers(1)` edge above.
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
        inc_claim_reduction: IncClaimReductionChallenges { gamma: inc_gamma },
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

    let input_points =
        stage6b_input_points_from_upstream(&sumchecks, stage2_points, stage4_points, stage5_points);

    if checked.zk {
        let consistency = sumchecks.verify_zk(&proof.stages.stage6b_sumcheck_proof, transcript)?;
        let cycle_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

        // The committed-claim count dedups runtime point aliases between the
        // booleanity bytecode-RA openings and the bytecode read-RAF openings,
        // so it cannot be derived from the output Exprs (stays hand-written).
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
        let bytecode_output_openings =
            bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
        let booleanity_output_openings =
            booleanity::booleanity_output_openings(formula_dimensions.ra_layout);
        let flat_instruction_ra_output_openings = instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        )
        .all();
        let bytecode_reduction_output_claims = bytecode_reduction_layout.map_or(0, |layout| {
            bytecode_reduction::cycle_phase_output_openings(
                layout.dimensions(),
                layout.chunk_count(),
            )
            .len()
        });
        let program_image_reduction_output_claims = program_image_reduction_layout
            .map_or(0, |layout| {
                program_image::cycle_phase_output_openings(layout.dimensions()).len()
            });
        let committed_output_claims = bytecode_output_openings.bytecode_ra.len()
            + booleanity_output_openings.len()
            - aliased_bytecode_ra_openings
            + 1
            + formula_dimensions
                .ram_ra_virtualization
                .num_committed_ra_polys()
            + flat_instruction_ra_output_openings.len()
            + 2
            + usize::from(sumchecks.trusted_advice.is_some())
            + usize::from(sumchecks.untrusted_advice.is_some())
            + bytecode_reduction_output_claims
            + program_image_reduction_output_claims;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage6b_sumcheck_proof,
            "stage6b_sumcheck_proof",
            committed_output_claims,
            JoltRelationId::BytecodeReadRaf,
        )?;

        return Ok(Stage6bOutput::Zk(Stage6bZkOutput {
            challenges: Stage6bCarriedChallenges {
                instruction_ra_gamma_powers,
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
    let claims_6a = &proof.clear_claims()?.stage6a;
    let claims = &proof.clear_claims()?.stage6b;

    // Reject opening claims supplied for cycle-phase members whose reduction did
    // not run (an absent `Option` member is silently skipped by
    // `expected_final_claim`, so a present claim for an uncommitted layout must be
    // rejected explicitly with the historical error ids). No transcript effect.
    if trusted_advice_layout.is_none() && claims.trusted_advice.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if untrusted_advice_layout.is_none() && claims.untrusted_advice.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted),
        });
    }
    if bytecode_reduction_layout.is_none() && claims.bytecode_reduction.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        });
    }
    if program_image_reduction_layout.is_none() && claims.program_image_reduction.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        });
    }

    let input_values = stage6b_input_values_from_upstream(
        &sumchecks,
        claims_6a,
        &stage2.output_values,
        stage4,
        &stage5.output_values,
    )?;
    let batch = sumchecks.verify_clear(
        &input_values,
        &cycle_challenges,
        &proof.stages.stage6b_sumcheck_proof,
        transcript,
    )?;

    validate_cycle_phase_claim_presence(
        formula_dimensions,
        claims,
        trusted_advice_layout,
        untrusted_advice_layout,
        bytecode_reduction_layout,
        program_image_reduction_layout,
    )?;

    let cycle_points =
        sumchecks.derive_opening_points(batch.reduction.point.as_slice(), &input_points)?;
    let expected_final_claim = sumchecks.expected_final_claim(
        &batch.coefficients,
        &input_points,
        claims,
        &cycle_points,
        &cycle_challenges,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 6 });
    }

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
        bytecode_reduction_weights: cycle_bytecode_reduction_weights,
    }))
}

/// The wire-shape and presence checks over the cycle-phase output claims that
/// the generated drivers cannot express: the bytecode RA claim count, the
/// per-member missing-claim guards (an absent `Option` member is silently
/// skipped by `expected_final_claim`, so a present member with missing claims
/// must be rejected explicitly with the historical error ids), and the bytecode
/// reduction's intermediate-vs-chunks shape.
fn validate_cycle_phase_claim_presence<F: Field>(
    formula_dimensions: &JoltFormulaDimensions,
    claims: &Stage6bOutputClaims<F>,
    trusted_advice_layout: Option<&jolt_claims::protocols::jolt::AdviceClaimReductionLayout>,
    untrusted_advice_layout: Option<&jolt_claims::protocols::jolt::AdviceClaimReductionLayout>,
    bytecode_reduction_layout: Option<&BytecodeClaimReductionLayout>,
    program_image_reduction_layout: Option<
        &jolt_claims::protocols::jolt::ProgramImageClaimReductionLayout,
    >,
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

    for (kind, layout, claim) in [
        (
            JoltAdviceKind::Trusted,
            trusted_advice_layout,
            &claims.trusted_advice,
        ),
        (
            JoltAdviceKind::Untrusted,
            untrusted_advice_layout,
            &claims.untrusted_advice,
        ),
    ] {
        let Some(layout) = layout else { continue };
        let claim = claim.as_ref().ok_or(VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_output_openings(kind, layout.dimensions())[0],
        })?;
        let opening_value = match kind {
            JoltAdviceKind::Trusted => claim.trusted,
            JoltAdviceKind::Untrusted => claim.untrusted,
        };
        if opening_value.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_advice_opening(kind),
            });
        }
    }

    if let Some(layout) = bytecode_reduction_layout {
        let output_claims =
            claims
                .bytecode_reduction
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim {
                    id: bytecode_reduction::cycle_phase_output_openings(
                        layout.dimensions(),
                        layout.chunk_count(),
                    )[0],
                })?;
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

    if let Some(layout) = program_image_reduction_layout {
        if claims.program_image_reduction.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: program_image::cycle_phase_output_openings(layout.dimensions())[0],
            });
        }
    }

    Ok(())
}

/// Assemble the stage-6b consumed opening *values* from the address-phase claims
/// and the upstream clear outputs into the generated `Stage6bInputClaims`
/// aggregate. The `Option` cells track member presence, so a present member always
/// has its input cell populated.
fn stage6b_input_values_from_upstream<F: Field>(
    sumchecks: &Stage6bSumchecks<F>,
    address_claims: &Stage6aOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5OutputClaims<F>,
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
        inc_claim_reduction: inc_claim_reduction_input_values_from_upstream(
            stage2,
            &stage4.output_values,
            stage5,
        ),
        trusted_advice: sumchecks.trusted_advice.as_ref().map(|_| {
            advice_cycle_phase_input_values_from_upstream(
                &stage4.ram_val_check_init,
                JoltAdviceKind::Trusted,
            )
        }),
        untrusted_advice: sumchecks.untrusted_advice.as_ref().map(|_| {
            advice_cycle_phase_input_values_from_upstream(
                &stage4.ram_val_check_init,
                JoltAdviceKind::Untrusted,
            )
        }),
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

/// Assemble the stage-6b consumed opening *points*. ZK-agnostic: the RA / inc
/// members read the upstream output-points aggregates (which both modes expose);
/// the remaining members derive their produced points from their own sumcheck
/// point and read no input point, so their cells are empty — but present for
/// present `Option` members, as the generated `derive_opening_points` requires.
fn stage6b_input_points_from_upstream<F: Field>(
    sumchecks: &Stage6bSumchecks<F>,
    stage2: &Stage2BatchOutputPoints<F>,
    stage4: &Stage4OutputPoints<F>,
    stage5: &Stage5OutputPoints<F>,
) -> Stage6bInputPoints<F> {
    Stage6bInputPoints {
        bytecode_read_raf: BytecodeReadRafInputClaims {
            address_phase: Vec::new(),
        },
        booleanity: BooleanityInputClaims {
            address_phase: Vec::new(),
        },
        ram_hamming_booleanity: RamHammingBooleanityInputClaims::default(),
        ram_ra_virtualization: ram_ra_virtualization_input_points_from_upstream(stage5),
        instruction_ra_virtualization: instruction_ra_virtualization_input_points_from_upstream(
            stage5,
        ),
        inc_claim_reduction: inc_claim_reduction_input_points_from_upstream(stage2, stage4, stage5),
        trusted_advice: sumchecks
            .trusted_advice
            .as_ref()
            .map(|_| AdviceCyclePhaseInputClaims {
                trusted: None,
                untrusted: None,
            }),
        untrusted_advice: sumchecks.untrusted_advice.as_ref().map(|_| {
            AdviceCyclePhaseInputClaims {
                trusted: None,
                untrusted: None,
            }
        }),
        bytecode_reduction: sumchecks.bytecode_reduction.as_ref().map(|_| {
            BytecodeReductionCyclePhaseInputClaims {
                val_stages: Vec::new(),
            }
        }),
        program_image_reduction: sumchecks.program_image_reduction.as_ref().map(|_| {
            ProgramImageReductionCyclePhaseInputClaims {
                contribution: Vec::new(),
            }
        }),
    }
}

fn stage6_stage1_cycle_binding<F: Field, C>(
    stage1: &Stage1Output<F, C>,
) -> Result<Vec<F>, VerifierError> {
    // The raw (un-reversed) remainder reduction point; its tail (`[1..]`) is the
    // cycle binding. This matches the ZK/BlindFold path, which slices
    // `remainder_consistency.challenges()[1..]` off the same raw point.
    let raw_point = stage1.remainder_point();
    let (_, cycle) =
        raw_point
            .split_first()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "Stage 1 remainder point is empty".to_string(),
            })?;
    Ok(cycle.to_vec())
}

fn stage6_checked_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
    stage: JoltRelationId,
) -> Result<(&'a [F], &'a [F]), VerifierError> {
    if point.len() < split_at {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: format!(
                "{label} has {} variables, expected at least {split_at}",
                point.len()
            ),
        });
    }
    Ok(point.split_at(split_at))
}

pub(crate) struct BytecodeReductionWeightInputs<'a, F: Field> {
    pub eta: F,
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    /// Full bytecode address point (the `BytecodeReadRafAddrClaim` opening).
    pub bytecode_r_address: &'a [F],
}

pub(crate) fn bytecode_reduction_weights<F: Field>(
    layout: &BytecodeClaimReductionLayout,
    inputs: BytecodeReductionWeightInputs<'_, F>,
) -> Result<BytecodeReductionWeights<F>, VerifierError> {
    let address_point = layout
        .split_address_point(inputs.bytecode_r_address)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let lane_weights = bytecode_reduction::lane_weights(BytecodeLaneWeightInputs {
        eta: inputs.eta,
        stage1_gammas: inputs.stage1_gammas,
        stage2_gammas: inputs.stage2_gammas,
        stage3_gammas: inputs.stage3_gammas,
        stage4_gammas: inputs.stage4_gammas,
        stage5_gammas: inputs.stage5_gammas,
        register_read_write_point: inputs.register_read_write_point,
        register_val_evaluation_point: inputs.register_val_evaluation_point,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
        reason: error.to_string(),
    })?;
    Ok(BytecodeReductionWeights {
        r_bc: address_point.r_bc,
        chunk_rbc_weights: address_point.chunk_rbc_weights,
        lane_weights,
    })
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
    // Full relations delegate to their derived `append_openings`, single-sourcing
    // the per-field Fiat-Shamir order from the `OutputClaims` derive. `booleanity`
    // stays explicit because its `bytecode_ra` openings are conditionally deduped
    // against the bytecode-read-RAF points.
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
    claims.ram_hamming_booleanity.append_openings(transcript);
    claims.ram_ra_virtualization.append_openings(transcript);
    claims
        .instruction_ra_virtualization
        .append_openings(transcript);
    claims.inc_claim_reduction.append_openings(transcript);
    if let Some(advice) = &claims.trusted_advice {
        if let Some(opening_claim) = &advice.trusted {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
    }
    if let Some(advice) = &claims.untrusted_advice {
        if let Some(opening_claim) = &advice.untrusted {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
    }
    if let Some(reduction) = &claims.bytecode_reduction {
        if let Some(opening_claim) = &reduction.intermediate {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
        for opening_claim in &reduction.chunks {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
    }
    if let Some(reduction) = &claims.program_image_reduction {
        transcript.append_labeled(b"opening_claim", &reduction.program_image);
    }
}

#[cfg(test)]
mod tests {
    use super::super::booleanity::BooleanityOutputClaims;
    use super::super::bytecode_read_raf::BytecodeReadRafOutputClaims;
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

    fn sample_claims() -> Stage6bOutputClaims<Fr> {
        Stage6bOutputClaims {
            bytecode_read_raf: BytecodeReadRafOutputClaims {
                bytecode_ra: vec![fr(1), fr(2)],
            },
            booleanity: BooleanityOutputClaims {
                instruction_ra: vec![fr(3)],
                bytecode_ra: vec![fr(4)],
                ram_ra: vec![fr(5)],
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                ram_hamming_weight: fr(6),
            },
            ram_ra_virtualization: RamRaVirtualizationOutputClaims {
                ram_ra: vec![fr(7)],
            },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra: vec![fr(8)],
            },
            inc_claim_reduction: IncClaimReductionOutputClaims {
                ram_inc: fr(9),
                rd_inc: fr(10),
            },
            trusted_advice: None,
            untrusted_advice: None,
            bytecode_reduction: None,
            program_image_reduction: None,
        }
    }

    /// Locks the stage-6b cycle-phase Fiat-Shamir append order against silent drift.
    /// The full relations are single-sourced via their `OutputClaims` derive;
    /// `booleanity` (conditional `bytecode_ra` dedup) and the optional reductions
    /// stay explicit. Points are empty so no `bytecode_ra` element is deduped;
    /// the `None` reductions carry absent sentinels to prove they are not appended.
    #[test]
    fn append_opening_claims_follows_canonical_order() {
        let claims = sample_claims();

        let mut got = RecordingTranscript::default();
        append_opening_claims(&mut got, &claims, &[], &[]);

        let mut want = RecordingTranscript::default();
        for value in (1..=10).map(fr) {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }
}
