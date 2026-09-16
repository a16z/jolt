//! Construction of the stage-6b cycle-phase sumcheck batch.
//!
//! [`Stage6bSumchecks::build`] assembles the batch members ONCE, after
//! stage 6a and the post-6a draws, directly from the upstream stage outputs. It
//! derives the mode-agnostic constructor legs (per-stage cycle bindings, reduced
//! points, the stage-6a address openings) plus the clear-only value aux
//! (`table_fold`, `address_val_stages`, base advice reference points — each
//! empty/`None` in ZK, where `expected_output` never runs) as a single contiguous
//! block before constructing the members. The four `Option` members are present
//! exactly when their precommitted layout needs a cycle-phase reduction, so the
//! batch's instance count matches the prover's.

#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_claims::protocols::jolt::{
    geometry::{
        booleanity::BooleanityDimensions,
        claim_reductions::bytecode::BytecodeLaneWeightInputs,
        dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
    },
    JoltRelationId,
};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
use jolt_riscv::JoltInstructionRow;
use jolt_transcript::Transcript;

use super::booleanity::{Booleanity, BooleanityCyclePhaseChallenges};
use super::bytecode_read_raf::{
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle, BytecodeReadRafCycleInputs,
    BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafTableFoldInputs,
    READ_RAF_CYCLE_STAGES,
};
#[cfg(not(feature = "akita"))]
use super::committed_reduction_cycle_phase::advice_reference_point_from_upstream;
use super::committed_reduction_cycle_phase::{
    bytecode_reduction_weights, BytecodeReductionCyclePhase, BytecodeReductionCyclePhaseChallenges,
    ProgramImageReductionCyclePhase,
};
#[cfg(not(feature = "akita"))]
use super::committed_reduction_cycle_phase::{TrustedAdviceCyclePhase, UntrustedAdviceCyclePhase};
#[cfg(feature = "field-inline")]
use super::field_registers_inc_claim_reduction::FieldRegistersIncClaimReductionChallenges;
#[cfg(not(feature = "akita"))]
use super::inc_claim_reduction::{IncClaimReduction, IncClaimReductionChallenges};
use super::instruction_ra_virtualization::{
    InstructionRaVirtualization, InstructionRaVirtualizationChallenges,
};
use super::outputs::{Stage6bChallenges, Stage6bSumchecks};
use super::ram_hamming_booleanity::RamHammingBooleanity;
use super::ram_ra_virtualization::RamRaVirtualization;
use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
#[cfg(feature = "field-inline")]
use crate::stages::field_inline_bytecode::FieldInlineBytecodeTable;
use crate::stages::stage1::Stage1Output;
use crate::stages::stage2::{Stage2BatchOutputPoints, Stage2Output};
use crate::stages::stage3::outputs::Stage3OutputPoints;
use crate::stages::stage3::Stage3Output;
use crate::stages::stage4::outputs::Stage4OutputPoints;
use crate::stages::stage4::Stage4Output;
use crate::stages::stage5::outputs::Stage5OutputPoints;
use crate::stages::stage5::Stage5Output;
use crate::stages::stage6a::bytecode_read_raf::bytecode_stage_points;
use crate::stages::stage6a::outputs::{Stage6aCarriedChallenges, Stage6aOutputPoints};
use crate::stages::stage6a::Stage6aOutput;
use crate::stages::{stage6_checked_split, PrecommittedSchedule};
use crate::verifier::CheckedInputs;
use crate::VerifierError;

/// The batch legs [`Stage6bSumchecks::build_from_parts`] assembles the members
/// from: protocol geometry, the precommitted schedule, the carried stage-6a
/// draws, the mode-agnostic upstream opening points, and the clear-only value
/// aux (each empty/`None` in ZK, where `input_claim`/`expected_output` never
/// run). Every field is data both the verifier and the prover hold.
pub struct Stage6bBuildParts<'a, F: JoltField> {
    pub formula_dimensions: &'a JoltFormulaDimensions,
    pub ram_log_k: usize,
    pub committed_chunk_bits: usize,
    pub precommitted: &'a PrecommittedSchedule,
    pub entry_bytecode_index: usize,
    /// The full bytecode rows backing the full-program table fold
    /// (`None` in ZK and committed-program modes).
    pub bytecode_table_rows: Option<&'a [JoltInstructionRow]>,
    /// The converted field-inline bytecode side table (required: the FR-on
    /// verifier rejects preprocessing without it before assembling parts).
    #[cfg(feature = "field-inline")]
    pub field_inline_bytecode: FieldInlineBytecodeTable,
    pub carried: &'a Stage6aCarriedChallenges<F>,
    pub eta: Option<F>,
    pub stage1_cycle_binding: Vec<F>,
    pub stage2_points: &'a Stage2BatchOutputPoints<F>,
    pub stage3_points: &'a Stage3OutputPoints<F>,
    pub stage4_points: &'a Stage4OutputPoints<F>,
    pub stage5_points: &'a Stage5OutputPoints<F>,
    pub stage6a_points: &'a Stage6aOutputPoints<F>,
    /// The staged `BytecodeValClaim` openings (clear committed-program mode;
    /// empty otherwise).
    pub address_val_stages: Vec<F>,
    #[cfg(not(feature = "akita"))]
    pub trusted_advice_reference_point: Option<Vec<F>>,
    #[cfg(not(feature = "akita"))]
    pub untrusted_advice_reference_point: Option<Vec<F>>,
}

/// The post-6a Fiat-Shamir draws, sampled before the batch is built (the batch
/// members carry them as constructor legs). Both fronts call
/// [`draw`](Self::draw), so the squeeze order is single-sourced.
pub struct Stage6bDraws<F> {
    pub instruction_ra_gamma: F,
    /// Base only: the packed batch has no inc claim-reduction member.
    #[cfg(not(feature = "akita"))]
    pub inc_gamma: F,
    /// The FR increment-reduction gamma (the spec's `eta`), member-drawn in
    /// declaration order: after the ordinary inc gamma, before the optional
    /// committed-bytecode eta.
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_gamma: F,
    /// The bytecode claim-reduction eta, drawn exactly when the bytecode
    /// layout is committed.
    pub eta: Option<F>,
}

impl<F: JoltField> Stage6bDraws<F> {
    pub fn draw<T: Transcript<Challenge = F>>(
        transcript: &mut T,
        committed_bytecode: bool,
    ) -> Self {
        // Field order is draw order: the struct literal evaluates in
        // declaration order.
        Self {
            instruction_ra_gamma: transcript.challenge_scalar(),
            #[cfg(not(feature = "akita"))]
            inc_gamma: transcript.challenge_scalar(),
            #[cfg(feature = "field-inline")]
            field_registers_inc_gamma: transcript.challenge_scalar(),
            eta: committed_bytecode.then(|| transcript.challenge_scalar()),
        }
    }
}

impl<F: JoltField> Stage6bSumchecks<F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6b's batch is built from the stage-6a output plus all five prior stage outputs directly; bundling them would reintroduce the removed `Stage6bParams` pack/unpack indirection."
    )]
    pub(super) fn build<PCS, VC, ZkProof>(
        checked: &CheckedInputs,
        preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
        proof: &JoltProof<PCS, VC, ZkProof>,
        formula_dimensions: &JoltFormulaDimensions,
        stage1: &Stage1Output<F, VC::Output>,
        stage2: &Stage2Output<F, VC::Output>,
        stage3: &Stage3Output<F, VC::Output>,
        stage4: &Stage4Output<F, VC::Output>,
        stage5: &Stage5Output<F, VC::Output>,
        stage6a: &Stage6aOutput<F, VC::Output>,
        eta: Option<F>,
    ) -> Result<Self, VerifierError>
    where
        PCS: CommitmentScheme<Field = F>,
        VC: VectorCommitment<Field = F>,
    {
        // The pre-/around-6a draws consumed by the legs ride on the stage-6a
        // output as typed upstream values; the mode-specific value aux (the
        // staged Val openings, the advice reference points, the full bytecode
        // rows) feeds only `input_claim` / `expected_output`, which never run
        // in ZK.
        let committed_program = checked.precommitted.bytecode.is_some();
        let stage1_cycle_binding = stage1.cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
        let entry_bytecode_index = preprocessing
            .program
            .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;
        let bytecode_table_rows = if checked.zk || committed_program {
            None
        } else {
            Some(
                preprocessing
                    .program
                    .as_full()
                    .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::BytecodeReadRaf,
                        reason: "full bytecode table is unavailable".to_string(),
                    })?
                    .bytecode
                    .bytecode
                    .as_slice(),
            )
        };
        let address_val_stages = if checked.zk {
            Vec::new()
        } else {
            stage6a
                .clear()?
                .output_values
                .bytecode_read_raf
                .val_stages
                .clone()
        };
        #[cfg(not(feature = "akita"))]
        let (trusted_advice_reference_point, untrusted_advice_reference_point) = if checked.zk {
            (None, None)
        } else {
            let stage4 = stage4.clear()?;
            (
                advice_reference_point_from_upstream(
                    &stage4.ram_val_check_init,
                    JoltAdviceKind::Trusted,
                ),
                advice_reference_point_from_upstream(
                    &stage4.ram_val_check_init,
                    JoltAdviceKind::Untrusted,
                ),
            )
        };

        #[cfg(feature = "field-inline")]
        let field_inline_bytecode =
            super::field_inline::preprocessed_bytecode_table(&preprocessing.program)?;

        Self::build_from_parts(Stage6bBuildParts {
            formula_dimensions,
            ram_log_k: crate::num::ilog2(checked.ram_K),
            committed_chunk_bits: proof.one_hot_config.committed_chunk_bits(),
            precommitted: &checked.precommitted,
            entry_bytecode_index,
            bytecode_table_rows,
            #[cfg(feature = "field-inline")]
            field_inline_bytecode,
            carried: stage6a.challenges(),
            eta,
            stage1_cycle_binding,
            stage2_points: stage2.batch_output_points(),
            stage3_points: stage3.output_points(),
            stage4_points: stage4.output_points(),
            stage5_points: stage5.output_points(),
            stage6a_points: stage6a.output_points(),
            address_val_stages,
            #[cfg(not(feature = "akita"))]
            trusted_advice_reference_point,
            #[cfg(not(feature = "akita"))]
            untrusted_advice_reference_point,
        })
    }

    /// The leg-assembly core of [`build`](Self::build), over data both sides
    /// hold: the prove-side stage-6b recipe constructs the batch through this
    /// same constructor from its clear carriers, so the ten member legs are
    /// single-sourced.
    pub fn build_from_parts(parts: Stage6bBuildParts<'_, F>) -> Result<Self, VerifierError> {
        let Stage6bBuildParts {
            formula_dimensions,
            ram_log_k: log_k,
            committed_chunk_bits,
            precommitted,
            entry_bytecode_index,
            bytecode_table_rows,
            #[cfg(feature = "field-inline")]
            field_inline_bytecode,
            carried,
            eta,
            stage1_cycle_binding,
            stage2_points,
            stage3_points,
            stage4_points,
            stage5_points,
            stage6a_points,
            address_val_stages,
            #[cfg(not(feature = "akita"))]
            trusted_advice_reference_point,
            #[cfg(not(feature = "akita"))]
            untrusted_advice_reference_point,
        } = parts;
        let log_t = formula_dimensions.trace.log_t();
        let trace_dimensions = formula_dimensions.trace;

        #[cfg(not(feature = "akita"))]
        let trusted_advice_layout = precommitted.trusted_advice.as_ref();
        #[cfg(not(feature = "akita"))]
        let untrusted_advice_layout = precommitted.untrusted_advice.as_ref();
        let bytecode_reduction_layout = precommitted.bytecode.as_ref();
        let program_image_reduction_layout = precommitted.program_image.as_ref();
        let committed_program = bytecode_reduction_layout.is_some();

        // (The verifier's own `build` already rejected at the metadata
        // requirement; this guards the shared parts-level entry too.)
        #[cfg(feature = "field-inline")]
        super::field_inline::require_full_program(committed_program)?;

        let booleanity_dimensions =
            BooleanityDimensions::new(formula_dimensions.ra_layout, log_t, committed_chunk_bits);

        // The bytecode folds below consume per-stage power VECTORS, expanded
        // once here from the carried scalars.
        let stage_gamma_powers = carried.bytecode_read_raf.stage_gamma_powers();
        let bytecode_r_address = stage6a_points.bytecode_read_raf.intermediate.clone();
        let booleanity_r_address = stage6a_points.booleanity.intermediate.clone();

        // Cycle-phase constructor legs, wired mode-agnostically off the upstream
        // outputs; the post-batch opening points are derived against these same
        // values through the relation objects.
        let stage5_instruction_cycle = stage5_points.instruction_r_cycle();
        let stage_points = bytecode_stage_points(
            &stage1_cycle_binding,
            stage2_points,
            stage3_points,
            stage4_points,
            stage5_points,
        )?;
        #[expect(
            clippy::indexing_slicing,
            reason = "bytecode_stage_points validated both register points against REGISTER_ADDRESS_BITS via stage6_checked_split"
        )]
        let register_read_write_address =
            &stage_points.register_read_write_point[..REGISTER_ADDRESS_BITS];
        #[expect(
            clippy::indexing_slicing,
            reason = "bytecode_stage_points validated both register points against REGISTER_ADDRESS_BITS via stage6_checked_split"
        )]
        let register_val_evaluation_address =
            &stage_points.register_val_evaluation_point[..REGISTER_ADDRESS_BITS];
        let ram_reduced = stage5_points.ram_reduced_opening_point();
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "log_k and log_t are ilog2 results (< 64); the sum cannot overflow usize"
        )]
        let ram_reduced_len = log_k + log_t;
        if ram_reduced.len() != ram_reduced_len {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: format!(
                    "Stage 6 RAM RA reduction opening point length mismatch: expected {ram_reduced_len}, got {}",
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
        let registers_read_write_cycle = stage_points.register_read_write_cycle().to_vec();
        let registers_val_evaluation_cycle = stage_points.register_val_evaluation_cycle().to_vec();
        // The FR opening sub-points: the stage-4/5 FR opening points split
        // past the FR address prefix. The cycle legs feed both the bytecode
        // FR public fold and the FR increment reduction's Eq publics.
        #[cfg(feature = "field-inline")]
        let field_inline_legs = super::field_inline::bytecode_fold_and_cycles(
            field_inline_bytecode,
            carried,
            stage4_points,
            stage5_points,
        )?;
        #[cfg(not(feature = "akita"))]
        let stage_cycle_points: [Vec<F>; READ_RAF_CYCLE_STAGES] = stage_points.stage_cycle_points;
        // The packed fused-inc consumer points appended to the shared five: the
        // four inc-producing relations' cycle bindings, in stage order (γ^5..8).
        // The register cycle vectors move in here (no clones): the akita build
        // fuses the inc reduction into the read-RAF legs, so no `IncClaimReduction`
        // member consumes them.
        #[cfg(feature = "akita")]
        let stage_cycle_points: [Vec<F>; READ_RAF_CYCLE_STAGES] = {
            let [stage1, stage2, stage3, stage4, stage5] = stage_points.stage_cycle_points;
            [
                stage1,
                stage2,
                stage3,
                stage4,
                stage5,
                ram_read_write_cycle.to_vec(),
                ram_val_check_cycle.to_vec(),
                registers_read_write_cycle,
                registers_val_evaluation_cycle,
            ]
        };
        // The full-program table fold is expected_output-only (absent rows mean
        // ZK or committed mode, where it never runs).
        let bytecode_table_fold =
            bytecode_table_rows.map(|bytecode| BytecodeReadRafTableFoldInputs {
                bytecode,
                register_read_write_point: register_read_write_address,
                register_val_evaluation_point: register_val_evaluation_address,
                stage_gammas: stage_gamma_powers.each_ref().map(Vec::as_slice),
            });
        // Both fronts draw `eta` exactly when the bytecode layout is committed;
        // a front that broke the coupling would otherwise surface only as a
        // downstream transcript mismatch, so reject it here by name.
        let cycle_bytecode_reduction_weights = match (bytecode_reduction_layout, eta) {
            (Some(layout), Some(eta)) => Some(bytecode_reduction_weights(
                layout,
                BytecodeLaneWeightInputs {
                    eta,
                    stage1_gammas: &stage_gamma_powers[0],
                    stage2_gammas: &stage_gamma_powers[1],
                    stage3_gammas: &stage_gamma_powers[2],
                    stage4_gammas: &stage_gamma_powers[3],
                    stage5_gammas: &stage_gamma_powers[4],
                    register_read_write_point: register_read_write_address,
                    register_val_evaluation_point: register_val_evaluation_address,
                },
                &bytecode_r_address,
            )?),
            (None, None) => None,
            (Some(_), None) | (None, Some(_)) => {
                return Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
                    reason: "the bytecode claim-reduction eta must be drawn exactly when the \
                             bytecode layout is committed"
                        .to_string(),
                })
            }
        };

        let bytecode_read_raf = if committed_program {
            BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                dimensions: formula_dimensions.bytecode_read_raf,
                r_address: bytecode_r_address,
                stage_cycle_points,
                entry_bytecode_index,
                committed_chunk_bits,
                val_stages: address_val_stages,
            })
        } else {
            BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                dimensions: formula_dimensions.bytecode_read_raf,
                r_address: bytecode_r_address,
                stage_cycle_points,
                entry_bytecode_index,
                committed_chunk_bits,
                table_fold: bytecode_table_fold,
                #[cfg(feature = "field-inline")]
                field_inline: field_inline_legs.fold,
            })?
        };

        #[cfg(feature = "akita")]
        let booleanity_dimensions =
            jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityDimensions::new(
                booleanity_dimensions,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            })?;
        // The little-endian reference cycle is construction geometry (the
        // reversed stage-5 instruction cycle, no draw of its own), so it is
        // rederived from the stage-5 point rather than carried with the
        // stage-6a draws.
        let booleanity = Booleanity::new(
            booleanity_dimensions,
            booleanity_r_address,
            carried.booleanity.reference_address.clone(),
            stage5_instruction_cycle.iter().rev().copied().collect(),
        );
        let ram_hamming_booleanity =
            RamHammingBooleanity::new(trace_dimensions, stage1_cycle_binding);
        let ram_ra_virtualization = RamRaVirtualization::new(
            formula_dimensions.ram_ra_virtualization,
            ram_reduced_address.to_vec(),
            ram_reduced_cycle.to_vec(),
            committed_chunk_bits,
        );
        let instruction_ra_virtualization = InstructionRaVirtualization::new(
            formula_dimensions.instruction_ra_virtualization,
            stage5_points.instruction_r_address(),
            stage5_instruction_cycle.to_vec(),
            committed_chunk_bits,
        );
        #[cfg(not(feature = "akita"))]
        let inc_claim_reduction = IncClaimReduction::new(
            trace_dimensions,
            ram_read_write_cycle.to_vec(),
            ram_val_check_cycle.to_vec(),
            registers_read_write_cycle,
            registers_val_evaluation_cycle,
        );
        #[cfg(feature = "field-inline")]
        let field_registers_inc_claim_reduction = super::field_inline::inc_claim_reduction_member(
            log_t,
            field_inline_legs.read_write_cycle,
            field_inline_legs.val_evaluation_cycle,
        );

        #[cfg(not(feature = "akita"))]
        let trusted_advice = trusted_advice_layout
            .map(|layout| TrustedAdviceCyclePhase::new(layout, trusted_advice_reference_point));
        #[cfg(not(feature = "akita"))]
        let untrusted_advice = untrusted_advice_layout
            .map(|layout| UntrustedAdviceCyclePhase::new(layout, untrusted_advice_reference_point));
        let bytecode_reduction = bytecode_reduction_layout
            .zip(cycle_bytecode_reduction_weights)
            .map(|(layout, weights)| BytecodeReductionCyclePhase::new(layout, weights));
        let program_image_reduction = program_image_reduction_layout.map(|layout| {
            ProgramImageReductionCyclePhase::new(layout, ram_val_check_address.to_vec())
        });

        Ok(Self {
            bytecode_read_raf,
            booleanity,
            ram_hamming_booleanity,
            ram_ra_virtualization,
            instruction_ra_virtualization,
            #[cfg(not(feature = "akita"))]
            inc_claim_reduction,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction,
            #[cfg(not(feature = "akita"))]
            trusted_advice,
            #[cfg(not(feature = "akita"))]
            untrusted_advice,
            bytecode_reduction,
            program_image_reduction,
        })
    }

    /// The stage challenges aggregate, hand-assembled for both fronts (the
    /// batch suppresses the generated `draw_challenges`): the bytecode gamma
    /// shares stage 6a's squeeze and the booleanity gamma was drawn pre-6a, so
    /// a generated per-member draw would squeeze for them at the wrong
    /// transcript position. The `Option` member slots mirror this batch's
    /// instance presence.
    pub fn cycle_challenges(
        &self,
        carried: &Stage6aCarriedChallenges<F>,
        draws: &Stage6bDraws<F>,
    ) -> Stage6bChallenges<F> {
        Stage6bChallenges {
            bytecode_read_raf: BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: carried.bytecode_read_raf.gamma,
            },
            booleanity: BooleanityCyclePhaseChallenges {
                gamma: carried.booleanity.gamma,
            },
            ram_hamming_booleanity: NoChallenges::default(),
            ram_ra_virtualization: NoChallenges::default(),
            instruction_ra_virtualization: InstructionRaVirtualizationChallenges {
                gamma: draws.instruction_ra_gamma,
            },
            #[cfg(not(feature = "akita"))]
            inc_claim_reduction: IncClaimReductionChallenges {
                gamma: draws.inc_gamma,
            },
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: FieldRegistersIncClaimReductionChallenges {
                gamma: draws.field_registers_inc_gamma,
            },
            #[cfg(not(feature = "akita"))]
            trusted_advice: self
                .trusted_advice
                .as_ref()
                .map(|_| NoChallenges::default()),
            #[cfg(not(feature = "akita"))]
            untrusted_advice: self
                .untrusted_advice
                .as_ref()
                .map(|_| NoChallenges::default()),
            bytecode_reduction: self
                .bytecode_reduction
                .as_ref()
                .zip(draws.eta)
                .map(|(_, eta)| BytecodeReductionCyclePhaseChallenges { eta }),
            program_image_reduction: self
                .program_image_reduction
                .as_ref()
                .map(|_| NoChallenges::default()),
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_field::Fr;

    /// Pins the post-6a draw schedule to member declaration order: the
    /// instruction-RA gamma, (base) the inc gamma, under `field-inline` the FR
    /// inc gamma (the spec's `eta` draw slot: after the ordinary inc gamma,
    /// before the optional committed-bytecode eta), then the committed
    /// bytecode eta exactly when the bytecode layout is committed.
    #[test]
    fn stage6b_draws_follow_member_declaration_order() {
        for committed_bytecode in [false, true] {
            let mut expected_squeezes = 1usize;
            #[cfg(not(feature = "akita"))]
            {
                expected_squeezes += 1;
            }
            #[cfg(feature = "field-inline")]
            {
                expected_squeezes += 1;
            }
            expected_squeezes += usize::from(committed_bytecode);

            let (inline_events, inline_values) = record(|t| {
                (0..expected_squeezes)
                    .map(|_| t.challenge_scalar())
                    .collect::<Vec<Fr>>()
            });
            let (draw_events, draws) = record(|t| Stage6bDraws::<Fr>::draw(t, committed_bytecode));

            assert_eq!(draw_events, inline_events);
            assert_eq!(
                draw_events,
                (1..=expected_squeezes as u64)
                    .map(DrawEvent::Squeeze)
                    .collect::<Vec<_>>()
            );
            let mut ordered = vec![draws.instruction_ra_gamma];
            #[cfg(not(feature = "akita"))]
            ordered.push(draws.inc_gamma);
            #[cfg(feature = "field-inline")]
            ordered.push(draws.field_registers_inc_gamma);
            ordered.extend(draws.eta);
            assert_eq!(ordered, inline_values);
            assert_eq!(draws.eta.is_some(), committed_bytecode);
        }
    }
}
