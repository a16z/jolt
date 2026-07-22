//! Construction of the stage-6b cycle-phase sumcheck batch.
//!
//! [`Stage6bSumchecks::build`] assembles the batch members ONCE, after
//! stage 6a and the post-6a draws, directly from the upstream stage outputs. It
//! derives the mode-agnostic constructor legs (per-stage cycle bindings, reduced
//! points, the stage-6a address openings) plus the clear-only value aux
//! (`table_fold`, `address_val_stages`, advice reference points — each
//! empty/`None` in ZK, where `expected_output` never runs) as a single contiguous
//! block, preserving the fallible-check precedence, before constructing the
//! members. The four `Option` members are present exactly when their precommitted
//! layout is committed, in both proving modes, so the batch's instance count
//! matches the prover's.

use jolt_claims::protocols::jolt::{
    geometry::{
        booleanity::BooleanityDimensions,
        claim_reductions::bytecode::BytecodeLaneWeightInputs,
        dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
    },
    JoltAdviceKind, JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_riscv::JoltInstructionRow;

use super::booleanity::Booleanity;
use super::bytecode_read_raf::{
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle, BytecodeReadRafCycleInputs,
    BytecodeReadRafTableFoldInputs, READ_RAF_CYCLE_STAGES,
};
use super::committed_reduction_cycle_phase::{
    advice_reference_point_from_upstream, bytecode_reduction_weights, BytecodeReductionCyclePhase,
    ProgramImageReductionCyclePhase, TrustedAdviceCyclePhase, UntrustedAdviceCyclePhase,
};
#[cfg(not(feature = "akita"))]
use super::inc_claim_reduction::IncClaimReduction;
use super::instruction_ra_virtualization::InstructionRaVirtualization;
use super::outputs::Stage6bSumchecks;
use super::ram_hamming_booleanity::RamHammingBooleanity;
use super::ram_ra_virtualization::RamRaVirtualization;
use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
use crate::stages::stage1::Stage1Output;
use crate::stages::stage2::{Stage2BatchOutputPoints, Stage2Output};
use crate::stages::stage3::outputs::Stage3OutputPoints;
use crate::stages::stage3::Stage3Output;
use crate::stages::stage4::outputs::Stage4OutputPoints;
use crate::stages::stage4::Stage4Output;
use crate::stages::stage5::outputs::Stage5OutputPoints;
use crate::stages::stage5::Stage5Output;
use crate::stages::stage6a::outputs::{Stage6aCarriedChallenges, Stage6aOutputPoints};
use crate::stages::stage6a::Stage6aOutput;
use crate::stages::PrecommittedSchedule;
use crate::verifier::CheckedInputs;
use crate::VerifierError;

/// The bytecode read-RAF upstream cycle points shared by the stage-6a address
/// phase and the stage-6b batch build: the five per-stage cycle bindings (the
/// stage-1 binding is the raw remainder tail, re-reversed), plus the register
/// opening points whose 7-var address prefixes feed the stage-value folds.
#[derive(Clone)]
pub struct BytecodeStagePoints<F: Field> {
    pub stage_cycle_points: [Vec<F>; 5],
    pub register_read_write_point: Vec<F>,
    pub register_val_evaluation_point: Vec<F>,
}

impl<F: Field> BytecodeStagePoints<F> {
    /// The stage-4 register read-write cycle leg (`stage_cycle_points[3]`).
    pub fn register_read_write_cycle(&self) -> &[F] {
        &self.stage_cycle_points[3]
    }

    /// The stage-5 register value-evaluation cycle leg (`stage_cycle_points[4]`).
    pub fn register_val_evaluation_cycle(&self) -> &[F] {
        &self.stage_cycle_points[4]
    }
}

/// Derive the [`BytecodeStagePoints`] from the mode-agnostic upstream opening
/// points. Shared by [`Stage6bSumchecks::build`] (both proving modes) and the
/// prove-side stage-6a/6b recipes, so the five-leg wiring cannot drift.
pub fn bytecode_stage_points<F: Field>(
    stage1_cycle_binding: &[F],
    stage2: &Stage2BatchOutputPoints<F>,
    stage3: &Stage3OutputPoints<F>,
    stage4: &Stage4OutputPoints<F>,
    stage5: &Stage5OutputPoints<F>,
) -> Result<BytecodeStagePoints<F>, VerifierError> {
    let register_read_write_point = stage4.registers_read_write_point().to_vec();
    let register_val_evaluation_point = stage5.registers_opening_point().to_vec();
    let (_, register_read_write_cycle) = stage6_checked_split(
        "Stage 6 stage4 register read-write opening",
        &register_read_write_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let (_, register_val_evaluation_cycle) = stage6_checked_split(
        "Stage 6 stage5 register value-evaluation opening",
        &register_val_evaluation_point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let stage_cycle_points = [
        stage1_cycle_binding.iter().rev().copied().collect(),
        stage2.product_remainder_point().to_vec(),
        stage3.shift_opening_point().to_vec(),
        register_read_write_cycle.to_vec(),
        register_val_evaluation_cycle.to_vec(),
    ];
    Ok(BytecodeStagePoints {
        stage_cycle_points,
        register_read_write_point,
        register_val_evaluation_point,
    })
}

/// The batch legs [`Stage6bSumchecks::build_from_parts`] assembles the members
/// from: protocol geometry, the precommitted schedule, the carried stage-6a
/// draws, the mode-agnostic upstream opening points, and the clear-only value
/// aux (each empty/`None` in ZK, where `input_claim`/`expected_output` never
/// run). Every field is data both the verifier and the prover hold.
pub struct Stage6bBuildParts<'a, F: Field> {
    pub formula_dimensions: &'a JoltFormulaDimensions,
    pub ram_log_k: usize,
    pub committed_chunk_bits: usize,
    pub precommitted: &'a PrecommittedSchedule,
    pub entry_bytecode_index: usize,
    /// The full bytecode rows backing the full-program table fold
    /// (`None` in ZK and committed-program modes).
    pub bytecode_table_rows: Option<&'a [JoltInstructionRow]>,
    pub carried: &'a Stage6aCarriedChallenges<F>,
    pub eta: Option<F>,
    pub stage1_cycle_binding: Vec<F>,
    pub stage2_points: &'a Stage2BatchOutputPoints<F>,
    pub stage3_points: &'a Stage3OutputPoints<F>,
    pub stage4_points: &'a Stage4OutputPoints<F>,
    pub stage5_points: &'a Stage5OutputPoints<F>,
    pub stage6a_points: &'a Stage6aOutputPoints<F>,
    /// The staged `BytecodeValStage` openings (clear committed-program mode;
    /// empty otherwise).
    pub address_val_stages: Vec<F>,
    pub trusted_advice_reference_point: Option<Vec<F>>,
    pub untrusted_advice_reference_point: Option<Vec<F>>,
}

impl<F: Field> Stage6bSumchecks<F> {
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
        let (address_val_stages, trusted_advice_reference_point, untrusted_advice_reference_point) =
            if checked.zk {
                (Vec::new(), None, None)
            } else {
                let stage4 = stage4.clear()?;
                (
                    stage6a
                        .clear()?
                        .output_values
                        .bytecode_read_raf
                        .val_stages
                        .clone(),
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

        Self::build_from_parts(Stage6bBuildParts {
            formula_dimensions,
            ram_log_k: checked.ram_K.ilog2() as usize,
            committed_chunk_bits: proof.one_hot_config.committed_chunk_bits(),
            precommitted: &checked.precommitted,
            entry_bytecode_index,
            bytecode_table_rows,
            carried: stage6a.challenges(),
            eta,
            stage1_cycle_binding,
            stage2_points: stage2.batch_output_points(),
            stage3_points: stage3.output_points(),
            stage4_points: stage4.output_points(),
            stage5_points: stage5.output_points(),
            stage6a_points: stage6a.output_points(),
            address_val_stages,
            trusted_advice_reference_point,
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
            carried,
            eta,
            stage1_cycle_binding,
            stage2_points,
            stage3_points,
            stage4_points,
            stage5_points,
            stage6a_points,
            address_val_stages,
            trusted_advice_reference_point,
            untrusted_advice_reference_point,
        } = parts;
        let log_t = formula_dimensions.trace.log_t();
        let trace_dimensions = formula_dimensions.trace;

        let trusted_advice_layout = precommitted.trusted_advice.as_ref();
        let untrusted_advice_layout = precommitted.untrusted_advice.as_ref();
        let bytecode_reduction_layout = precommitted.bytecode.as_ref();
        let program_image_reduction_layout = precommitted.program_image.as_ref();
        let committed_program = bytecode_reduction_layout.is_some();

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
        let register_read_write_address =
            &stage_points.register_read_write_point[..REGISTER_ADDRESS_BITS];
        let register_val_evaluation_address =
            &stage_points.register_val_evaluation_point[..REGISTER_ADDRESS_BITS];
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
        let registers_read_write_cycle = stage_points.register_read_write_cycle().to_vec();
        let registers_val_evaluation_cycle = stage_points.register_val_evaluation_cycle().to_vec();
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
        let booleanity = Booleanity::new(
            booleanity_dimensions,
            booleanity_r_address,
            carried.booleanity_reference_address.clone(),
            carried.booleanity_reference_cycle.clone(),
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

        let trusted_advice = trusted_advice_layout
            .map(|layout| TrustedAdviceCyclePhase::new(layout, trusted_advice_reference_point));
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
            trusted_advice,
            untrusted_advice,
            bytecode_reduction,
            program_image_reduction,
        })
    }
}

pub(super) fn stage6_checked_split<'a, F: Field>(
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
