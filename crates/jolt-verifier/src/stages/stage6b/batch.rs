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

use super::booleanity::Booleanity;
use super::bytecode_read_raf::{
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle, BytecodeReadRafCycleInputs,
    BytecodeReadRafTableFoldInputs,
};
use super::committed_reduction_cycle_phase::{
    bytecode_reduction_weights, BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase,
    TrustedAdviceCyclePhase, UntrustedAdviceCyclePhase,
};
use super::inc_claim_reduction::IncClaimReduction;
use super::instruction_ra_virtualization::InstructionRaVirtualization;
use super::outputs::Stage6bSumchecks;
use super::ram_hamming_booleanity::RamHammingBooleanity;
use super::ram_ra_virtualization::RamRaVirtualization;
use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
use crate::stages::stage1::Stage1Output;
use crate::stages::stage2::Stage2Output;
use crate::stages::stage3::Stage3Output;
use crate::stages::stage4::Stage4Output;
use crate::stages::stage5::Stage5Output;
use crate::stages::stage6a::Stage6aOutput;
use crate::verifier::CheckedInputs;
use crate::VerifierError;

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

        // The pre-/around-6a draws consumed here ride on the stage-6a output as
        // typed upstream values: the bytecode fold gamma (shared with stage 6a's
        // squeeze), the per-stage folding gammas, and the booleanity address / cycle
        // reference points. The bytecode folds below consume per-stage power
        // VECTORS, expanded once here from the carried scalars.
        let carried = stage6a.challenges();
        let stage_gamma_powers = carried.bytecode_read_raf.stage_gamma_powers();
        let bytecode_r_address = stage6a
            .output_points()
            .bytecode_read_raf
            .intermediate
            .clone();
        let booleanity_r_address = stage6a.output_points().booleanity.intermediate.clone();

        // Cycle-phase constructor legs, wired mode-agnostically off the upstream
        // outputs; the post-batch opening points are derived against these same
        // values through the relation objects.
        let stage4_points = stage4.output_points();
        let stage5_points = stage5.output_points();
        let stage5_instruction_address = stage5.instruction_r_address();
        let stage5_instruction_cycle = stage5_points.instruction_r_cycle();
        let stage1_cycle_binding =
            stage1
                .cycle_binding()
                .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::BytecodeReadRaf,
                    reason: "Stage 1 remainder point is empty".to_string(),
                })?;
        let stage2_points = stage2.batch_output_points();
        let stage3_points = stage3.output_points();
        let (register_read_write_address, register_read_write_cycle) = stage6_checked_split(
            "Stage 6 stage4 register read-write opening",
            stage4_points.registers_read_write_point(),
            REGISTER_ADDRESS_BITS,
            JoltRelationId::BytecodeReadRaf,
        )?;
        let (register_val_evaluation_address, register_val_evaluation_cycle) =
            stage6_checked_split(
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
        let entry_bytecode_index =
            preprocessing
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
                stage_gammas: stage_gamma_powers.each_ref().map(Vec::as_slice),
            })
        };
        // `eta` is drawn exactly when the bytecode layout is committed, so a
        // committed layout always carries weights and the member's `(Some, None)`
        // case is unreachable.
        let cycle_bytecode_reduction_weights = bytecode_reduction_layout
            .zip(eta)
            .map(|(layout, eta)| {
                bytecode_reduction_weights(
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
                )
            })
            .transpose()?;
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

        let bytecode_read_raf = if committed_program {
            BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                dimensions: formula_dimensions.bytecode_read_raf,
                r_address: bytecode_r_address,
                stage_cycle_points,
                entry_bytecode_index,
                committed_chunk_bits: proof.one_hot_config.committed_chunk_bits(),
                val_stages: address_val_stages,
            })
        } else {
            BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                dimensions: formula_dimensions.bytecode_read_raf,
                r_address: bytecode_r_address,
                stage_cycle_points,
                entry_bytecode_index,
                committed_chunk_bits: proof.one_hot_config.committed_chunk_bits(),
                table_fold: bytecode_table_fold,
            })?
        };

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
            proof.one_hot_config.committed_chunk_bits(),
        );
        let instruction_ra_virtualization = InstructionRaVirtualization::new(
            formula_dimensions.instruction_ra_virtualization,
            stage5_instruction_address.to_vec(),
            stage5_instruction_cycle.to_vec(),
            proof.one_hot_config.committed_chunk_bits(),
        );
        let [ram_read_write_cycle, ram_val_check_cycle, registers_read_write_cycle, registers_val_evaluation_cycle] =
            inc_cycle_points;
        let inc_claim_reduction = IncClaimReduction::new(
            trace_dimensions,
            ram_read_write_cycle,
            ram_val_check_cycle,
            registers_read_write_cycle,
            registers_val_evaluation_cycle,
        );

        let trusted_advice = trusted_advice_layout
            .map(|layout| TrustedAdviceCyclePhase::new(layout, trusted_advice_reference_point));
        let untrusted_advice = untrusted_advice_layout
            .map(|layout| UntrustedAdviceCyclePhase::new(layout, untrusted_advice_reference_point));
        let bytecode_reduction = match (bytecode_reduction_layout, cycle_bytecode_reduction_weights)
        {
            (Some(layout), Some(weights)) => {
                Some(BytecodeReductionCyclePhase::new(layout, weights))
            }
            _ => None,
        };
        let program_image_reduction = program_image_reduction_layout.map(|layout| {
            ProgramImageReductionCyclePhase::new(layout, ram_val_check_address.to_vec())
        });

        Ok(Self {
            bytecode_read_raf,
            booleanity,
            ram_hamming_booleanity,
            ram_ra_virtualization,
            instruction_ra_virtualization,
            inc_claim_reduction,
            trusted_advice,
            untrusted_advice,
            bytecode_reduction,
            program_image_reduction,
        })
    }
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
