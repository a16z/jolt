//! Construction of the stage-6b cycle-phase sumcheck batch.
//!
//! [`Stage6CyclePhaseSumchecks::build`] assembles the batch members ONCE, after
//! stage 6a and the post-6a draws, from mode-agnostic constructor legs (per-stage
//! cycle bindings, reduced points, the stage-6a address openings) plus the
//! clear-only value aux (`table_fold`, `address_val_stages`, advice reference
//! points — each empty/`None` in ZK, where `expected_output` never runs). The
//! four `Option` members are present exactly when their precommitted layout is
//! committed, in both proving modes, so the batch's instance count matches the
//! prover's.

use jolt_claims::protocols::jolt::{
    geometry::{
        booleanity::BooleanityDimensions, bytecode::BytecodeReadRafDimensions,
        dimensions::TraceDimensions, instruction::InstructionRaVirtualizationDimensions,
        ram::RamRaVirtualizationDimensions,
    },
    AdviceClaimReductionLayout, BytecodeClaimReductionChallenge, BytecodeClaimReductionLayout,
    JoltAdviceKind, JoltChallengeId, ProgramImageClaimReductionLayout,
};
use jolt_field::Field;

use super::booleanity::Booleanity;
use super::bytecode_read_raf::{
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle, BytecodeReadRafCycleInputs,
    BytecodeReadRafTableFoldInputs,
};
use super::committed_reduction_cycle_phase::{
    AdviceCyclePhase, BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase,
};
use super::inc_claim_reduction::IncClaimReduction;
use super::instruction_ra_virtualization::InstructionRaVirtualization;
use super::outputs::{BytecodeReductionWeights, Stage6CyclePhaseSumchecks};
use super::ram_hamming_booleanity::RamHammingBooleanity;
use super::ram_ra_virtualization::RamRaVirtualization;
use crate::VerifierError;

/// Construction inputs for [`Stage6CyclePhaseSumchecks::build`]: the formula
/// dimensions, the stage-6a address openings, the upstream cycle/reduced points,
/// and the committed-program reduction layouts. Clear-only fields are documented
/// as such; everything else is available in both proving modes.
pub(super) struct Stage6CyclePhaseParams<'a, F: Field> {
    pub bytecode_dimensions: BytecodeReadRafDimensions,
    pub booleanity_dimensions: BooleanityDimensions,
    pub trace_dimensions: TraceDimensions,
    pub ram_ra_dimensions: RamRaVirtualizationDimensions,
    pub instruction_ra_dimensions: InstructionRaVirtualizationDimensions,
    pub committed_chunk_bits: usize,
    pub entry_bytecode_index: usize,
    /// Clear-only, full-program mode: the bytecode-table fold aux (folded at
    /// construction; `None` in ZK and in committed mode).
    pub bytecode_table_fold: Option<BytecodeReadRafTableFoldInputs<'a, F>>,
    /// The stage-6a bytecode read-RAF address opening (`bytecode_r_address`).
    pub bytecode_r_address: Vec<F>,
    /// The stage-6a booleanity address opening (`booleanity_r_address`).
    pub booleanity_r_address: Vec<F>,
    /// Clear-only, committed mode: the staged `BytecodeValStage` opening values
    /// from the address phase (empty in ZK).
    pub address_val_stages: Vec<F>,
    /// Per-stage (1..=5) cycle bindings used by the bytecode read-RAF publics.
    pub stage_cycle_points: [Vec<F>; 5],
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    /// The stage-1 Spartan-outer cycle binding (RAM hamming booleanity reference).
    pub stage1_cycle_binding: Vec<F>,
    /// The stage-5 reduced RAM address prefix / cycle suffix.
    pub ram_reduced_address: Vec<F>,
    pub ram_reduced_cycle: Vec<F>,
    /// The stage-5 instruction RA reduced address prefix / cycle suffix.
    pub instruction_r_address: Vec<F>,
    pub instruction_r_cycle: Vec<F>,
    /// Increment claim-reduction per-source cycle bindings, in source order:
    /// RAM read-write, RAM value-check, register read-write, register value-eval.
    pub inc_cycle_points: [Vec<F>; 4],
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub bytecode_reduction_layout: Option<&'a BytecodeClaimReductionLayout>,
    pub program_image_reduction_layout: Option<&'a ProgramImageClaimReductionLayout>,
    /// The bytecode claim-reduction weights, computed by the caller from the
    /// post-6a `eta` (present iff the bytecode layout is committed).
    pub bytecode_reduction_weights: Option<BytecodeReductionWeights<F>>,
    /// The RAM read-write `RamVal` address prefix (program-image `FinalScale`).
    pub program_image_r_addr_rw: Vec<F>,
    /// Clear-only: the stage-4 staged advice opening points (advice `FinalScale`
    /// references); `None` in ZK.
    pub trusted_advice_reference_point: Option<Vec<F>>,
    pub untrusted_advice_reference_point: Option<Vec<F>>,
}

impl<F: Field> Stage6CyclePhaseSumchecks<F> {
    pub(super) fn build(params: Stage6CyclePhaseParams<'_, F>) -> Result<Self, VerifierError> {
        let committed_program = params.bytecode_reduction_layout.is_some();
        let bytecode_read_raf = if committed_program {
            BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                dimensions: params.bytecode_dimensions,
                r_address: params.bytecode_r_address,
                stage_cycle_points: params.stage_cycle_points,
                entry_bytecode_index: params.entry_bytecode_index,
                committed_chunk_bits: params.committed_chunk_bits,
                val_stages: params.address_val_stages,
            })
        } else {
            BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                dimensions: params.bytecode_dimensions,
                r_address: params.bytecode_r_address,
                stage_cycle_points: params.stage_cycle_points,
                entry_bytecode_index: params.entry_bytecode_index,
                committed_chunk_bits: params.committed_chunk_bits,
                table_fold: params.bytecode_table_fold,
            })?
        };

        let booleanity = Booleanity::new(
            params.booleanity_dimensions,
            params.booleanity_r_address,
            params.booleanity_reference_address,
            params.booleanity_reference_cycle,
        );
        let ram_hamming_booleanity =
            RamHammingBooleanity::new(params.trace_dimensions, params.stage1_cycle_binding);
        let ram_ra_virtualization = RamRaVirtualization::new(
            params.ram_ra_dimensions,
            params.ram_reduced_address,
            params.ram_reduced_cycle,
            params.committed_chunk_bits,
        );
        let instruction_ra_virtualization = InstructionRaVirtualization::new(
            params.instruction_ra_dimensions,
            params.instruction_r_address,
            params.instruction_r_cycle,
            params.committed_chunk_bits,
        );
        let [ram_read_write_cycle, ram_val_check_cycle, registers_read_write_cycle, registers_val_evaluation_cycle] =
            params.inc_cycle_points;
        let inc_claim_reduction = IncClaimReduction::new(
            params.trace_dimensions,
            ram_read_write_cycle,
            ram_val_check_cycle,
            registers_read_write_cycle,
            registers_val_evaluation_cycle,
        );

        let trusted_advice = params.trusted_advice_layout.map(|layout| {
            AdviceCyclePhase::new(
                JoltAdviceKind::Trusted,
                layout,
                params.trusted_advice_reference_point,
            )
        });
        let untrusted_advice = params.untrusted_advice_layout.map(|layout| {
            AdviceCyclePhase::new(
                JoltAdviceKind::Untrusted,
                layout,
                params.untrusted_advice_reference_point,
            )
        });
        // `eta` is drawn exactly when the bytecode layout is committed, so a
        // committed layout without weights is unreachable from `verify`.
        let bytecode_reduction = match (
            params.bytecode_reduction_layout,
            params.bytecode_reduction_weights,
        ) {
            (Some(layout), Some(weights)) => {
                Some(BytecodeReductionCyclePhase::new(layout, weights))
            }
            (Some(_), None) => {
                return Err(VerifierError::MissingStageClaimChallenge {
                    id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
                })
            }
            (None, _) => None,
        };
        let program_image_reduction = params.program_image_reduction_layout.map(|layout| {
            ProgramImageReductionCyclePhase::new(layout, params.program_image_r_addr_rw)
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
