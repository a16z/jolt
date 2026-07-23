//! Typed verifier stage entry points.

use jolt_claims::protocols::jolt::{
    geometry::claim_reductions::{advice, bytecode, program_image},
    geometry::dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
    geometry::error::JoltFormulaPointError,
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind, JoltOneHotConfig,
    JoltRelationId, PrecommittedClaimReduction, ProgramImageClaimReductionLayout,
    TracePolynomialOrder,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;

use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
use crate::stages::stage2::Stage2BatchOutputPoints;
use crate::stages::stage3::outputs::Stage3OutputPoints;
use crate::stages::stage4::outputs::Stage4OutputPoints;
use crate::stages::stage5::outputs::Stage5OutputPoints;
use crate::verifier::CheckedInputs;
use crate::VerifierError;

pub mod relations;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6a;
pub mod stage6b;
pub mod stage7;
pub mod stage8;
pub mod uniskip;
#[doc(hidden)]
pub mod zk;

/// Build the one-hot [`JoltFormulaDimensions`] from the proof's one-hot config and
/// the verifier-trusted geometry (trace length, lookup operand width, bytecode
/// length, RAM size), mapping the layout error to `stage`. Built once during
/// verification and shared by the stages (5-8) that derive their RA layouts from
/// it, and reused by the BlindFold input derivation.
pub fn build_formula_dimensions<PCS, VC, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    checked: &CheckedInputs,
    log_t: usize,
    stage: JoltRelationId,
) -> Result<JoltFormulaDimensions, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    formula_dimensions_from_parts(
        proof.one_hot_config,
        log_t,
        preprocessing.program.bytecode_len(),
        checked.ram_K,
        stage,
    )
}

/// Core [`JoltFormulaDimensions`] constructor over primitive geometry inputs,
/// shared by the verifier's [`build_formula_dimensions`] and the prover's
/// `formula_dimensions` so the two sides cannot drift apart.
pub fn formula_dimensions_from_parts(
    one_hot_config: JoltOneHotConfig,
    log_t: usize,
    bytecode_len: usize,
    ram_k: usize,
    stage: JoltRelationId,
) -> Result<JoltFormulaDimensions, VerifierError> {
    JoltFormulaDimensions::try_from(one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        bytecode_len,
        ram_k,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: error.to_string(),
    })
}

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
/// points. Shared by the stage-6a and stage-6b batch builds (both proving
/// modes, both fronts), single-sourcing the five-leg wiring on the clear-mode
/// paths. The BlindFold ZK input derivation (`crate::stages::zk::blindfold`)
/// assembles its own legs from the committed consistency points and does not
/// route through this helper.
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

pub(crate) fn stage6_checked_split<'a, F: Field>(
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

/// Committed-program geometry feeding the bytecode and program-image
/// claim-reduction layouts, derived from the committed preprocessing during
/// input validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedProgramSchedule {
    pub bytecode_len: usize,
    pub bytecode_chunk_count: usize,
    pub program_image_len_words: usize,
    /// Remapped RAM word address of the first program-image word
    /// (`remap(min_bytecode_address)`).
    pub program_image_start_index: usize,
}

/// Per-polynomial claim-reduction layouts over the shared precommitted
/// scheduling reference, derived once during input validation.
///
/// The reference spans all present precommitted polynomials, so the layouts
/// must be built together; stages read them from `CheckedInputs` instead of
/// re-deriving the schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecommittedSchedule {
    pub trusted_advice: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice: Option<AdviceClaimReductionLayout>,
    pub bytecode: Option<BytecodeClaimReductionLayout>,
    pub program_image: Option<ProgramImageClaimReductionLayout>,
}

impl PrecommittedSchedule {
    pub fn new(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        trusted_max_advice_bytes: Option<usize>,
        untrusted_max_advice_bytes: Option<usize>,
        committed_program: Option<CommittedProgramSchedule>,
    ) -> Result<Self, JoltFormulaPointError> {
        let mut candidates =
            advice::candidate_total_vars(trusted_max_advice_bytes, untrusted_max_advice_bytes);
        if let Some(committed) = committed_program {
            candidates.push(bytecode::precommitted_candidate(
                committed.bytecode_len,
                committed.bytecode_chunk_count,
            )?);
            candidates.push(program_image::precommitted_candidate(
                committed.program_image_len_words,
            ));
        }
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
            log_t + log_k_chunk,
            &candidates,
            log_k_chunk,
        );
        let layout = |max_bytes: Option<usize>| {
            max_bytes
                .map(|max_bytes| {
                    AdviceClaimReductionLayout::balanced(
                        trace_order,
                        log_t,
                        scheduling_reference,
                        max_bytes,
                    )
                })
                .transpose()
        };
        let bytecode = committed_program
            .map(|committed| {
                BytecodeClaimReductionLayout::balanced(
                    trace_order,
                    log_t,
                    scheduling_reference,
                    committed.bytecode_len,
                    committed.bytecode_chunk_count,
                )
            })
            .transpose()?;
        let program_image = committed_program
            .map(|committed| {
                ProgramImageClaimReductionLayout::balanced(
                    trace_order,
                    log_t,
                    scheduling_reference,
                    committed.program_image_len_words,
                    committed.program_image_start_index,
                )
            })
            .transpose()?;
        Ok(Self {
            trusted_advice: layout(trusted_max_advice_bytes)?,
            untrusted_advice: layout(untrusted_max_advice_bytes)?,
            bytecode,
            program_image,
        })
    }

    pub fn advice(&self, kind: JoltAdviceKind) -> Option<&AdviceClaimReductionLayout> {
        match kind {
            JoltAdviceKind::Trusted => self.trusted_advice.as_ref(),
            JoltAdviceKind::Untrusted => self.untrusted_advice.as_ref(),
        }
    }
}
