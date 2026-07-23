//! Construction of the stage-6a address-phase sumcheck batch.
//!
//! [`Stage6aSumchecks::build_from_parts`] assembles the two members over data
//! both the verifier and the prover hold, so the member legs (the bytecode
//! stage points, the booleanity dimensions, the stage-5 instruction points)
//! are single-sourced across the two fronts — the same idiom as stage 6b's
//! `Stage6bSumchecks::build_from_parts`.

use jolt_claims::protocols::jolt::geometry::{
    booleanity::BooleanityDimensions, dimensions::JoltFormulaDimensions,
};
use jolt_field::Field;

use super::booleanity::BooleanityAddressPhase;
use super::bytecode_read_raf::BytecodeReadRafAddressPhase;
use super::outputs::Stage6aSumchecks;
use crate::stages::bytecode_stage_points;
use crate::stages::stage2::Stage2BatchOutputPoints;
use crate::stages::stage3::outputs::Stage3OutputPoints;
use crate::stages::stage4::outputs::Stage4OutputPoints;
use crate::stages::stage5::outputs::Stage5OutputPoints;
use crate::VerifierError;

/// The batch legs [`Stage6aSumchecks::build_from_parts`] assembles the members
/// from: protocol geometry and the mode-agnostic upstream opening points.
/// Every field is data both the verifier and the prover hold.
pub struct Stage6aBuildParts<'a, F: Field> {
    pub formula_dimensions: &'a JoltFormulaDimensions,
    pub committed_chunk_bits: usize,
    pub committed_program: bool,
    pub entry_bytecode_index: usize,
    pub stage1_cycle_binding: &'a [F],
    pub stage2_points: &'a Stage2BatchOutputPoints<F>,
    pub stage3_points: &'a Stage3OutputPoints<F>,
    pub stage4_points: &'a Stage4OutputPoints<F>,
    pub stage5_points: &'a Stage5OutputPoints<F>,
}

impl<F: Field> Stage6aSumchecks<F> {
    /// Assemble the address-phase batch: the bytecode member carries the
    /// upstream cycle/register points and the entry index (full geometry at
    /// construction — the prover's kernel read path; the verifier itself
    /// never evaluates them), the booleanity member the stage-5 instruction
    /// address/cycle points.
    pub fn build_from_parts(parts: Stage6aBuildParts<'_, F>) -> Result<Self, VerifierError> {
        let Stage6aBuildParts {
            formula_dimensions,
            committed_chunk_bits,
            committed_program,
            entry_bytecode_index,
            stage1_cycle_binding,
            stage2_points,
            stage3_points,
            stage4_points,
            stage5_points,
        } = parts;
        let stage_points = bytecode_stage_points(
            stage1_cycle_binding,
            stage2_points,
            stage3_points,
            stage4_points,
            stage5_points,
        )?;
        let booleanity_dimensions = BooleanityDimensions::new(
            formula_dimensions.ra_layout,
            formula_dimensions.trace.log_t(),
            committed_chunk_bits,
        );
        Ok(Self {
            bytecode_read_raf: BytecodeReadRafAddressPhase::new(
                formula_dimensions.bytecode_read_raf,
                committed_program,
                stage_points,
                entry_bytecode_index,
            ),
            booleanity: BooleanityAddressPhase::new(
                booleanity_dimensions,
                stage5_points.instruction_r_address(),
                stage5_points.instruction_r_cycle().to_vec(),
            ),
        })
    }
}
