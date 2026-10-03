use jolt_claims::protocols::jolt::geometry::{
    booleanity::BooleanityDimensions, dimensions::JoltFormulaDimensions,
};
use jolt_field::JoltField;

use super::booleanity::BooleanityAddressPhase;
use super::bytecode_read_raf::{bytecode_stage_points, BytecodeReadRafAddressPhase};
use super::outputs::Stage6aSumchecks;
use crate::stages::stage2::Stage2BatchOutputPoints;
use crate::stages::stage3::outputs::Stage3OutputPoints;
use crate::stages::stage4::outputs::Stage4OutputPoints;
use crate::stages::stage5::outputs::Stage5OutputPoints;
use crate::VerifierError;

pub struct Stage6aBuildParts<'a, F: JoltField> {
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

impl<F: JoltField> Stage6aSumchecks<F> {
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
