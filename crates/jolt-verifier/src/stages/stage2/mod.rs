//! Stage 2 product uni-skip and batch verifier.

#[cfg(feature = "field-inline")]
pub mod field_registers_claim_reduction;
pub mod instruction_claim_reduction;
pub mod outputs;
pub mod product_remainder;
pub mod product_uniskip;
pub mod ram_output_check;
pub mod ram_raf_evaluation;
pub mod ram_read_write_checking;
mod verify;

pub use outputs::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output, Stage2ZkOutput};
pub use verify::{product_tau_low, stage2_batch_input_values_from_upstream, verify};

use jolt_claims::protocols::jolt::{geometry::dimensions::ReadWriteDimensions, JoltRelationId};

use crate::VerifierError;

fn phase1_instance_point_offset(
    dimensions: ReadWriteDimensions,
    stage: JoltRelationId,
    batch_num_vars: usize,
) -> Result<usize, VerifierError> {
    let window_offset = batch_num_vars
        .checked_sub(dimensions.read_write_rounds())
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: format!("{stage:?}"),
            reason: format!(
                "batch challenge vector has {batch_num_vars} entries, fewer than the \
                     active stage-2 window's {} rounds",
                dimensions.read_write_rounds()
            ),
        })?;
    window_offset
        .checked_add(dimensions.phase1_num_rounds())
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: format!("{stage:?}"),
            reason: format!(
                "stage-2 window offset {window_offset} plus {} phase-1 rounds overflows usize",
                dimensions.phase1_num_rounds()
            ),
        })
}
