//! Stage 2 product uni-skip and five-instance batch verifier.

pub mod instruction_claim_reduction;
pub mod outputs;
pub mod product_remainder;
pub mod product_uniskip;
pub mod ram_output_check;
pub mod ram_raf_evaluation;
pub mod ram_read_write_checking;
mod verify;

pub use outputs::{Stage2BatchOutputClaims, Stage2BatchOutputPoints, Stage2Output, Stage2ZkOutput};
pub use verify::verify;

use jolt_claims::protocols::jolt::{geometry::dimensions::ReadWriteDimensions, JoltRelationId};

use crate::VerifierError;

/// The batch-point offset shared by the two stage-2 RAM relations whose openings sit
/// at the phase-1 sub-point: the active stage-2 window (the RAM read-write leader's
/// `log_t + log_k` rounds) starts at `batch_num_vars - read_write_rounds`, and each
/// relation joins it after the leader's `phase1_num_rounds` cycle rounds — the
/// pre-port verifier's `try_round_offset(log_t + log_k) + phase1_num_rounds()`
/// slicing. `stage` attributes the underflow error to the calling relation.
fn phase1_instance_point_offset(
    dimensions: ReadWriteDimensions,
    stage: JoltRelationId,
    batch_num_vars: usize,
) -> Result<usize, VerifierError> {
    let window_offset = batch_num_vars
        .checked_sub(dimensions.read_write_rounds())
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: format!(
                "batch challenge vector has {batch_num_vars} entries, fewer than the \
                     active stage-2 window's {} rounds",
                dimensions.read_write_rounds()
            ),
        })?;
    Ok(window_offset + dimensions.phase1_num_rounds())
}
