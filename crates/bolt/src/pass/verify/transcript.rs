mod ops;
mod thread;

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Phase};
use crate::schema::operation_name;

use super::VerifyError;
use ops::TranscriptOp;
use thread::TranscriptThread;

pub fn verify_concrete_transcript<P>(module: &BoltModule<'_, P>) -> Result<(), VerifyError>
where
    P: Phase,
{
    let mut thread = TranscriptThread::default();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        let name = operation_name(op);
        match TranscriptOp::classify(&name) {
            TranscriptOp::State => thread.initialize(op)?,
            TranscriptOp::Absorb => {
                thread.require_state_input(
                    op,
                    "transcript absorb requires a prior transcript.state result",
                    format!("{name} requires transcript-state operand 0"),
                )?;
                thread.require_operand(
                    op,
                    1,
                    format!("{name} requires commitment artifact operand 1"),
                )?;
                thread.advance_from_result(
                    op,
                    format!("{name} requires one transcript-state result"),
                )?;
            }
            TranscriptOp::AbsorbBytes => {
                thread.require_state_input(
                    op,
                    "transcript absorb_bytes requires a prior transcript.state result",
                    "transcript.absorb_bytes requires transcript-state operand 0",
                )?;
                thread.advance_from_result(
                    op,
                    "transcript.absorb_bytes requires one transcript-state result",
                )?;
            }
            TranscriptOp::ChallengeDriver => {
                thread.require_state_input(
                    op,
                    format!("{name} requires a prior transcript.state result"),
                    format!("{name} requires transcript-state operand 0"),
                )?;
                thread.advance_from_result(
                    op,
                    format!("{name} requires transcript-state result 0"),
                )?;
            }
            TranscriptOp::Ignored => {}
        }
    }

    Ok(())
}
