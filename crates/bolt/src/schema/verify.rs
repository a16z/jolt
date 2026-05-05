use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Phase};

use super::kernels::KernelReferenceTracker;
use super::ops::{validate_op, validate_verifier_lowering_op};
use super::phase::{verify_module_phase_attr, ModulePhase};
use super::SchemaError;

pub(super) fn verify_schema<P>(
    module: &BoltModule<'_, P>,
    phase: ModulePhase,
) -> Result<(), SchemaError>
where
    P: Phase,
{
    verify_module_phase_attr(module)?;

    let mut kernel_refs = KernelReferenceTracker::default();
    let role = module.role();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        validate_op(op)?;
        if phase.requires_verifier_lowering_policy(role.as_ref()) {
            validate_verifier_lowering_op(op)?;
        }
        kernel_refs.record(op)?;
    }

    if phase.is_lowered() {
        kernel_refs.verify()?;
    }

    Ok(())
}
