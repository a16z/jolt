use melior::ir::operation::OperationLike;

use crate::ir::{string_attribute_value, BoltModule, Phase, Role};

use super::SchemaError;

#[derive(Clone, Copy)]
pub(super) enum ModulePhase {
    Protocol,
    Concrete,
    Party,
    Compute,
    Cpu,
}

impl ModulePhase {
    pub(super) fn is_lowered(self) -> bool {
        matches!(self, Self::Compute | Self::Cpu)
    }

    pub(super) fn requires_verifier_lowering_policy(self, role: Option<&Role>) -> bool {
        self.is_lowered() && matches!(role, Some(Role::Verifier))
    }
}

pub(super) fn verify_module_phase_attr<P: Phase>(
    module: &BoltModule<'_, P>,
) -> Result<(), SchemaError> {
    let phase_attr = module
        .as_mlir_module()
        .as_operation()
        .attribute("bolt.phase")
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| SchemaError::new("module missing required attr `bolt.phase`"))?;
    if phase_attr != P::NAME {
        return Err(SchemaError::new(format!(
            "module phase `{phase_attr}` does not match expected `{}`",
            P::NAME
        )));
    }
    Ok(())
}
