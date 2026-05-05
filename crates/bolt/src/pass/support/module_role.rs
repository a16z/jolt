use crate::ir::{BoltModule, Phase, Role};
use crate::mlir::MlirError;

use super::diagnostic::schema_error;

pub(in crate::pass) fn require_module_role<P: Phase>(
    module: &BoltModule<'_, P>,
    message: impl Into<String>,
) -> Result<Role, MlirError> {
    module.role().ok_or_else(|| schema_error(message))
}
