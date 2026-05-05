use crate::ir::{BoltModule, Concrete, Phase, Role};
use crate::mlir::{MeliorContext, MlirError};

use super::super::module_copy::{phase_copy_source, PhaseCopyRole};

pub(super) fn derive_role<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
    role: Role,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    let source = phase_copy_source(module, Concrete::NAME, PhaseCopyRole::present(&role), &[]);
    context.parse_module::<Concrete>(&source)
}
