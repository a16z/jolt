use crate::ir::{BoltModule, Concrete, Party, Phase, Role};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_concrete_schema, verify_party_schema};

use super::super::module_copy::{phase_copy_source, PhaseCopyRole};
use super::boundary::require_declared_role;

pub fn project_party<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
    role: Role,
) -> Result<BoltModule<'c, Party>, MlirError> {
    verify_concrete_schema(module)?;
    require_declared_role(module, &role)?;
    let party_function = party_function_op(module, &role);
    let source = phase_copy_source(
        module,
        Party::NAME,
        PhaseCopyRole::present(&role),
        &[party_function],
    );
    let party = context.parse_module::<Party>(&source)?;
    verify_module(&party)?;
    verify_party_schema(&party)?;
    Ok(party)
}

fn party_function_op(module: &BoltModule<'_, Concrete>, role: &Role) -> String {
    format!(
        "  \"party.function\"() {{role = \"{}\", source = @{}, sym_name = \"{}.{}\"}} : () -> ()",
        role.as_str(),
        module.name(),
        module.name(),
        role.as_str()
    )
}
