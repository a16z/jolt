use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Concrete, Role};
use crate::mlir::MlirError;
use crate::schema::operation_name;

pub(super) fn require_declared_role(
    module: &BoltModule<'_, Concrete>,
    role: &Role,
) -> Result<(), MlirError> {
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if operation_name(op) != "protocol.boundary" {
            continue;
        }
        let roles = op
            .attribute("roles")
            .ok()
            .and_then(|attribute| parse_string_array(&attribute.to_string()))
            .ok_or_else(|| MlirError::Schema {
                message: "protocol.boundary requires string array attr `roles`".to_owned(),
            })?;
        if roles.iter().any(|declared| declared == role.as_str()) {
            return Ok(());
        }
        return Err(MlirError::Schema {
            message: format!("protocol.boundary does not declare role `{role}`"),
        });
    }

    Err(MlirError::Schema {
        message: "module missing required op `protocol.boundary`".to_owned(),
    })
}

fn parse_string_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| {
            item.trim()
                .strip_prefix('"')
                .and_then(|item| item.strip_suffix('"'))
                .map(ToOwned::to_owned)
        })
        .collect()
}
