use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::super::super::diagnostic::schema_error;
use super::super::keys::operation_result_key_at;

pub(super) fn insert_result_mapping<'c, 'a>(
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    source: OperationRef<'_, '_>,
    target: OperationRef<'c, 'a>,
    source_index: usize,
    target_index: usize,
) -> Result<(), MlirError> {
    let key = operation_result_key_at(source, source_index)?;
    let value = target.result(target_index).map(Into::into).map_err(|_| {
        schema_error(format!(
            "{} requires result {target_index}",
            operation_name(target)
        ))
    })?;
    let _ = value_map.insert(key, value);
    Ok(())
}
