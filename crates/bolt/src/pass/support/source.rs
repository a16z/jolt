use melior::ir::operation::OperationRef;

use crate::schema::operation_name;

pub(in crate::pass) fn compute_to_cpu_op_name(operation: OperationRef<'_, '_>) -> String {
    operation_name(operation).replacen("compute.", "cpu.", 1)
}

pub(in crate::pass) fn symbol_ref(value: &str) -> String {
    format!("@{value}")
}

pub(in crate::pass) fn string_attr_source(value: &str) -> String {
    format!("{value:?}")
}

pub(in crate::pass) fn symbol_array_attr_source(values: &[String]) -> String {
    let values = values
        .iter()
        .map(|value| format!("@{value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

pub(in crate::pass) fn int_attr_source(value: usize) -> String {
    format!("{value} : i64")
}

pub(in crate::pass) fn bool_attr_source(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}
