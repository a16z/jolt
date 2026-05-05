use melior::ir::operation::{OperationLike, OperationRef};

use crate::schema::SchemaError;

pub(crate) fn operation_name(operation: OperationRef<'_, '_>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}

pub(crate) fn missing_module_op(name: &str) -> SchemaError {
    SchemaError::new(format!("module missing required op `{name}`"))
}

pub(crate) fn missing_symbol(symbol: &str) -> SchemaError {
    SchemaError::new(format!("module missing required symbol @{symbol}"))
}

pub(in crate::schema) fn is_bolt_dialect_op(name: &str) -> bool {
    matches!(
        name.split_once('.').map(|(dialect, _)| dialect),
        Some(
            "field"
                | "poly"
                | "hash"
                | "transcript"
                | "commit"
                | "pcs"
                | "protocol"
                | "piop"
                | "party"
                | "compute"
                | "cpu"
        )
    )
}
