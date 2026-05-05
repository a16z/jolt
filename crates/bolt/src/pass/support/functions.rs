use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;

use super::attr_sources::{lower_attr_sources, LoweredAttr};

pub(in crate::pass) const COMPUTE_FUNCTION_ATTRS: &[&str] = &["source"];

const CPU_FUNCTION_ATTRS: &[LoweredAttr] = &[LoweredAttr::symbol_ref("source")];

pub(in crate::pass) fn compute_function_attrs(source: &str) -> [(&'static str, &str); 1] {
    [("source", source)]
}

pub(in crate::pass) fn cpu_function_symbol_ref_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    lower_attr_sources(operation, CPU_FUNCTION_ATTRS)
}
