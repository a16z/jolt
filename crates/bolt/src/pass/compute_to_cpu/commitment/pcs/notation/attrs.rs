use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;

use super::super::super::super::super::support::{lower_attr_sources, LoweredAttr};

const BATCH_ATTRS: &[LoweredAttr] = &[
    LoweredAttr::symbol_ref("artifact"),
    LoweredAttr::int("count"),
    LoweredAttr::symbol_ref("domain"),
    LoweredAttr::string("label"),
    LoweredAttr::int("num_vars"),
    LoweredAttr::symbol_ref("oracle_family"),
    LoweredAttr::symbol_array("ordered_oracles"),
    LoweredAttr::symbol_ref("pcs"),
];

const OPTIONAL_ATTRS: &[LoweredAttr] = &[
    LoweredAttr::symbol_ref("artifact"),
    LoweredAttr::symbol_ref("domain"),
    LoweredAttr::string("label"),
    LoweredAttr::int("num_vars"),
    LoweredAttr::symbol_ref("oracle"),
    LoweredAttr::symbol_ref("pcs"),
    LoweredAttr::string("skip_policy"),
];

pub(in crate::pass::compute_to_cpu::commitment::pcs) fn batch_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    lower_attr_sources(operation, BATCH_ATTRS)
}

pub(in crate::pass::compute_to_cpu::commitment::pcs) fn optional_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    lower_attr_sources(operation, OPTIONAL_ATTRS)
}
