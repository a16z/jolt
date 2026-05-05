use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationRef};

use crate::ir::{BoltModule, Phase};
use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::attr_sources::{lower_attr_sources, LoweredAttr};
use super::attrs::copy_attrs;
use super::diagnostic::schema_error;

pub(in crate::pass) const PROTOCOL_PARAM_ATTRS: &[&str] = &["field", "pcs", "transcript"];
const COMPUTE_PARAM_SYMBOL_REF_ATTRS: &[LoweredAttr] = &[
    LoweredAttr::symbol_ref("field"),
    LoweredAttr::symbol_ref("pcs"),
    LoweredAttr::symbol_ref("transcript"),
];

pub(in crate::pass) fn protocol_params_attrs<P: Phase>(
    module: &BoltModule<'_, P>,
) -> Result<Vec<(String, String)>, MlirError> {
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if operation_name(op) == "protocol.params" {
            return copy_attrs(op, PROTOCOL_PARAM_ATTRS);
        }
    }
    Err(schema_error("module missing protocol.params"))
}

pub(in crate::pass) fn compute_params_symbol_ref_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    lower_attr_sources(operation, COMPUTE_PARAM_SYMBOL_REF_ATTRS)
}
