use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;

use super::super::super::support::{lower_attr_sources, LoweredAttr};

const TRANSCRIPT_INIT_ATTRS: &[LoweredAttr] = &[LoweredAttr::symbol_ref("scheme")];
const TRANSCRIPT_ABSORB_ATTRS: &[LoweredAttr] =
    &[LoweredAttr::string("label"), LoweredAttr::bool("optional")];

pub(super) fn transcript_init_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    lower_attr_sources(operation, TRANSCRIPT_INIT_ATTRS)
}

pub(super) fn transcript_absorb_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    lower_attr_sources(operation, TRANSCRIPT_ABSORB_ATTRS)
}
