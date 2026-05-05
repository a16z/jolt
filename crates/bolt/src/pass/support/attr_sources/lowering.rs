use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;
use crate::schema::{int_attr, symbol_array_attr};

use super::super::attrs::{bool_attr, string_attr, symbol_attr};
use super::super::source::{
    bool_attr_source, int_attr_source, string_attr_source, symbol_array_attr_source, symbol_ref,
};
use super::descriptor::{LoweredAttr, LoweredAttrKind};

pub(in crate::pass) fn lower_attr_sources(
    operation: OperationRef<'_, '_>,
    attrs: &[LoweredAttr],
) -> Result<Vec<(String, String)>, MlirError> {
    attrs
        .iter()
        .map(|attr| Ok((attr.name.to_owned(), attr_value(*attr, operation)?)))
        .collect()
}

fn attr_value(attr: LoweredAttr, operation: OperationRef<'_, '_>) -> Result<String, MlirError> {
    match attr.kind {
        LoweredAttrKind::SymbolRef => Ok(symbol_ref(&symbol_attr(operation, attr.name)?)),
        LoweredAttrKind::SymbolArray => Ok(symbol_array_attr_source(&symbol_array_attr(
            operation, attr.name,
        )?)),
        LoweredAttrKind::String => Ok(string_attr_source(&string_attr(operation, attr.name)?)),
        LoweredAttrKind::Int => Ok(int_attr_source(int_attr(operation, attr.name)?)),
        LoweredAttrKind::Bool => Ok(bool_attr_source(bool_attr(operation, attr.name)?).to_owned()),
    }
}
