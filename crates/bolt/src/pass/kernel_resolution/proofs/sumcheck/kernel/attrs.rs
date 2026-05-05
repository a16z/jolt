use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;
use crate::pass::support::{copy_attrs, symbol_ref};

pub(super) fn kernel_attrs(
    operation: OperationRef<'_, '_>,
    source_attrs: &[&str],
    kernel: &str,
) -> Result<Vec<(String, String)>, MlirError> {
    let mut attrs = copy_attrs(operation, source_attrs)?;
    attrs.push(("kernel".to_owned(), symbol_ref(kernel)));
    Ok(attrs)
}
