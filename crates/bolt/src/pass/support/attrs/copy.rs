use melior::ir::operation::{OperationLike, OperationRef};

use crate::mlir::MlirError;

pub(in crate::pass) fn copy_attrs(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
) -> Result<Vec<(String, String)>, MlirError> {
    attrs
        .iter()
        .filter_map(|attr| {
            operation
                .attribute(attr)
                .ok()
                .map(|value| Ok(((*attr).to_owned(), value.to_string())))
        })
        .collect()
}
