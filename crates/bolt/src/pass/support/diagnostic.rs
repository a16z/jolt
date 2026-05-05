use crate::mlir::MlirError;
use crate::schema::SchemaError;

pub(in crate::pass) fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}
