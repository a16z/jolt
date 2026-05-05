use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::mlir::MlirError;
use crate::pass::VerifyError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaError {
    message: String,
}

impl SchemaError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for SchemaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for SchemaError {}

impl From<SchemaError> for MlirError {
    fn from(error: SchemaError) -> Self {
        Self::Schema {
            message: error.to_string(),
        }
    }
}

impl From<VerifyError> for SchemaError {
    fn from(error: VerifyError) -> Self {
        Self::new(error.to_string())
    }
}
