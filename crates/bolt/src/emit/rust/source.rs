use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::schema::SchemaError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustSourceFile {
    pub filename: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitError {
    message: String,
}

impl EmitError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for EmitError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for EmitError {}

impl From<SchemaError> for EmitError {
    fn from(error: SchemaError) -> Self {
        Self::new(error.to_string())
    }
}
