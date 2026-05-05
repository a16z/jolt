mod transcript;

use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub use transcript::verify_concrete_transcript;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyError {
    message: String,
}

impl VerifyError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for VerifyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for VerifyError {}
