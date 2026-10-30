use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatNoOperands {}

impl InstructionFormat for FormatNoOperands {
    fn parse(_word: u32) -> Self {
        FormatNoOperands {}
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(_: &mut rand::rngs::StdRng) -> Self {
        Self {}
    }
}

impl From<NormalizedOperands> for FormatNoOperands {
    fn from(_: NormalizedOperands) -> Self {
        Self {}
    }
}

impl From<FormatNoOperands> for NormalizedOperands {
    fn from(_: FormatNoOperands) -> Self {
        NormalizedOperands::default()
    }
}
