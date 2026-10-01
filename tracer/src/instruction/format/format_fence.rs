use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatFence {}

impl InstructionFormat for FormatFence {
    fn parse(_word: u32) -> Self {
        FormatFence {}
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(_: &mut rand::rngs::StdRng) -> Self {
        Self {}
    }
}

impl From<NormalizedOperands> for FormatFence {
    fn from(_: NormalizedOperands) -> Self {
        Self {}
    }
}

impl From<FormatFence> for NormalizedOperands {
    fn from(_: FormatFence) -> Self {
        NormalizedOperands::default()
    }
}
