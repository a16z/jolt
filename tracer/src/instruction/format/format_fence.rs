use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, InstructionRegisterState, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatFence {}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatFence {}

impl InstructionRegisterState for RegisterStateFormatFence {
    #[cfg(any(feature = "test-utils", test))]
    fn random(_: &mut rand::rngs::StdRng, _: &NormalizedOperands) -> Self {
        Self {}
    }
}

impl InstructionFormat for FormatFence {
    type RegisterState = RegisterStateFormatFence;

    fn parse(_word: u32) -> Self {
        FormatFence {}
    }

    fn capture_pre_execution_state(&self, _state: &mut Self::RegisterState, _cpu: &mut Cpu) {
        // No-op
    }

    fn capture_post_execution_state(&self, _state: &mut Self::RegisterState, _cpu: &mut Cpu) {
        // No-op
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
