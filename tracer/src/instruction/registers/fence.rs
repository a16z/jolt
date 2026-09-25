use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_fence::FormatFence, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFence {}

impl InstructionRegisterState for RegisterStateFence {
    #[cfg(any(feature = "test-utils", test))]
    fn random(_: &mut StdRng, _: &NormalizedOperands) -> Self {
        Self {}
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateFence
where
    I: RISCVInstruction<Format = FormatFence>,
{
    fn capture_pre(_instruction: &I, _cpu: &Cpu) -> Self {
        Self {}
    }

    fn capture_post(&mut self, _instruction: &I, _cpu: &Cpu) {}
}
