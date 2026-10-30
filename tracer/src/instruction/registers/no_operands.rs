use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_no_operands::FormatNoOperands, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateNoOperands {}

impl InstructionRegisterState for RegisterStateNoOperands {
    #[cfg(any(feature = "test-utils", test))]
    fn random(_: &mut StdRng, _: &NormalizedOperands) -> Self {
        Self {}
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateNoOperands
where
    I: RISCVInstruction<Format = FormatNoOperands>,
{
    fn capture_pre(_instruction: &I, _cpu: &Cpu) -> Self {
        Self {}
    }

    fn capture_post(&mut self, _instruction: &I, _cpu: &Cpu) {}
}
