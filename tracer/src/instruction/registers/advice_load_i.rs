use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_advice_load_i::FormatAdviceLoadI, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateAdviceLoadI {
    pub rd: (u64, u64), // (old_value, new_value)
}

impl InstructionRegisterState for RegisterStateAdviceLoadI {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, _operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        Self {
            rd: (rng.next_u64(), rng.next_u64()),
        }
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateAdviceLoadI
where
    I: RISCVInstruction<Format = FormatAdviceLoadI>,
{
    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self {
        let operands = instruction.operands();
        let rd = normalize_register_value(cpu, operands.rd as usize);
        Self { rd: (rd, rd) }
    }

    fn capture_post(&mut self, instruction: &I, cpu: &Cpu) {
        self.rd.1 = normalize_register_value(cpu, instruction.operands().rd as usize);
    }
}
