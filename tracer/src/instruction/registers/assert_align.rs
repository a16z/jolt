use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_assert_align::FormatAssert, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateAssert {
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStateAssert {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        Self {
            rs1: if operands.rs1.unwrap() == 0 {
                0
            } else {
                rng.next_u64()
            },
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateAssert
where
    I: RISCVInstruction<Format = FormatAssert>,
{
    type Before = Self;

    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self::Before {
        let operands = instruction.operands();
        Self {
            rs1: normalize_register_value(cpu, operands.rs1 as usize),
        }
    }

    fn capture_post(_instruction: &I, before: Self::Before, _cpu: &Cpu) -> Self {
        before
    }
}
