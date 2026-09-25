use crate::{
    emulator::cpu::Cpu,
    instruction::{format::format_amo::FormatAMO, RISCVInstruction},
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateAMO {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,       // Memory address
    pub rs2: u64,       // Value to use in atomic operation
}

impl InstructionRegisterState for RegisterStateAMO {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self {
        use crate::instruction::test::{DRAM_BASE, TEST_MEMORY_CAPACITY};
        use rand::RngCore;

        // Sample naturally aligned addresses inside the emulated DRAM region.
        let alignment = 8; // Fits both AMO.W and AMO.D.
        let max_offset = (TEST_MEMORY_CAPACITY / 2).min(0x10000) - alignment;
        let offset = (rng.next_u64() % (max_offset / alignment)) * alignment;
        let address = DRAM_BASE + offset;

        debug_assert_ne!(operands.rs1.unwrap(), 0);

        let rs2_value = match operands.rs2.unwrap() {
            0 => 0,
            _ if operands.rs2 == operands.rs1 => address,
            _ => rng.next_u64(),
        };

        Self {
            rd: (
                match operands.rd {
                    _ if operands.rd == operands.rs1 => address,
                    _ if operands.rd == operands.rs2 => rs2_value,
                    _ => rng.next_u64(),
                },
                rng.next_u64(),
            ),
            rs1: address,
            rs2: rs2_value,
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rs2_value(&self) -> Option<u64> {
        Some(self.rs2)
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl<I> RegisterSnapshot<I> for RegisterStateAMO
where
    I: RISCVInstruction<Format = FormatAMO>,
{
    type Before = (u64, u64, u64);

    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self::Before {
        let operands = instruction.operands();
        (
            normalize_register_value(cpu, operands.rs1 as usize),
            normalize_register_value(cpu, operands.rs2 as usize),
            normalize_register_value(cpu, operands.rd as usize),
        )
    }

    fn capture_post(instruction: &I, before: Self::Before, cpu: &Cpu) -> Self {
        let (rs1, rs2, rd_pre) = before;
        Self {
            rd: (
                rd_pre,
                normalize_register_value(cpu, instruction.operands().rd as usize),
            ),
            rs1,
            rs2,
        }
    }
}
