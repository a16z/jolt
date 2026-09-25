use crate::{
    emulator::cpu::Cpu,
    instruction::{
        format::format_virtual_right_shift_r::FormatVirtualRightShiftR, RISCVInstruction,
    },
};
#[cfg(any(feature = "test-utils", test))]
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateVirtualRightShiftR<const MASK_WIDTH: usize = 64> {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
    pub rs2: u64,
}

impl<const MASK_WIDTH: usize> InstructionRegisterState
    for RegisterStateVirtualRightShiftR<MASK_WIDTH>
{
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1.unwrap() == 0 {
            0
        } else {
            rng.next_u64()
        };

        assert!((1..=64).contains(&MASK_WIDTH));
        let shift = rng.next_u64() % MASK_WIDTH as u64;

        debug_assert_ne!(
            operands.rs2.unwrap(),
            0,
            "rs2 cannot be 0 in VirtualRightShift instruction"
        );
        debug_assert_ne!(
            operands.rs2, operands.rs1,
            "rs2 cannot equal rs1 in VirtualRightShift instruction"
        );

        let rs2_value = ((1u128 << MASK_WIDTH) - (1u128 << shift)) as u64;

        Self {
            rd: (
                match operands.rd {
                    _ if operands.rd == operands.rs1 => rs1_value,
                    _ if operands.rd == operands.rs2 => rs2_value,
                    _ => rng.next_u64(),
                },
                rng.next_u64(),
            ),
            rs1: rs1_value,
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

impl<I, const MASK_WIDTH: usize> RegisterSnapshot<I> for RegisterStateVirtualRightShiftR<MASK_WIDTH>
where
    I: RISCVInstruction<Format = FormatVirtualRightShiftR<MASK_WIDTH>>,
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
