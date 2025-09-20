use serde::{Deserialize, Serialize};

use super::{RISCVInstruction, RISCVTrace};
use crate::instruction::format::format_r::FormatR;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

macro_rules! declare_xorrot {
    ($name:ident, $rotation:expr) => {
        declare_riscv_instr!(
            name = $name,
            mask = 0,
            match = 0,
            format = FormatR,
            ram = (),
            is_virtual = true
        );

        impl $name {
            fn exec(&self, cpu: &mut Cpu, _: &mut <$name as RISCVInstruction>::RAMAccess) {
                let xor_result = cpu.x[self.operands.rs1 as usize] ^ cpu.x[self.operands.rs2 as usize];
                let rotated = xor_result.rotate_right($rotation);
                cpu.x[self.operands.rd as usize] = rotated as i64;
            }
        }

        impl RISCVTrace for $name {}
    };
}
declare_xorrot!(VirtualXORROT32, 32);
declare_xorrot!(VirtualXORROT24, 24);
declare_xorrot!(VirtualXORROT16, 16);
declare_xorrot!(VirtualXORROT63, 63);
