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
            ram = ()
        );

        impl $name {
            fn exec(&self, cpu: &mut Cpu, _: &mut <$name as RISCVInstruction>::RAMAccess) {
                let xor_result = cpu.x[self.operands.rs1 as usize] ^ cpu.x[self.operands.rs2 as usize];
                let rotated = xor_result.rotate_right($rotation);
                cpu.write_register(self.operands.rd as usize, rotated as i64);
            }
        }

        impl RISCVTrace for $name {}
    };
}
declare_xorrot!(VirtualXORROT32, 32);
declare_xorrot!(VirtualXORROT24, 24);
declare_xorrot!(VirtualXORROT16, 16);
declare_xorrot!(VirtualXORROT63, 63);
declare_xorrot!(VirtualXORROT2, 2);
declare_xorrot!(VirtualXORROT3, 3);
declare_xorrot!(VirtualXORROT8, 8);
declare_xorrot!(VirtualXORROT9, 9);
declare_xorrot!(VirtualXORROT19, 19);
declare_xorrot!(VirtualXORROT20, 20);
declare_xorrot!(VirtualXORROT21, 21);
declare_xorrot!(VirtualXORROT23, 23);
declare_xorrot!(VirtualXORROT25, 25);
declare_xorrot!(VirtualXORROT28, 28);
declare_xorrot!(VirtualXORROT36, 36);
declare_xorrot!(VirtualXORROT37, 37);
declare_xorrot!(VirtualXORROT39, 39);
declare_xorrot!(VirtualXORROT43, 43);
declare_xorrot!(VirtualXORROT44, 44);
declare_xorrot!(VirtualXORROT46, 46);
declare_xorrot!(VirtualXORROT49, 49);
declare_xorrot!(VirtualXORROT50, 50);
declare_xorrot!(VirtualXORROT54, 54);
declare_xorrot!(VirtualXORROT56, 56);
declare_xorrot!(VirtualXORROT58, 58);
declare_xorrot!(VirtualXORROT61, 61);
declare_xorrot!(VirtualXORROT62, 62);
