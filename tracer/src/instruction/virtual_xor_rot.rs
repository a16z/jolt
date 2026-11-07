use crate::instruction::registers::r::RegisterStateR;
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
            registers = RegisterStateR,
            ram = ()
        );

        impl $name {
            fn exec(&self, cpu: &mut Cpu, _: &mut <$name as RISCVInstruction>::RamAccess) {
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
declare_xorrot!(VirtualXORROT3, 3);
declare_xorrot!(VirtualXORROT4, 4);
declare_xorrot!(VirtualXORROT6, 6);
declare_xorrot!(VirtualXORROT9, 9);
declare_xorrot!(VirtualXORROT10, 10);
declare_xorrot!(VirtualXORROT14, 14);
declare_xorrot!(VirtualXORROT18, 18);
declare_xorrot!(VirtualXORROT21, 21);
declare_xorrot!(VirtualXORROT22, 22);
declare_xorrot!(VirtualXORROT23, 23);
declare_xorrot!(VirtualXORROT26, 26);
declare_xorrot!(VirtualXORROT27, 27);
declare_xorrot!(VirtualXORROT33, 33);
declare_xorrot!(VirtualXORROT43, 43);
declare_xorrot!(VirtualXORROT44, 44);
declare_xorrot!(VirtualXORROT45, 45);
declare_xorrot!(VirtualXORROT47, 47);
declare_xorrot!(VirtualXORROT49, 49);
declare_xorrot!(VirtualXORROT52, 52);
declare_xorrot!(VirtualXORROT60, 60);
declare_xorrot!(VirtualXORROT61, 61);
