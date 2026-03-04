use serde::{Deserialize, Serialize};

use super::{RISCVInstruction, RISCVTrace};
use crate::instruction::format::format_r::FormatR;
use crate::{declare_riscv_instr, emulator::cpu::Cpu, emulator::cpu::Xlen};

macro_rules! declare_xorrotw {
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
                match cpu.xlen {
                    Xlen::Bit32 => {
                        panic!("XORROTW instructions are not supported in 32-bit mode");
                    }
                    Xlen::Bit64 => {
                        let rs1_val = cpu.x[self.operands.rs1 as usize] as u32;
                        let rs2_val = cpu.x[self.operands.rs2 as usize] as u32;
                        let xor_result = rs1_val ^ rs2_val;
                        let rotated = xor_result.rotate_right($rotation);
                        cpu.write_register(self.operands.rd as usize, rotated as i64);
                    }
                }
            }
        }

        impl RISCVTrace for $name {}
    };
}

declare_xorrotw!(VirtualXORROTW16, 16);
declare_xorrotw!(VirtualXORROTW12, 12);
declare_xorrotw!(VirtualXORROTW8, 8);
declare_xorrotw!(VirtualXORROTW7, 7);
