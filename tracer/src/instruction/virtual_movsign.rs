use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_i::FormatI, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

// Constants for 32-bit and 64-bit word sizes
const ALL_ONES_32: u64 = 0xFFFF_FFFF;
const ALL_ONES_64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT_32: u64 = 0x8000_0000;
const SIGN_BIT_64: u64 = 0x8000_0000_0000_0000;

declare_riscv_instr!(
    name = VirtualMovsign,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualMovsign {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualMovsign as RISCVInstruction>::RAMAccess) {
        let val = cpu.x[self.operands.rs1] as u64;
        cpu.x[self.operands.rd] = match cpu.xlen {
            Xlen::Bit32 => {
                if val & SIGN_BIT_32 != 0 {
                    // Should this be ALL_ONES_64?
                    ALL_ONES_32 as i64
                } else {
                    0
                }
            }
            Xlen::Bit64 => {
                if val & SIGN_BIT_64 != 0 {
                    ALL_ONES_64 as i64
                } else {
                    0
                }
            }
        };
    }
}

impl RISCVTrace for VirtualMovsign {}
