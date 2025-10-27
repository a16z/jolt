use serde::{Deserialize, Serialize};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

declare_riscv_instr!(
    name   = MULHU,
    mask   = 0xfe00707f,
    match  = 0x02003033,
    format = FormatR,
    ram    = ()
);

impl MULHU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULHU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                (((cpu.x[self.operands.rs1 as usize] as u32 as u64)
                    * (cpu.x[self.operands.rs2 as usize] as u32 as u64))
                    >> 32) as i64,
            ),
            Xlen::Bit64 => {
                ((cpu.x[self.operands.rs1 as usize] as u64 as u128)
                    .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u64 as u128)
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULHU {}
