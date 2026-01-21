use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualChangeDivisorW,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualChangeDivisorW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualChangeDivisorW as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => {
                panic!("VirtualChangeDivisorW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                let dividend = cpu.x[self.operands.rs1 as usize] as i32;
                let divisor = cpu.x[self.operands.rs2 as usize] as i32;
                if dividend == i32::MIN && divisor == -1 {
                    cpu.x[self.operands.rd as usize] = 1;
                } else {
                    cpu.x[self.operands.rd as usize] = divisor as i64;
                }
            }
        }
    }
}

impl RISCVTrace for VirtualChangeDivisorW {}
