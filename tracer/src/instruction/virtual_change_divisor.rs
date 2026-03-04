use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualChangeDivisor,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualChangeDivisor {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualChangeDivisor as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let dividend = cpu.x[self.operands.rs1 as usize] as i32;
                let divisor = cpu.x[self.operands.rs2 as usize] as i32;
                if dividend == i32::MIN && divisor == -1 {
                    cpu.write_register(self.operands.rd as usize, 1);
                } else {
                    cpu.write_register(self.operands.rd as usize, divisor as i64);
                }
            }
            Xlen::Bit64 => {
                let dividend = cpu.x[self.operands.rs1 as usize];
                let divisor = cpu.x[self.operands.rs2 as usize];
                if dividend == i64::MIN && divisor == -1 {
                    cpu.write_register(self.operands.rd as usize, 1);
                } else {
                    cpu.write_register(self.operands.rd as usize, divisor);
                }
            }
        }
    }
}

impl RISCVTrace for VirtualChangeDivisor {}
