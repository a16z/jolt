use crate::instruction::registers::r::RegisterStateR;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualShiftDataH,
    mask = 0,
    match = 0,
    format = FormatR,
    registers = RegisterStateR,
    ram = ()
);

impl VirtualShiftDataH {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualShiftDataH as RISCVInstruction>::RAMAccess) {
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let ea = cpu.x[self.operands.rs2 as usize] as u64;
        let v = (x & 0xFFFF) << (8 * (ea & 6));
        cpu.write_register(self.operands.rd as usize, v as i64);
    }
}

impl RISCVTrace for VirtualShiftDataH {}
