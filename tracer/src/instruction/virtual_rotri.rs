use serde::{Deserialize, Serialize};

use crate::instruction::format::format_i::FormatI;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::InstructionFormat, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualROTRI,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualROTRI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualROTRI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1])
                .rotate_right(self.operands.imm as u32) as i64,
        );
    }
}

impl RISCVTrace for VirtualROTRI {}
