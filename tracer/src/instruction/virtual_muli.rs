use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{
        format::{format_i::FormatI, InstructionFormat},
        RISCVInstruction, RISCVTrace,
    },
};

declare_riscv_instr!(
    name = VirtualMULI,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = (),
    is_virtual = true
);

impl VirtualMULI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualMULI as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = cpu
            .sign_extend(cpu.x[self.operands.rs1 as usize].wrapping_mul(self.operands.imm as i64))
    }
}

impl RISCVTrace for VirtualMULI {}
