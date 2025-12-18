use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
    instruction::{format::format_i::FormatI, RISCVInstruction, RISCVTrace},
};

declare_riscv_instr!(
    name = VirtualMULI,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualMULI {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualMULI as RISCVInstruction>::RAMAccess,
    ) {
        cpu.x[self.operands.rd as usize] = cpu
            .sign_extend(cpu.x[self.operands.rs1 as usize].wrapping_mul(self.operands.imm as i64))
    }
}

impl RISCVTrace for VirtualMULI {}
