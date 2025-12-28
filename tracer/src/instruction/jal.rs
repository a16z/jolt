use serde::{Deserialize, Serialize};

use super::{
    format::{format_j::FormatJ, normalize_imm},
    RISCVInstruction, RISCVTrace,
};
use crate::declare_riscv_instr;
use crate::{
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
    instruction::format::NormalizedOperands,
};

declare_riscv_instr!(
    name   = JAL,
    mask   = 0x0000_007f,
    match  = 0x0000_006f,
    format = FormatJ,
    ram    = ()
);

impl JAL {
    // cpu.pc is pre-incremented by 4 (or 2 for compressed) in tick_operate() before execution,
    // self.address is the instruction address.
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <JAL as RISCVInstruction>::RAMAccess,
    ) {
        if self.operands.rd != 0 {
            if self.operands.rd == 1 {
                // Track function call if we're saving a return address (rd != 0)
                cpu.track_call(self.address, NormalizedOperands::from(self.operands));
            }
            cpu.x[self.operands.rd as usize] = cpu.sign_extend(cpu.pc as i64);
        }
        cpu.pc = ((self.address as i64).wrapping_add(normalize_imm(self.operands.imm, &cpu.xlen)))
            as u64;
    }
}

impl RISCVTrace for JAL {}
