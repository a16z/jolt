use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_r::FormatR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

// rd = rs1 & !rs2
declare_riscv_instr!(
    name   = ANDN,
    mask   = 0xfe00707f,
    match  = 0x40007033,
    format = FormatR,
    ram    = ()
);

impl ANDN {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ANDN as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] =
            cpu.sign_extend(cpu.x[self.operands.rs1 as usize] & !cpu.x[self.operands.rs2 as usize]);
    }
}

impl RISCVTrace for ANDN {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_constant() {
        let instr = ANDN::new(0x40007033, 0, true, false);
        assert_eq!(instr.operands.rs1, 0);
        assert_eq!(instr.operands.rs2, 0);
        assert_eq!(instr.operands.rd, 0);
    }
}
