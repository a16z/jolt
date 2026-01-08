use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_u::FormatU, normalize_imm},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AUIPC,
    mask   = 0x0000007f,
    match  = 0x00000017,
    format = FormatU,
    ram    = ()
);

impl AUIPC {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AUIPC as RISCVInstruction>::RAMAccess) {
        let pc = self.address as i64;
        let imm = normalize_imm(self.operands.imm, &cpu.xlen);
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(pc.wrapping_add(imm));
    }
}

impl RISCVTrace for AUIPC {}
