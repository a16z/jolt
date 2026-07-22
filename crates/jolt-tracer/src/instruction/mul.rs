use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MUL,
    mask   = 0xfe00707f,
    match  = 0x02000033,
    format = FormatR,
    ram    = ()
);

impl MUL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MUL as RISCVInstruction>::RAMAccess) {
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize].wrapping_mul(cpu.x[self.operands.rs2 as usize]),
            ),
        );
    }
}

impl RISCVTrace for MUL {}
