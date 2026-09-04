use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = ADDC,
    mask   = 0,
    match  = 0,
    format = FormatR,
    ram    = (),
    produces_carry = true,
);

impl ADDC {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADDC as RISCVInstruction>::RAMAccess) {
        let sum = cpu.x[self.operands.rs1 as usize] as u64 as u128
            + cpu.x[self.operands.rs2 as usize] as u64 as u128
            + cpu.carry as u128;
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(sum as u64 as i64),
        );
        cpu.carry = (sum >> 64) as u64;
    }
}

impl RISCVTrace for ADDC {}
