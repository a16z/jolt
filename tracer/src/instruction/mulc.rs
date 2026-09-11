use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MULC,
    mask   = 0,
    match  = 0,
    format = FormatR,
    ram    = (),
    produces_carry = true,
);

impl MULC {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULC as RISCVInstruction>::RAMAccess) {
        // Unsigned 64x64 widening product plus carry-in; cannot overflow 128
        // bits: (2^64-1)^2 + (2^64-1) < 2^128.
        let product = cpu.x[self.operands.rs1 as usize] as u64 as u128
            * (cpu.x[self.operands.rs2 as usize] as u64 as u128)
            + cpu.carry as u128;
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(product as u64 as i64),
        );
        cpu.carry = (product >> 64) as u64;
    }
}

impl RISCVTrace for MULC {}
