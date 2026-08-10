use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MUL,
    mask   = 0xfe00707f,
    match  = 0x02000033,
    format = FormatR,
    ram    = (),
    produces_carry = true,
);

impl MUL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MUL as RISCVInstruction>::RAMAccess) {
        let rs1 = cpu.x[self.operands.rs1 as usize];
        let rs2 = cpu.x[self.operands.rs2 as usize];
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(rs1.wrapping_mul(rs2)),
        );
        // Carry-out: the high 64 bits of the unsigned 64x64 widening product
        // of the raw words (equals `MULHU rs1, rs2`).
        #[cfg(feature = "implicit-carry")]
        {
            cpu.carry = ((rs1 as u64 as u128 * rs2 as u64 as u128) >> 64) as u64;
        }
    }
}

impl RISCVTrace for MUL {}
