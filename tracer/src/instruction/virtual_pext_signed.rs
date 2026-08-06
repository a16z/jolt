use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualPextSigned,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualPextSigned {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPextSigned as RISCVInstruction>::RAMAccess) {
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;
        cpu.write_register(self.operands.rd as usize, pext_signed(x, y) as i64);
    }
}

/// Packs `x`'s bits at `y`'s set positions (MSB-first), then sign-extends by
/// the extracted window's top bit: `pext(x, y) + σ·(2^64 − 2^popcount(y))`.
pub(crate) fn pext_signed(x: u64, y: u64) -> u64 {
    let mut pext = 0u64;
    let mut sign = 0u64;
    let mut pc = 0u32;
    for i in (0..64).rev() {
        if (y >> i) & 1 == 1 {
            let x_i = (x >> i) & 1;
            if pc == 0 {
                sign = x_i;
            }
            pext = (pext << 1) | x_i;
            pc += 1;
        }
    }
    let ext = if sign == 1 && pc < 64 {
        u64::MAX << pc
    } else {
        0
    };
    pext | ext
}

impl RISCVTrace for VirtualPextSigned {}
