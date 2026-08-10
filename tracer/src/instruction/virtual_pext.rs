use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualPext,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualPext {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPext as RISCVInstruction>::RAMAccess) {
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;
        cpu.write_register(self.operands.rd as usize, pext(x, y) as i64);
    }
}

/// Packs `x`'s bits at `y`'s set positions (MSB-first), zero-extended.
pub(crate) fn pext(x: u64, y: u64) -> u64 {
    let mut pext = 0u64;
    for i in (0..64).rev() {
        if (y >> i) & 1 == 1 {
            pext = (pext << 1) | ((x >> i) & 1);
        }
    }
    pext
}

impl RISCVTrace for VirtualPext {}
