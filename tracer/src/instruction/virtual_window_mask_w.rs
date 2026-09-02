use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualWindowMaskW,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualWindowMaskW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualWindowMaskW as RISCVInstruction>::RAMAccess) {
        // Byte mask of the word at the effective address's offset within its
        // containing doubleword. Only bit 2 is read; bits 0-1 are zero on the
        // word-aligned addresses the surrounding sequence asserts. The
        // effective address is `rs1 + imm`.
        let ea =
            cpu.x[self.operands.rs1 as usize].wrapping_add(normalize_imm(self.operands.imm)) as u64;
        let mask = 0xFFFF_FFFFu64 << (32 * ((ea >> 2) & 1));
        cpu.write_register(self.operands.rd as usize, mask as i64);
    }
}

impl RISCVTrace for VirtualWindowMaskW {}
