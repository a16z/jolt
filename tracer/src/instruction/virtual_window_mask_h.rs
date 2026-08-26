use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualWindowMaskH,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualWindowMaskH {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualWindowMaskH as RISCVInstruction>::RAMAccess) {
        // Byte mask of the halfword at the effective address's offset within
        // its containing doubleword. Bit 0 is ignored; it is zero on the
        // halfword-aligned addresses the surrounding sequence asserts. The
        // effective address is `rs1 + imm`.
        let ea =
            cpu.x[self.operands.rs1 as usize].wrapping_add(normalize_imm(self.operands.imm)) as u64;
        let mask = 0xFFFFu64 << (8 * (ea & 6));
        cpu.write_register(self.operands.rd as usize, mask as i64);
    }
}

impl RISCVTrace for VirtualWindowMaskH {}
