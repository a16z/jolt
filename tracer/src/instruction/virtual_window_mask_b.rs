use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_i::FormatI, normalize_imm},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualWindowMaskB,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualWindowMaskB {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualWindowMaskB as RISCVInstruction>::RAMAccess) {
        // Byte mask of the byte at the effective address's offset within its
        // containing doubleword; the effective address is `rs1 + imm`.
        let ea =
            cpu.x[self.operands.rs1 as usize].wrapping_add(normalize_imm(self.operands.imm)) as u64;
        let mask = 0xFFu64 << (8 * (ea & 7));
        cpu.write_register(self.operands.rd as usize, mask as i64);
    }
}

impl RISCVTrace for VirtualWindowMaskB {}
