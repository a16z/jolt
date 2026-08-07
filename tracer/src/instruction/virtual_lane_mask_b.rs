use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(name = VirtualLaneMaskB, mask = 0, match = 0, format = FormatI, ram = ());

impl VirtualLaneMaskB {
    fn exec(&self, cpu: &mut Cpu, _: &mut <Self as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm as i64);
        cpu.write_register(
            self.operands.rd as usize,
            0xffu64.wrapping_shl(8 * (address as u32 & 7)) as i64,
        );
    }
}

impl RISCVTrace for VirtualLaneMaskB {}
