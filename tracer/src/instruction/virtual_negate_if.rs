use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualNegateIf,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualNegateIf {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualNegateIf as RISCVInstruction>::RAMAccess) {
        let sign_source = cpu.x[self.operands.rs1 as usize];
        let value = cpu.x[self.operands.rs2 as usize];
        cpu.write_register(
            self.operands.rd as usize,
            if sign_source < 0 {
                value.wrapping_neg()
            } else {
                value
            },
        );
    }
}

impl RISCVTrace for VirtualNegateIf {}
