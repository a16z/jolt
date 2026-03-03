use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualSpoilProof,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualSpoilProof {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSpoilProof as RISCVInstruction>::RAMAccess) {
        tracing::warn!(
            "VirtualSpoilProof: rs1={} != rs2={}, proof will be unsatisfiable",
            cpu.x[self.operands.rs1 as usize],
            cpu.x[self.operands.rs2 as usize]
        );
    }
}

impl RISCVTrace for VirtualSpoilProof {}
