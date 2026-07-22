use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_b::FormatB, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertEQ,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = ()
);

impl VirtualAssertEQ {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAssertEQ as RISCVInstruction>::RAMAccess) {
        if self.operands.imm == 0 {
            assert_eq!(
                cpu.x[self.operands.rs1 as usize],
                cpu.x[self.operands.rs2 as usize]
            );
        } else {
            let rs1 = cpu.x[self.operands.rs1 as usize];
            let rs2 = cpu.x[self.operands.rs2 as usize];
            if rs1 != rs2 {
                tracing::warn!(
                    "VirtualAssertEQ (spoil): rs1={rs1} != rs2={rs2}, proof will be unsatisfiable",
                );
            }
        }
    }
}

impl RISCVTrace for VirtualAssertEQ {}
