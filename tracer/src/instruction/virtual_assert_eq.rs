use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
};

use super::{format::format_b::FormatB, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertEQ,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = ()
);

impl VirtualAssertEQ {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualAssertEQ as RISCVInstruction>::RAMAccess,
    ) {
        assert_eq!(
            cpu.x[self.operands.rs1 as usize],
            cpu.x[self.operands.rs2 as usize]
        );
    }
}

impl RISCVTrace for VirtualAssertEQ {}
