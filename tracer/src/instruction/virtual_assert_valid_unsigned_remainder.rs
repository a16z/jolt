use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{
        cpu::{GeneralizedCpu, Xlen},
        memory::MemoryData,
    },
};

use super::{format::format_b::FormatB, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertValidUnsignedRemainder,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = ()
);

impl VirtualAssertValidUnsignedRemainder {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualAssertValidUnsignedRemainder as RISCVInstruction>::RAMAccess,
    ) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let remainder = cpu.x[self.operands.rs1 as usize] as i32 as u32;
                let divisor = cpu.x[self.operands.rs2 as usize] as i32 as u32;
                assert!(divisor == 0 || remainder < divisor);
            }
            Xlen::Bit64 => {
                let remainder = cpu.x[self.operands.rs1 as usize] as u64;
                let divisor = cpu.x[self.operands.rs2 as usize] as u64;
                assert!(divisor == 0 || remainder < divisor);
            }
        }
    }
}

impl RISCVTrace for VirtualAssertValidUnsignedRemainder {}
