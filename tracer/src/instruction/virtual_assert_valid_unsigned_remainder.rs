use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::{format_b::FormatB, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualAssertValidUnsignedRemainder,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = (),
    is_virtual = true
);

impl VirtualAssertValidUnsignedRemainder {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertValidUnsignedRemainder as RISCVInstruction>::RAMAccess,
    ) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let remainder = cpu.x[self.operands.rs1] as i32 as u32;
                let divisor = cpu.x[self.operands.rs2] as i32 as u32;
                assert!(divisor == 0 || remainder < divisor);
            }
            Xlen::Bit64 => {
                let remainder = cpu.x[self.operands.rs1] as u64;
                let divisor = cpu.x[self.operands.rs2] as u64;
                assert!(divisor == 0 || remainder < divisor);
            }
        }
    }
}

impl RISCVTrace for VirtualAssertValidUnsignedRemainder {}
