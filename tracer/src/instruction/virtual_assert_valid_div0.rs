use serde::{Deserialize, Serialize};

use super::{format::format_b::FormatB, RISCVInstruction, RISCVTrace};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

declare_riscv_instr!(
    name = VirtualAssertValidDiv0,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = ()
);

impl VirtualAssertValidDiv0 {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAssertValidDiv0 as RISCVInstruction>::RAMAccess) {
        let divisor = cpu.x[self.operands.rs1 as usize];
        let quotient = cpu.x[self.operands.rs2 as usize];
        match cpu.xlen {
            Xlen::Bit32 => {
                if divisor == 0 {
                    assert!(quotient as u64 as u32 == u32::MAX);
                }
            }
            Xlen::Bit64 => {
                if divisor == 0 {
                    assert!(quotient as u64 == u64::MAX);
                }
            }
        }
    }
}

impl RISCVTrace for VirtualAssertValidDiv0 {}
