//! Virtual instruction that asserts unsigned multiplication of two operands does not overflow.

use super::{format::format_b::FormatB, RISCVInstruction, RISCVTrace};
use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name = VirtualAssertMulUNoOverflow,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = ()
);

impl VirtualAssertMulUNoOverflow {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertMulUNoOverflow as RISCVInstruction>::RAMAccess,
    ) {
        let rs1_val = cpu.x[self.operands.rs1 as usize] as u64;
        let rs2_val = cpu.x[self.operands.rs2 as usize] as u64;
        assert!(rs1_val.checked_mul(rs2_val).is_some());
    }
}

impl RISCVTrace for VirtualAssertMulUNoOverflow {}
