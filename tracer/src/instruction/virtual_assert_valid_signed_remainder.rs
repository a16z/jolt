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
    name = VirtualAssertValidSignedRemainder,
    mask = 0,
    match = 0,
    format = FormatB,
    ram = (),
    is_virtual = true
);

impl VirtualAssertValidSignedRemainder {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertValidSignedRemainder as RISCVInstruction>::RAMAccess,
    ) {
        match cpu.xlen {
            Xlen::Bit32 => {
                let remainder = cpu.x[self.operands.rs1] as i32;
                let divisor = cpu.x[self.operands.rs2] as i32;
                if remainder != 0 && divisor != 0 {
                    let remainder_sign = remainder >> 31;
                    let divisor_sign = divisor >> 31;
                    assert!(
                        remainder.unsigned_abs() < divisor.unsigned_abs()
                            && remainder_sign == divisor_sign
                    );
                }
            }
            Xlen::Bit64 => {
                let remainder = cpu.x[self.operands.rs1];
                let divisor = cpu.x[self.operands.rs2];
                if remainder != 0 && divisor != 0 {
                    let remainder_sign = remainder >> 63;
                    let divisor_sign = divisor >> 63;
                    assert!(
                        remainder.unsigned_abs() < divisor.unsigned_abs()
                            && remainder_sign == divisor_sign
                    );
                }
            }
        }
    }
}

impl RISCVTrace for VirtualAssertValidSignedRemainder {}
