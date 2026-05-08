use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    fill_virtual_advice, format::format_r::FormatR, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = REMW,
    mask   = 0xfe00707f,
    match  = 0x200603b,
    format = FormatR,
    ram    = ()
);

impl REMW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1 as usize] as i32;
        let divisor = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.write_register(
            self.operands.rd as usize,
            (if divisor == 0 {
                dividend
            } else if dividend == i32::MIN && divisor == -1 {
                0
            } else {
                dividend.wrapping_rem(divisor)
            }) as i64,
        );
    }
}

impl RISCVTrace for REMW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // REMW operands
        let x = cpu.x[self.operands.rs1 as usize] as i32;
        let y = cpu.x[self.operands.rs2 as usize] as i32;

        let (quotient, remainder) = if y == 0 {
            (-1i32, x.unsigned_abs())
        } else if y == -1 && x == i32::MIN {
            (i32::MIN, 0) //overflow
        } else {
            let quotient = x / y;
            let remainder = x % y;
            (quotient, remainder.unsigned_abs())
        };

        let mut inline_sequence =
            Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        fill_virtual_advice(&mut inline_sequence, &[quotient as u64, remainder as u64]);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
