use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    fill_virtual_advice, format::format_r::FormatR, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = DIVW,
    mask   = 0xfe00707f,
    match  = 0x200403b,
    format = FormatR,
    ram    = ()
);

impl DIVW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVW as RISCVInstruction>::RAMAccess) {
        // DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower
        // 32 bits of rs2, treating them as signed and unsigned integers, placing the 32-bit
        // quotient in rd, sign-extended to 64 bits.
        let dividend = cpu.x[self.operands.rs1 as usize] as i32;
        let divisor = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.write_register(
            self.operands.rd as usize,
            (if divisor == 0 {
                -1i32
            } else if dividend == i32::MIN && divisor == -1 {
                dividend
            } else {
                dividend.wrapping_div(divisor)
            }) as i64,
        );
    }
}

impl RISCVTrace for DIVW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // DIVW operands
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
