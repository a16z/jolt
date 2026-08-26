use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, Cycle, Instruction, RISCVInstruction, RISCVTrace};

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

        let quotient = if y == 0 {
            (-1i32) as u64
        } else if y == -1 && x == i32::MIN {
            0x8000_0000
        } else {
            (x / y) as u64
        };

        super::trace_inline_sequence_with_advice(
            &Instruction::from(*self),
            cpu,
            &[quotient],
            trace,
        );
    }
}
