use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    fill_virtual_advice, format::format_r::FormatR, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = DIV,
    mask   = 0xfe00707f,
    match  = 0x02004033,
    format = FormatR,
    ram    = ()
);

impl DIV {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIV as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1 as usize];
        let divisor = cpu.x[self.operands.rs2 as usize];
        if divisor == 0 {
            cpu.write_register(self.operands.rd as usize, -1);
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.write_register(self.operands.rd as usize, dividend);
        } else {
            cpu.write_register(
                self.operands.rd as usize,
                cpu.sign_extend(dividend.wrapping_div(divisor)),
            );
        }
    }
}

impl RISCVTrace for DIV {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // RISCV spec: For REM, the sign of a nonzero result equals the sign of the dividend.
        // DIV operands
        let x = cpu.x[self.operands.rs1 as usize];
        let y = cpu.x[self.operands.rs2 as usize];

        let (quotient, remainder) = if y == 0 {
            (u64::MAX, x.unsigned_abs())
        } else if x == cpu.most_negative() && y == -1 {
            (x as u64, 0)
        } else {
            let quotient = x / y;
            let remainder = (x % y).unsigned_abs();
            (quotient as u64, remainder)
        };

        let mut inline_sequence =
            Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        fill_virtual_advice(&mut inline_sequence, &[quotient, remainder]);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
