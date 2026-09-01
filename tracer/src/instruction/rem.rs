use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = REM,
    mask   = 0xfe00707f,
    match  = 0x02006033,
    format = FormatR,
    ram    = ()
);

impl REM {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REM as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1 as usize];
        let divisor = cpu.x[self.operands.rs2 as usize];
        if divisor == 0 {
            cpu.write_register(self.operands.rd as usize, dividend);
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.write_register(self.operands.rd as usize, 0);
        } else {
            cpu.write_register(
                self.operands.rd as usize,
                cpu.sign_extend(
                    cpu.x[self.operands.rs1 as usize]
                        .wrapping_rem(cpu.x[self.operands.rs2 as usize]),
                ),
            );
        }
    }
}

impl RISCVTrace for REM {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // RISCV spec: For REM, the sign of a nonzero result equals the sign of the dividend.
        // REM operands
        let x = cpu.x[self.operands.rs1 as usize];
        let y = cpu.x[self.operands.rs2 as usize];

        let quotient_magnitude = if y == 0 {
            0
        } else if x == cpu.most_negative() && y == -1 {
            1 << 63
        } else {
            (x / y).unsigned_abs()
        };

        super::trace_inline_sequence_with_advice(
            &Instruction::from(*self),
            cpu,
            &[quotient_magnitude],
            trace,
        );
    }
}
