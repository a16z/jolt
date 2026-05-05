use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    fill_virtual_advice, format::format_r::FormatR, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = DIVU,
    mask   = 0xfe00707f,
    match  = 0x02005033,
    format = FormatR,
    ram    = ()
);

impl DIVU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVU as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]);
        if divisor == 0 {
            cpu.write_register(self.operands.rd as usize, -1);
        } else {
            cpu.write_register(
                self.operands.rd as usize,
                cpu.sign_extend(dividend.wrapping_div(divisor) as i64),
            );
        }
    }
}

impl RISCVTrace for DIVU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // DIV operands
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;

        let quotient = if y == 0 {
            match cpu.xlen {
                Xlen::Bit32 => u32::MAX as u64,
                Xlen::Bit64 => u64::MAX,
            }
        } else {
            match cpu.xlen {
                Xlen::Bit32 => ((x as u32) / (y as u32)) as u64,
                Xlen::Bit64 => x / y,
            }
        };

        let mut inline_sequence =
            Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        fill_virtual_advice(&mut inline_sequence, &[quotient]);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
