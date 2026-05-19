use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    fill_virtual_advice, format::format_r::FormatR, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = REMU,
    mask   = 0xfe00707f,
    match  = 0x02007033,
    format = FormatR,
    ram    = ()
);

impl REMU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMU as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]);
        cpu.write_register(
            self.operands.rd as usize,
            match divisor {
                0 => cpu.sign_extend(dividend as i64),
                _ => cpu.sign_extend(dividend.wrapping_rem(divisor) as i64),
            },
        );
    }
}

impl RISCVTrace for REMU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let mut inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let quotient = if cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) == 0 {
            u64::MAX
        } else {
            cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                / cpu.unsigned_data(cpu.x[self.operands.rs2 as usize])
        };
        fill_virtual_advice(&mut inline_sequence, &[quotient]);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
