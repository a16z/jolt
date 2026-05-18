use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    fill_virtual_advice, format::format_r::FormatR, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = REMUW,
    mask   = 0xfe00707f,
    match  = 0x200703b,
    format = FormatR,
    ram    = ()
);

impl REMUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMUW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1 as usize] as u32;
        let divisor = cpu.x[self.operands.rs2 as usize] as u32;
        cpu.write_register(
            self.operands.rd as usize,
            (if divisor == 0 {
                dividend
            } else {
                dividend.wrapping_rem(divisor)
            }) as i32 as i64,
        );
    }
}

impl RISCVTrace for REMUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // REMUW operands
        let x = cpu.x[self.operands.rs1 as usize] as u32;
        let y = cpu.x[self.operands.rs2 as usize] as u32;

        let quotient = x.checked_div(y).map_or(u32::MAX as u64, u64::from);

        let mut inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        fill_virtual_advice(&mut inline_sequence, &[quotient]);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
