use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, mul::MUL, virtual_sign_extend::VirtualSignExtend, RISCVInstruction,
    RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = MULW,
    mask   = 0xfe00707f,
    match  = 0x0200003b,
    format = FormatR,
    ram    = ()
);

impl MULW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULW as RISCVInstruction>::RAMAccess) {
        // MULW is an RV64 instruction that multiplies the lower 32 bits of the source registers,
        // placing the sign extension of the lower 32 bits of the result into the destination
        // register.
        let a = cpu.x[self.operands.rs1 as usize] as i32;
        let b = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.x[self.operands.rd as usize] = a.wrapping_mul(b) as i64;
    }
}

impl RISCVTrace for MULW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_r::<MUL>(self.operands.rd, self.operands.rs1, self.operands.rs2);
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
