use crate::emulator::cpu::Cpu;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Xlen};

use super::{
    format::format_r::FormatR, mul::MUL, virtual_sign_extend_word::VirtualSignExtendWord, Cycle,
    Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = MULW,
    mask   = 0xfe00707f,
    match  = 0x0200003b,
    format = FormatR,
    ram    = ()
);

impl MULW {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <MULW as RISCVInstruction>::RAMAccess,
    ) {
        // MULW is an RV64 instruction that multiplies the lower 32 bits of the source registers,
        // placing the sign extension of the lower 32 bits of the result into the destination
        // register.
        let a = cpu.x[self.operands.rs1 as usize] as i32;
        let b = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.x[self.operands.rd as usize] = a.wrapping_mul(b) as i64;
    }
}

impl RISCVTrace for MULW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        asm.emit_r::<MUL>(self.operands.rd, self.operands.rs1, self.operands.rs2);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);

        asm.finalize()
    }
}
