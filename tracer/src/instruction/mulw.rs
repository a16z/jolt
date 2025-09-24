use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence for 32-bit multiplication on 64-bit systems.
    ///
    /// MULW is an RV64 instruction that multiplies the lower 32 bits of rs1 and rs2,
    /// treating them as 32-bit signed integers, and sign-extends the lower 32 bits
    /// of the result to 64 bits.
    ///
    /// The implementation:
    /// 1. Performs full 64-bit multiplication (which includes the 32-bit result)
    /// 2. Sign-extends the lower 32 bits to get the final 64-bit result
    ///
    /// This works because the lower 32 bits of a 64-bit multiplication are identical
    /// to a 32-bit multiplication, regardless of the upper bits of the operands.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Perform full 64-bit multiplication
        // The lower 32 bits contain the 32-bit multiplication result
        asm.emit_r::<MUL>(self.operands.rd, self.operands.rs1, self.operands.rs2);

        // Step 2: Sign-extend the 32-bit result to 64 bits
        // This ensures bits 32-63 match bit 31 (sign bit of 32-bit result)
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);

        asm.finalize()
    }
}
