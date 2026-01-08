use crate::emulator::cpu::Cpu;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Xlen};

use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::{
    format::format_r::FormatR, mul::MUL, virtual_pow2_w::VirtualPow2W, Cycle, Instruction,
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SLLW,
    mask   = 0xfe00707f,
    match  = 0x0000003b | (0b001 << 12),
    format = FormatR,
    ram    = ()
);

impl SLLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) << shamt) as i32 as i64;
    }
}

impl RISCVTrace for SLLW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Shift left logical for 32-bit words with sign extension.
    ///
    /// SLLW is an RV64I-only instruction that shifts the lower 32 bits of rs1
    /// left by the amount in the lower 5 bits of rs2, then sign-extends the
    /// 32-bit result to 64 bits.
    ///
    /// Implementation:
    /// 1. Compute 2^(rs2[4:0]) using VirtualPow2W
    /// 2. Multiply rs1 by this power of 2 (equivalent to left shift)
    /// 3. Sign-extend the lower 32 bits of the result
    ///
    /// The multiplication approach allows zkVM-friendly verification.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_pow2 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<VirtualPow2W>(*v_pow2, self.operands.rs2, 0);
        asm.emit_r::<MUL>(self.operands.rd, self.operands.rs1, *v_pow2);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
