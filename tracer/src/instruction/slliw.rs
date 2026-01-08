use crate::emulator::cpu::Cpu;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{declare_riscv_instr, emulator::cpu::Xlen, instruction::virtual_muli::VirtualMULI};
use serde::{Deserialize, Serialize};

use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SLLIW,
    mask   = 0xfc00707f,
    match  = 0x0000101b,
    format = FormatI,
    ram    = ()
);

impl SLLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLLIW as RISCVInstruction>::RAMAccess) {
        // SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but
        // operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW,
        // and SRAIW encodings with imm[5] ≠ 0 are reserved.
        let shamt = (self.operands.imm & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) << shamt) as i32 as i64;
    }
}

impl RISCVTrace for SLLIW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Shift left logical immediate for 32-bit words with sign extension.
    ///
    /// SLLIW is an RV64I-only instruction that shifts the lower 32 bits of rs1
    /// left by a constant amount, then sign-extends the 32-bit result to 64 bits.
    ///
    /// Implementation:
    /// 1. Multiply by 2^shift_amount (equivalent to left shift)
    /// 2. Sign-extend the lower 32 bits to 64 bits
    ///
    /// The shift amount is restricted to 5 bits (0-31) for 32-bit operations.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let mask = match xlen {
            Xlen::Bit32 => panic!("SLLIW is invalid in 32b mode"),
            Xlen::Bit64 => 0x1f, //low 5bits
        };
        let shift = self.operands.imm & mask;

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<VirtualMULI>(self.operands.rd, self.operands.rs1, 1 << shift);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
