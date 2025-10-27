use serde::{Deserialize, Serialize};

use super::{
    format::format_i::FormatI,
    slli::SLLI,
    virtual_sign_extend_word::VirtualSignExtendWord,
    Cycle,
    Instruction,
    RISCVInstruction,
    RISCVTrace,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::virtual_srli::VirtualSRLI,
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};

declare_riscv_instr!(
    name   = SRLIW,
    mask   = 0xfc00707f,
    match  = 0x0000501b,
    format = FormatI,
    ram    = ()
);

impl SRLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLIW as RISCVInstruction>::RAMAccess) {
        // SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but
        // operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW,
        // and SRAIW encodings with imm[5] ≠ 0 are reserved.
        let shamt = (self.operands.imm & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) >> shamt) as i32 as i64;
    }
}

impl RISCVTrace for SRLIW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Logical right shift immediate for 32-bit words with sign extension.
    ///
    /// SRLIW is an RV64I-only instruction that logically shifts the lower 32 bits
    /// of rs1 right by a constant amount, then sign-extends the 32-bit result to 64 bits.
    ///
    /// Implementation:
    /// 1. Shift rs1 left by 32 to position the lower 32 bits in the upper half
    /// 2. Apply a 64-bit logical right shift by (shift_amount + 32)
    /// 3. Sign-extend the resulting 32-bit value
    ///
    /// This technique ensures proper 32-bit logical shift semantics on 64-bit system.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rs1 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<SLLI>(*v_rs1, self.operands.rs1, 32);
        let (shift, len) = match xlen {
            Xlen::Bit32 => panic!("SRLIW is invalid in 32b mode"),
            Xlen::Bit64 => ((self.operands.imm & 0x1f) + 32, 64),
        };
        let ones = (1u128 << (len - shift)) - 1;
        let bitmask = (ones << shift) as u64;
        asm.emit_vshift_i::<VirtualSRLI>(self.operands.rd, *v_rs1, bitmask);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
