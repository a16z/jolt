use crate::emulator::cpu::Cpu;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Xlen};

use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::{
    andi::ANDI, format::format_r::FormatR, virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_sra::VirtualSRA, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SRAW,
    mask   = 0xfe00707f,
    match  = 0x4000003b | (0b101 << 12),
    format = FormatR,
    ram    = ()
);

impl SRAW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRAW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as i32) >> shamt) as i64;
    }
}

impl RISCVTrace for SRAW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Arithmetic right shift for 32-bit words with sign extension.
    ///
    /// SRAW is an RV64I-only instruction that arithmetically shifts the lower 32 bits
    /// of rs1 right by the amount in the lower 5 bits of rs2, preserving the sign bit,
    /// then sign-extends the 32-bit result to 64 bits.
    ///
    /// Implementation:
    /// 1. Sign-extend rs1 from 32 to 64 bits to prepare for arithmetic shift
    /// 2. Mask rs2 to get only the lower 5 bits (shift amount 0-31)
    /// 3. Generate bitmask for the shift amount
    /// 4. Apply arithmetic right shift (preserves sign bit)
    /// 5. Sign-extend the result to ensure proper 32-bit semantics
    ///
    /// The double sign-extension ensures correct handling of negative 32-bit values.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rs1 = allocator.allocate();
        let v_bitmask = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<VirtualSignExtendWord>(*v_rs1, self.operands.rs1, 0);
        asm.emit_i::<ANDI>(*v_bitmask, self.operands.rs2, 0x1f);
        asm.emit_i::<VirtualShiftRightBitmask>(*v_bitmask, *v_bitmask, 0);
        asm.emit_vshift_r::<VirtualSRA>(self.operands.rd, *v_rs1, *v_bitmask);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
