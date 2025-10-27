use serde::{Deserialize, Serialize};

use super::{
    format::format_r::FormatR,
    ori::ORI,
    slli::SLLI,
    virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_sign_extend_word::VirtualSignExtendWord,
    virtual_srl::VirtualSRL,
    Cycle,
    Instruction,
    RISCVInstruction,
    RISCVTrace,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};

declare_riscv_instr!(
    name   = SRLW,
    mask   = 0xfe00707f,
    match  = 0x0000003b | (0b101 << 12),
    format = FormatR,
    ram    = ()
);

impl SRLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRLW as RISCVInstruction>::RAMAccess) {
        // SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate
        // on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is
        // given by rs2[4:0].
        let shamt = (cpu.x[self.operands.rs2 as usize] & 0x1f) as u32;
        cpu.x[self.operands.rd as usize] =
            ((cpu.x[self.operands.rs1 as usize] as u32) >> shamt) as i32 as i64;
    }
}

impl RISCVTrace for SRLW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Logical right shift for 32-bit words with sign extension.
    ///
    /// SRLW is an RV64I-only instruction that logically shifts the lower 32 bits
    /// of rs1 right by the amount in the lower 5 bits of rs2, then sign-extends
    /// the 32-bit result to 64 bits.
    ///
    /// Implementation:
    /// 1. Shift rs1 left by 32 to position the lower 32 bits in the upper half
    /// 2. OR rs2 with 32 to create a shift amount for 64-bit logical right shift
    /// 3. Generate bitmask for the adjusted shift amount
    /// 4. Apply logical right shift (which now operates on the upper 32 bits)
    /// 5. Sign-extend the resulting 32-bit value
    ///
    /// This approach ensures proper 32-bit logical shift semantics on 64-bit system.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_bitmask = allocator.allocate();
        let v_rs1 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<SLLI>(*v_rs1, self.operands.rs1, 32);
        asm.emit_i::<ORI>(*v_bitmask, self.operands.rs2, 32);
        asm.emit_i::<VirtualShiftRightBitmask>(*v_bitmask, *v_bitmask, 0);
        asm.emit_vshift_r::<VirtualSRL>(self.operands.rd, *v_rs1, *v_bitmask);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
