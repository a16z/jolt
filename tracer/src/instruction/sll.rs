use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use serde::{Deserialize, Serialize};

use super::{
    format::format_r::FormatR, mul::MUL, virtual_pow2::VirtualPow2, Cycle, Instruction,
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SLL,
    mask   = 0xfe00707f,
    match  = 0x00001033,
    format = FormatR,
    ram    = ()
);

impl SLL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SLL as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize]
                .wrapping_shl(cpu.x[self.operands.rs2 as usize] as u32 & mask),
        );
    }
}

impl RISCVTrace for SLL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence for logical left shift operation.
    ///
    /// SLL (Shift Left Logical) shifts rs1 left by the shift amount in the lower
    /// 5 bits (RV32) or 6 bits (RV64) of rs2, filling the vacated bits with zeros.
    ///
    /// The implementation uses the mathematical equivalence:
    /// x << n = x × 2^n
    ///
    /// This allows the shift to be verified using multiplication, which is
    /// more amenable to proof systems that work with field arithmetic.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_pow2 = allocator.allocate(); // holds 2^shift_amount

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Compute 2^shift_amount
        // VirtualPow2 computes 2^(rs2 & mask) where mask is 0x1f (RV32) or 0x3f (RV64)
        asm.emit_i::<VirtualPow2>(*v_pow2, self.operands.rs2, 0);

        // Step 2: Multiply rs1 by 2^shift_amount
        // This achieves the left shift: rs1 << shift_amount = rs1 × 2^shift_amount
        asm.emit_r::<MUL>(self.operands.rd, self.operands.rs1, *v_pow2);

        asm.finalize()
    }
}
