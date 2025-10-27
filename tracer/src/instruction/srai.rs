use serde::{Deserialize, Serialize};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::virtual_srai::VirtualSRAI,
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};

declare_riscv_instr!(
    name   = SRAI,
    mask   = 0xfc00707f,
    match  = 0x40005013,
    format = FormatI,
    ram    = ()
);

impl SRAI {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRAI as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize].wrapping_shr(self.operands.imm as u32 & mask),
        );
    }
}

impl RISCVTrace for SRAI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Arithmetic right shift immediate using bitmask approach.
    ///
    /// SRAI (Shift Right Arithmetic Immediate) shifts rs1 right by a constant amount,
    /// filling the vacated bits with copies of the sign bit.
    ///
    /// Implementation:
    /// 1. Calculate a bitmask for the shift amount
    /// 2. Apply arithmetic shift using VirtualSRAI with the bitmask
    ///
    /// The arithmetic shift preserves the sign bit, extending it into the
    /// high-order bits, which is essential for signed integer division by powers of 2.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let (shift, len) = match xlen {
            Xlen::Bit32 => (self.operands.imm & 0x1f, 32),
            Xlen::Bit64 => (self.operands.imm & 0x3f, 64),
        };
        let ones = (1u128 << (len - shift)) - 1;
        let bitmask = (ones << shift) as u64;

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_vshift_i::<VirtualSRAI>(self.operands.rd, self.operands.rs1, bitmask);
        asm.finalize()
    }
}
