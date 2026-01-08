use crate::emulator::cpu::Cpu;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Xlen, instruction::virtual_muli::VirtualMULI};

use super::{format::format_i::FormatI, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SLLI,
    mask   = 0xfc00707f,
    match  = 0x00001013,
    format = FormatI,
    ram    = ()
);

impl SLLI {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <SLLI as RISCVInstruction>::RAMAccess,
    ) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize].wrapping_shl(self.operands.imm as u32 & mask),
        );
    }
}

impl RISCVTrace for SLLI {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for shift left logical immediate.
    ///
    /// SLLI (Shift Left Logical Immediate) shifts rs1 left by a constant amount.
    /// The implementation uses multiplication by 2^shift_amount, leveraging the
    /// mathematical equivalence: x << n = x * 2^n
    ///
    /// This approach is zkVM-friendly as multiplication is a native operation
    /// that can be efficiently verified in the constraint system.
    ///
    /// The shift amount is masked to 5 bits on RV32 (0-31) or 6 bits on RV64 (0-63),
    /// ensuring the shift stays within valid bounds for the architecture.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // Determine word size based on immediate value and instruction encoding
        // For SLLI: RV32 uses 5-bit immediates (0-31), RV64 uses 6-bit immediates (0-63)
        let mask = match xlen {
            Xlen::Bit32 => 0x1f, //low 5bits
            Xlen::Bit64 => 0x3f, //low 6bits
        };
        let shift = self.operands.imm & mask;

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<VirtualMULI>(self.operands.rd, self.operands.rs1, 1 << shift);
        asm.finalize()
    }
}
