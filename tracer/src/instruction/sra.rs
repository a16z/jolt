use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_sra::VirtualSRA, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SRA,
    mask   = 0xfe00707f,
    match  = 0x40005033,
    format = FormatR,
    ram    = ()
);

impl SRA {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRA as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.x[self.operands.rs1 as usize]
                .wrapping_shr(cpu.x[self.operands.rs2 as usize] as u32 & mask),
        );
    }
}

impl RISCVTrace for SRA {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence for arithmetic right shift operation.
    ///
    /// SRA (Shift Right Arithmetic) shifts rs1 right by the shift amount in rs2,
    /// filling the vacated bits with copies of the sign bit (bit XLEN-1).
    ///
    /// The implementation uses a bitmask approach where:
    /// 1. A bitmask is computed based on the shift amount
    /// 2. Virtual SRA instruction applies the shift using the bitmask
    ///
    /// This approach allows the shift to be verified in a way that's compatible
    /// with zkVM's constraint system.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_bitmask = allocator.allocate(); // holds the shift bitmask

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Generate bitmask for right shift
        // Creates a mask that represents which bits to keep/shift
        asm.emit_i::<VirtualShiftRightBitmask>(*v_bitmask, self.operands.rs2, 0);

        // Step 2: Apply arithmetic right shift using the bitmask
        // Preserves sign bit during the shift operation
        asm.emit_vshift_r::<VirtualSRA>(self.operands.rd, self.operands.rs1, *v_bitmask);

        asm.finalize()
    }
}
