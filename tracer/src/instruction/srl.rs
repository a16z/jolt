use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_srl::VirtualSRL, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SRL,
    mask   = 0xfe00707f,
    match  = 0x00005033,
    format = FormatR,
    ram    = ()
);

impl SRL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRL as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                .wrapping_shr(cpu.x[self.operands.rs2 as usize] as u32 & mask) as i64,
        );
    }
}

impl RISCVTrace for SRL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence for logical right shift operation.
    ///
    /// SRL (Shift Right Logical) shifts rs1 right by the shift amount in rs2,
    /// filling the vacated bits with zeros (unlike SRA which fills with sign bit).
    ///
    /// The implementation uses a bitmask approach where:
    /// 1. A bitmask is computed based on the shift amount
    /// 2. Virtual SRL instruction applies the shift using the bitmask
    ///
    /// This approach allows the shift to be verified in a way that's compatible
    /// with zkVM's constraint system, avoiding direct bit manipulation.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_bitmask = allocator.allocate(); // holds the shift bitmask

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Generate bitmask for right shift
        // Creates a mask that represents which bits to keep after shifting
        asm.emit_i::<VirtualShiftRightBitmask>(*v_bitmask, self.operands.rs2, 0);

        // Step 2: Apply logical right shift using the bitmask
        // Fills high bits with zeros during the shift operation
        asm.emit_vshift_r::<VirtualSRL>(self.operands.rd, self.operands.rs1, *v_bitmask);

        asm.finalize()
    }
}
