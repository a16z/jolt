use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::addi::ADDI;
use super::format::format_r::FormatR;
use super::sw::SW;
use super::virtual_assert_eq::VirtualAssertEQ;
use super::{Cycle, Instruction, RAMWrite, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SCW,
    mask   = 0xf800707f,
    match  = 0x1800202f,
    format = FormatR,
    ram    = RAMWrite
);

impl SCW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SCW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let value = cpu.x[self.operands.rs2 as usize] as u32;

        // Check if reservation is set and matches the address
        if cpu.has_reservation(address) {
            // Store the word to memory
            let result = cpu.mmu.store_word(address, value);

            match result {
                Ok(memory_write) => {
                    *ram_access = memory_write;
                    // Clear the reservation
                    cpu.clear_reservation();
                    // Return 0 to indicate success
                    cpu.x[self.operands.rd as usize] = 0;
                }
                Err(_) => panic!("MMU store error"),
            }
        } else {
            // Reservation failed, return 1 to indicate failure
            cpu.x[self.operands.rd as usize] = 1;
        }
    }
}

impl RISCVTrace for SCW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // IMPORTANT: trace() and inline_sequence() must produce the SAME sequence.
        // We always assume success - the constraint system verifies correctness.
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }

        // Clear reservation in CPU state after SC
        cpu.clear_reservation();
    }

    /// SC.W: Store Conditional Word
    /// Always assumes success - the bytecode and trace must match.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_reservation = allocator.reservation_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // SC.W sequence - if reservation doesn't match, proof is invalid
        // 1. Assert reservation address matches rs1 (panics/invalidates proof if not)
        asm.emit_b::<VirtualAssertEQ>(v_reservation, self.operands.rs1, 0);
        // 2. Store the word (32-bit)
        asm.emit_s::<SW>(self.operands.rs1, self.operands.rs2, 0);
        // 3. Clear reservation (set v_reservation to 0)
        asm.emit_i::<ADDI>(v_reservation, 0, 0);
        // 4. Write 0 to rd to indicate success
        asm.emit_i::<ADDI>(self.operands.rd, 0, 0);

        asm.finalize()
    }
}
