use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::ld::LD;
use super::sd::SD;
use super::virtual_move::VirtualMove;
use super::Instruction;
use crate::utils::inline_helpers::InstrAssembler;

use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOADDD,
    mask   = 0xf800707f,
    match  = 0x0000302f,
    format = FormatR,
    ram    = ()
);

impl AMOADDD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOADDD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let add_value = cpu.x[self.operands.rs2 as usize];

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Add the values and store back to memory
        let new_value = original_value.wrapping_add(add_value) as u64;
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOADDD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic memory operation add (64-bit).
    ///
    /// AMOADD.D atomically loads a 64-bit value from memory, adds rs2 to it,
    /// stores the result back to memory, and returns the original value in rd.
    ///
    /// The implementation sequence:
    /// 1. Load the current value from memory address in rs1
    /// 2. Add rs2 to the loaded value
    /// 3. Store the result back to the same memory address
    /// 4. Move the original value to rd
    ///
    /// Note: In zkVM, atomicity is guaranteed by the execution model rather than
    /// hardware synchronization, as there's only one execution thread.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rs2 = allocator.allocate(); // holds the sum result
        let v_rd = allocator.allocate(); // holds the original memory value

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Load the current 64-bit value from memory
        asm.emit_ld::<LD>(*v_rd, self.operands.rs1, 0);

        // Step 2: Add rs2 to the loaded value
        asm.emit_r::<ADD>(*v_rs2, *v_rd, self.operands.rs2);

        // Step 3: Store the sum back to memory
        asm.emit_s::<SD>(self.operands.rs1, *v_rs2, 0);

        // Step 4: Move original value to destination register
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_rd, 0);

        asm.finalize()
    }
}
