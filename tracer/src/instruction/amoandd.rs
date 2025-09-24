use serde::{Deserialize, Serialize};

use super::and::AND;
use super::ld::LD;
use super::sd::SD;
use super::virtual_move::VirtualMove;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOANDD,
    mask   = 0xf800707f,
    match  = 0x6000302f,
    format = FormatR,
    ram    = ()
);

impl AMOANDD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOANDD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let and_value = cpu.x[self.operands.rs2 as usize] as u64;

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // AND the values and store back to memory
        let new_value = (original_value as u64) & and_value;
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOANDD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic memory operation AND (64-bit).
    ///
    /// AMOAND.D atomically loads a 64-bit value from memory, performs bitwise AND
    /// with rs2, stores the result back to memory, and returns the original value in rd.
    ///
    /// The implementation sequence:
    /// 1. Load the current value from memory address in rs1
    /// 2. Perform bitwise AND with rs2
    /// 3. Store the result back to the same memory address
    /// 4. Move the original value to rd
    ///
    /// This is useful for atomically clearing bits in shared memory locations.
    /// In zkVM, atomicity is guaranteed by the single-threaded execution model.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rs2 = allocator.allocate(); // holds the AND result
        let v_rd = allocator.allocate(); // holds the original memory value

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Load the current 64-bit value from memory
        asm.emit_ld::<LD>(*v_rd, self.operands.rs1, 0);

        // Step 2: Perform bitwise AND with rs2
        asm.emit_r::<AND>(*v_rs2, *v_rd, self.operands.rs2);

        // Step 3: Store the AND result back to memory
        asm.emit_s::<SD>(self.operands.rs1, *v_rs2, 0);

        // Step 4: Move original value to destination register
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_rd, 0);

        asm.finalize()
    }
}
