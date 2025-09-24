use serde::{Deserialize, Serialize};

use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::and::AND;
use super::Instruction;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOANDW,
    mask   = 0xf800707f,
    match  = 0x6000202f,
    format = FormatR,
    ram    = ()
);

impl AMOANDW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOANDW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let and_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // AND the values and store back to memory
        let new_value = (original_value as u32) & and_value;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOANDW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic memory operation AND (32-bit).
    ///
    /// AMOAND.W atomically loads a 32-bit word from memory, performs bitwise AND
    /// with the lower 32 bits of rs2, stores the result back to memory, and
    /// returns the original value sign-extended in rd.
    ///
    /// The implementation differs between 32-bit and 64-bit systems:
    /// - On RV32: Direct word operations using amo_pre32/post32 helpers
    /// - On RV64: More complex due to potential misalignment within 64-bit words
    ///
    /// For RV64, the sequence handles:
    /// 1. Address alignment to 64-bit boundaries
    /// 2. Extracting the 32-bit word from the aligned 64-bit load
    /// 3. Performing the AND operation on the 32-bit value
    /// 4. Merging the result back into the 64-bit word for storage
    ///
    /// The amo_pre/post helpers encapsulate the complexity of:
    /// - Memory alignment and word extraction
    /// - Sign extension of the result
    /// - Proper masking and shifting for sub-word operations
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rd = allocator.allocate(); // holds original 32-bit value
        let v_rs2 = allocator.allocate(); // holds AND result

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        match xlen {
            Xlen::Bit32 => {
                // Simple case: direct 32-bit word operations
                amo_pre32(&mut asm, self.operands.rs1, *v_rd); // Load word from memory
                asm.emit_r::<AND>(*v_rs2, *v_rd, self.operands.rs2); // AND with rs2
                amo_post32(&mut asm, *v_rs2, self.operands.rs1, self.operands.rd, *v_rd);
                // Store and return
            }
            Xlen::Bit64 => {
                // Complex case: handle 32-bit operations within 64-bit system
                let v_mask = allocator.allocate(); // mask for word extraction
                let v_dword_address = allocator.allocate(); // aligned 64-bit address
                let v_dword = allocator.allocate(); // 64-bit loaded value
                let v_word = allocator.allocate(); // extracted 32-bit word
                let v_shift = allocator.allocate(); // shift amount for alignment

                // Pre-operation: load and extract 32-bit word from 64-bit memory
                amo_pre64(
                    &mut asm,
                    self.operands.rs1,
                    *v_rd,
                    *v_dword_address,
                    *v_dword,
                    *v_shift,
                );

                // Perform AND on the 32-bit value
                asm.emit_r::<AND>(*v_rs2, *v_rd, self.operands.rs2);

                // Post-operation: merge result back and store to memory
                amo_post64(
                    &mut asm,
                    *v_rs2,
                    *v_dword_address,
                    *v_dword,
                    *v_shift,
                    *v_mask,
                    *v_word,
                    self.operands.rd,
                    *v_rd,
                );
            }
        }

        asm.finalize()
    }
}
