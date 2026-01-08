use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::Instruction;
use crate::emulator::cpu::Cpu;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{declare_riscv_instr, emulator::cpu::Xlen};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOADDW,
    mask   = 0xf800707f,
    match  = 0x0000202f,
    format = FormatAMO,
    ram    = ()
);

impl AMOADDW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOADDW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let add_value = cpu.x[self.operands.rs2 as usize] as i32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Add the values and store back to memory
        let new_value = (original_value as i32).wrapping_add(add_value) as u32;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOADDW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic memory add operation (32-bit).
    ///
    /// AMOADD.W atomically loads a 32-bit word from memory, adds the lower 32 bits
    /// of rs2 to it, stores the result back to memory, and returns the original
    /// value sign-extended in rd.
    ///
    /// The implementation differs between RV32 and RV64:
    /// - On RV32: Direct word operations with amo_pre32/post32 helpers
    /// - On RV64: Complex handling for 32-bit operations in 64-bit system
    ///
    /// For RV64, the sequence must:
    /// 1. Align address to 64-bit boundaries
    /// 2. Extract the 32-bit word from the aligned 64-bit load
    /// 3. Perform 32-bit addition with proper overflow wrapping
    /// 4. Merge the result back into the 64-bit word for storage
    /// 5. Sign-extend the original 32-bit value to 64 bits for rd
    ///
    /// The amo_pre/post helpers handle the memory alignment complexity,
    /// ensuring atomic semantics even though zkVM execution is single-threaded.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rd = allocator.allocate();
        let v_rs2 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        match xlen {
            Xlen::Bit32 => {
                amo_pre32(&mut asm, self.operands.rs1, *v_rd);
                asm.emit_r::<ADD>(*v_rs2, *v_rd, self.operands.rs2);
                amo_post32(&mut asm, *v_rs2, self.operands.rs1, self.operands.rd, *v_rd);
            }
            Xlen::Bit64 => {
                let v_mask = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_shift = allocator.allocate();

                amo_pre64(&mut asm, self.operands.rs1, *v_rd, *v_dword, *v_shift);
                asm.emit_r::<ADD>(*v_rs2, *v_rd, self.operands.rs2);
                amo_post64(
                    &mut asm,
                    self.operands.rs1,
                    *v_rs2,
                    *v_dword,
                    *v_shift,
                    *v_mask,
                    self.operands.rd,
                    *v_rd,
                );
            }
        }

        asm.finalize()
    }
}
