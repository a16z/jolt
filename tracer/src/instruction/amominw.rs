use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::mul::MUL;
use super::slt::SLT;
use super::sub::SUB;
use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::Instruction;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOMINW,
    mask   = 0xf800707f,
    match  = 0xa000202f,
    format = FormatAMO,
    ram    = ()
);

impl AMOMINW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMINW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize] as i32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the minimum and store back to memory
        let new_value = if original_value as i32 <= compare_value {
            original_value as i32
        } else {
            compare_value
        };
        cpu.mmu
            .store_word(address, new_value as u32)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOMINW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic minimum operation (signed 32-bit).
    ///
    /// AMOMIN.W atomically loads a 32-bit word from memory, computes the minimum
    /// of that value and the lower 32 bits of rs2 (treating both as signed),
    /// stores the minimum back to memory, and returns the original value
    /// sign-extended in rd.
    ///
    /// Uses branchless minimum computation with signed comparison:
    /// 1. Load and prepare operands with proper sign-extension for comparison
    /// 2. Use the approach from AMOMIN.D to select the minimum without branching
    /// 3. Store result and return original value sign-extended
    ///
    /// On RV64, requires sign-extending both operands to 64 bits before
    /// comparison to ensure correct signed comparison semantics.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        match xlen {
            Xlen::Bit32 => {
                let v_rd = allocator.allocate();
                let v_rs2 = allocator.allocate();
                let v0 = allocator.allocate();
                let v1 = allocator.allocate();
                amo_pre32(&mut asm, self.operands.rs1, *v_rd);
                asm.emit_r::<SLT>(*v0, self.operands.rs2, *v_rd);
                asm.emit_r::<SUB>(*v1, self.operands.rs2, *v_rd);
                asm.emit_r::<MUL>(*v1, *v1, *v0);
                asm.emit_r::<ADD>(*v_rs2, *v1, *v_rd);
                amo_post32(&mut asm, *v_rs2, self.operands.rs1, self.operands.rd, *v_rd);
            }
            Xlen::Bit64 => {
                let v_rd = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_shift = allocator.allocate();

                amo_pre64(&mut asm, self.operands.rs1, *v_rd, *v_dword, *v_shift);

                let v_rs2 = allocator.allocate();
                let v0 = allocator.allocate();
                let v1 = allocator.allocate();
                let v2 = allocator.allocate();
                // Sign-extend rs2 into v_rs2
                asm.emit_i::<VirtualSignExtendWord>(*v_rs2, self.operands.rs2, 0);
                // Sign-extend v_rd into v0
                asm.emit_i::<VirtualSignExtendWord>(*v0, *v_rd, 0);
                // Put min(v_rs2, rs2) in v_rs2
                asm.emit_r::<SLT>(*v1, *v_rs2, *v_rd);
                asm.emit_r::<SUB>(*v2, *v_rs2, *v0);
                asm.emit_r::<MUL>(*v2, *v2, *v1);
                asm.emit_r::<ADD>(*v_rs2, *v2, *v0);
                // post processing, use v0 as v_mask in amo_post64
                amo_post64(
                    &mut asm,
                    self.operands.rs1,
                    *v_rs2,
                    *v_dword,
                    *v_shift,
                    *v0,
                    self.operands.rd,
                    *v_rd,
                );
            }
        }

        asm.finalize()
    }
}
