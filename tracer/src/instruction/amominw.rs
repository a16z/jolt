use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::mul::MUL;
use super::slt::SLT;
use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::xori::XORI;
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
    match  = 0x8000202f,
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
    /// Uses branchless minimum selection with signed comparison:
    /// 1. Load word and prepare operands with sign extension (on RV64)
    /// 2. Use SLT with rs2 < current to determine which is smaller
    /// 3. Use multiplication to select minimum without branches
    /// 4. Store result and return original value sign-extended
    ///
    /// On RV64, requires sign-extending both operands for correct comparison.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rd = allocator.allocate();
        let v_rs2 = allocator.allocate();
        let v_sel_rs2 = allocator.allocate();
        let v_sel_rd = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        match xlen {
            Xlen::Bit32 => {
                amo_pre32(&mut asm, self.operands.rs1, *v_rd);
                asm.emit_r::<SLT>(*v_sel_rs2, self.operands.rs2, *v_rd);
                asm.emit_i::<XORI>(*v_sel_rd, *v_sel_rs2, 1);
                asm.emit_r::<MUL>(*v_rs2, *v_sel_rs2, self.operands.rs2);
                asm.emit_r::<MUL>(*v_sel_rd, *v_sel_rd, *v_rd);
                asm.emit_r::<ADD>(*v_rs2, *v_sel_rd, *v_rs2);
                amo_post32(&mut asm, *v_rs2, self.operands.rs1, self.operands.rd, *v_rd);
            }
            Xlen::Bit64 => {
                let v_mask = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_shift = allocator.allocate();

                amo_pre64(&mut asm, self.operands.rs1, *v_rd, *v_dword, *v_shift);
                // Sign-extend rs2 into v_rs2
                asm.emit_i::<VirtualSignExtendWord>(*v_rs2, self.operands.rs2, 0);
                // Sign-extend v_rd in place into v_sel_rd (temporarily)
                asm.emit_i::<VirtualSignExtendWord>(*v_sel_rd, *v_rd, 0);
                // Compare: v_rs2 < v_sel_rd (sign-extended v_rd)
                asm.emit_r::<SLT>(*v_sel_rs2, *v_rs2, *v_sel_rd);
                // Invert selector to get selector for v_rd
                asm.emit_i::<XORI>(*v_sel_rd, *v_sel_rs2, 1);
                // Select minimum using multiplication
                asm.emit_r::<MUL>(*v_rs2, *v_sel_rs2, self.operands.rs2);
                asm.emit_r::<MUL>(*v_sel_rd, *v_sel_rd, *v_rd);
                asm.emit_r::<ADD>(*v_rs2, *v_sel_rd, *v_rs2);
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
