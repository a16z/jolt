use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::mul::MUL;
use super::sltu::SLTU;
use super::sub::SUB;
use super::virtual_zero_extend_word::VirtualZeroExtendWord;
use super::Instruction;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOMINUW,
    mask   = 0xf800707f,
    match  = 0xe000202f,
    format = FormatAMO,
    ram    = ()
);

impl AMOMINUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMINUW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the minimum (unsigned comparison) and store back to memory
        let new_value = if (original_value as u32) <= compare_value {
            original_value as u32
        } else {
            compare_value
        };
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value (sign extended)
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOMINUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates inline sequence for atomic minimum operation (unsigned 32-bit).
    ///
    /// AMOMINU.W atomically loads a 32-bit word from memory, computes the minimum
    /// of that value and the lower 32 bits of rs2 (treating both as unsigned),
    /// stores the minimum back to memory, and returns the original value
    /// sign-extended in rd.
    ///
    /// The implementation uses a branchless minimum selection:
    /// 1. Load and prepare operands with proper zero-extension for comparison
    /// 2. Use the approach from AMOMINU.D to select the minimum without branching
    /// 3. Store result and return original value sign-extended
    ///
    /// On RV64, additional complexity arises from:
    /// - Need to zero-extend both operands for correct unsigned comparison
    /// - Use amo_pre64/post64 helpers for word alignment within doublewords
    /// - Proper sign extension of the original value for rd
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
                asm.emit_r::<SLTU>(*v0, self.operands.rs2, *v_rd);
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
                // Zero-extend rs2 into v_rs2
                asm.emit_i::<VirtualZeroExtendWord>(*v_rs2, self.operands.rs2, 0);
                // Zero-extend v_rd into v0
                asm.emit_i::<VirtualZeroExtendWord>(*v0, *v_rd, 0);
                // Put min(v_rs2, rs2) in v_rs2
                asm.emit_r::<SLTU>(*v1, *v_rs2, *v_rd);
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
