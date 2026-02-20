use serde::{Deserialize, Serialize};

use super::xor::XOR;
use super::Instruction;
use crate::instruction::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOXORW,
    mask   = 0xf800707f,
    match  = 0x2000202f,
    format = FormatAMO,
    ram    = ()
);

impl AMOXORW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOXORW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let xor_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // XOR the values and store back to memory
        let new_value = (original_value as u32) ^ xor_value;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.write_register(self.operands.rd as usize, original_value);
    }
}

impl RISCVTrace for AMOXORW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// AMOXOR.W atomically performs bitwise XOR on a 32-bit word in memory with rs2.
    ///
    /// This atomic memory operation (AMO) instruction atomically loads a 32-bit word from
    /// the memory address in rs1, performs a bitwise XOR with the lower 32 bits of rs2,
    /// stores the result back to memory, and returns the original value sign-extended in rd.
    ///
    /// Return value handling:
    /// - The original 32-bit value is sign-extended to XLEN bits
    /// - Consistent behavior across different XLEN implementations
    ///
    /// XOR properties in concurrent contexts:
    /// - Commutative: order of XOR operations doesn't matter
    /// - Self-canceling: applying same XOR twice restores original
    /// - Useful for temporary locks that auto-release on second application
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let effective_rd = if self.operands.rd == 0 {
            *allocator.allocate()
        } else {
            self.operands.rd
        };
        let v_rd = allocator.allocate();
        let v_rs2 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        match xlen {
            Xlen::Bit32 => {
                amo_pre32(&mut asm, self.operands.rs1, *v_rd);
                asm.emit_r::<XOR>(*v_rs2, *v_rd, self.operands.rs2);
                amo_post32(&mut asm, *v_rs2, self.operands.rs1, effective_rd, *v_rd);
            }
            Xlen::Bit64 => {
                let v_mask = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_shift = allocator.allocate();

                amo_pre64(&mut asm, self.operands.rs1, *v_rd, *v_dword, *v_shift);
                asm.emit_r::<XOR>(*v_rs2, *v_rd, self.operands.rs2);
                amo_post64(
                    &mut asm,
                    self.operands.rs1,
                    *v_rs2,
                    *v_dword,
                    *v_shift,
                    *v_mask,
                    effective_rd,
                    *v_rd,
                );
            }
        }

        asm.finalize()
    }
}
