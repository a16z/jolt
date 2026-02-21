use serde::{Deserialize, Serialize};

use crate::instruction::addi::ADDI;
use crate::instruction::and::AND;
use crate::instruction::andi::ANDI;
use crate::instruction::ld::LD;
use crate::instruction::ori::ORI;
use crate::instruction::sd::SD;
use crate::instruction::sll::SLL;
use crate::instruction::slli::SLLI;
use crate::instruction::srl::SRL;
use crate::instruction::srli::SRLI;
use crate::instruction::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use crate::instruction::virtual_lw::VirtualLW;
use crate::instruction::virtual_sign_extend_word::VirtualSignExtendWord;
use crate::instruction::virtual_sw::VirtualSW;
use crate::instruction::xor::XOR;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_amo::FormatAMO, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOSWAPW,
    mask   = 0xf800707f,
    match  = 0x0800202f,
    format = FormatAMO,
    ram    = ()
);

impl AMOSWAPW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOSWAPW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let new_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Store the new value to memory
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.write_register(self.operands.rd as usize, original_value);
    }
}

impl RISCVTrace for AMOSWAPW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// AMOSWAP.W atomically swaps a 32-bit word in memory with rs2.
    ///
    /// This atomic memory operation (AMO) instruction atomically loads a 32-bit word from
    /// the memory address in rs1, stores the lower 32 bits of rs2 to that location, and
    /// returns the original value sign-extended in rd. This is an unconditional atomic exchange.
    ///
    /// Implementation differences:
    /// - RV32: Direct word swap using amo_pre32/post32 helpers
    ///   - Simple load-store sequence on word-aligned addresses
    ///   - Natural 32-bit operations throughout
    /// - RV64: Complex handling for 32-bit operations within 64-bit doublewords
    ///   - Uses amo_pre64 to extract the word from containing doubleword
    ///   - Preserves other 32 bits when storing back
    ///   - Uses amo_post64 to merge and sign-extend result
    ///
    /// AMOSWAP.W is commonly used for:
    /// - 32-bit mutex implementations
    /// - Atomic flag exchanges in embedded systems
    /// - Producer-consumer buffer management
    /// - Atomic state machine transitions
    /// - Legacy 32-bit atomic operations on 64-bit systems
    ///
    /// Return value handling:
    /// - The original 32-bit value is sign-extended to XLEN bits
    /// - Ensures consistent signed interpretation across architectures
    ///
    /// Memory ordering: Like all AMO operations, provides acquire-release
    /// semantics, though this is implicit in zkVM's single-threaded model.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        match xlen {
            Xlen::Bit32 => {
                let v_rd = allocator.allocate();
                let mut asm =
                    InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
                asm.emit_align::<super::virtual_assert_word_alignment::VirtualAssertWordAlignment>(
                    self.operands.rs1,
                    0,
                );
                asm.emit_i::<VirtualLW>(*v_rd, self.operands.rs1, 0);
                asm.emit_s::<VirtualSW>(self.operands.rs1, self.operands.rs2, 0);
                asm.emit_i::<ADDI>(self.operands.rd, *v_rd, 0);
                asm.finalize()
            }
            Xlen::Bit64 => {
                let v_mask = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_shift = allocator.allocate();
                let v_rd = allocator.allocate();
                let mut asm =
                    InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
                asm.emit_align::<VirtualAssertWordAlignment>(self.operands.rs1, 0);
                asm.emit_i::<ANDI>(*v_shift, self.operands.rs1, -8i64 as u64);
                asm.emit_ld::<LD>(*v_dword, *v_shift, 0);
                asm.emit_i::<SLLI>(*v_shift, self.operands.rs1, 3);
                asm.emit_r::<SRL>(*v_rd, *v_dword, *v_shift);
                asm.emit_i::<ORI>(*v_mask, 0, -1i64 as u64);
                asm.emit_i::<SRLI>(*v_mask, *v_mask, 32);
                asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
                asm.emit_r::<SLL>(*v_shift, self.operands.rs2, *v_shift);
                asm.emit_r::<XOR>(*v_shift, *v_dword, *v_shift);
                asm.emit_r::<AND>(*v_shift, *v_shift, *v_mask);
                asm.emit_r::<XOR>(*v_dword, *v_dword, *v_shift);
                asm.emit_i::<ANDI>(*v_mask, self.operands.rs1, -8i64 as u64);
                asm.emit_s::<SD>(*v_mask, *v_dword, 0);
                asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *v_rd, 0);
                asm.finalize()
            }
        }
    }
}
