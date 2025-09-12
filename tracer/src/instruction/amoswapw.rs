use serde::{Deserialize, Serialize};

use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = AMOSWAPW,
    mask   = 0xf800707f,
    match  = 0x0800202f,
    format = FormatR,
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
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOSWAPW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => {
                let v_rd = allocator.allocate();
                let mut asm =
                    InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
                asm.emit_halign::<super::virtual_assert_word_alignment::VirtualAssertWordAlignment>(self.operands.rs1, 0);
                asm.emit_i::<super::virtual_lw::VirtualLW>(*v_rd, self.operands.rs1, 0);
                asm.emit_s::<super::virtual_sw::VirtualSW>(self.operands.rs1, self.operands.rs2, 0);
                asm.emit_i::<super::virtual_move::VirtualMove>(self.operands.rd, *v_rd, 0);
                asm.finalize()
            }
            Xlen::Bit64 => {
                let v_mask = allocator.allocate();
                let v_dword_address = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_word = allocator.allocate();
                let v_shift = allocator.allocate();
                let v_rd = allocator.allocate();
                let mut asm =
                    InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
                asm.emit_halign::<super::virtual_assert_word_alignment::VirtualAssertWordAlignment>(self.operands.rs1, 0);
                asm.emit_i::<super::andi::ANDI>(*v_dword_address, self.operands.rs1, -8i64 as u64);
                asm.emit_ld::<super::ld::LD>(*v_dword, *v_dword_address, 0);
                asm.emit_i::<super::slli::SLLI>(*v_shift, self.operands.rs1, 3);
                asm.emit_r::<super::srl::SRL>(*v_rd, *v_dword, *v_shift);
                asm.emit_i::<super::ori::ORI>(*v_mask, 0, -1i64 as u64);
                asm.emit_i::<super::srli::SRLI>(*v_mask, *v_mask, 32);
                asm.emit_r::<super::sll::SLL>(*v_mask, *v_mask, *v_shift);
                asm.emit_r::<super::sll::SLL>(*v_word, self.operands.rs2, *v_shift);
                asm.emit_r::<super::xor::XOR>(*v_word, *v_dword, *v_word);
                asm.emit_r::<super::and::AND>(*v_word, *v_word, *v_mask);
                asm.emit_r::<super::xor::XOR>(*v_dword, *v_dword, *v_word);
                asm.emit_s::<super::sd::SD>(*v_dword_address, *v_dword, 0);
                asm.emit_i::<super::virtual_sign_extend_word::VirtualSignExtendWord>(
                    self.operands.rd,
                    *v_rd,
                    0,
                );
                asm.finalize()
            }
        }
    }
}
