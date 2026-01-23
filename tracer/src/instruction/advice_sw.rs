use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu, Xlen},
    instruction::{ori::ORI, srli::SRLI},
    utils::inline_helpers::InstrAssembler,
};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::ld::LD;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::virtual_advice_load::VirtualAdviceLoad;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;
use super::Instruction;
use super::RAMWrite;
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use super::{format::format_advice_s::FormatAdviceS, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AdviceSW,
    mask   = 0x0000707f,
    match  = 0x0000205b,  // opcode=0x5B (custom instruction), funct3=2
    format = FormatAdviceS,
    ram    = RAMWrite
);

impl AdviceSW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AdviceSW as RISCVInstruction>::RAMAccess) {
        // Read 4 bytes (word) from the advice tape
        let advice_value = advice_tape_read(4).expect("Failed to read from advice tape");

        // Store the advice value to memory at address rs1 + imm
        *ram_access = cpu
            .mmu
            .store_word(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                advice_value as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for AdviceSW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Store word (32-bit) from advice tape to aligned memory.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        match xlen {
            Xlen::Bit32 => self.inline_sequence_32(allocator),
            Xlen::Bit64 => self.inline_sequence_64(allocator),
        }
    }
}

impl AdviceSW {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_word = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);
        // Read 4 bytes from advice tape into v_word register
        asm.emit_j::<VirtualAdviceLoad>(*v_word, 4);
        // Store v_word to memory at rs1 + imm
        asm.emit_s::<VirtualSW>(self.operands.rs1, *v_word, self.operands.imm);
        asm.finalize()
    }

    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_address = allocator.allocate();
        let v_dword_address = allocator.allocate();
        let v_dword = allocator.allocate();
        let v_shift = allocator.allocate();
        let v_mask = allocator.allocate();
        let v_word = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        asm.emit_align::<VirtualAssertWordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_i::<ORI>(*v_mask, 0, -1i64 as u64);
        asm.emit_i::<SRLI>(*v_mask, *v_mask, 32);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        // Read word from advice tape into v_word register (imm=4 means 4 bytes)
        asm.emit_j::<VirtualAdviceLoad>(*v_word, 4);
        asm.emit_r::<SLL>(*v_word, *v_word, *v_shift);
        asm.emit_r::<XOR>(*v_word, *v_dword, *v_word);
        asm.emit_r::<AND>(*v_word, *v_word, *v_mask);
        asm.emit_r::<XOR>(*v_dword, *v_dword, *v_word);
        asm.emit_s::<SD>(*v_dword_address, *v_dword, 0);
        asm.finalize()
    }
}
