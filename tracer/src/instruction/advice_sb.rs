use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu, Xlen},
};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::ld::LD;
use super::lui::LUI;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::virtual_advice_load::VirtualAdviceLoad;
use super::virtual_lw::VirtualLW;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;
use super::{Instruction, RAMWrite};

use super::{format::format_advice_s::FormatAdviceS, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AdviceSB,
    mask   = 0,
    match  = 0,
    format = FormatAdviceS,
    ram    = RAMWrite
);

impl AdviceSB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <AdviceSB as RISCVInstruction>::RAMAccess) {
        // Read 1 byte from the advice tape
        let advice_value = advice_tape_read(1).expect("Failed to read from advice tape");

        // Store the advice value to memory at address rs1 + imm
        *ram_access = cpu
            .mmu
            .store(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                advice_value as u8,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for AdviceSB {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Store byte from advice tape to memory using word-aligned access.
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

impl AdviceSB {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_address = allocator.allocate();
        let v_word_address = allocator.allocate();
        let v_word = allocator.allocate();
        let v_shift = allocator.allocate();
        let v_mask = allocator.allocate();
        let v_byte = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_word_address, *v_address, -4i64 as u64);
        asm.emit_i::<VirtualLW>(*v_word, *v_word_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_u::<LUI>(*v_mask, 0xff);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        // Read byte from advice tape into v_byte register (imm=1 means 1 byte)
        asm.emit_i::<VirtualAdviceLoad>(*v_byte, 0, 1);
        asm.emit_r::<SLL>(*v_byte, *v_byte, *v_shift);
        asm.emit_r::<XOR>(*v_byte, *v_word, *v_byte);
        asm.emit_r::<AND>(*v_byte, *v_byte, *v_mask);
        asm.emit_r::<XOR>(*v_word, *v_word, *v_byte);
        asm.emit_s::<VirtualSW>(*v_word_address, *v_word, 0);
        asm.finalize()
    }

    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_address = allocator.allocate();
        let v_dword_address = allocator.allocate();
        let v_dword = allocator.allocate();
        let v_shift = allocator.allocate();
        let v_mask = allocator.allocate();
        let v_byte = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_u::<LUI>(*v_mask, 0xff);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        // Read byte from advice tape into v_byte register (imm=1 means 1 byte)
        asm.emit_i::<VirtualAdviceLoad>(*v_byte, 0, 1);
        asm.emit_r::<SLL>(*v_byte, *v_byte, *v_shift);
        asm.emit_r::<XOR>(*v_byte, *v_dword, *v_byte);
        asm.emit_r::<AND>(*v_byte, *v_byte, *v_mask);
        asm.emit_r::<XOR>(*v_dword, *v_dword, *v_byte);
        asm.emit_s::<SD>(*v_dword_address, *v_dword, 0);
        asm.finalize()
    }
}
