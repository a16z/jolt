use super::andi::ANDI;
use super::format::format_load::FormatLoad;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srli::SRLI;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::xori::XORI;
use super::{addi::ADDI, Instruction};
use super::{Cycle, RISCVInstruction, RISCVTrace};
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = LWU,
    mask   = 0x0000707f,
    match  = 0x00006003,
    format = FormatLoad,
    ram    = super::RAMRead
);

impl LWU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LWU as RISCVInstruction>::RAMAccess) {
        // The LWU instruction, on the other hand, zero-extends the 32-bit value from memory for
        // RV64I.
        let address = cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64;
        let value = cpu.mmu.load_word(address);

        cpu.x[self.operands.rd as usize] = match value {
            Ok((word, memory_read)) => {
                *ram_access = memory_read;
                // Zero extension for unsigned word load
                word as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LWU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
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
    ) -> Vec<Instruction> {
        match xlen {
            Xlen::Bit32 => panic!("LWU is invalid in 32b mode"),
            Xlen::Bit64 => self.inline_sequence_64(allocator, xlen),
        }
    }
}

impl LWU {
    fn inline_sequence_64(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_address = allocator.allocate();
        let v_dword_address = allocator.allocate();
        let v_dword = allocator.allocate();
        let v_shift = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_halign::<VirtualAssertWordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<XORI>(*v_shift, *v_address, 4);
        asm.emit_i::<SLLI>(*v_shift, *v_shift, 3);
        asm.emit_r::<SLL>(self.operands.rd, *v_dword, *v_shift);
        asm.emit_i::<SRLI>(self.operands.rd, self.operands.rd, 32);
        asm.finalize()
    }
}
