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

        cpu.write_register(
            self.operands.rd as usize,
            match value {
                Ok((word, memory_read)) => {
                    *ram_access = memory_read;
                    // Zero extension for unsigned word load
                    word as i64
                }
                Err(_) => panic!("MMU load error"),
            },
        );
    }
}

impl RISCVTrace for LWU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Load unsigned word (32-bit) with zero extension to 64-bit.
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
        let v0 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_align::<VirtualAssertWordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(self.operands.rd, *v0, -8i64 as u64);
        asm.emit_ld::<LD>(self.operands.rd, self.operands.rd, 0);
        asm.emit_i::<XORI>(*v0, *v0, 4);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SLL>(self.operands.rd, self.operands.rd, *v0);
        asm.emit_i::<SRLI>(self.operands.rd, self.operands.rd, 32);

        asm.finalize()
    }
}
