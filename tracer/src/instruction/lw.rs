use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::ld::LD,
};

use super::andi::ANDI;
use super::format::format_load::FormatLoad;
use super::slli::SLLI;
use super::srl::SRL;
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::virtual_lw::VirtualLW;
use super::virtual_sign_extend_word::VirtualSignExtendWord;
use super::RAMRead;
use super::{addi::ADDI, Instruction};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use super::{Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LW,
    mask   = 0x0000707f,
    match  = 0x00002003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LW as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu
            .mmu
            .load_word(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((word, memory_read)) => {
                *ram_access = memory_read;
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Load word (32-bit) from aligned memory.
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

impl LW {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);
        asm.emit_i::<VirtualLW>(
            self.operands.rd,
            self.operands.rs1,
            self.operands.imm as u64,
        );
        asm.finalize()
    }

    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        asm.emit_align::<VirtualAssertWordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(self.operands.rd, *v0, -8i64 as u64);
        asm.emit_ld::<LD>(self.operands.rd, self.operands.rd, 0);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SRL>(self.operands.rd, self.operands.rd, *v0);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);

        asm.finalize()
    }
}
