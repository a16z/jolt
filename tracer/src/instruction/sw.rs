use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
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
use super::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;
use super::Instruction;
use super::RAMWrite;
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use super::{format::format_s::FormatS, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SW,
    mask   = 0x0000707f,
    match  = 0x00002023,
    format = FormatS,
    ram    = RAMWrite
);

impl SW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SW as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_word(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Store word (32-bit) to aligned memory.
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

impl SW {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);
        asm.emit_s::<VirtualSW>(self.operands.rs1, self.operands.rs2, self.operands.imm);
        asm.finalize()
    }

    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();
        let v2 = allocator.allocate();
        let v3 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        asm.emit_align::<VirtualAssertWordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v1, *v0, -8i64 as u64);
        asm.emit_ld::<LD>(*v2, *v1, 0);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_i::<ORI>(*v3, 0, -1i64 as u64);
        asm.emit_i::<SRLI>(*v3, *v3, 32);
        asm.emit_r::<SLL>(*v3, *v3, *v0);
        asm.emit_r::<SLL>(*v0, self.operands.rs2, *v0);
        asm.emit_r::<XOR>(*v0, *v2, *v0);
        asm.emit_r::<AND>(*v0, *v0, *v3);
        asm.emit_r::<XOR>(*v2, *v2, *v0);
        asm.emit_s::<SD>(*v1, *v2, 0);

        asm.finalize()
    }
}
