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
use super::RAMWrite;
use super::RV32IMInstruction;
use crate::utils::virtual_registers::allocate_virtual_register;

use super::{format::format_s::FormatS, RISCVInstruction, RISCVTrace, RV32IMCycle};

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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        match xlen {
            Xlen::Bit32 => self.inline_sequence_32(),
            Xlen::Bit64 => self.inline_sequence_64(),
        }
    }
}

impl SW {
    fn inline_sequence_32(&self) -> Vec<RV32IMInstruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32);
        asm.emit_s::<VirtualSW>(self.operands.rs1, self.operands.rs2, self.operands.imm);
        asm.finalize()
    }

    fn inline_sequence_64(&self) -> Vec<RV32IMInstruction> {
        let v_address = allocate_virtual_register();
        let v_dword_address = allocate_virtual_register();
        let v_dword = allocate_virtual_register();
        let v_shift = allocate_virtual_register();
        let v_mask = allocate_virtual_register();
        let v_word = allocate_virtual_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64);

        asm.emit_halign::<VirtualAssertWordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_i::<ORI>(*v_mask, 0, -1i64 as u64);
        asm.emit_i::<SRLI>(*v_mask, *v_mask, 32);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        asm.emit_r::<SLL>(*v_word, self.operands.rs2, *v_shift);
        asm.emit_r::<XOR>(*v_word, *v_dword, *v_word);
        asm.emit_r::<AND>(*v_word, *v_word, *v_mask);
        asm.emit_r::<XOR>(*v_dword, *v_dword, *v_word);
        asm.emit_s::<SD>(*v_dword_address, *v_dword, 0);
        asm.finalize()
    }
}
