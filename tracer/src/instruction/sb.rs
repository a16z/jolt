use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::and::AND;
use super::andi::ANDI;
use super::ld::LD;
use super::lui::LUI;
use super::sd::SD;
use super::sll::SLL;
use super::slli::SLLI;
use super::virtual_lw::VirtualLW;
use super::virtual_sw::VirtualSW;
use super::xor::XOR;
use super::{RAMWrite, RV32IMInstruction};

use super::{format::format_s::FormatS, RISCVInstruction, RISCVTrace, RV32IMCycle};

declare_riscv_instr!(
    name   = SB,
    mask   = 0x0000707f,
    match  = 0x00000023,
    format = FormatS,
    ram    = RAMWrite
);

impl SB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SB as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u8,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SB {
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

impl SB {
    fn inline_sequence_32(&self) -> Vec<RV32IMInstruction> {
        let v_address = allocate_virtual_register();
        let v_word_address = allocate_virtual_register();
        let v_word = allocate_virtual_register();
        let v_shift = allocate_virtual_register();
        let v_mask = allocate_virtual_register();
        let v_byte = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_word_address, *v_address, -4i64 as u64);
        asm.emit_i::<VirtualLW>(*v_word, *v_word_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_u::<LUI>(*v_mask, 0xff);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        asm.emit_r::<SLL>(*v_byte, self.operands.rs2, *v_shift);
        asm.emit_r::<XOR>(*v_byte, *v_word, *v_byte);
        asm.emit_r::<AND>(*v_byte, *v_byte, *v_mask);
        asm.emit_r::<XOR>(*v_word, *v_word, *v_byte);
        asm.emit_s::<VirtualSW>(*v_word_address, *v_word, 0);
        asm.finalize()
    }

    fn inline_sequence_64(&self) -> Vec<RV32IMInstruction> {
        let v_address = allocate_virtual_register();
        let v_dword_address = allocate_virtual_register();
        let v_dword = allocate_virtual_register();
        let v_shift = allocate_virtual_register();
        let v_mask = allocate_virtual_register();
        let v_byte = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_u::<LUI>(*v_mask, 0xff);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        asm.emit_r::<SLL>(*v_byte, self.operands.rs2, *v_shift);
        asm.emit_r::<XOR>(*v_byte, *v_dword, *v_byte);
        asm.emit_r::<AND>(*v_byte, *v_byte, *v_mask);
        asm.emit_r::<XOR>(*v_dword, *v_dword, *v_byte);
        asm.emit_s::<SD>(*v_dword_address, *v_dword, 0);
        asm.finalize()
    }
}
