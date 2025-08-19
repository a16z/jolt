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
use super::virtual_sign_extend::VirtualSignExtend;
use super::RAMRead;
use super::{addi::ADDI, RV32IMInstruction};
use crate::utils::virtual_registers::allocate_virtual_register;

use super::{RISCVInstruction, RISCVTrace, RV32IMCycle};

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

impl LW {
    fn inline_sequence_32(&self) -> Vec<RV32IMInstruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32);
        asm.emit_i::<VirtualLW>(
            self.operands.rd,
            self.operands.rs1,
            self.operands.imm as u64,
        );
        asm.finalize()
    }

    fn inline_sequence_64(&self) -> Vec<RV32IMInstruction> {
        // Virtual registers used in sequence
        let v_address = allocate_virtual_register();
        let v_dword_address = allocate_virtual_register();
        let v_dword = allocate_virtual_register();
        let v_shift = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64);
        asm.emit_halign::<VirtualAssertWordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_r::<SRL>(self.operands.rd, *v_dword, *v_shift);
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
