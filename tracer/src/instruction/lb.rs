use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::andi::ANDI;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srai::SRAI;
use super::virtual_lw::VirtualLW;
use super::xori::XORI;
use super::{Instruction, RAMRead};

use super::{format::format_load::FormatLoad, Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LB,
    mask   = 0x0000707f,
    match  = 0x00000003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LB {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LB as RISCVInstruction>::RAMAccess) {
        let value = match cpu
            .mmu
            .load(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((byte, memory_read)) => {
                *ram_access = memory_read;
                byte as i8 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, value);
    }
}

impl RISCVTrace for LB {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// LB loads a byte from memory and sign-extends it to XLEN bits.
    ///
    /// The zkVM constraint system requires word-aligned memory access, so loading
    /// individual bytes requires extracting them from the containing word/doubleword.
    /// The byte is then sign-extended to fill the destination register.
    ///
    /// Different implementations for RV32 and RV64 due to different word sizes.
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

impl LB {
    /// RV32 implementation: Extracts and sign-extends a byte from a 32-bit word.
    ///
    /// Steps:
    /// 1. Calculate byte address
    /// 2. Align to word boundary
    /// 3. Load the containing word
    /// 4. Calculate shift amount based on byte position (XOR with 3 for little-endian)
    /// 5. Shift byte to MSB position
    /// 6. Arithmetic right shift by 24 to sign-extend to 32 bits
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v1, *v0, -4i64 as u64);
        asm.emit_i::<VirtualLW>(*v1, *v1, 0);
        asm.emit_i::<XORI>(*v0, *v0, 3);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SLL>(*v1, *v1, *v0);
        asm.emit_i::<SRAI>(self.operands.rd, *v1, 24);

        asm.finalize()
    }

    /// RV64 implementation: Extracts and sign-extends a byte from a 64-bit doubleword.
    ///
    /// Steps:
    /// 1. Calculate byte address
    /// 2. Align to doubleword boundary
    /// 3. Load the containing doubleword
    /// 4. Calculate shift amount based on byte position (XOR with 7 for little-endian)
    /// 5. Shift byte to MSB position
    /// 6. Arithmetic right shift by 56 to sign-extend to 64 bits
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v1, *v0, -8i64 as u64);
        asm.emit_ld::<LD>(*v1, *v1, 0);
        asm.emit_i::<XORI>(*v0, *v0, 7);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SLL>(*v1, *v1, *v0);
        asm.emit_i::<SRAI>(self.operands.rd, *v1, 56);

        asm.finalize()
    }
}
