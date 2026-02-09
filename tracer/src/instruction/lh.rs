use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::andi::ANDI;
use super::format::format_load::FormatLoad;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srai::SRAI;
use super::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use super::virtual_lw::VirtualLW;
use super::xori::XORI;
use super::RAMRead;
use super::{addi::ADDI, Instruction};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use super::{Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LH,
    mask   = 0x0000707f,
    match  = 0x00001003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LH {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LH as RISCVInstruction>::RAMAccess) {
        let value = match cpu.mmu.load_halfword(
            cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
        ) {
            Ok((halfword, memory_read)) => {
                *ram_access = memory_read;
                halfword as i16 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, value);
    }
}

impl RISCVTrace for LH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Load halfword with sign extension from aligned memory.
    ///
    /// LH loads a 16-bit value from memory at address rs1+imm and sign-extends
    /// it to the full register width. Since zkVM uses word-aligned memory,
    /// this requires:
    /// 1. Assert halfword alignment of the source address
    /// 2. Load the aligned word/doubleword containing the halfword
    /// 3. Shift the halfword to the high bits
    /// 4. Arithmetic right shift to sign-extend to full width
    ///
    /// The clever trick here is to shift the halfword to the MSB position
    /// then use arithmetic right shift to simultaneously extract and sign-extend.
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

impl LH {
    /// 32-bit implementation of load halfword with sign extension.
    ///
    /// Algorithm:
    /// 1. Assert halfword alignment (address must be multiple of 2)
    /// 2. Align address to 4-byte boundary to get word address
    /// 3. Load the aligned word containing the halfword
    /// 4. XOR address with 2 to get opposite halfword position
    /// 5. Calculate shift amount (position * 8 bits)
    /// 6. Shift halfword to bits [31:16]
    /// 7. Arithmetic right shift by 16 to sign-extend to 32 bits
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        asm.emit_align::<VirtualAssertHalfwordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(self.operands.rd, *v0, -4i64 as u64);
        asm.emit_i::<VirtualLW>(self.operands.rd, self.operands.rd, 0);
        asm.emit_i::<XORI>(*v0, *v0, 2);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SLL>(self.operands.rd, self.operands.rd, *v0);
        asm.emit_i::<SRAI>(self.operands.rd, self.operands.rd, 16);

        asm.finalize()
    }

    /// 64-bit implementation of load halfword with sign extension.
    ///
    /// Similar to 32-bit but works with doublewords:
    /// 1. Assert halfword alignment
    /// 2. Align to 8-byte boundary
    /// 3. XOR address with 6 to handle 4 possible halfword positions
    /// 4. Shift halfword to bits [63:48]
    /// 5. Arithmetic right shift by 48 to sign-extend to 64 bits
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        asm.emit_align::<VirtualAssertHalfwordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(self.operands.rd, *v0, -8i64 as u64);
        asm.emit_ld::<LD>(self.operands.rd, self.operands.rd, 0);
        asm.emit_i::<XORI>(*v0, *v0, 6);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SLL>(self.operands.rd, self.operands.rd, *v0);
        asm.emit_i::<SRAI>(self.operands.rd, self.operands.rd, 48);

        asm.finalize()
    }
}
