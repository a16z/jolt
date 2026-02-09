use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::{Cpu, Xlen};

use super::andi::ANDI;
use super::format::format_load::FormatLoad;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srli::SRLI;
use super::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use super::virtual_lw::VirtualLW;
use super::xori::XORI;
use super::RAMRead;
use super::{addi::ADDI, Instruction};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use super::{Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LHU,
    mask   = 0x0000707f,
    match  = 0x00005003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LHU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LHU as RISCVInstruction>::RAMAccess) {
        let value = match cpu.mmu.load_halfword(
            cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
        ) {
            Ok((halfword, memory_read)) => {
                *ram_access = memory_read;
                halfword as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, value);
    }
}

impl RISCVTrace for LHU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Load unsigned halfword without sign extension.
    ///
    /// LHU loads a 16-bit value from memory at address rs1+imm and zero-extends
    /// it to the full register width. Since zkVM uses word-aligned memory:
    /// 1. Assert halfword alignment of the source address
    /// 2. Load the aligned word/doubleword containing the halfword
    /// 3. Shift halfword to the high bits
    /// 4. Logical right shift to extract and zero-extend
    ///
    /// Differs from LH only in using logical (not arithmetic) right shift.
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

impl LHU {
    /// 32-bit implementation of load unsigned halfword.
    ///
    /// Algorithm:
    /// 1. Assert halfword alignment
    /// 2. Align address to 4-byte boundary
    /// 3. Load word containing the halfword
    /// 4. XOR with 2 to get opposite halfword position
    /// 5. Shift halfword to bits [31:16]
    /// 6. Logical right shift by 16 to zero-extend
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
        asm.emit_i::<SRLI>(self.operands.rd, self.operands.rd, 16);

        asm.finalize()
    }

    /// 64-bit implementation of load unsigned halfword.
    ///
    /// Similar to 32-bit but handles 4 halfword positions:
    /// 1. XOR with 6 for position calculation
    /// 2. Shift halfword to bits [63:48]
    /// 3. Logical right shift by 48 to zero-extend
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
        asm.emit_i::<SRLI>(self.operands.rd, self.operands.rd, 48);

        asm.finalize()
    }
}
