use serde::{Deserialize, Serialize};

use super::{
    addi::ADDI,
    andi::ANDI,
    format::format_load::FormatLoad,
    ld::LD,
    sll::SLL,
    slli::SLLI,
    srli::SRLI,
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment,
    virtual_lw::VirtualLW,
    xori::XORI,
    Cycle,
    Instruction,
    RAMRead,
    RISCVInstruction,
    RISCVTrace,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};

declare_riscv_instr!(
    name   = LHU,
    mask   = 0x0000707f,
    match  = 0x00005003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LHU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LHU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu
            .mmu
            .load_halfword(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((halfword, memory_read)) => {
                *ram_access = memory_read;
                halfword as i64
            }
            Err(_) => panic!("MMU load error"),
        };
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
        // Virtual registers used in sequence
        let v_address = allocator.allocate();
        let v_word_address = allocator.allocate();
        let v_word = allocator.allocate();
        let v_shift = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);
        asm.emit_halign::<VirtualAssertHalfwordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_word_address, *v_address, -4i64 as u64);
        asm.emit_i::<VirtualLW>(*v_word, *v_word_address, 0);
        asm.emit_i::<XORI>(*v_shift, *v_address, 2);
        asm.emit_i::<SLLI>(*v_shift, *v_shift, 3);
        asm.emit_r::<SLL>(self.operands.rd, *v_word, *v_shift);
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
        // Virtual registers used in sequence
        let v_address = allocator.allocate();
        let v_dword_address = allocator.allocate();
        let v_dword = allocator.allocate();
        let v_shift = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);
        asm.emit_halign::<VirtualAssertHalfwordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<XORI>(*v_shift, *v_address, 6);
        asm.emit_i::<SLLI>(*v_shift, *v_shift, 3);
        asm.emit_r::<SLL>(self.operands.rd, *v_dword, *v_shift);
        asm.emit_i::<SRLI>(self.operands.rd, self.operands.rd, 48);
        asm.finalize()
    }
}
