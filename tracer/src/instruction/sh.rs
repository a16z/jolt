use serde::{Deserialize, Serialize};

use super::{
    addi::ADDI,
    and::AND,
    andi::ANDI,
    format::format_s::FormatS,
    ld::LD,
    lui::LUI,
    sd::SD,
    sll::SLL,
    slli::SLLI,
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment,
    virtual_lw::VirtualLW,
    virtual_sw::VirtualSW,
    xor::XOR,
    Cycle,
    Instruction,
    RAMWrite,
    RISCVInstruction,
    RISCVTrace,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};

declare_riscv_instr!(
    name   = SH,
    mask   = 0x0000707f,
    match  = 0x00001023,
    format = FormatS,
    ram    = RAMWrite
);

impl SH {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SH as RISCVInstruction>::RAMAccess) {
        *ram_access = cpu
            .mmu
            .store_halfword(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                cpu.x[self.operands.rs2 as usize] as u16,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for SH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Store halfword to memory using word-aligned access.
    ///
    /// SH stores the lower 16 bits of rs2 to memory at address rs1+imm.
    /// Since zkVM uses word-aligned memory, this requires:
    /// 1. Assert halfword alignment of the target address
    /// 2. Load the aligned word/doubleword containing the target halfword
    /// 3. Mask and replace the specific 16-bit halfword
    /// 4. Store the modified word/doubleword back to memory
    ///
    /// The implementation uses the XOR technique: (word ^ halfword) & mask ^ word
    /// This clears the original halfword bits and sets the new halfword value
    /// in a single sequence without branches.
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

impl SH {
    /// 32-bit implementation of store halfword.
    ///
    /// Algorithm:
    /// 1. Assert halfword alignment (address must be multiple of 2)
    /// 2. Calculate target address and align to 4-byte boundary
    /// 3. Load the aligned word containing the target halfword
    /// 4. Calculate shift amount based on halfword position (bit 1 of address)
    /// 5. Create 16-bit mask (0xFFFF) shifted to halfword position
    /// 6. Shift halfword value to correct position
    /// 7. Use XOR operations to replace the target halfword
    /// 8. Store the modified word back to memory
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        // Virtual registers used in sequence
        let v_address = allocator.allocate();
        let v_word_address = allocator.allocate();
        let v_word = allocator.allocate();
        let v_shift = allocator.allocate();
        let v_mask = allocator.allocate();
        let v_halfword = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);
        asm.emit_halign::<VirtualAssertHalfwordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_word_address, *v_address, -4i64 as u64);
        asm.emit_i::<VirtualLW>(*v_word, *v_word_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_u::<LUI>(*v_mask, 0xffff);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        asm.emit_r::<SLL>(*v_halfword, self.operands.rs2, *v_shift);
        asm.emit_r::<XOR>(*v_halfword, *v_word, *v_halfword);
        asm.emit_r::<AND>(*v_halfword, *v_halfword, *v_mask);
        asm.emit_r::<XOR>(*v_word, *v_word, *v_halfword);
        asm.emit_s::<VirtualSW>(*v_word_address, *v_word, 0);
        asm.finalize()
    }

    /// 64-bit implementation of store halfword.
    ///
    /// Similar to 32-bit version but operates on 64-bit doublewords.
    /// The halfword position is determined by bits 1-2 of the address
    /// (4 possible halfword positions within an 8-byte doubleword).
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        // Virtual registers used in sequence
        let v_address = allocator.allocate();
        let v_dword_address = allocator.allocate();
        let v_dword = allocator.allocate();
        let v_shift = allocator.allocate();
        let v_mask = allocator.allocate();
        let v_halfword = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);
        asm.emit_halign::<VirtualAssertHalfwordAlignment>(self.operands.rs1, self.operands.imm);
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);
        asm.emit_i::<SLLI>(*v_shift, *v_address, 3);
        asm.emit_u::<LUI>(*v_mask, 0xffff);
        asm.emit_r::<SLL>(*v_mask, *v_mask, *v_shift);
        asm.emit_r::<SLL>(*v_halfword, self.operands.rs2, *v_shift);
        asm.emit_r::<XOR>(*v_halfword, *v_dword, *v_halfword);
        asm.emit_r::<AND>(*v_halfword, *v_halfword, *v_mask);
        asm.emit_r::<XOR>(*v_dword, *v_dword, *v_halfword);
        asm.emit_s::<SD>(*v_dword_address, *v_dword, 0);
        asm.finalize()
    }
}
