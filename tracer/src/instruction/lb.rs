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
        cpu.x[self.operands.rd as usize] = match cpu
            .mmu
            .load(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((byte, memory_read)) => {
                *ram_access = memory_read;
                byte as i8 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
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

    /// Generates an inline sequence to load a sign-extended byte from memory.
    ///
    /// LB (Load Byte) loads an 8-bit value from memory and sign-extends it to XLEN bits.
    /// The implementation loads a naturally-aligned word/dword containing the byte,
    /// then extracts and sign-extends the specific byte.
    ///
    /// This approach is used because the zkVM works with word-aligned memory accesses,
    /// so byte loads must be emulated using word loads and bit manipulation.
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
    /// 32-bit implementation of byte load with sign extension.
    ///
    /// Algorithm:
    /// 1. Calculate effective address (base + offset)
    /// 2. Align address to 4-byte boundary to get word address
    /// 3. Load the aligned 32-bit word containing the target byte
    /// 4. Calculate shift amount based on byte position within word
    /// 5. Shift byte to MSB position and sign-extend by arithmetic right shift
    ///
    /// The byte position within the word is determined by the lower 2 bits of the address.
    /// For little-endian systems, XOR with 3 reverses the byte order for shifting.
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_address = allocator.allocate(); // effective address
        let v_word_address = allocator.allocate(); // aligned word address
        let v_word = allocator.allocate(); // loaded word
        let v_shift = allocator.allocate(); // shift amount

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        // Step 1: Calculate effective address = base + immediate
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);

        // Step 2: Align to 4-byte boundary (clear lower 2 bits)
        asm.emit_i::<ANDI>(*v_word_address, *v_address, -4i64 as u64);

        // Step 3: Load the aligned 32-bit word
        asm.emit_i::<VirtualLW>(*v_word, *v_word_address, 0);

        // Step 4: Calculate shift amount to position byte at MSB
        // XOR with 3 handles little-endian byte ordering
        // Multiply by 8 converts byte offset to bit offset
        asm.emit_i::<XORI>(*v_shift, *v_address, 3); // reverse byte position
        asm.emit_i::<SLLI>(*v_shift, *v_shift, 3); // multiply by 8 (bits per byte)

        // Step 5: Shift byte to MSB and sign-extend
        asm.emit_r::<SLL>(self.operands.rd, *v_word, *v_shift); // shift byte to top
        asm.emit_i::<SRAI>(self.operands.rd, self.operands.rd, 24); // sign-extend from bit 7

        asm.finalize()
    }

    /// 64-bit implementation of byte load with sign extension.
    ///
    /// Algorithm:
    /// 1. Calculate effective address (base + offset)
    /// 2. Align address to 8-byte boundary to get dword address
    /// 3. Load the aligned 64-bit dword containing the target byte
    /// 4. Calculate shift amount based on byte position within dword
    /// 5. Shift byte to MSB position and sign-extend by arithmetic right shift
    ///
    /// The byte position within the dword is determined by the lower 3 bits of the address.
    /// For little-endian systems, XOR with 7 reverses the byte order for shifting.
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_address = allocator.allocate(); // effective address
        let v_dword_address = allocator.allocate(); // aligned dword address
        let v_dword = allocator.allocate(); // loaded dword
        let v_shift = allocator.allocate(); // shift amount

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        // Step 1: Calculate effective address = base + immediate
        asm.emit_i::<ADDI>(*v_address, self.operands.rs1, self.operands.imm as u64);

        // Step 2: Align to 8-byte boundary (clear lower 3 bits)
        asm.emit_i::<ANDI>(*v_dword_address, *v_address, -8i64 as u64);

        // Step 3: Load the aligned 64-bit dword
        asm.emit_ld::<LD>(*v_dword, *v_dword_address, 0);

        // Step 4: Calculate shift amount to position byte at MSB
        // XOR with 7 handles little-endian byte ordering
        // Multiply by 8 converts byte offset to bit offset
        asm.emit_i::<XORI>(*v_shift, *v_address, 7); // reverse byte position
        asm.emit_i::<SLLI>(*v_shift, *v_shift, 3); // multiply by 8 (bits per byte)

        // Step 5: Shift byte to MSB and sign-extend
        asm.emit_r::<SLL>(self.operands.rd, *v_dword, *v_shift); // shift byte to top
        asm.emit_i::<SRAI>(self.operands.rd, self.operands.rd, 56); // sign-extend from bit 7

        asm.finalize()
    }
}
