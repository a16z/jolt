use crate::utils::inline_helpers::InstrAssembler;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::andi::ANDI;
use super::format::format_load::FormatLoad;
use super::ld::LD;
use super::sll::SLL;
use super::slli::SLLI;
use super::srli::SRLI;
use super::virtual_lw::VirtualLW;
use super::xori::XORI;
use super::{Instruction, RAMRead};
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use super::{Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LBU,
    mask   = 0x0000707f,
    match  = 0x00004003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LBU {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LBU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu
            .mmu
            .load(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((byte, memory_read)) => {
                *ram_access = memory_read;
                byte as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LBU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Load unsigned byte without sign extension.
    ///
    /// LBU loads an 8-bit value from memory at address rs1+imm and zero-extends
    /// it to the full register width. Since zkVM uses word-aligned memory:
    /// 1. Load the aligned word/doubleword containing the byte
    /// 2. Shift byte to the high bits using XOR-based position calculation
    /// 3. Logical right shift to extract and zero-extend
    ///
    /// Unlike LB, uses logical (not arithmetic) right shift for zero extension.
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

impl LBU {
    /// 32-bit implementation of load unsigned byte.
    ///
    /// Algorithm:
    /// 1. Calculate target address
    /// 2. Align to 4-byte boundary for word access
    /// 3. Load word containing the byte
    /// 4. XOR address with 3 to get opposite byte position (little-endian)
    /// 5. Shift byte to bits [31:24]
    /// 6. Logical right shift by 24 to zero-extend to 32 bits
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(self.operands.rd, *v0, -4i64 as u64);
        asm.emit_i::<VirtualLW>(self.operands.rd, self.operands.rd, 0);
        asm.emit_i::<XORI>(*v0, *v0, 3);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SLL>(self.operands.rd, self.operands.rd, *v0);
        asm.emit_i::<SRLI>(self.operands.rd, self.operands.rd, 24);

        asm.finalize()
    }

    /// 64-bit implementation of load unsigned byte.
    ///
    /// Similar to 32-bit but handles 8 possible byte positions:
    /// 1. XOR address with 7 for position calculation
    /// 2. Shift byte to bits [63:56]
    /// 3. Logical right shift by 56 to zero-extend to 64 bits
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(self.operands.rd, *v0, -8i64 as u64);
        asm.emit_ld::<LD>(self.operands.rd, self.operands.rd, 0);
        asm.emit_i::<XORI>(*v0, *v0, 7);
        asm.emit_i::<SLLI>(*v0, *v0, 3);
        asm.emit_r::<SLL>(self.operands.rd, self.operands.rd, *v0);
        asm.emit_i::<SRLI>(self.operands.rd, self.operands.rd, 56);

        asm.finalize()
    }
}
