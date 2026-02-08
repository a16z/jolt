use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
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
use super::{Instruction, RAMWrite};

use super::{format::format_s::FormatS, Cycle, RISCVInstruction, RISCVTrace};

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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Store byte to memory using word-aligned access.
    ///
    /// SB stores the lower 8 bits of rs2 to memory at address rs1+imm.
    /// Since zkVM uses word-aligned memory, this requires:
    /// 1. Loading the aligned word containing the target byte
    /// 2. Masking and replacing the specific byte
    /// 3. Storing the modified word back to memory
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

impl SB {
    /// 32-bit implementation of store byte.
    ///
    /// Algorithm:
    /// 1. Calculate target address and align to 4-byte boundary
    /// 2. Load the aligned word containing the target byte
    /// 3. Create a mask for the specific byte position (0xFF shifted to position)
    /// 4. Shift the byte value to the correct position
    /// 5. Use XOR operations to replace only the target byte
    /// 6. Store the modified word back to memory
    ///
    /// The XOR technique: (word ^ byte) & mask ^ word
    /// This clears the original byte and sets the new byte value.
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();
        let v2 = allocator.allocate();
        let v3 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v1, *v0, -4i64 as u64);
        asm.emit_i::<VirtualLW>(*v2, *v1, 0);
        asm.emit_i::<SLLI>(*v3, *v0, 3);
        asm.emit_u::<LUI>(*v0, 0xff);
        asm.emit_r::<SLL>(*v0, *v0, *v3);
        asm.emit_r::<SLL>(*v3, self.operands.rs2, *v3);
        asm.emit_r::<XOR>(*v3, *v2, *v3);
        asm.emit_r::<AND>(*v3, *v3, *v0);
        asm.emit_r::<XOR>(*v2, *v2, *v3);
        asm.emit_s::<VirtualSW>(*v1, *v2, 0);

        asm.finalize()
    }

    /// 64-bit implementation of store byte.
    ///
    /// Similar to 32-bit version but operates on 64-bit doublewords.
    /// The byte position is determined by the lower 3 bits of the address,
    /// and the shift amount is multiplied by 8 to convert to bit positions.
    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();
        let v2 = allocator.allocate();
        let v3 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        asm.emit_i::<ADDI>(*v0, self.operands.rs1, self.operands.imm as u64);
        asm.emit_i::<ANDI>(*v1, *v0, -8i64 as u64);
        asm.emit_ld::<LD>(*v2, *v1, 0);
        asm.emit_i::<SLLI>(*v3, *v0, 3);
        asm.emit_u::<LUI>(*v0, 0xff);
        asm.emit_r::<SLL>(*v0, *v0, *v3);
        asm.emit_r::<SLL>(*v3, self.operands.rs2, *v3);
        asm.emit_r::<XOR>(*v3, *v2, *v3);
        asm.emit_r::<AND>(*v3, *v3, *v0);
        asm.emit_r::<XOR>(*v2, *v2, *v3);
        asm.emit_s::<SD>(*v1, *v2, 0);

        asm.finalize()
    }
}
