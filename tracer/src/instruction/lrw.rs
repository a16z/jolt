use serde::{Deserialize, Serialize};

use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::format::format_r::FormatR;
use super::lw::LW;
use super::virtual_lw::VirtualLW;
use super::{Cycle, Instruction, RAMRead, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LRW,
    mask   = 0xf9f0707f,
    match  = 0x1000202f,
    format = FormatR,
    ram    = RAMRead
);

impl LRW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <LRW as RISCVInstruction>::RAMAccess) {
        if cpu.is_reservation_set() {
            println!("LRW: Reservation is already set");
        }

        let address = cpu.x[self.operands.rs1 as usize] as u64;

        // Load the word from memory
        let value = cpu.mmu.load_word(address);

        cpu.x[self.operands.rd as usize] = match value {
            Ok((word, _memory_read)) => {
                // Set reservation for this address
                cpu.set_reservation(address);

                // Sign extend the 32-bit value
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LRW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Set reservation in CPU state (for SC.W to check later)
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        cpu.set_reservation(address);

        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// LR.W: Load Reserved Word
    /// Loads a 32-bit word from memory at address rs1, sign-extends it to 64 bits,
    /// stores it in rd, and sets a reservation on the address for use by SC.W.
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

impl LRW {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_reservation = allocator.reservation_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        // Store reservation address (rs1 -> register 32) using ADDI rd, rs1, 0
        asm.emit_i::<ADDI>(v_reservation, self.operands.rs1, 0);
        // Load word (same as VirtualLW but with imm=0)
        asm.emit_i::<VirtualLW>(self.operands.rd, self.operands.rs1, 0);

        asm.finalize()
    }

    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_reservation = allocator.reservation_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        // 1. Store reservation address (rs1 -> v_reservation) using ADDI
        asm.emit_i::<ADDI>(v_reservation, self.operands.rs1, 0);
        // 2. Load word - LW will auto-expand into the proper sequence
        asm.emit_ld::<LW>(self.operands.rd, self.operands.rs1, 0);

        asm.finalize()
    }
}
