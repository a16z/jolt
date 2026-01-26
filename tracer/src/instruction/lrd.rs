use serde::{Deserialize, Serialize};

use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::addi::ADDI;
use super::format::format_r::FormatR;
use super::ld::LD;
use super::{Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LRD,
    mask   = 0xf9f0707f,
    match  = 0x1000302f,
    format = FormatR,
    ram    = ()
);

impl LRD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <LRD as RISCVInstruction>::RAMAccess) {
        if cpu.is_reservation_set() {
            println!("LRD: Reservation is already set");
        }

        let address = cpu.x[self.operands.rs1 as usize] as u64;

        // Load the doubleword from memory
        let value = cpu.mmu.load_doubleword(address);

        cpu.x[self.operands.rd as usize] = match value {
            Ok((doubleword, _memory_read)) => {
                // Set reservation for this address
                cpu.set_reservation(address);

                // Return the 64-bit value
                doubleword as i64
            }
            Err(_) => panic!("MMU load error"),
        };
    }
}

impl RISCVTrace for LRD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // Set reservation in CPU state (for SC.D to check later)
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        cpu.set_reservation(address);

        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// LR.D: Load Reserved Doubleword
    /// Loads a 64-bit doubleword from memory at address rs1, stores it in rd,
    /// and sets a reservation on the address for use by SC.D.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // LR.D is only available in RV64A, so we only implement the 64-bit path
        assert_eq!(xlen, Xlen::Bit64, "LR.D is only available in RV64");

        let v_reservation = allocator.reservation_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Store reservation address (rs1 -> register 32) using ADDI rd, rs1, 0
        asm.emit_i::<ADDI>(v_reservation, self.operands.rs1, 0);
        // Load doubleword directly (no alignment/shifting needed for 64-bit aligned access)
        asm.emit_ld::<LD>(self.operands.rd, self.operands.rs1, 0);

        asm.finalize()
    }
}
