use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, ReservationWidth},
};

use super::format::format_r::FormatR;
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

        let write_value = match value {
            Ok((doubleword, _memory_read)) => {
                cpu.set_reservation(address, ReservationWidth::Doubleword);
                // Return the 64-bit value
                doubleword as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, write_value);
    }
}

impl RISCVTrace for LRD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        cpu.set_reservation(address, ReservationWidth::Doubleword);

        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
