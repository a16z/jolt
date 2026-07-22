use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::format_amo::FormatAMO, Cycle, Instruction, RAMWrite, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AMOANDD,
    mask   = 0xf800707f,
    match  = 0x6000302f,
    format = FormatAMO,
    ram    = RAMWrite
);

impl AMOANDD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOANDD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let and_value = cpu.x[self.operands.rs2 as usize] as u64;

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // AND the values and store back to memory
        let new_value = (original_value as u64) & and_value;
        cpu.mmu
            .store_doubleword(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.write_register(self.operands.rd as usize, original_value);
    }
}

impl RISCVTrace for AMOANDD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}
