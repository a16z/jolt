use crate::instruction::registers::amo::RegisterStateAMO;
use serde::{Deserialize, Serialize};

use super::Instruction;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_amo::FormatAMO, Cycle, RAMWrite, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AMOMINW,
    mask   = 0xf800707f,
    match  = 0x8000202f,
    format = FormatAMO,
    registers = RegisterStateAMO,
    ram    = RAMWrite
);

impl AMOMINW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMINW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize] as i32;

        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        let new_value = if original_value as i32 <= compare_value {
            original_value as i32
        } else {
            compare_value
        };
        cpu.mmu
            .store_word(address, new_value as u32)
            .expect("MMU store error");

        cpu.write_register(self.operands.rd as usize, original_value);
    }
}

impl RISCVTrace for AMOMINW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        super::trace_inline_sequence(&Instruction::from(*self), cpu, trace);
    }
}
