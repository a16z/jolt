use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::format::format_load::FormatLoad;
use super::Instruction;
use super::RAMRead;

use super::{Cycle, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LH,
    mask   = 0x0000707f,
    match  = 0x00001003,
    format = FormatLoad,
    ram    = RAMRead
);

impl LH {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LH as RISCVInstruction>::RAMAccess) {
        let value = match cpu
            .mmu
            .load_halfword(cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64)
        {
            Ok((halfword, memory_read)) => {
                *ram_access = memory_read;
                halfword as i16 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, value);
    }
}

impl RISCVTrace for LH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl LH {}
