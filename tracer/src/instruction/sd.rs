use crate::instruction::registers::s::RegisterStateS;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::RamWrite;

use super::{
    floor_access, format::format_s::FormatS, runs_misaligned_sequence, trace_final_row,
    trace_inline_sequence, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};
use jolt_program::expand::{DualAccess, DualWidth};

declare_riscv_instr!(
    name   = SD,
    mask   = 0x0000707f,
    match  = 0x00003023,
    format = FormatS,
    registers = RegisterStateS,
    ram    = RamWrite
);

impl SD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SD as RISCVInstruction>::RamAccess) {
        let address = cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64;
        let write = cpu
            .mmu
            .store_doubleword(
                floor_access::native_doubleword(address),
                cpu.x[self.operands.rs2 as usize] as u64,
            )
            .ok()
            .unwrap();
        *ram_access = RamWrite { address, ..write };
    }
}

impl RISCVTrace for SD {
    /// An `SD` runs as its native row, which aborts on a misaligned address,
    /// except that under `zicclsm` a misaligned one runs the misaligned
    /// sequence of its dual expansion.
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let access = DualAccess {
            base: self.operands.rs1,
            offset: self.operands.imm.into(),
            width: DualWidth::Doubleword,
        };
        if runs_misaligned_sequence(access, &cpu.x) {
            trace_inline_sequence(&Instruction::SD(*self), cpu, trace);
        } else {
            trace_final_row(self, cpu, trace);
        }
    }
}
