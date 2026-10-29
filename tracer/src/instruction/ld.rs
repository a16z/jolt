use super::{
    floor_access, runs_misaligned_sequence, trace_final_row, trace_inline_sequence, Cycle,
    Instruction, RISCVInstruction, RISCVTrace, RamRead,
};
use crate::instruction::registers::load::RegisterStateLoad;
use crate::{
    declare_riscv_instr, emulator::cpu::Cpu, instruction::format::format_load::FormatLoad,
};
use jolt_program::expand::{DualAccess, DualWidth};
use serde::{Deserialize, Serialize};

declare_riscv_instr!(
    name   = LD,
    mask   = 0x0000707f,
    match  = 0x00003003,
    format = FormatLoad,
    registers = RegisterStateLoad,
    ram    = RamRead
);

impl LD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <LD as RISCVInstruction>::RamAccess) {
        let address = (cpu.x[self.operands.rs1 as usize] as u64)
            .wrapping_add(self.operands.imm as i32 as u64);
        let value = cpu
            .get_mut_mmu()
            .load_doubleword(floor_access::native_doubleword(address));
        cpu.write_register(
            self.operands.rd as usize,
            match value {
                Ok((value, memory_read)) => {
                    *ram_access = RamRead {
                        address,
                        ..memory_read
                    };
                    value as i64
                }
                Err(_) => panic!("MMU load error"),
            },
        );
    }
}
impl RISCVTrace for LD {
    /// An `LD` runs as its native row, which aborts on a misaligned address,
    /// except that under `zicclsm` a misaligned one runs the misaligned
    /// sequence of its dual expansion.
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let access = DualAccess {
            base: self.operands.rs1,
            offset: self.operands.imm.into(),
            width: DualWidth::Doubleword,
        };
        if runs_misaligned_sequence(access, &cpu.x) {
            trace_inline_sequence(&Instruction::LD(*self), cpu, trace);
        } else {
            trace_final_row(self, cpu, trace);
        }
    }
}
