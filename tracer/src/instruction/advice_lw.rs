use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu},
    instruction::format::format_advice_load_i::FormatAdviceLoadI,
};

use super::{Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AdviceLW,
    mask   = 0,
    match  = 0,
    format = FormatAdviceLoadI,
    ram    = ()
);

impl AdviceLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AdviceLW as RISCVInstruction>::RAMAccess) {
        // Read 4 bytes (word) from the advice tape
        let advice_value = advice_tape_read(cpu, 4).expect("Failed to read from advice tape");
        // Store the sign extended advice value to register rd
        cpu.write_register(self.operands.rd as usize, advice_value as i32 as i64);
    }
}

impl RISCVTrace for AdviceLW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl AdviceLW {}
