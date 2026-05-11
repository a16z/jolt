use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu},
    instruction::format::format_advice_load_i::FormatAdviceLoadI,
};

use super::{Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = AdviceLD,
    mask   = 0,
    match  = 0,
    format = FormatAdviceLoadI,
    ram    = ()
);

impl AdviceLD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AdviceLD as RISCVInstruction>::RAMAccess) {
        // Read 8 bytes (doubleword) from the advice tape
        let advice_value = advice_tape_read(cpu, 8).expect("Failed to read from advice tape");
        // Store the advice value to register rd
        cpu.write_register(self.operands.rd as usize, advice_value as i64);
    }
}

impl RISCVTrace for AdviceLD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

impl AdviceLD {}
