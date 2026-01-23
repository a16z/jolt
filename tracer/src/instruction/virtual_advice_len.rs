use serde::{Deserialize, Serialize};

use super::{format::format_j::FormatJ, RISCVInstruction, RISCVTrace};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_remaining, Cpu},
};

declare_riscv_instr!(
    name = VirtualAdviceLen,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = ()
);

impl VirtualAdviceLen {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAdviceLen as RISCVInstruction>::RAMAccess) {
        // Get the number of bytes remaining in the advice tape and write to rd register
        let remaining = advice_tape_remaining();
        cpu.x[self.operands.rd as usize] = remaining as i64;
    }
}

impl RISCVTrace for VirtualAdviceLen {}
