use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr, emulator::cpu::Cpu, instruction::format::format_fence::FormatFence,
};

use super::{RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = FENCE,
    mask   = 0x0000707f,
    match  = 0x0000000f,
    format = FormatFence,
    ram    = ()
);

impl FENCE {
    fn exec(&self, _: &mut Cpu, _: &mut <FENCE as RISCVInstruction>::RAMAccess) {
        // no-op
    }
}

impl RISCVTrace for FENCE {}
