use serde::{Deserialize, Serialize};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};
use crate::{declare_riscv_instr, emulator::cpu::{Cpu, advice_tape_read}};

declare_riscv_instr!(
    name = VirtualAdviceLoad,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualAdviceLoad {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAdviceLoad as RISCVInstruction>::RAMAccess) {
        // Read from advice tape and write to rd register
        // The imm field specifies how many bytes to read (1, 2, 4, or 8)
        let num_bytes = self.operands.imm as usize;
        let advice_value = advice_tape_read(num_bytes).expect("Failed to read from advice tape");
        cpu.x[self.operands.rd as usize] = advice_value as i64;
    }
}

impl RISCVTrace for VirtualAdviceLoad {}
