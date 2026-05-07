use serde::{Deserialize, Serialize};

use super::{format::format_j::FormatJ, RISCVInstruction, RISCVTrace};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{advice_tape_read, Cpu},
};

declare_riscv_instr!(
    name = VirtualAdviceLoad,
    mask = 0,
    match = 0,
    format = FormatJ,
    ram = ()
);

impl VirtualAdviceLoad {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualAdviceLoad as RISCVInstruction>::RAMAccess) {
        // Read from advice tape and write to rd register
        // The imm field specifies how many bytes to read (1, 2, 4, or 8)
        let num_bytes = self.operands.imm as usize;
        debug_assert!([1, 2, 4, 8].contains(&num_bytes));
        let advice_value =
            advice_tape_read(cpu, num_bytes).expect("Failed to read from advice tape");
        cpu.write_register(self.operands.rd as usize, advice_value as i64);
    }
}

impl RISCVTrace for VirtualAdviceLoad {}
