use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::format_virtual_right_shift_i::FormatVirtualRightShiftI, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualSRAIW,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI<32>,
    ram = ()
);

impl VirtualSRAIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSRAIW as RISCVInstruction>::RAMAccess) {
        let shift = self.operands.imm.trailing_zeros();
        let result = (cpu.x[self.operands.rs1 as usize] as i32) >> shift;
        cpu.write_register(self.operands.rd as usize, result as i64);
    }
}

impl RISCVTrace for VirtualSRAIW {}
