use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::format_virtual_right_shift_i::FormatVirtualRightShiftI, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualSRLIW,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI<32>,
    ram = ()
);

impl VirtualSRLIW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSRLIW as RISCVInstruction>::RAMAccess) {
        let shift = self.operands.imm.trailing_zeros();
        let result = (cpu.x[self.operands.rs1 as usize] as u32) >> shift;
        cpu.write_register(self.operands.rd as usize, result as i32 as i64);
    }
}

impl RISCVTrace for VirtualSRLIW {}
