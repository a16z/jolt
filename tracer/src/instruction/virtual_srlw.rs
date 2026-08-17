use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::format_virtual_right_shift_r::FormatVirtualRightShiftR, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualSRLW,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftR<32>,
    ram = ()
);

impl VirtualSRLW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSRLW as RISCVInstruction>::RAMAccess) {
        let shift = cpu.x[self.operands.rs2 as usize].trailing_zeros();
        let result = (cpu.x[self.operands.rs1 as usize] as u32) >> shift;
        cpu.write_register(self.operands.rd as usize, result as i32 as i64);
    }
}

impl RISCVTrace for VirtualSRLW {}
