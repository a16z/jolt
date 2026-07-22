use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::format_virtual_right_shift_r::FormatVirtualRightShiftR, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualSRA,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftR,
    ram = ()
);

impl VirtualSRA {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSRA as RISCVInstruction>::RAMAccess) {
        let shift = cpu.x[self.operands.rs2 as usize].trailing_zeros();
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(cpu.x[self.operands.rs1 as usize].wrapping_shr(shift)),
        );
    }
}

impl RISCVTrace for VirtualSRA {}
