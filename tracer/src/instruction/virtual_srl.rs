use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::{format_virtual_right_shift_r::FormatVirtualRightShiftR, InstructionFormat},
    RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualSRL,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftR,
    ram = (),
    is_virtual = true
);

impl VirtualSRL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSRL as RISCVInstruction>::RAMAccess) {
        let shift = cpu.x[self.operands.rs2].trailing_zeros();
        cpu.x[self.operands.rd] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1])
                .wrapping_shr(shift) as i64,
        );
    }
}

impl RISCVTrace for VirtualSRL {}
