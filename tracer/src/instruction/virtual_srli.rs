use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::{cpu::GeneralizedCpu, memory::MemoryData},
    instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI,
};

use super::{RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualSRLI,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftI,
    ram = ()
);

impl VirtualSRLI {
    fn exec<D: MemoryData>(
        &self,
        cpu: &mut GeneralizedCpu<D>,
        _: &mut <VirtualSRLI as RISCVInstruction>::RAMAccess,
    ) {
        let shift = self.operands.imm.trailing_zeros();
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                .wrapping_shr(shift) as i64,
        );
    }
}

impl RISCVTrace for VirtualSRLI {}
