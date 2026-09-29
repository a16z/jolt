use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr, emulator::cpu::Cpu,
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
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSRLI as RISCVInstruction>::RAMAccess) {
        let shift = self.operands.imm.trailing_zeros();
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(
                cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                    .checked_shr(shift)
                    .unwrap_or(0) as i64,
            ),
        );
    }
}

impl RISCVTrace for VirtualSRLI {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;

    #[test]
    fn zero_mask_outputs_zero() {
        let instruction = VirtualSRLI {
            address: 0,
            operands: FormatVirtualRightShiftI {
                rd: 2,
                rs1: 1,
                imm: 0,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: true,
            is_compressed: false,
        };
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = -1;
        cpu.x[2] = 1;
        instruction.trace(&mut cpu, None);
        assert_eq!(cpu.x[2], 0);
    }
}
