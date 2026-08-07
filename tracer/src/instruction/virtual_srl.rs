use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::format_virtual_right_shift_r::FormatVirtualRightShiftR, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualSRL,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftR,
    ram = ()
);

impl VirtualSRL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualSRL as RISCVInstruction>::RAMAccess) {
        let mask = cpu.x[self.operands.rs2 as usize] as u64;
        let shift = mask.trailing_zeros();
        cpu.write_register(
            self.operands.rd as usize,
            cpu.sign_extend(
                (cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                    .wrapping_shr(shift)
                    & mask.wrapping_shr(shift)) as i64,
            ),
        );
    }
}

impl RISCVTrace for VirtualSRL {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;

    #[test]
    fn extracts_contiguous_masked_lanes() {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = 0x8070_6050_4030_2010u64 as i64;
        let instruction = VirtualSRL {
            address: 0,
            operands: FormatVirtualRightShiftR {
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };

        for offset in 0..8 {
            cpu.x[2] = (0xffu64 << (8 * offset)) as i64;
            instruction.execute(&mut cpu, &mut ());
            assert_eq!(cpu.x[3] as u64, 0x10 * (offset + 1));
        }
    }
}
