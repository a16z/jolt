use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    format::format_virtual_right_shift_w_r::FormatVirtualRightShiftWR, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name = VirtualSRLW,
    mask = 0,
    match = 0,
    format = FormatVirtualRightShiftWR,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::{
        format::format_virtual_right_shift_w_i::FormatVirtualRightShiftWI,
        virtual_sraiw::VirtualSRAIW, virtual_sraw::VirtualSRAW, virtual_srliw::VirtualSRLIW,
    };

    fn mask(shift: u32) -> i64 {
        ((1u64 << 32) - (1u64 << shift)) as i64
    }

    fn srlw_register(x: i64, shift: u32) -> i64 {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = x;
        cpu.x[2] = mask(shift);
        let instruction = VirtualSRLW {
            address: 0,
            operands: FormatVirtualRightShiftWR {
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        instruction.execute(&mut cpu, &mut ());
        cpu.x[3]
    }

    fn sraw_register(x: i64, shift: u32) -> i64 {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = x;
        cpu.x[2] = mask(shift);
        let instruction = VirtualSRAW {
            address: 0,
            operands: FormatVirtualRightShiftWR {
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        instruction.execute(&mut cpu, &mut ());
        cpu.x[3]
    }

    fn srliw_immediate(x: i64, shift: u32) -> i64 {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = x;
        let instruction = VirtualSRLIW {
            address: 0,
            operands: FormatVirtualRightShiftWI {
                rd: 3,
                rs1: 1,
                imm: mask(shift) as u64,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        instruction.execute(&mut cpu, &mut ());
        cpu.x[3]
    }

    fn sraiw_immediate(x: i64, shift: u32) -> i64 {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = x;
        let instruction = VirtualSRAIW {
            address: 0,
            operands: FormatVirtualRightShiftWI {
                rd: 3,
                rs1: 1,
                imm: mask(shift) as u64,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        instruction.execute(&mut cpu, &mut ());
        cpu.x[3]
    }

    #[test]
    fn word_shift_torture() {
        for x in [
            0x6a5a_5a5a_7fff_0001u64 as i64,
            0xa5a5_a5a5_8000_0001u64 as i64,
        ] {
            for shift in [0, 1, 31] {
                let logical = ((x as u32) >> shift) as i32 as i64;
                let arithmetic = ((x as i32) >> shift) as i64;
                assert_eq!(srlw_register(x, shift), logical);
                assert_eq!(srliw_immediate(x, shift), logical);
                assert_eq!(sraw_register(x, shift), arithmetic);
                assert_eq!(sraiw_immediate(x, shift), arithmetic);
            }
        }
    }
}
