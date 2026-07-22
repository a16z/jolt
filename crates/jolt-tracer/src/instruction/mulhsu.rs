use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = MULHSU,
    mask   = 0xfe00707f,
    match  = 0x02002033,
    format = FormatR,
    ram    = ()
);

impl MULHSU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULHSU as RISCVInstruction>::RAMAccess) {
        cpu.write_register(
            self.operands.rd as usize,
            ((cpu.x[self.operands.rs1 as usize] as i128 as u128)
                .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u64 as u128)
                >> 64) as i64,
        );
    }
}

impl MULHSU {
    /// Construct a MULHSU with given register operands.
    #[cfg(test)]
    fn with_regs(rd: u8, rs1: u8, rs2: u8) -> Self {
        // Build a valid instruction word from the MATCH constant.
        let word = Self::MATCH | ((rd as u32) << 7) | ((rs1 as u32) << 15) | ((rs2 as u32) << 20);
        Self::new(word, 0x1000, true, false)
    }
}

impl RISCVTrace for MULHSU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};

    /// Regression test: MULHSU with negative rs1.
    ///
    /// MULHSU computes the upper 64 bits of (rs1 as signed) * (rs2 as unsigned).
    /// Before the fix, rs2 was zero-extended incorrectly (as i64 as u128 instead
    /// of as u64 as u128), causing sign-extension of negative rs1 to leak into
    /// the unsigned operand, producing wrong results for negative rs1 values.
    #[test]
    fn test_mulhsu_negative_rs1() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));

        // MULHSU rd=x1, rs1=x2, rs2=x3
        let instr = MULHSU::with_regs(1, 2, 3);

        // rs1 = -1 (signed), rs2 = 2 (unsigned)
        cpu.x[2] = -1_i64;
        cpu.x[3] = 2;

        instr.exec(&mut cpu, &mut ());

        // (-1_i128) * (2_u128 treated as 128-bit) = -2_i128
        // -2 in two's complement 128-bit = 0xFFFF...FFFE
        // upper 64 bits = 0xFFFFFFFFFFFFFFFF
        //
        // Before the fix, the incorrect zero-extension computed:
        //   0xFFFFFFFFFFFFFFFF_u128 * 2_u128 = 0x1_FFFFFFFFFFFFFFFE
        //   upper 64 = 1 (WRONG)
        assert_eq!(
            cpu.x[1] as u64, 0xFFFFFFFFFFFFFFFF,
            "MULHSU(-1, 2) upper 64 bits should be all-ones"
        );
    }
}
