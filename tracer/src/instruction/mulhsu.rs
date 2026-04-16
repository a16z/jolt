use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, andi::ANDI, format::format_r::FormatR, mul::MUL, mulhu::MULHU, sltu::SLTU,
    virtual_movsign::VirtualMovsign, xor::XOR, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

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
            match cpu.xlen {
                Xlen::Bit32 => cpu.sign_extend(
                    cpu.x[self.operands.rs1 as usize]
                        .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u32 as i64)
                        >> 32,
                ),
                Xlen::Bit64 => {
                    ((cpu.x[self.operands.rs1 as usize] as i128 as u128)
                        .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u64 as u128)
                        >> 64) as i64
                }
            },
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
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();
        let v2 = allocator.allocate();
        let v3 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        asm.emit_i::<VirtualMovsign>(*v0, self.operands.rs1, 0);
        asm.emit_i::<ANDI>(*v1, *v0, 1);
        asm.emit_r::<XOR>(*v2, self.operands.rs1, *v0);
        asm.emit_r::<ADD>(*v2, *v2, *v1);
        asm.emit_r::<MULHU>(*v3, *v2, self.operands.rs2);
        asm.emit_r::<MUL>(*v2, *v2, self.operands.rs2);
        asm.emit_r::<XOR>(*v3, *v3, *v0);
        asm.emit_r::<XOR>(*v2, *v2, *v0);
        asm.emit_r::<ADD>(*v0, *v2, *v1);
        asm.emit_r::<SLTU>(*v0, *v0, *v2);
        asm.emit_r::<ADD>(self.operands.rd, *v3, *v0);

        asm.finalize()
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
        cpu.update_xlen(Xlen::Bit64);

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
