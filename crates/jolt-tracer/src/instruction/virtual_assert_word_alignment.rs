use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_assert_align::FormatAssert, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertWordAlignment,
    mask = 0,
    match = 0,
    format = FormatAssert,
    ram = ()
);

impl VirtualAssertWordAlignment {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertWordAlignment as RISCVInstruction>::RAMAccess,
    ) {
        let address = cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm);
        assert!(
            address & 3 == 0,
            "RAM access (LW or LWU) is not word aligned: {address:x}"
        );
    }
}

impl RISCVTrace for VirtualAssertWordAlignment {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::cpu::Cpu;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::format::format_assert_align::FormatAssert;

    /// `rs1 + imm` must use wrapping arithmetic to match RV64 semantics
    /// (effective addresses are mod 2^64) and the convention used by every
    /// other tracer instruction (LB/LBU/LH/LHU/LW/LWU/LD, SB/SH/SW/SD,
    /// VirtualLW/VirtualSW, AMO*). Plain `+` panics on i64 overflow in
    /// debug builds for any rs1/imm pair whose signed sum falls outside
    /// `[i64::MIN, i64::MAX]`.
    #[test]
    fn wraps_on_overflow() {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = i64::MAX;
        let instr = VirtualAssertWordAlignment {
            address: 0,
            operands: FormatAssert { rs1: 1, imm: 1 },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        let mut ram_access = ();
        // i64::MAX.wrapping_add(1) == i64::MIN, whose low 2 bits are 0,
        // so this must execute without panicking.
        instr.execute(&mut cpu, &mut ram_access);
    }
}
