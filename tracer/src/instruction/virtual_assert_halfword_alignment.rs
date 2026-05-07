use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_assert_align::FormatAssert, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualAssertHalfwordAlignment,
    mask = 0,
    match = 0,
    format = FormatAssert,
    ram = ()
);

impl VirtualAssertHalfwordAlignment {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <VirtualAssertHalfwordAlignment as RISCVInstruction>::RAMAccess,
    ) {
        let address = cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm);
        assert!(
            address & 1 == 0,
            "RAM access (LH or LHU) is not halfword aligned: {address:x}"
        );
    }
}

impl RISCVTrace for VirtualAssertHalfwordAlignment {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::cpu::Cpu;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::format::format_assert_align::FormatAssert;

    /// See [`VirtualAssertWordAlignment`] tests for the rationale.
    #[test]
    fn wraps_on_overflow() {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.x[1] = i64::MAX;
        let instr = VirtualAssertHalfwordAlignment {
            address: 0,
            operands: FormatAssert { rs1: 1, imm: 1 },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        let mut ram_access = ();
        // i64::MAX.wrapping_add(1) == i64::MIN, whose low bit is 0, so this
        // must execute without panicking.
        instr.execute(&mut cpu, &mut ram_access);
    }
}
