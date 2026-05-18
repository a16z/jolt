//! BN254 Fr FieldAdd: `field_regs[frd] = field_regs[frs1] + field_regs[frs2]`.
//!
//! Encoded as opcode `0x0B` + funct7 `0x40` + funct3 `0x03`.

use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    field_arith_common::{
        field_arith_match, fr_from_limbs, fr_to_limbs, trace_field_arith_cycle,
        FIELD_ARITH_MASK, FUNCT3_FADD,
    },
    format::format_r::FormatR,
    Cycle, RISCVTrace,
};

declare_riscv_instr!(
    name   = FieldAdd,
    mask   = FIELD_ARITH_MASK,
    match  = field_arith_match(FUNCT3_FADD),
    format = FormatR,
    ram    = ()
);

impl FieldAdd {
    fn exec(&self, cpu: &mut Cpu, _: &mut <FieldAdd as crate::instruction::RISCVInstruction>::RAMAccess) {
        let frd = self.operands.rd as usize;
        let frs1 = self.operands.rs1 as usize;
        let frs2 = self.operands.rs2 as usize;
        debug_assert!(frd < cpu.field_regs.len(), "frd out of range");
        debug_assert!(frs1 < cpu.field_regs.len(), "frs1 out of range");
        debug_assert!(frs2 < cpu.field_regs.len(), "frs2 out of range");

        let a = fr_from_limbs(&cpu.field_regs[frs1]);
        let b = fr_from_limbs(&cpu.field_regs[frs2]);
        cpu.field_regs[frd] = fr_to_limbs(&(a + b));
    }
}

impl RISCVTrace for FieldAdd {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let operands = self.operands;
        trace_field_arith_cycle(self, &operands, cpu, trace);
    }
}
