//! BN254 Fr FieldInv: `field_regs[frd] = field_regs[frs1]⁻¹`.
//!
//! Encoded as opcode `0x0B` + funct7 `0x40` + funct3 `0x04`. FINV(0) is
//! unsatisfiable — the SDK guards it via `Fr::inverse() -> Option<Fr>`;
//! `execute()` panics if called with frs1 holding the zero element so a
//! malicious inline-asm path can't silently fill in zero and corrupt the
//! Stage 4 FR Twist downstream.

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{
    field_arith_common::{
        field_arith_match, fr_from_limbs, fr_to_limbs, trace_field_arith_cycle, FIELD_ARITH_MASK,
        FUNCT3_FINV,
    },
    format::format_r::FormatR,
    Cycle, RISCVTrace,
};

declare_riscv_instr!(
    name   = FieldInv,
    mask   = FIELD_ARITH_MASK,
    match  = field_arith_match(FUNCT3_FINV),
    format = FormatR,
    ram    = ()
);

impl FieldInv {
    fn exec(
        &self,
        cpu: &mut Cpu,
        _: &mut <FieldInv as crate::instruction::RISCVInstruction>::RAMAccess,
    ) {
        let frd = self.operands.rd as usize;
        let frs1 = self.operands.rs1 as usize;
        debug_assert!(frd < cpu.field_regs.len(), "frd out of range");
        debug_assert!(frs1 < cpu.field_regs.len(), "frs1 out of range");

        let a = fr_from_limbs(&cpu.field_regs[frs1]);
        let inv = a
            .inverse()
            .expect("FieldInv on zero input; the SDK guards this via Fr::inverse() -> Option<Fr>");
        cpu.field_regs[frd] = fr_to_limbs(&inv);
    }
}

impl RISCVTrace for FieldInv {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let operands = self.operands;
        trace_field_arith_cycle(self, &operands, cpu, trace);
    }
}
