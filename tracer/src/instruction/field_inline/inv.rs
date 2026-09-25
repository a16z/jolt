use jolt_field::Field;
use jolt_program::field_inline::{FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite};
use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::Cpu,
    instruction::{format::format_field_inline::FormatFieldInline, RISCVInstruction, RISCVTrace},
};

use super::{decode_field, encode_field, FieldInlineCycleData, ProofField};

declare_riscv_instr!(
    name   = FIELD_INV,
    mask   = FieldInlineOp::Inv.instruction_mask(),
    match  = FieldInlineOp::Inv.instruction_match(),
    format = FormatFieldInline,
    ram    = FieldInlineCycleData
);

impl FIELD_INV {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <FIELD_INV as RISCVInstruction>::RAMAccess) {
        let rs1_register = self.operands.rs1.unwrap_or(0);
        let rd_register = self.operands.rd.unwrap_or(0);
        let rs1_value = cpu.field_registers.read(rs1_register);
        let pre_value = cpu.field_registers.read(rd_register);
        // inv(0) fails closed: the R1CS row `IsFieldInv * (FieldInvProduct - 1) = 0`
        // demands `rs1 * rd == 1`, which is unsatisfiable for rs1 = 0, so a trace
        // containing FIELD_INV(0) can never be proven. Trapping here surfaces the
        // guest bug at trace time instead of as a downstream sumcheck failure; the
        // guest-side API is responsible for guarding zero before emitting FIELD_INV.
        let inverse = decode_field::<ProofField>(rs1_value)
            .inverse()
            .unwrap_or_else(|| {
                panic!(
                    "FIELD_INV of zero at pc 0x{:x} (field register {}): the inverse constraint is \
                     unsatisfiable for a zero operand; guard the guest-side inverse",
                    cpu.read_pc(),
                    rs1_register,
                )
            });
        let post_value = encode_field(inverse);
        cpu.field_registers.write(rd_register, post_value);
        *ram_access = FieldInlineTraceData {
            op: Some(FieldInlineOp::Inv),
            rs1: Some(FieldRegisterRead {
                register: rs1_register,
                value: rs1_value,
            }),
            rd: Some(FieldRegisterWrite {
                register: rd_register,
                pre_value,
                post_value,
            }),
            inv_product: Some(encode_field(
                decode_field::<ProofField>(rs1_value) * inverse,
            )),
            ..Default::default()
        }
        .into();
    }
}

impl RISCVTrace for FIELD_INV {}
