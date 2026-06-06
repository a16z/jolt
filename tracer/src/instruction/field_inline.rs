#![expect(
    non_camel_case_types,
    reason = "Tracer concrete instruction names mirror generated Jolt instruction constants"
)]

use jolt_field::{CanonicalBytes, CanonicalU64, Fr, Invertible, ReducingBytes};
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead,
    FieldRegisterWrite,
};
use jolt_riscv::{FieldInlineOp, SourceInstructionKind, FIELD_INLINE_OPCODE};
use serde::{Deserialize, Serialize};

use super::{
    format::{format_field_inline::FormatFieldInline, InstructionFormat},
    RAMAccess, RISCVInstruction, RISCVTrace,
};
use crate::emulator::cpu::Cpu;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineCycleData {
    pub trace: Option<FieldInlineTraceData>,
}

impl From<FieldInlineCycleData> for RAMAccess {
    fn from(_value: FieldInlineCycleData) -> Self {
        Self::NoOp
    }
}

macro_rules! field_instruction {
    ($name:ident, $op:expr, $source_kind:expr) => {
        #[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
        pub struct $name {
            pub address: u64,
            pub operands: FormatFieldInline,
            pub virtual_sequence_remaining: Option<u16>,
            pub is_first_in_sequence: bool,
            pub is_compressed: bool,
        }

        impl RISCVInstruction for $name {
            const MASK: u32 = 0x0000_707f;
            const MATCH: u32 = (FIELD_INLINE_OPCODE as u32) | (($op.funct3() as u32) << 12);

            type Format = FormatFieldInline;
            type RAMAccess = FieldInlineCycleData;

            fn operands(&self) -> &Self::Format {
                &self.operands
            }

            fn source_kind(&self) -> SourceInstructionKind {
                $source_kind
            }

            fn new(word: u32, address: u64, _validate: bool, is_compressed: bool) -> Self {
                Self {
                    address,
                    operands: FormatFieldInline::parse(word),
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed,
                }
            }

            fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess) {
                ram_access.trace = Some(execute_field_inline($op, self.operands, cpu));
            }
        }

        impl RISCVTrace for $name {}

        impl From<super::SourceInstructionRow> for $name {
            fn from(row: super::SourceInstructionRow) -> Self {
                let mut operands = FormatFieldInline::from(row.operands);
                operands.op = Some($op);
                Self {
                    address: row.address as u64,
                    operands,
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: row.is_compressed,
                }
            }
        }
    };
}

field_instruction!(
    FIELD_ADD,
    FieldInlineOp::Add,
    SourceInstructionKind::FIELD_ADD
);
field_instruction!(
    FIELD_SUB,
    FieldInlineOp::Sub,
    SourceInstructionKind::FIELD_SUB
);
field_instruction!(
    FIELD_MUL,
    FieldInlineOp::Mul,
    SourceInstructionKind::FIELD_MUL
);
field_instruction!(
    FIELD_INV,
    FieldInlineOp::Inv,
    SourceInstructionKind::FIELD_INV
);
field_instruction!(
    FIELD_ASSERT_EQ,
    FieldInlineOp::AssertEq,
    SourceInstructionKind::FIELD_ASSERT_EQ
);
field_instruction!(
    FIELD_LOAD_FROM_X,
    FieldInlineOp::LoadFromX,
    SourceInstructionKind::FIELD_LOAD_FROM_X
);
field_instruction!(
    FIELD_STORE_TO_X,
    FieldInlineOp::StoreToX,
    SourceInstructionKind::FIELD_STORE_TO_X
);
field_instruction!(
    FIELD_LOAD_IMM,
    FieldInlineOp::LoadImm,
    SourceInstructionKind::FIELD_LOAD_IMM
);

fn execute_field_inline(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    match op {
        FieldInlineOp::Add => execute_binary(op, operands, cpu, |left, right| left + right),
        FieldInlineOp::Sub => execute_binary(op, operands, cpu, |left, right| left - right),
        FieldInlineOp::Mul => {
            let mut trace = execute_binary(op, operands, cpu, |left, right| left * right);
            trace.product = trace.rd.map(|write| write.post_value);
            trace
        }
        FieldInlineOp::Inv => execute_inverse(op, operands, cpu),
        FieldInlineOp::AssertEq => execute_assert_eq(op, operands, cpu),
        FieldInlineOp::LoadFromX => execute_load_from_x(op, operands, cpu),
        FieldInlineOp::StoreToX => execute_store_to_x(op, operands, cpu),
        FieldInlineOp::LoadImm => execute_load_imm(op, operands, cpu),
    }
}

fn execute_binary(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
    f: impl FnOnce(Fr, Fr) -> Fr,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rs2_register = operands.rs2.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let rs2_value = cpu.field_registers.read(rs2_register);
    let pre_value = cpu.field_registers.read(rd_register);
    let post_value = encode_field(f(decode_field(rs1_value), decode_field(rs2_value)));
    cpu.field_registers.write(rd_register, post_value);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rs2: Some(FieldRegisterRead {
            register: rs2_register,
            value: rs2_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value,
        }),
        ..Default::default()
    }
}

fn execute_inverse(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let pre_value = cpu.field_registers.read(rd_register);
    let inverse = decode_field(rs1_value)
        .inverse()
        .unwrap_or_else(|| Fr::from(0u64));
    let post_value = encode_field(inverse);
    cpu.field_registers.write(rd_register, post_value);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value,
        }),
        inv_product: Some(encode_field(decode_field(rs1_value) * inverse)),
        ..Default::default()
    }
}

fn execute_assert_eq(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rs2_register = operands.rs2.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let rs2_value = cpu.field_registers.read(rs2_register);
    assert_eq!(
        decode_field(rs1_value),
        decode_field(rs2_value),
        "field-inline assert_eq failed"
    );
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rs2: Some(FieldRegisterRead {
            register: rs2_register,
            value: rs2_value,
        }),
        ..Default::default()
    }
}

fn execute_load_from_x(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let x_register = operands.rs1.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let x_value = cpu.read_register(x_register) as u64;
    let field_value = encode_field(Fr::from(x_value));
    let pre_value = cpu.field_registers.read(rd_register);
    cpu.field_registers.write(rd_register, field_value);
    FieldInlineTraceData {
        op: Some(op),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value: field_value,
        }),
        bridge: Some(FieldInlineBridge::LoadFromX {
            x_register,
            x_value,
            field_value,
        }),
        ..Default::default()
    }
}

fn execute_store_to_x(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let field_register = operands.rs1.unwrap_or(0);
    let x_register = operands.rd.unwrap_or(0);
    let field_value = cpu.field_registers.read(field_register);
    let x_value = decode_field(field_value)
        .to_canonical_u64_checked()
        .unwrap_or_else(|| {
            u64::from_le_bytes(field_value.bytes_le[..8].try_into().unwrap_or([0; 8]))
        });
    cpu.write_register(x_register as usize, x_value as i64);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: field_register,
            value: field_value,
        }),
        bridge: Some(FieldInlineBridge::StoreToX {
            field_register,
            field_value,
            x_register,
            x_value,
        }),
        ..Default::default()
    }
}

fn execute_load_imm(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rd_register = operands.rd.unwrap_or(0);
    let value = u64::try_from(operands.imm)
        .ok()
        .map_or_else(FieldEncodedValue::zero, FieldEncodedValue::from_u64);
    let pre_value = cpu.field_registers.read(rd_register);
    cpu.field_registers.write(rd_register, value);
    FieldInlineTraceData {
        op: Some(op),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value: value,
        }),
        ..Default::default()
    }
}

fn decode_field(value: FieldEncodedValue) -> Fr {
    <Fr as ReducingBytes>::from_le_bytes_mod_order(&value.bytes_le)
}

fn encode_field(value: Fr) -> FieldEncodedValue {
    let mut bytes_le = [0u8; 32];
    value.to_bytes_le(&mut bytes_le);
    FieldEncodedValue { bytes_le }
}
