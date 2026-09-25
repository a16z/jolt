//! Program and trace artifacts for field-inline execution.
//!
//! These types describe row shape and encoded values at the program boundary.
//! They intentionally avoid importing proving-field types; conversion into a
//! concrete field belongs to witness generation.

#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_riscv::{field_inline_operand_shape, FieldInlineOp, FieldRegister, JoltInstructionRow};

/// A field element in canonical little-endian bytes.
///
/// The buffer is 32 bytes under every proof field. Narrower fields occupy
/// the low bytes and leave the remainder zero, so trace consumers can decode
/// the same fixed-size register values using their concrete proof field.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct FieldEncodedValue {
    pub bytes_le: [u8; 32],
}

impl FieldEncodedValue {
    pub const BYTE_LEN: u16 = 32;

    pub const fn zero() -> Self {
        Self { bytes_le: [0; 32] }
    }

    pub fn from_u64(value: u64) -> Self {
        let mut bytes_le = [0u8; 32];
        bytes_le[..8].copy_from_slice(&value.to_le_bytes());
        Self { bytes_le }
    }
}

/// Validate field and integer operand roles at the ordinary bytecode boundary.
pub fn validate_field_inline_instruction(
    row: &JoltInstructionRow,
) -> Result<(), FieldInlineInstructionError> {
    let Some(shape) = field_inline_operand_shape(row.instruction_kind) else {
        return Ok(());
    };
    let field_rd_name = if shape.field_rd_in_rs2_slot {
        "rs2"
    } else {
        "rd"
    };
    let field_operands = row.field_operands();
    if shape.writes_field_rd {
        validate_field_register(field_operands.rd, field_rd_name)?;
    }
    if shape.reads_field_rs1 {
        let operand = if shape.field_rs1_is_field_rd {
            field_rd_name
        } else {
            "rs1"
        };
        validate_field_register(field_operands.rs1, operand)?;
    }
    if shape.reads_field_rs2 {
        validate_field_register(field_operands.rs2, "rs2")?;
    }
    let integer_operands = row.integer_operands();
    match shape.op {
        FieldInlineOp::LoadAccumulateFromRegister => {
            let _ = x_register(integer_operands.rs1, "rs1")?;
        }
        FieldInlineOp::LoadAccumulateFromMemory | FieldInlineOp::AdviceLimb => {
            if x_register(integer_operands.rd, "rd")? == 0 {
                return Err(FieldInlineInstructionError::ZeroWriteRegister);
            }
            if shape.op == FieldInlineOp::LoadAccumulateFromMemory {
                let _ = x_register(integer_operands.rs1, "rs1")?;
            }
        }
        FieldInlineOp::Add
        | FieldInlineOp::Sub
        | FieldInlineOp::Mul
        | FieldInlineOp::Inv
        | FieldInlineOp::AssertEq
        | FieldInlineOp::AssertZero
        | FieldInlineOp::LoadImm => {}
    }
    let reads_rs1 =
        (shape.reads_field_rs1 && !shape.field_rs1_is_field_rd) || integer_operands.rs1.is_some();
    let reads_rs2 = shape.reads_field_rs2 || shape.field_rd_in_rs2_slot;
    let writes_rd =
        (shape.writes_field_rd && !shape.field_rd_in_rs2_slot) || integer_operands.rd.is_some();
    if row.operands.rs1.is_some() != reads_rs1
        || row.operands.rs2.is_some() != reads_rs2
        || row.operands.rd.is_some() != writes_rd
    {
        return Err(FieldInlineInstructionError::OperandShapeMismatch { op: shape.op });
    }
    if shape.has_immediate {
        let _ = u64::try_from(row.operands.imm)
            .map_err(|_| FieldInlineInstructionError::InvalidImmediate(row.operands.imm))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct FieldRegisterRead {
    pub register: u8,
    pub value: FieldEncodedValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct FieldRegisterWrite {
    pub register: u8,
    pub pre_value: FieldEncodedValue,
    pub post_value: FieldEncodedValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum FieldInlineBridge {
    LoadAccumulateFromRegister {
        x_register: u8,
        x_value: u64,
        field_value: FieldEncodedValue,
    },
    AdviceLimb {
        field_register: u8,
        field_value: FieldEncodedValue,
        x_register: u8,
        x_value: u64,
    },
    /// The word at `x[x_base] + offset` is written to integer register
    /// `x_register` and accumulated into the field destination. The integer
    /// write lets the ordinary RV64 load constraints bind the memory value.
    LoadAccumulateFromMemory {
        x_base: u8,
        x_register: u8,
        word: u64,
        field_value: FieldEncodedValue,
    },
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct FieldInlineTraceData {
    pub op: Option<FieldInlineOp>,
    pub rs1: Option<FieldRegisterRead>,
    pub rs2: Option<FieldRegisterRead>,
    pub rd: Option<FieldRegisterWrite>,
    pub bridge: Option<FieldInlineBridge>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FieldInlineInstructionError {
    #[error("field-inline row does not match operand shape for {op:?}")]
    OperandShapeMismatch { op: FieldInlineOp },
    #[error("field-inline row is missing {operand}")]
    MissingOperand { operand: &'static str },
    #[error("field-inline field register operand {operand} is out of bounds: {register}")]
    InvalidFieldRegister { operand: &'static str, register: u8 },
    #[error("field-inline x-register operand {operand} is out of bounds: {register}")]
    InvalidXRegister { operand: &'static str, register: u8 },
    #[error("field-inline write bridge targets x0, which discards the write")]
    ZeroWriteRegister,
    #[error("field-inline immediate must be non-negative and fit in u64: {0}")]
    InvalidImmediate(i128),
}

fn validate_field_register(
    register: Option<u8>,
    operand: &'static str,
) -> Result<(), FieldInlineInstructionError> {
    let register = register.ok_or(FieldInlineInstructionError::MissingOperand { operand })?;
    FieldRegister::new(register)
        .map(|_| ())
        .ok_or(FieldInlineInstructionError::InvalidFieldRegister { operand, register })
}

fn x_register(
    register: Option<u8>,
    operand: &'static str,
) -> Result<u8, FieldInlineInstructionError> {
    let register = register.ok_or(FieldInlineInstructionError::MissingOperand { operand })?;
    if register < common::constants::RISCV_REGISTER_COUNT {
        Ok(register)
    } else {
        Err(FieldInlineInstructionError::InvalidXRegister { operand, register })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_riscv::{JoltInstructionKind as Kind, NormalizedOperands};

    #[test]
    fn field_write_bridges_require_nonzero_integer_destinations() {
        for instruction_kind in [
            Kind::FIELD_LOAD_ACCUMULATE_FROM_MEMORY,
            Kind::FIELD_ADVICE_LIMB,
        ] {
            let mut row = JoltInstructionRow {
                instruction_kind,
                operands: NormalizedOperands {
                    rd: Some(0),
                    rs1: Some(1),
                    rs2: Some(2),
                    imm: 0,
                },
                ..Default::default()
            };
            assert!(validate_field_inline_instruction(&row).is_err());
            row.operands.rd = Some(3);
            assert!(validate_field_inline_instruction(&row).is_ok());
        }
    }

    #[test]
    fn field_operand_shape_rejects_unused_slots() {
        let mut row = JoltInstructionRow {
            instruction_kind: Kind::FIELD_ASSERT_ZERO,
            operands: NormalizedOperands {
                rs1: Some(3),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(validate_field_inline_instruction(&row).is_ok());
        row.operands.rd = Some(1);
        assert_eq!(
            validate_field_inline_instruction(&row),
            Err(FieldInlineInstructionError::OperandShapeMismatch {
                op: FieldInlineOp::AssertZero
            })
        );
    }
}
