//! Program and trace artifacts for field-inline execution.
//!
//! These types describe row shape and encoded values at the program boundary.
//! They intentionally avoid importing proving-field types; conversion into a
//! concrete field belongs to witness generation.

#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_riscv::{
    field_inline_operand_shape, FieldInlineOp, FieldInlineXRegisterRole, FieldRegister,
    JoltInstructionRow, FIELD_REGISTER_LOG_K,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct FieldValueEncoding {
    pub byte_len: u16,
    pub limb_bits: u16,
    pub limb_count: u16,
    pub canonical: bool,
}

impl FieldValueEncoding {
    pub const BN254_SCALAR_CANONICAL: Self = Self {
        byte_len: FieldEncodedValue::BYTE_LEN,
        limb_bits: 64,
        limb_count: 4,
        canonical: true,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct FieldInlineBytecodeMetadata {
    pub rows: Vec<FieldInlineBytecodeRow>,
    pub field_register_log_k: u8,
    pub value_encoding: FieldValueEncoding,
    pub profile_fingerprint: u64,
}

impl FieldInlineBytecodeMetadata {
    pub fn from_bytecode(
        bytecode: &[JoltInstructionRow],
        profile_fingerprint: u64,
    ) -> Result<Self, FieldInlineMetadataError> {
        let mut rows = Vec::with_capacity(bytecode.len());
        for row in bytecode {
            rows.push(FieldInlineBytecodeRow::from_instruction(row)?);
        }
        let metadata = Self {
            rows,
            field_register_log_k: FIELD_REGISTER_LOG_K,
            value_encoding: FieldValueEncoding::BN254_SCALAR_CANONICAL,
            profile_fingerprint,
        };
        metadata.validate(bytecode.len())?;
        Ok(metadata)
    }

    pub fn validate(&self, expected_len: usize) -> Result<(), FieldInlineMetadataError> {
        if self.rows.len() != expected_len {
            return Err(FieldInlineMetadataError::LengthMismatch {
                expected: expected_len,
                actual: self.rows.len(),
            });
        }
        if self.field_register_log_k != FIELD_REGISTER_LOG_K {
            return Err(FieldInlineMetadataError::InvalidFieldRegisterLogK {
                log_k: self.field_register_log_k,
            });
        }
        if self.value_encoding.byte_len != FieldEncodedValue::BYTE_LEN
            || self.value_encoding.limb_bits != 64
            || self.value_encoding.limb_count != 4
            || !self.value_encoding.canonical
        {
            return Err(FieldInlineMetadataError::InvalidValueEncoding(
                self.value_encoding,
            ));
        }
        for (index, row) in self.rows.iter().enumerate() {
            row.validate(index, self.field_register_log_k)?;
        }
        Ok(())
    }
}

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
pub struct FieldInlineBytecodeRow {
    pub active: bool,
    pub op: Option<FieldInlineOp>,
    pub rs1: Option<FieldRegister>,
    pub rs2: Option<FieldRegister>,
    pub rd: Option<FieldRegister>,
    pub bridge_x_register: Option<u8>,
    pub immediate: Option<FieldEncodedValue>,
}

impl FieldInlineBytecodeRow {
    pub fn from_instruction(row: &JoltInstructionRow) -> Result<Self, FieldInlineMetadataError> {
        let Some(shape) = field_inline_operand_shape(row.instruction_kind) else {
            return Ok(Self::default());
        };
        let rs1 = if shape.reads_fr_rs1 {
            Some(field_register(row.operands.rs1, "rs1")?)
        } else {
            None
        };
        let rs2 = if shape.reads_fr_rs2 {
            Some(field_register(row.operands.rs2, "rs2")?)
        } else {
            None
        };
        let rd = if shape.writes_fr_rd {
            Some(field_register(row.operands.rd, "rd")?)
        } else {
            None
        };
        let bridge_x_register = match shape.bridge_x_register_role {
            Some(FieldInlineXRegisterRole::ReadRs1) => Some(x_register(row.operands.rs1, "rs1")?),
            Some(FieldInlineXRegisterRole::WriteRd) => Some(x_register(row.operands.rd, "rd")?),
            None => None,
        };
        let immediate = if shape.has_immediate {
            Some(encoded_immediate(row.operands.imm)?)
        } else {
            None
        };
        Ok(Self {
            active: true,
            op: Some(shape.op),
            rs1,
            rs2,
            rd,
            bridge_x_register,
            immediate,
        })
    }

    fn validate(&self, index: usize, log_k: u8) -> Result<(), FieldInlineMetadataError> {
        if !self.active {
            if self.op.is_some()
                || self.rs1.is_some()
                || self.rs2.is_some()
                || self.rd.is_some()
                || self.bridge_x_register.is_some()
                || self.immediate.is_some()
            {
                return Err(FieldInlineMetadataError::InactiveRowHasData { index });
            }
            return Ok(());
        }
        let Some(op) = self.op else {
            return Err(FieldInlineMetadataError::ActiveRowMissingOp { index });
        };
        let max_register = 1u8
            .checked_shl(u32::from(log_k))
            .ok_or(FieldInlineMetadataError::InvalidFieldRegisterLogK { log_k })?;
        for register in [self.rs1, self.rs2, self.rd].into_iter().flatten() {
            if register.index() >= max_register {
                return Err(FieldInlineMetadataError::FieldRegisterOutOfBounds {
                    index,
                    register: register.index(),
                    log_k,
                });
            }
        }
        let expected = shape_for_op(op);
        if expected.reads_fr_rs1 != self.rs1.is_some()
            || expected.reads_fr_rs2 != self.rs2.is_some()
            || expected.writes_fr_rd != self.rd.is_some()
            || expected.bridge_x_register_role.is_some() != self.bridge_x_register.is_some()
            || expected.has_immediate != self.immediate.is_some()
        {
            return Err(FieldInlineMetadataError::OperandShapeMismatch { index, op });
        }
        Ok(())
    }
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
    LoadFromX {
        x_register: u8,
        x_value: u64,
        field_value: FieldEncodedValue,
    },
    StoreToX {
        field_register: u8,
        field_value: FieldEncodedValue,
        x_register: u8,
        x_value: u64,
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
    pub product: Option<FieldEncodedValue>,
    pub inv_product: Option<FieldEncodedValue>,
    pub bridge: Option<FieldInlineBridge>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FieldInlineMetadataError {
    #[error("field-inline metadata length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("invalid field-register log_k in field-inline metadata: {log_k}")]
    InvalidFieldRegisterLogK { log_k: u8 },
    #[error("invalid field value encoding in field-inline metadata: {0:?}")]
    InvalidValueEncoding(FieldValueEncoding),
    #[error("field-inline inactive metadata row {index} carries data")]
    InactiveRowHasData { index: usize },
    #[error("field-inline active metadata row {index} is missing its op")]
    ActiveRowMissingOp { index: usize },
    #[error(
        "field-inline metadata row {index} has field register {register} outside log_k {log_k}"
    )]
    FieldRegisterOutOfBounds {
        index: usize,
        register: u8,
        log_k: u8,
    },
    #[error("field-inline metadata row {index} does not match operand shape for {op:?}")]
    OperandShapeMismatch { index: usize, op: FieldInlineOp },
    #[error("field-inline row is missing {operand}")]
    MissingOperand { operand: &'static str },
    #[error("field-inline field register operand {operand} is out of bounds: {register}")]
    InvalidFieldRegister { operand: &'static str, register: u8 },
    #[error("field-inline x-register operand {operand} is out of bounds: {register}")]
    InvalidXRegister { operand: &'static str, register: u8 },
    #[error("field-inline immediate must be non-negative and fit in u64: {0}")]
    InvalidImmediate(i128),
}

fn field_register(
    register: Option<u8>,
    operand: &'static str,
) -> Result<FieldRegister, FieldInlineMetadataError> {
    let register = register.ok_or(FieldInlineMetadataError::MissingOperand { operand })?;
    FieldRegister::new(register)
        .ok_or(FieldInlineMetadataError::InvalidFieldRegister { operand, register })
}

fn x_register(register: Option<u8>, operand: &'static str) -> Result<u8, FieldInlineMetadataError> {
    let register = register.ok_or(FieldInlineMetadataError::MissingOperand { operand })?;
    if register < common::constants::RISCV_REGISTER_COUNT {
        Ok(register)
    } else {
        Err(FieldInlineMetadataError::InvalidXRegister { operand, register })
    }
}

fn encoded_immediate(value: i128) -> Result<FieldEncodedValue, FieldInlineMetadataError> {
    let value =
        u64::try_from(value).map_err(|_| FieldInlineMetadataError::InvalidImmediate(value))?;
    Ok(FieldEncodedValue::from_u64(value))
}

fn shape_for_op(op: FieldInlineOp) -> jolt_riscv::FieldInlineOperandShape {
    match op {
        FieldInlineOp::Add | FieldInlineOp::Sub | FieldInlineOp::Mul => {
            jolt_riscv::FieldInlineOperandShape {
                op,
                reads_fr_rs1: true,
                reads_fr_rs2: true,
                writes_fr_rd: true,
                bridge_x_register_role: None,
                has_immediate: false,
            }
        }
        FieldInlineOp::Inv => jolt_riscv::FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: None,
            has_immediate: false,
        },
        FieldInlineOp::AssertEq => jolt_riscv::FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: true,
            writes_fr_rd: false,
            bridge_x_register_role: None,
            has_immediate: false,
        },
        FieldInlineOp::LoadFromX => jolt_riscv::FieldInlineOperandShape {
            op,
            reads_fr_rs1: false,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: Some(FieldInlineXRegisterRole::ReadRs1),
            has_immediate: false,
        },
        FieldInlineOp::StoreToX => jolt_riscv::FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: false,
            writes_fr_rd: false,
            bridge_x_register_role: Some(FieldInlineXRegisterRole::WriteRd),
            has_immediate: false,
        },
        FieldInlineOp::LoadImm => jolt_riscv::FieldInlineOperandShape {
            op,
            reads_fr_rs1: false,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: None,
            has_immediate: true,
        },
    }
}
