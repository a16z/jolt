//! Program and trace artifacts for field-inline execution.
//!
//! These types describe row shape and encoded values at the program boundary.
//! They intentionally avoid importing proving-field types; conversion into a
//! concrete field belongs to witness generation.

#[cfg(feature = "serialization")]
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
};
use jolt_riscv::{
    field_inline_operand_shape, FieldInlineOp, FieldInlineXRegisterRole, FieldRegister,
    JoltInstructionRow, FIELD_REGISTER_LOG_K,
};

/// A field element in canonical little-endian bytes.
///
/// The buffer is 32 bytes under every encoding: a narrower field (e.g.
/// [`FieldValueEncoding::TWO_LIMB_128_CANONICAL`]) occupies the low
/// `byte_len` bytes and leaves the rest zero. Sizing the buffer per encoding
/// would push a width parameter through the register file, trace rows, and
/// bytecode metadata for no versioning benefit; the encoding (plus the
/// profile fingerprint) already stamps preprocessing artifacts, so the valid
/// width is tagged by [`FieldValueEncoding::byte_len`] instead.
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

    /// 128-bit canonical two-limb encoding: two 64-bit limbs in the low 16
    /// bytes of the value buffer, upper 16 bytes zero. Active under
    /// `fp128-field-inline` (the packed/akita configuration's proof field).
    pub const TWO_LIMB_128_CANONICAL: Self = Self {
        byte_len: 16,
        limb_bits: 64,
        limb_count: 2,
        canonical: true,
    };

    /// The encoding this build emits and accepts. Metadata carrying any other
    /// encoding fails validation, so preprocessing built under a different
    /// proof field is rejected at load time rather than misdecoded during
    /// proving. The `fp128-field-inline` feature (the packed/akita chain)
    /// selects the two-limb encoding together with the tracer's `ProofField`
    /// alias; homomorphic (Dory) builds keep the BN254 form.
    #[cfg(not(feature = "fp128-field-inline"))]
    pub const ACTIVE: Self = Self::BN254_SCALAR_CANONICAL;
    #[cfg(feature = "fp128-field-inline")]
    pub const ACTIVE: Self = Self::TWO_LIMB_128_CANONICAL;
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(CanonicalSerialize, serde::Serialize, serde::Deserialize)
)]
pub struct FieldInlineBytecodeMetadata {
    pub rows: Vec<FieldInlineBytecodeRow>,
    pub field_register_log_k: u8,
    pub value_encoding: FieldValueEncoding,
    pub profile_fingerprint: u64,
}

#[cfg(feature = "serialization")]
impl Valid for FieldInlineBytecodeMetadata {
    fn check(&self) -> Result<(), SerializationError> {
        // Metadata loaded from disk would otherwise be trusted unvalidated. Re-run the
        // structural checks (register bounds, value encoding, per-row operand shape) so a
        // tampered artifact is rejected at deserialize time rather than during proving.
        self.validate(self.rows.len())
            .map_err(|_| SerializationError::InvalidData)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for FieldInlineBytecodeMetadata {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = Self {
            rows: Vec::<FieldInlineBytecodeRow>::deserialize_with_mode(
                &mut reader,
                compress,
                Validate::No,
            )?,
            field_register_log_k: u8::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            value_encoding: FieldValueEncoding::deserialize_with_mode(
                &mut reader,
                compress,
                Validate::No,
            )?,
            profile_fingerprint: u64::deserialize_with_mode(&mut reader, compress, Validate::No)?,
        };
        if let Validate::Yes = validate {
            value.check()?;
        }
        Ok(value)
    }
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
            value_encoding: FieldValueEncoding::ACTIVE,
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
        // Fail closed on any encoding other than the build's own, including
        // well-formed ones: row immediates only decode correctly under the
        // field ACTIVE describes, so metadata from a different-field build
        // must never reach proving.
        if self.value_encoding != FieldValueEncoding::ACTIVE {
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
            Some(FieldInlineXRegisterRole::WriteRd) => {
                // x0 discards writes, so the bridge row `RdWriteValue =
                // FieldRs1Value` could hold only for a zero field value; the
                // tracer traps on the same encoding, keeping the two in
                // agreement instead of leaving an honest trace unprovable.
                let register = x_register(row.operands.rd, "rd")?;
                if register == 0 {
                    return Err(FieldInlineMetadataError::StoreToXZeroRegister);
                }
                Some(register)
            }
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
        let expected = jolt_riscv::field_inline_operand_shape_for_op(op);
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
    #[error("field-inline store bridge targets x0, which discards the write")]
    StoreToXZeroRegister,
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

#[cfg(all(test, feature = "serialization"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

    fn roundtrip(
        metadata: &FieldInlineBytecodeMetadata,
        validate: Validate,
    ) -> Result<FieldInlineBytecodeMetadata, SerializationError> {
        let mut bytes = Vec::new();
        metadata
            .serialize_with_mode(&mut bytes, Compress::No)
            .unwrap();
        FieldInlineBytecodeMetadata::deserialize_with_mode(&bytes[..], Compress::No, validate)
    }

    #[test]
    fn metadata_roundtrips() {
        let metadata = FieldInlineBytecodeMetadata::from_bytecode(&[], 0).unwrap();
        assert_eq!(roundtrip(&metadata, Validate::Yes).unwrap(), metadata);
    }

    #[test]
    fn metadata_deserialize_reruns_validation() {
        // `from_bytecode` is the only validated constructor; a metadata assembled directly
        // with an invalid field-register width must still be rejected when deserialized.
        let tampered = FieldInlineBytecodeMetadata {
            rows: Vec::new(),
            field_register_log_k: FIELD_REGISTER_LOG_K + 1,
            value_encoding: FieldValueEncoding::ACTIVE,
            profile_fingerprint: 0,
        };
        assert!(roundtrip(&tampered, Validate::Yes).is_err());
        assert!(roundtrip(&tampered, Validate::No).is_ok());
    }

    fn metadata_with_encoding(value_encoding: FieldValueEncoding) -> FieldInlineBytecodeMetadata {
        FieldInlineBytecodeMetadata {
            rows: Vec::new(),
            field_register_log_k: FIELD_REGISTER_LOG_K,
            value_encoding,
            profile_fingerprint: 0,
        }
    }

    // The declared encoding this build is NOT on: the other side of the
    // ACTIVE equality gate, so the mismatch tests cover both directions
    // (BN254 rejects two-limb, and under fp128-field-inline vice versa).
    const FOREIGN: FieldValueEncoding = if cfg!(feature = "fp128-field-inline") {
        FieldValueEncoding::BN254_SCALAR_CANONICAL
    } else {
        FieldValueEncoding::TWO_LIMB_128_CANONICAL
    };

    #[test]
    fn metadata_with_foreign_encoding_rejects_fail_closed() {
        // A well-formed encoding that is not the build's own must reject: this
        // metadata decodes correctly only under the field it was built for.
        let foreign = metadata_with_encoding(FOREIGN);
        assert!(matches!(
            foreign.validate(0),
            Err(FieldInlineMetadataError::InvalidValueEncoding(encoding))
                if encoding == FOREIGN
        ));
        assert!(roundtrip(&foreign, Validate::Yes).is_err());
    }

    #[test]
    fn active_encoding_tracks_the_proof_field_feature() {
        let expected = if cfg!(feature = "fp128-field-inline") {
            FieldValueEncoding::TWO_LIMB_128_CANONICAL
        } else {
            FieldValueEncoding::BN254_SCALAR_CANONICAL
        };
        assert_eq!(FieldValueEncoding::ACTIVE, expected);
    }

    #[test]
    fn foreign_encoding_metadata_roundtrips_unvalidated() {
        // The other build's variant must survive the wire byte-faithfully so
        // that build can read back what it wrote.
        let metadata = metadata_with_encoding(FOREIGN);
        assert_eq!(roundtrip(&metadata, Validate::No).unwrap(), metadata);
    }

    #[test]
    fn declared_encodings_fit_the_value_buffer() {
        for encoding in [
            FieldValueEncoding::BN254_SCALAR_CANONICAL,
            FieldValueEncoding::TWO_LIMB_128_CANONICAL,
        ] {
            assert!(encoding.byte_len <= FieldEncodedValue::BYTE_LEN);
            assert_eq!(
                u32::from(encoding.byte_len) * 8,
                u32::from(encoding.limb_bits) * u32::from(encoding.limb_count)
            );
            assert!(encoding.canonical);
        }
    }
}
