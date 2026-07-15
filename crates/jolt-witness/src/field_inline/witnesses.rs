//! Atomic witnesses of the field-inline protocol: one newtype per witness
//! with its single-sourced derivation from the row's field-inline payload —
//! the jolt-vm pattern in miniature.
//!
//! Field-inline witness values are decoded field elements, so the newtypes
//! carry `F` and the value accessor is [`FieldValue`] (the analog of the
//! scalar witnesses' `ToField`). Rows without a field-inline payload
//! extract to zero / false.

use jolt_claims::protocols::field_inline::FieldInlineOpFlag;
use jolt_field::{Field, ReducingBytes};
use jolt_program::{execution::TraceRow, field_inline::FieldEncodedValue};
use jolt_riscv::FieldInlineOp;

use crate::witnesses::{Extract, ExtractIndexed, WitnessEnv};
use crate::WitnessError;

/// Decoded field value read from field-register rs1; zero when absent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldRs1Value<F>(pub F);

/// Decoded field value read from field-register rs2; zero when absent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldRs2Value<F>(pub F);

/// Decoded field value written to field-register rd; zero when absent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldRdValue<F>(pub F);

/// Product of the decoded rs1 and rs2 values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldProduct<F>(pub F);

/// Product of the decoded rs1 value and the decoded rd post-value (the
/// inverse relation's constraint input).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldInvProduct<F>(pub F);

/// Whether the row performs the field-inline op bound at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldOpFlag(pub bool);

/// Signed field delta written to field-register rd; zero when absent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldRdInc<F>(pub F);

/// Unwraps an atomic field-inline witness into its field value — the
/// oracle-table boundary, like the scalar witnesses' `ToField`.
pub trait FieldValue<F> {
    fn value(self) -> F;
}

macro_rules! field_value {
    ($($name:ident),* $(,)?) => {
        $(impl<F> FieldValue<F> for $name<F> {
            fn value(self) -> F {
                self.0
            }
        })*
    };
}

field_value!(
    FieldRs1Value,
    FieldRs2Value,
    FieldRdValue,
    FieldProduct,
    FieldInvProduct,
    FieldRdInc,
);

impl<F: Field> FieldValue<F> for FieldOpFlag {
    fn value(self) -> F {
        F::from_bool(self.0)
    }
}

impl<F: Field> Extract for FieldRs1Value<F> {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.field_inline.as_deref().map_or_else(F::zero, {
            |data| {
                data.rs1
                    .map_or_else(F::zero, |read| decode_value(read.value))
            }
        })))
    }
}

impl<F: Field> Extract for FieldRs2Value<F> {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.field_inline.as_deref().map_or_else(F::zero, {
            |data| {
                data.rs2
                    .map_or_else(F::zero, |read| decode_value(read.value))
            }
        })))
    }
}

impl<F: Field> Extract for FieldRdValue<F> {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.field_inline.as_deref().map_or_else(F::zero, {
            |data| {
                data.rd
                    .map_or_else(F::zero, |write| decode_value(write.post_value))
            }
        })))
    }
}

impl<F: Field> Extract for FieldProduct<F> {
    fn extract(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let rs1 = FieldRs1Value::<F>::extract(row, next, env)?.0;
        let rs2 = FieldRs2Value::<F>::extract(row, next, env)?.0;
        Ok(Self(rs1 * rs2))
    }
}

impl<F: Field> Extract for FieldInvProduct<F> {
    fn extract(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let rs1 = FieldRs1Value::<F>::extract(row, next, env)?.0;
        let rd = FieldRdValue::<F>::extract(row, next, env)?.0;
        Ok(Self(rs1 * rd))
    }
}

impl ExtractIndexed<FieldInlineOpFlag> for FieldOpFlag {
    fn extract_indexed(
        flag: FieldInlineOpFlag,
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            row.field_inline
                .as_deref()
                .is_some_and(|data| data.op == Some(op(flag))),
        ))
    }
}

impl<F: Field> Extract for FieldRdInc<F> {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            row.field_inline
                .as_deref()
                .and_then(|data| data.rd)
                .map_or_else(F::zero, |write| {
                    decode_value::<F>(write.post_value) - decode_value::<F>(write.pre_value)
                }),
        ))
    }
}

pub(crate) fn decode_value<F: Field>(value: FieldEncodedValue) -> F {
    if value.bytes_le[8..].iter().all(|byte| *byte == 0) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&value.bytes_le[..8]);
        return F::from_u64(u64::from_le_bytes(bytes));
    }
    <F as ReducingBytes>::from_le_bytes_mod_order(&value.bytes_le)
}

pub(crate) const fn op(flag: FieldInlineOpFlag) -> FieldInlineOp {
    match flag {
        FieldInlineOpFlag::Add => FieldInlineOp::Add,
        FieldInlineOpFlag::Sub => FieldInlineOp::Sub,
        FieldInlineOpFlag::Mul => FieldInlineOp::Mul,
        FieldInlineOpFlag::Inv => FieldInlineOp::Inv,
        FieldInlineOpFlag::AssertEq => FieldInlineOp::AssertEq,
        FieldInlineOpFlag::LoadFromX => FieldInlineOp::LoadFromX,
        FieldInlineOpFlag::StoreToX => FieldInlineOp::StoreToX,
        FieldInlineOpFlag::LoadImm => FieldInlineOp::LoadImm,
    }
}
