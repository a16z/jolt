//! Atomic witnesses of the field-inline protocol: one newtype per witness
//! with its single-sourced derivation from the row's field-inline payload —
//! the jolt-vm pattern in miniature.
//!
//! Field-inline witness values are decoded field elements, so the newtypes
//! carry `F` and the value accessor is [`FieldValue`] (the analog of the
//! scalar witnesses' `ToField`). Rows without a field-inline payload
//! extract to zero / false.

use jolt_field::{CanonicalEncoding, JoltField};
use jolt_program::{execution::TraceRow, field_inline::FieldEncodedValue};

use crate::witnesses::{Extract, WitnessEnv};
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

impl<F: JoltField> Extract<TraceRow> for FieldRs1Value<F> {
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

impl<F: JoltField> Extract<TraceRow> for FieldRs2Value<F> {
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

impl<F: JoltField> Extract<TraceRow> for FieldRdValue<F> {
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

impl<F: JoltField> Extract<TraceRow> for FieldProduct<F> {
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

impl<F: JoltField> Extract<TraceRow> for FieldInvProduct<F> {
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

impl<F: JoltField> Extract<TraceRow> for FieldRdInc<F> {
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

pub(crate) fn decode_value<F: JoltField>(value: FieldEncodedValue) -> F {
    if value.bytes_le[8..].iter().all(|byte| *byte == 0) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&value.bytes_le[..8]);
        return F::from_u64(u64::from_le_bytes(bytes));
    }
    <F as CanonicalEncoding>::from_bytes_le_reduced(&value.bytes_le)
}
