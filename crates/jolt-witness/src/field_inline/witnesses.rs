//! Atomic witnesses of the field-inline protocol: one newtype per witness
//! with its single-sourced derivation from the row's field-inline payload —
//! the jolt-vm pattern in miniature.
//!
//! Field-inline witness values are decoded field elements, so the newtypes
//! carry `F` and the value accessor is [`FieldValue`] (the analog of the
//! scalar witnesses' `ToField`). The trace view supplies zero for cycles
//! without a field-inline payload.

use jolt_field::{CanonicalEncoding, JoltField};
use jolt_program::field_inline::{FieldEncodedValue, FieldInlineTraceData};

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

impl<F: JoltField> Extract<FieldInlineTraceData> for FieldRs1Value<F> {
    fn extract(
        row: &FieldInlineTraceData,
        _next: Option<&FieldInlineTraceData>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            row.rs1
                .map_or_else(F::zero, |read| decode_value(read.value)),
        ))
    }
}

impl<F: JoltField> Extract<FieldInlineTraceData> for FieldRs2Value<F> {
    fn extract(
        row: &FieldInlineTraceData,
        _next: Option<&FieldInlineTraceData>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            row.rs2
                .map_or_else(F::zero, |read| decode_value(read.value)),
        ))
    }
}

impl<F: JoltField> Extract<FieldInlineTraceData> for FieldRdValue<F> {
    fn extract(
        row: &FieldInlineTraceData,
        _next: Option<&FieldInlineTraceData>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.rd.map_or_else(F::zero, |write| {
            decode_value(write.post_value)
        })))
    }
}

impl<F: JoltField> Extract<FieldInlineTraceData> for FieldProduct<F> {
    fn extract(
        row: &FieldInlineTraceData,
        next: Option<&FieldInlineTraceData>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let rs1 = FieldRs1Value::<F>::extract(row, next, env)?.0;
        let rs2 = FieldRs2Value::<F>::extract(row, next, env)?.0;
        Ok(Self(rs1 * rs2))
    }
}

impl<F: JoltField> Extract<FieldInlineTraceData> for FieldInvProduct<F> {
    fn extract(
        row: &FieldInlineTraceData,
        next: Option<&FieldInlineTraceData>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let rs1 = FieldRs1Value::<F>::extract(row, next, env)?.0;
        let rd = FieldRdValue::<F>::extract(row, next, env)?.0;
        Ok(Self(rs1 * rd))
    }
}

impl<F: JoltField> Extract<FieldInlineTraceData> for FieldRdInc<F> {
    fn extract(
        row: &FieldInlineTraceData,
        _next: Option<&FieldInlineTraceData>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.rd.map_or_else(F::zero, |write| {
            decode_value::<F>(write.post_value) - decode_value::<F>(write.pre_value)
        })))
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
