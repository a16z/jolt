use jolt_field::Field;
use jolt_program::execution::TraceRow;

use super::{Extract, ToField, WitnessEnv};
use crate::WitnessError;

/// Value read from rs1; 0 when the instruction has no rs1 operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rs1Value(pub u64);

/// Value read from rs2; 0 when the instruction has no rs2 operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rs2Value(pub u64);

/// Value written to rd; 0 when the instruction has no rd operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdWriteValue(pub u64);

impl ToField for Rs1Value {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for Rs1Value {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rs1.map_or(0, |read| read.value)))
    }
}

impl ToField for Rs2Value {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for Rs2Value {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rs2.map_or(0, |read| read.value)))
    }
}

impl ToField for RdWriteValue {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RdWriteValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rd.map_or(0, |write| write.post_value)))
    }
}
