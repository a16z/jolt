use jolt_field::JoltField;
use jolt_riscv::{CircuitFlags, JoltTraceRow as TraceRow};

use super::{Extract, ToField, WitnessEnv};
use crate::WitnessError;

/// The row's incoming implicit carry (the previous row's carry-out): the
/// committed `Carry` column. Zero on padding rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Carry(pub u64);

/// [`Carry`] of the successor row, i.e. this row's carry-out; 0 at the last
/// cycle, where the shift relation requires the next-value to vanish.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextCarry(pub u64);

/// The carry the row actually consumes: [`Carry`] on `UsesCarry` rows
/// (`ADDC`, `MULC`), 0 elsewhere. Product-virtualized as
/// `OpFlags(UsesCarry) · Carry`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CarryUsed(pub u64);

macro_rules! u64_to_field {
    ($($name:ident),* $(,)?) => {
        $(impl ToField for $name {
            fn to_field<F: JoltField>(self) -> F {
                F::from_u64(self.0)
            }
        })*
    };
}

u64_to_field!(Carry, NextCarry, CarryUsed);

impl Extract for Carry {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.carry()))
    }
}

impl Extract for NextCarry {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.map_or(0, TraceRow::carry)))
    }
}

impl Extract for CarryUsed {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(if row.circuit_flags()[CircuitFlags::UsesCarry] {
            row.carry()
        } else {
            0
        }))
    }
}
