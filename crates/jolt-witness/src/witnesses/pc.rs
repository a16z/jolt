use jolt_field::Field;
use jolt_riscv::JoltTraceRow as TraceRow;

use super::{row_is_noop, Extract, ToField, WitnessEnv};
use crate::WitnessError;

/// Bytecode PC with the read-RAF pushforward convention: no-op rows land on
/// slot 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BytecodePc(pub usize);

/// Bytecode PC for the committed one-hot convention (no-ops map to slot 0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MappedPc(pub Option<usize>);

impl Extract for BytecodePc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        if row_is_noop(row) {
            return Ok(Self(0));
        }
        Ok(Self(row.pc() as usize))
    }
}

impl Extract for MappedPc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(Some(row.pc() as usize)))
    }
}

/// Bytecode-expanded program counter (the preprocessing PC index, not the
/// instruction's memory address).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pc(pub u64);

/// The instruction's memory address (virtual-sequence entries share it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnexpandedPc(pub u64);

/// [`Pc`] of the successor row; 0 at the last cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextPc(pub u64);

/// [`UnexpandedPc`] of the successor row; 0 at the last cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextUnexpandedPc(pub u64);

impl ToField for Pc {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for Pc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.pc()))
    }
}

impl ToField for UnexpandedPc {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for UnexpandedPc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.unexpanded_pc()))
    }
}

impl ToField for NextPc {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for NextPc {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.map_or(0, TraceRow::pc)))
    }
}

impl ToField for NextUnexpandedPc {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for NextUnexpandedPc {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.map_or(0, TraceRow::unexpanded_pc)))
    }
}
