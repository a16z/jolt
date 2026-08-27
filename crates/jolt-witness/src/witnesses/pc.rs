use jolt_field::JoltField;
use jolt_riscv::JoltTraceRow as TraceRow;

use super::{Extract, ToField, WitnessEnv};
use crate::WitnessError;

/// The cycle's bytecode slot, for both the read-RAF pushforward and the
/// committed one-hot.
///
/// Total, and total on purpose: a row whose instruction has no bytecode
/// mapping cannot be materialized (`BytecodePreprocessing::get_pc` returning
/// `None` is a hard trace error), and every no-op already sits on slot 0, so
/// there is no cold cycle for this column to represent. The two conventions
/// this used to be split across — `BytecodePc`, which zeroed no-op rows, and
/// `MappedPc`, which did not — were the same value on every row. See
/// `jolt-program`'s `noop_maps_to_bytecode_slot_zero`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BytecodePc(pub usize);

impl Extract for BytecodePc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.pc() as usize))
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
    fn to_field<F: JoltField>(self) -> F {
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
    fn to_field<F: JoltField>(self) -> F {
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
    fn to_field<F: JoltField>(self) -> F {
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
    fn to_field<F: JoltField>(self) -> F {
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
