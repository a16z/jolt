use jolt_field::Field;
use jolt_program::execution::TraceRow;

use super::{pc_for_row, row_is_noop, Extract, ToField, WitnessEnv};
use crate::WitnessError;

/// Bytecode PC with the read-RAF pushforward convention: no-op rows and rows
/// without a bytecode mapping land on slot 0 (unlike [`Pc`], which requires
/// the mapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BytecodePc(pub usize);

impl Extract for BytecodePc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        if row_is_noop(row) {
            return Ok(Self(0));
        }
        Ok(Self(
            env.preprocessing
                .bytecode
                .get_pc(&row.instruction)
                .unwrap_or(0),
        ))
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
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        pc_for_row(row, env.preprocessing).map(|pc| Self(pc as u64))
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
        Ok(Self(row.instruction.address as u64))
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
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            next.map(|row| pc_for_row(row, env.preprocessing))
                .transpose()?
                .map_or(0, |pc| pc as u64),
        ))
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
        Ok(Self(next.map_or(0, |row| row.instruction.address as u64)))
    }
}
