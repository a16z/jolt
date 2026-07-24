use jolt_field::Field;
use jolt_program::execution::{RamAccess, TraceRow};

use super::{Extract, ToField, WitnessEnv};
use crate::WitnessError;

/// Signed delta written to rd this cycle; 0 when the instruction has no rd
/// operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdInc(pub i128);

/// Signed delta written to RAM this cycle; 0 for reads and no-ops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamInc(pub i128);

impl ToField for RdInc {
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for RdInc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(match row.registers.rd {
            Some(write) => write.post_value as i128 - write.pre_value as i128,
            None => 0,
        }))
    }
}

impl ToField for RamInc {
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for RamInc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(match row.ram_access {
            RamAccess::Write(write) => write.post_value as i128 - write.pre_value as i128,
            RamAccess::Read(_) | RamAccess::NoOp => 0,
        }))
    }
}
