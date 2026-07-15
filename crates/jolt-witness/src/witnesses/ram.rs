use jolt_field::Field;
use jolt_program::execution::{RamAccess, TraceRow};

use super::{Extract, WitnessEnv};
use crate::WitnessError;

pub(crate) const fn ram_access_address(access: RamAccess) -> Option<u64> {
    match access {
        RamAccess::Read(read) => Some(read.address),
        RamAccess::Write(write) => Some(write.address),
        RamAccess::NoOp => None,
    }
}

/// Raw (unremapped) RAM access address; 0 when the cycle makes no RAM
/// access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamAddress(pub u64);

/// Pre-access RAM word value; 0 when the cycle makes no RAM access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamReadValue(pub u64);

/// Post-access RAM word value (equals [`RamReadValue`] for reads); 0 when the
/// cycle makes no RAM access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamWriteValue(pub u64);

/// Whether the cycle accesses a nonzero RAM address.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamHammingWeight(pub bool);

impl RamAddress {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RamAddress {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(ram_access_address(row.ram_access).unwrap_or(0)))
    }
}

impl RamReadValue {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RamReadValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.pre_value,
            RamAccess::NoOp => 0,
        }))
    }
}

impl RamWriteValue {
    pub fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RamWriteValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(match row.ram_access {
            RamAccess::Read(read) => read.value,
            RamAccess::Write(write) => write.post_value,
            RamAccess::NoOp => 0,
        }))
    }
}

impl RamHammingWeight {
    pub fn to_field<F: Field>(self) -> F {
        F::from_bool(self.0)
    }
}

impl Extract for RamHammingWeight {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            ram_access_address(row.ram_access).is_some_and(|address| address != 0),
        ))
    }
}
