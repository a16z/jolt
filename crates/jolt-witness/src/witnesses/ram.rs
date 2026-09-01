use jolt_field::JoltField;
use jolt_riscv::JoltTraceRow as TraceRow;

use super::{Extract, ToField, WitnessEnv};
use crate::WitnessError;

pub(crate) fn ram_access_address(row: &TraceRow) -> Option<u64> {
    (row.is_load() || row.is_store()).then(|| row.ram_address())
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

/// The cycle's RAM access address remapped to a word index; `None` for
/// no-ops and — the committed streams' convention — for unremappable
/// addresses (the grid materializers bound-check instead).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RemappedRamAddress(pub Option<u64>);

impl ToField for RamAddress {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RamAddress {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.ram_address()))
    }
}

impl ToField for RamReadValue {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RamReadValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.ram_read_value()))
    }
}

impl ToField for RamWriteValue {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RamWriteValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.ram_write_value()))
    }
}

impl ToField for RamHammingWeight {
    fn to_field<F: JoltField>(self) -> F {
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
            ram_access_address(row).is_some_and(|address| address != 0),
        ))
    }
}

impl Extract for RemappedRamAddress {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            ram_access_address(row)
                .and_then(|address| {
                    env.preprocessing
                        .memory_layout
                        .remap_word_address(address)
                        .ok()
                })
                .flatten(),
        ))
    }
}
