use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::{Extract, WitnessEnv};
use jolt_witness::WitnessError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdAddress(pub Option<u8>);

impl Extract for RdAddress {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.rd_index()))
    }
}
