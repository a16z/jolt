use jolt_field::Field;
use jolt_lookup_tables::LookupQuery;
use jolt_program::execution::TraceRow;

use super::{lookup_query, Extract, ToField, WitnessEnv};
use crate::WitnessError;
use crate::RV64_XLEN;

/// Output of the instruction's lookup query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupOutput(pub u64);

impl ToField for LookupOutput {
    fn to_field<F: Field>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for LookupOutput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(LookupQuery::<RV64_XLEN>::to_lookup_output(
            &lookup_query(row),
        )))
    }
}
