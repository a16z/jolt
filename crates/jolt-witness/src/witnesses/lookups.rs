use jolt_field::Field;
use jolt_lookup_tables::{InstructionLookupTable, LookupQuery};
use jolt_program::execution::TraceRow;
use jolt_riscv::JoltInstruction;

use super::{decode_instruction, lookup_query, Extract, ToField, WitnessEnv};
use crate::WitnessError;
use crate::RV64_XLEN;

/// Output of the instruction's lookup query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupOutput(pub u64);

/// The instruction's 128-bit lookup index (its interleaved or concatenated
/// lookup operands).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupIndex(pub u128);

/// Which lookup table the instruction's lookup targets, if any.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableIndex(pub Option<usize>);

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

impl ToField for LookupIndex {
    fn to_field<F: Field>(self) -> F {
        F::from_u128(self.0)
    }
}

impl Extract for LookupIndex {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(LookupQuery::<RV64_XLEN>::to_lookup_index(
            &lookup_query(row),
        )))
    }
}

impl Extract for TableIndex {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let instruction = decode_instruction(row)?;
        Ok(Self(
            <JoltInstruction as InstructionLookupTable<RV64_XLEN>>::lookup_table(&instruction)
                .map(|kind| kind.index()),
        ))
    }
}
