//! Instruction lookup-index derivation.
//!
//! Re-homed from `jolt-program` (whose `lookup` module was removed so the
//! program crate no longer depends on `jolt-lookup-tables`). The witness owns
//! this thin wrapper because it already builds `JoltLookupQuery`/`LookupQuery`
//! for the lookup argument.

use jolt_lookup_tables::{JoltLookupQuery, LookupQuery};
use jolt_program::execution::TraceRow;
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub(crate) enum LookupIndexError {
    #[error("unsupported XLEN for Jolt lookup index derivation: {xlen}")]
    UnsupportedXlen { xlen: usize },
}

pub(crate) fn instruction_lookup_index<const XLEN: usize>(
    row: &TraceRow,
) -> Result<u128, LookupIndexError> {
    if XLEN == 0 || XLEN > 64 {
        return Err(LookupIndexError::UnsupportedXlen { xlen: XLEN });
    }
    let query = JoltLookupQuery::new(row.instruction.instruction_kind, row);
    Ok(LookupQuery::<XLEN>::to_lookup_index(&query))
}
