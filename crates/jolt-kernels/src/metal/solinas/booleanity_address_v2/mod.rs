//! Source-only design packet for the log-27 Booleanity-address blocker.
//!
//! The candidate moves all optional selectors into the original-row pass,
//! making the retained hot projection sufficient without a validity plane.
//! The protocol boundary and Fiat--Shamir transcript remain on the host.

mod abi;
pub mod model;
pub mod oracle;

pub use abi::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BooleanityAddressV2Error {
    InvalidRows(usize),
    ShaderIndexOverflow {
        name: &'static str,
        value: usize,
    },
    ArithmeticOverflow,
    BufferLength {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    ReceiptMismatch {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    MissingIdentity(&'static str),
    AliasedAllocations,
    MissingGeneration,
    ProducerIncomplete,
    IncompleteOverwrite,
    CommandIncomplete,
    MissingGpuTimestamp,
    LifecycleMismatch {
        components_ns: u64,
        complete_member_ns: u64,
    },
    WeightShape {
        rows: usize,
        e_in: usize,
        e_out: usize,
    },
    RowStorageLength {
        expected: usize,
        got: usize,
    },
    HotStorageLength {
        expected: usize,
        got: usize,
    },
    RowOutOfBounds {
        rows: usize,
        row: usize,
    },
    InvalidSelector(usize),
    InvalidModulus(u64),
    InvalidCensus {
        name: &'static str,
        rows: u64,
        got: u64,
    },
    CampaignSize(usize),
    CampaignOrder(usize),
    CampaignEvidence(usize),
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "fixed test fixtures fail loudly")]
mod tests;
