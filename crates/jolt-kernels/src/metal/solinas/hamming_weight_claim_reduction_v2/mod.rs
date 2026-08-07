//! Promotion control for the retained-hot Hamming-weight kernel.
//!
//! The accepted-row and retained-hot implementations compute the same 7,424
//! recentered masses. This slice fixes the retained ABI, proves the traffic
//! distinction, and rejects stale producer leases before the existing Metal
//! implementation is selected. It deliberately contains no replacement
//! shader: the retained implementation already has exact production evidence.

mod abi;
pub mod model;

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle;

pub use abi::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HammingWeightV2Error {
    #[error("Hamming v2 rows must be a power of two at least 2^15, got {0}")]
    InvalidRows(usize),
    #[error("Hamming v2 {name} value {value} does not fit the shader ABI")]
    ShaderIndexOverflow { name: &'static str, value: usize },
    #[error("Hamming v2 size arithmetic overflow")]
    ArithmeticOverflow,
    #[error("Hamming v2 {name} identity must be nonzero")]
    MissingIdentity { name: &'static str },
    #[error("Hamming v2 hot and source allocations must not alias")]
    AliasedAllocations,
    #[error("Hamming v2 proof generation must be nonzero")]
    MissingGeneration,
    #[error("Hamming v2 lease has {got} rows, expected {expected}")]
    LeaseRows { expected: u64, got: u64 },
    #[error("Hamming v2 lease has {got} hot bytes, expected {expected}")]
    LeaseBytes { expected: u64, got: u64 },
    #[error("Hamming v2 selector schedule version is {got}, expected {expected}")]
    SelectorSchedule { expected: u32, got: u32 },
    #[error("Hamming v2 producer command did not complete")]
    ProducerIncomplete,
    #[error("Hamming v2 producer did not completely overwrite the hot allocation")]
    IncompleteOverwrite,
    #[error("Hamming v2 producer performed {0} private projection dispatches")]
    PrivateProjectionDispatches(u32),
    #[error("Hamming v2 producer uploaded {0} row bytes")]
    RowUpload(u64),
    #[error("Hamming v2 {name} is {got}, expected {expected}")]
    ReceiptMismatch {
        name: &'static str,
        expected: u64,
        got: u64,
    },
    #[error("Hamming v2 consumer command did not complete")]
    ConsumerIncomplete,
    #[error("Hamming v2 consumer GPU-active duration is zero")]
    MissingGpuTimestamp,
    #[error("Hamming v2 campaign needs five alternating pairs, got {0}")]
    CampaignLength(usize),
    #[error("Hamming v2 campaign pair {index} has the wrong order")]
    CampaignOrder { index: usize },
    #[error("Hamming v2 campaign pair {index} failed {guard}")]
    CampaignGuard { index: usize, guard: &'static str },
    #[error("Hamming v2 campaign pair {index} does not clear the 5x floor")]
    PairBelowFloor { index: usize },
    #[error("Hamming v2 campaign does not clear the 5.3x robust bar in {0}")]
    CampaignBelowRobustBar(&'static str),
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests;
