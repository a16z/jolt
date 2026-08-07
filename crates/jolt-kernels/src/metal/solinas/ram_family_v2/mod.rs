//! Shared sparse RAM-family inputs and exact host-side reference evaluation.
//!
//! The owner keeps proof-relevant access rows and nonzero increments as
//! separate ordered streams. Its topology is the union of those streams and
//! records every low-to-high merge, including missing even or odd children.
//! The claim-reduction oracle deliberately ignores that topology and rebuilds
//! each round from the access records, giving future Metal kernels an
//! independent parity target.
//!
//! Integration must assign a stable source identity and generation, and must
//! certify that retained access rows are exactly the RAM Hamming support before
//! using Hamming-dependent relations. Device ABI and allocation ownership are
//! intentionally left to the backend integration slice.

#![expect(
    dead_code,
    unused_imports,
    reason = "the staged owner and oracle are not registered with production consumers yet"
)]

mod oracle;
mod owner;

pub(crate) use oracle::{
    ram_ra_claim_reduction, RamRaClaimOracleInputs, RamRaClaimOracleResult, RAM_RA_CLAIM_TERMS,
};
pub(crate) use owner::{
    HammingSupportCertificate, SparseCycleLeaf, SparseCycleNode, SparseCycleTopology,
    SparseRamAccess, SparseRamCertificates, SparseRamIncrement, SparseRamOwner,
    SparseRamProvenance, SparseRamSource,
};

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid fixtures")]
mod tests;

use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum RamFamilyV2Error {
    #[error("RAM-family source identity must be nonzero")]
    ZeroSourceIdentity,
    #[error("RAM-family source generation must be nonzero")]
    ZeroSourceGeneration,
    #[error("RAM-family log_T {log_t} exceeds the 32-bit retained-cycle domain")]
    InvalidLogT { log_t: usize },
    #[error("RAM-family address domain {address_domain} must be a nonzero power of two below the reserved u32 sentinel")]
    InvalidAddressDomain { address_domain: usize },
    #[error("retained RAM access tape failed validation")]
    TapeRejected,
    #[error("retained RAM access records are unavailable for {access_count} accesses")]
    AccessRecordsUnavailable { access_count: usize },
    #[error("retained RAM access count is {expected}, but {got} records were supplied")]
    AccessCountMismatch { expected: usize, got: usize },
    #[error("RAM access cycle {cycle} is outside the 2^{log_t} cycle domain")]
    AccessCycleOutOfRange { cycle: u32, log_t: usize },
    #[error("RAM access records are not strictly ordered at cycle {cycle}")]
    AccessesOutOfOrder { cycle: u32 },
    #[error("RAM address {address} is outside the {address_domain}-element domain or is reserved")]
    AddressOutOfRange { address: u32, address_domain: usize },
    #[error("RAM increment cycle {cycle} is outside the 2^{log_t} cycle domain")]
    IncrementCycleOutOfRange { cycle: u32, log_t: usize },
    #[error("RAM increment cycle {cycle} cannot be represented by the retained 32-bit format")]
    IncrementCycleEncodingOverflow { cycle: usize },
    #[error("RAM increments are not strictly ordered at cycle {cycle}")]
    IncrementsOutOfOrder { cycle: u32 },
    #[error("zero RAM increment was retained at cycle {cycle}")]
    ZeroIncrement { cycle: u32 },
    #[error(
        "RAM access delta at cycle {cycle} is {expected}, but retained activity is {actual:?}"
    )]
    IncrementDeltaMismatch {
        cycle: u32,
        expected: i128,
        actual: Option<i128>,
    },
    #[error("sparse RAM topology exceeds its 32-bit relative-index representation")]
    TopologyIndexOverflow,
    #[error("RAM Hamming support has not been certified as identical to retained access support")]
    HammingSupportUncertified,
    #[error("{point} has length {got}; expected {expected}")]
    PointLength {
        point: &'static str,
        expected: usize,
        got: usize,
    },
}
