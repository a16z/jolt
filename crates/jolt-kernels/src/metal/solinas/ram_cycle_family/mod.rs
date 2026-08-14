//! Sparse RAM-cycle state shared by the Metal hybrid backend.
//!
//! The owner is built once from the RAM access tape and borrowed by the RAM
//! read-write, value, RA, and Hamming members.

mod oracle;
mod owner;
mod ram_hamming_booleanity;
mod ram_ra_claim_reduction;
mod ram_ra_virtualization;
mod ram_val_check;
mod topology;

pub use oracle::DenseRamValCheckOracle;
pub use owner::{
    OwnerConfig, OwnerError, RamAccessRecord, RamCycleFamilyOwner, RamCycleFamilyOwnerBuilder,
    RamCycleFamilyReceipt, RamCycleRow, RamIncrementRecord, RAM_CYCLE_FAMILY_SCHEMA_VERSION,
};
pub use ram_hamming_booleanity::{
    estimated_ram_hamming_products, HostSparseRamHammingBooleanity, RamHammingError,
    RamHammingMessage, RamHammingSparsePlan, RamHammingTerminal,
};
pub use ram_ra_claim_reduction::{
    estimated_ram_ra_claim_products, HostSparseRamRaClaimReduction, RamRaClaimError,
    RamRaClaimMessage, RamRaClaimTerminal,
};
pub use ram_ra_virtualization::{
    estimated_ram_ra_virtualization_products, HostSparseRamRaVirtualization,
    RamRaVirtualizationError, RamRaVirtualizationMessage, RamRaVirtualizationTerminal,
};
pub use ram_val_check::{
    HostSparseRamValCheck, RamValError, RamValFrontierEntry, RamValMessage, RamValTerminalFactors,
};
pub use topology::{
    BlockMerge, LevelCensus, RamBlockTopology, RamRwGroupEvent, RamRwMergeEvent,
    RamRwMergeTopology, TopologyError,
};

#[cfg(test)]
mod tests;
