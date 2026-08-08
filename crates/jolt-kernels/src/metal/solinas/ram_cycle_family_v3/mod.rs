//! Shared sparse RAM-cycle state for the Metal hybrid backend.
//!
//! It provides the checked owner and sparse host sequences shared by the RAM
//! family, plus dense oracles and analytical selectors for successor kernels.

mod model;
mod oracle;
mod owner;
mod ram_hamming_booleanity;
mod ram_ra_claim_reduction;
mod ram_ra_virtualization;
mod ram_val_check;
mod selector;
mod topology;

pub use model::{
    owner_byte_accounting, read_write_accounting, value_check_accounting, AccountingError,
    OwnerByteAccounting, ReadWriteAccounting, RoofAccounting, RoofRates, ValueCheckAccounting,
};
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
pub use selector::{
    select_read_write, select_value_check, CycleCutoffPlan, ExecutionLane, ExecutionOverheads,
    ExecutionProfile, RwLevelSchedule, SelectionError,
};
pub use topology::{
    BlockMerge, LevelCensus, RamBlockTopology, RamRwGroupEvent, RamRwMergeEvent,
    RamRwMergeTopology, TopologyError,
};

#[cfg(test)]
mod tests;
