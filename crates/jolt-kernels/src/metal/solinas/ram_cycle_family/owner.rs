use std::mem::size_of;

use thiserror::Error;

use super::topology::{LevelCensus, RamBlockTopology, TopologyError};

pub const RAM_CYCLE_FAMILY_SCHEMA_VERSION: u32 = 4;

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamAccessRecord {
    cycle: u32,
    address: u32,
    pre_value: u64,
    post_value: u64,
}

const _: [(); 24] = [(); size_of::<RamAccessRecord>()];

impl RamAccessRecord {
    pub const fn new(cycle: u32, address: u32, pre_value: u64, post_value: u64) -> Self {
        Self {
            cycle,
            address,
            pre_value,
            post_value,
        }
    }

    copy_field_getters! { pub, {
        cycle: u32,
        address: u32,
        pre_value: u64,
        post_value: u64,
    }}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamIncrementRecord {
    cycle: u64,
    increment: i128,
}

impl RamIncrementRecord {
    pub const fn new(cycle: u64, increment: i128) -> Self {
        Self { cycle, increment }
    }

    copy_field_getters! { pub, {
        cycle: u64,
        increment: i128,
    }}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnerConfig {
    log_t: usize,
    log_k: usize,
    source_generation: u64,
    max_sparse_records: usize,
}

impl OwnerConfig {
    pub fn new(
        log_t: usize,
        log_k: usize,
        source_generation: u64,
        max_sparse_records: usize,
    ) -> Result<Self, OwnerError> {
        if log_t == 0 || log_t > u32::BITS as usize {
            return Err(OwnerError::InvalidLogT { log_t });
        }
        if log_k == 0 || log_k >= u32::BITS as usize {
            return Err(OwnerError::InvalidLogK { log_k });
        }
        if source_generation == 0 {
            return Err(OwnerError::ZeroSourceGeneration);
        }
        if max_sparse_records == 0 {
            return Err(OwnerError::ZeroSparseCapacity);
        }
        let _ = domain_size(log_t)?;
        let _ = domain_size(log_k)?;
        Ok(Self {
            log_t,
            log_k,
            source_generation,
            max_sparse_records,
        })
    }

    copy_field_getters! { pub, {
        log_t: usize,
        log_k: usize,
        source_generation: u64,
        max_sparse_records: usize,
    }}
}

pub struct RamCycleFamilyReceipt {
    schema_version: u32,
    source_generation: u64,
    log_t: usize,
    log_k: usize,
    cycles: usize,
    address_domain: usize,
    access_count: usize,
    increment_count: usize,
    block_census: Box<[LevelCensus]>,
    fingerprint: u64,
}

impl RamCycleFamilyReceipt {
    copy_field_getters! { pub, {
        schema_version: u32,
        source_generation: u64,
        log_t: usize,
        log_k: usize,
        cycles: usize,
        address_domain: usize,
        access_count: usize,
        increment_count: usize,
        fingerprint: u64,
    } }

    pub fn block_census(&self) -> &[LevelCensus] {
        &self.block_census
    }
}

pub struct RamCycleFamilyOwner {
    receipt: RamCycleFamilyReceipt,
    records: Box<[RamAccessRecord]>,
    increment_cycles: Box<[u64]>,
    increments: Box<[i128]>,
    final_memory: Box<[u64]>,
    block_topology: RamBlockTopology,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamCycleFamilyOwner {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("access_records"),
            std::mem::size_of_val(self.records.as_ref()),
        );
        visitor.visit_simple(
            allocative::Key::new("increment_cycles"),
            std::mem::size_of_val(self.increment_cycles.as_ref()),
        );
        visitor.visit_simple(
            allocative::Key::new("increments"),
            std::mem::size_of_val(self.increments.as_ref()),
        );
        visitor.visit_simple(
            allocative::Key::new("final_memory"),
            std::mem::size_of_val(self.final_memory.as_ref()),
        );
        visitor.visit_simple(
            allocative::Key::new("topology"),
            self.block_topology.owned_heap_bytes(),
        );
        visitor.visit_simple(
            allocative::Key::new("receipt_census"),
            std::mem::size_of_val(self.receipt.block_census.as_ref()),
        );
        visitor.exit();
    }
}

impl RamCycleFamilyOwner {
    pub fn from_sparse_records(
        config: OwnerConfig,
        records: Vec<RamAccessRecord>,
        increments: Vec<RamIncrementRecord>,
        final_memory: Vec<u64>,
    ) -> Result<Self, OwnerError> {
        let cycles = domain_size(config.log_t)?;
        let address_domain = domain_size(config.log_k)?;
        if records.len() > config.max_sparse_records {
            return Err(OwnerError::SparseCapacityExceeded {
                maximum: config.max_sparse_records,
            });
        }
        if increments.len() > config.max_sparse_records {
            return Err(OwnerError::SparseCapacityExceeded {
                maximum: config.max_sparse_records,
            });
        }
        if final_memory.len() != address_domain {
            return Err(OwnerError::FinalMemoryLength {
                expected: address_domain,
                got: final_memory.len(),
            });
        }

        let mut last_cycle = None;
        let mut last_post = vec![None; address_domain];
        for record in &records {
            let cycle = record.cycle as usize;
            let address = record.address as usize;
            if cycle >= cycles {
                return Err(OwnerError::CycleIndexTooLarge);
            }
            if last_cycle.is_some_and(|previous| previous >= cycle) {
                return Err(OwnerError::AccessRecordsOutOfOrder { cycle });
            }
            if address >= address_domain {
                return Err(OwnerError::AddressOutOfRange {
                    cycle,
                    address: record.address,
                    address_domain,
                });
            }
            if last_post[address].is_some_and(|previous| previous != record.pre_value) {
                return Err(OwnerError::CheckpointDiscontinuity {
                    cycle,
                    address: record.address,
                });
            }
            last_post[address] = Some(record.post_value);
            last_cycle = Some(cycle);
        }
        for (address, expected) in last_post.into_iter().enumerate() {
            if let Some(expected) = expected {
                let got = final_memory[address];
                if got != expected {
                    return Err(OwnerError::FinalMemoryMismatch {
                        address,
                        expected,
                        got,
                    });
                }
            }
        }

        let mut increment_cycles = Vec::with_capacity(increments.len());
        let mut increment_values = Vec::with_capacity(increments.len());
        let mut last_increment_cycle = None;
        for increment in increments {
            let cycle =
                usize::try_from(increment.cycle).map_err(|_| OwnerError::CycleIndexTooLarge)?;
            if cycle >= cycles {
                return Err(OwnerError::CycleIndexTooLarge);
            }
            if last_increment_cycle.is_some_and(|previous| previous >= cycle) {
                return Err(OwnerError::IncrementRecordsOutOfOrder { cycle });
            }
            if increment.increment == 0 {
                return Err(OwnerError::ZeroIncrement { cycle });
            }
            increment_cycles.push(increment.cycle);
            increment_values.push(increment.increment);
            last_increment_cycle = Some(cycle);
        }

        for record in &records {
            let cycle = u64::from(record.cycle);
            let expected = i128::from(record.post_value) - i128::from(record.pre_value);
            let got = increment_cycles
                .binary_search(&cycle)
                .ok()
                .map_or(0, |index| increment_values[index]);
            if got != expected {
                return Err(OwnerError::IncrementMismatch {
                    cycle: record.cycle as usize,
                    expected,
                    got,
                });
            }
        }

        build_owner(
            config,
            cycles,
            address_domain,
            records,
            increment_cycles,
            increment_values,
            final_memory,
        )
    }

    pub fn receipt(&self) -> &RamCycleFamilyReceipt {
        &self.receipt
    }

    pub fn access_records(&self) -> &[RamAccessRecord] {
        &self.records
    }

    pub fn increment_records(&self) -> impl ExactSizeIterator<Item = RamIncrementRecord> + '_ {
        self.increment_cycles
            .iter()
            .copied()
            .zip(self.increments.iter().copied())
            .map(|(cycle, increment)| RamIncrementRecord { cycle, increment })
    }

    pub fn final_memory(&self) -> &[u64] {
        &self.final_memory
    }

    pub fn block_topology(&self) -> &RamBlockTopology {
        &self.block_topology
    }

    pub fn owned_heap_bytes(&self) -> usize {
        std::mem::size_of_val(self.records.as_ref())
            + std::mem::size_of_val(self.increment_cycles.as_ref())
            + std::mem::size_of_val(self.increments.as_ref())
            + std::mem::size_of_val(self.final_memory.as_ref())
            + self.block_topology.owned_heap_bytes()
            + std::mem::size_of_val(self.receipt.block_census.as_ref())
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn verify_integrity(&self) -> Result<(), OwnerError> {
        if self.receipt.schema_version != RAM_CYCLE_FAMILY_SCHEMA_VERSION {
            return Err(OwnerError::UnsupportedSchema {
                got: self.receipt.schema_version,
            });
        }
        let cycles = domain_size(self.receipt.log_t)?;
        let address_domain = domain_size(self.receipt.log_k)?;
        if self.receipt.cycles != cycles
            || self.receipt.address_domain != address_domain
            || self.receipt.access_count != self.records.len()
            || self.receipt.increment_count != self.increments.len()
            || self.increment_cycles.len() != self.increments.len()
            || self.final_memory.len() != address_domain
            || self.receipt.source_generation == 0
            || self.block_topology.log_t() != self.receipt.log_t
            || self.receipt.block_census.as_ref() != self.block_topology.census()
        {
            return Err(OwnerError::ReceiptMismatch);
        }
        let mut last_cycle = None;
        let mut last_post = vec![None; address_domain];
        for record in &self.records {
            let cycle = usize::try_from(record.cycle).map_err(|_| OwnerError::ReceiptMismatch)?;
            let address = record.address as usize;
            if cycle >= cycles
                || address >= address_domain
                || last_cycle.is_some_and(|previous| previous >= cycle)
                || last_post
                    .get(address)
                    .copied()
                    .flatten()
                    .is_some_and(|previous| previous != record.pre_value)
            {
                return Err(OwnerError::ReceiptMismatch);
            }
            let destination = last_post
                .get_mut(address)
                .ok_or(OwnerError::ReceiptMismatch)?;
            *destination = Some(record.post_value);
            last_cycle = Some(cycle);
        }
        for (address, expected) in last_post.into_iter().enumerate() {
            if expected.is_some_and(|expected| self.final_memory.get(address) != Some(&expected)) {
                return Err(OwnerError::ReceiptMismatch);
            }
        }
        let mut last_increment_cycle = None;
        for (&cycle, &increment) in self.increment_cycles.iter().zip(&self.increments) {
            let cycle_in_range = match usize::try_from(cycle) {
                Ok(cycle) => cycle < cycles,
                Err(_) => false,
            };
            if increment == 0
                || !cycle_in_range
                || last_increment_cycle.is_some_and(|previous| previous >= cycle)
            {
                return Err(OwnerError::ReceiptMismatch);
            }
            last_increment_cycle = Some(cycle);
        }
        let config = OwnerConfig {
            log_t: self.receipt.log_t,
            log_k: self.receipt.log_k,
            source_generation: self.receipt.source_generation,
            max_sparse_records: self.records.len().max(self.increments.len()).max(1),
        };
        let fingerprint = owner_fingerprint(
            config,
            &self.records,
            &self.increment_cycles,
            &self.increments,
            &self.final_memory,
            &self.block_topology,
        );
        if fingerprint != self.receipt.fingerprint {
            return Err(OwnerError::ReceiptMismatch);
        }
        Ok(())
    }

    pub(crate) fn increment_slices(&self) -> (&[u64], &[i128]) {
        (&self.increment_cycles, &self.increments)
    }
}

fn build_owner(
    config: OwnerConfig,
    cycles: usize,
    address_domain: usize,
    records: Vec<RamAccessRecord>,
    increment_cycles: Vec<u64>,
    increments: Vec<i128>,
    final_memory: Vec<u64>,
) -> Result<RamCycleFamilyOwner, OwnerError> {
    let block_topology = RamBlockTopology::build(config.log_t, &records, &increment_cycles)?;
    let final_memory = final_memory.into_boxed_slice();
    let records = records.into_boxed_slice();
    let increment_cycles = increment_cycles.into_boxed_slice();
    let increments = increments.into_boxed_slice();
    let fingerprint = owner_fingerprint(
        config,
        &records,
        &increment_cycles,
        &increments,
        &final_memory,
        &block_topology,
    );
    let receipt = RamCycleFamilyReceipt {
        schema_version: RAM_CYCLE_FAMILY_SCHEMA_VERSION,
        source_generation: config.source_generation,
        log_t: config.log_t,
        log_k: config.log_k,
        cycles,
        address_domain,
        access_count: records.len(),
        increment_count: increments.len(),
        block_census: block_topology.census().to_vec().into_boxed_slice(),
        fingerprint,
    };
    Ok(RamCycleFamilyOwner {
        receipt,
        records,
        increment_cycles,
        increments,
        final_memory,
        block_topology,
    })
}

fn owner_fingerprint(
    config: OwnerConfig,
    records: &[RamAccessRecord],
    increment_cycles: &[u64],
    increments: &[i128],
    final_memory: &[u64],
    block_topology: &RamBlockTopology,
) -> u64 {
    let mut state = 0x6a09_e667_f3bc_c909u64;
    for value in [
        u64::from(RAM_CYCLE_FAMILY_SCHEMA_VERSION),
        config.source_generation,
        config.log_t as u64,
        config.log_k as u64,
        records.len() as u64,
        increments.len() as u64,
    ] {
        state = mix(state, value);
    }
    for record in records {
        state = mix(state, u64::from(record.cycle));
        state = mix(state, u64::from(record.address));
        state = mix(state, record.pre_value);
        state = mix(state, record.post_value);
    }
    for (&cycle, &increment) in increment_cycles.iter().zip(increments) {
        state = mix(state, cycle);
        let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] =
            increment.to_le_bytes();
        let low = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
        let high = u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]);
        state = mix(mix(state, low), high);
    }
    for &word in final_memory {
        state = mix(state, word);
    }
    for level in block_topology.census() {
        state = mix(state, level.entries());
    }
    state
}

fn mix(state: u64, value: u64) -> u64 {
    let mixed = state ^ value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb)
}

fn domain_size(log_size: usize) -> Result<usize, OwnerError> {
    1usize
        .checked_shl(u32::try_from(log_size).map_err(|_| OwnerError::Overflow)?)
        .ok_or(OwnerError::Overflow)
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum OwnerError {
    #[error("RAM owner schema version {got} is unsupported")]
    UnsupportedSchema { got: u32 },
    #[error("RAM owner log_T {log_t} is unsupported")]
    InvalidLogT { log_t: usize },
    #[error("RAM owner log_K {log_k} is unsupported")]
    InvalidLogK { log_k: usize },
    #[error("RAM owner source generation must be nonzero")]
    ZeroSourceGeneration,
    #[error("RAM owner sparse capacity must be nonzero")]
    ZeroSparseCapacity,
    #[error("RAM owner cycle index exceeds the u32 ABI")]
    CycleIndexTooLarge,
    #[error("RAM owner access records are not strictly ordered at cycle {cycle}")]
    AccessRecordsOutOfOrder { cycle: usize },
    #[error("RAM owner increment records are not strictly ordered at cycle {cycle}")]
    IncrementRecordsOutOfOrder { cycle: usize },
    #[error("RAM owner retained a zero increment at cycle {cycle}")]
    ZeroIncrement { cycle: usize },
    #[error(
        "RAM owner address {address} at cycle {cycle} is outside the {address_domain}-word domain"
    )]
    AddressOutOfRange {
        cycle: usize,
        address: u32,
        address_domain: usize,
    },
    #[error("RAM owner increment at cycle {cycle} is {got}, expected {expected}")]
    IncrementMismatch {
        cycle: usize,
        expected: i128,
        got: i128,
    },
    #[error("RAM owner checkpoints diverge at cycle {cycle}, address {address}")]
    CheckpointDiscontinuity { cycle: usize, address: u32 },
    #[error("RAM owner sparse payload exceeds its {maximum}-record capacity")]
    SparseCapacityExceeded { maximum: usize },
    #[error("RAM owner final-memory length is {got}, expected {expected}")]
    FinalMemoryLength { expected: usize, got: usize },
    #[error(
        "RAM owner final memory at address {address} is {got}, expected the last post-value {expected}"
    )]
    FinalMemoryMismatch {
        address: usize,
        expected: u64,
        got: u64,
    },
    #[error("RAM owner arithmetic overflowed")]
    Overflow,
    #[error("RAM owner receipt does not match its payload")]
    ReceiptMismatch,
    #[error(transparent)]
    Topology(#[from] TopologyError),
}
