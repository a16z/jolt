use std::mem::size_of;

use thiserror::Error;

use super::topology::{LevelCensus, RamBlockTopology, RamRwMergeTopology, TopologyError};

pub const RAM_CYCLE_FAMILY_SCHEMA_VERSION: u32 = 3;

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

    pub const fn cycle(self) -> u32 {
        self.cycle
    }

    pub const fn address(self) -> u32 {
        self.address
    }

    pub const fn pre_value(self) -> u64 {
        self.pre_value
    }

    pub const fn post_value(self) -> u64 {
        self.post_value
    }
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

    pub const fn cycle(self) -> u64 {
        self.cycle
    }

    pub const fn increment(self) -> i128 {
        self.increment
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AddressKind {
    NoAccess,
    RawZero,
    Remapped(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamCycleRow {
    address: AddressKind,
    pre_value: u64,
    post_value: u64,
    ram_increment: i128,
}

impl RamCycleRow {
    pub const fn no_access() -> Self {
        Self {
            address: AddressKind::NoAccess,
            pre_value: 0,
            post_value: 0,
            ram_increment: 0,
        }
    }

    pub const fn raw_address_zero(ram_increment: i128) -> Self {
        Self {
            address: AddressKind::RawZero,
            pre_value: 0,
            post_value: 0,
            ram_increment,
        }
    }

    pub const fn remapped(
        address: u32,
        pre_value: u64,
        post_value: u64,
        ram_increment: i128,
    ) -> Self {
        Self {
            address: AddressKind::Remapped(address),
            pre_value,
            post_value,
            ram_increment,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnerConfig {
    log_t: usize,
    log_k: usize,
    source_generation: u64,
    threadgroup_width: usize,
    max_sparse_records: usize,
}

impl OwnerConfig {
    pub fn new(
        log_t: usize,
        log_k: usize,
        source_generation: u64,
        threadgroup_width: usize,
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
        if threadgroup_width == 0 {
            return Err(OwnerError::ZeroThreadgroupWidth);
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
            threadgroup_width,
            max_sparse_records,
        })
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn source_generation(self) -> u64 {
        self.source_generation
    }

    pub const fn threadgroup_width(self) -> usize {
        self.threadgroup_width
    }

    pub const fn max_sparse_records(self) -> usize {
        self.max_sparse_records
    }
}

pub struct RamCycleFamilyOwnerBuilder {
    config: OwnerConfig,
    cycles: usize,
    address_domain: usize,
    next_cycle: usize,
    records: Vec<RamAccessRecord>,
    increment_cycles: Vec<u64>,
    increments: Vec<i128>,
    last_post: Vec<Option<u64>>,
}

impl RamCycleFamilyOwnerBuilder {
    pub fn new(config: OwnerConfig) -> Result<Self, OwnerError> {
        let cycles = domain_size(config.log_t)?;
        let address_domain = domain_size(config.log_k)?;
        Ok(Self {
            config,
            cycles,
            address_domain,
            next_cycle: 0,
            records: Vec::new(),
            increment_cycles: Vec::new(),
            increments: Vec::new(),
            last_post: vec![None; address_domain],
        })
    }

    pub fn push_cycle(&mut self, row: RamCycleRow) -> Result<(), OwnerError> {
        if self.next_cycle >= self.cycles {
            return Err(OwnerError::TooManyCycles {
                expected: self.cycles,
            });
        }
        let cycle = self.next_cycle;
        let next_cycle = cycle.checked_add(1).ok_or(OwnerError::Overflow)?;
        let cycle_u32 = u32::try_from(cycle).map_err(|_| OwnerError::CycleIndexTooLarge)?;
        let cycle_u64 = u64::try_from(cycle).map_err(|_| OwnerError::CycleIndexTooLarge)?;
        let needs_access_record = matches!(row.address, AddressKind::Remapped(_));
        let needs_increment_record = row.ram_increment != 0;
        if needs_access_record && self.records.len() >= self.config.max_sparse_records {
            return Err(OwnerError::SparseCapacityExceeded {
                maximum: self.config.max_sparse_records,
            });
        }
        if needs_increment_record && self.increments.len() >= self.config.max_sparse_records {
            return Err(OwnerError::SparseCapacityExceeded {
                maximum: self.config.max_sparse_records,
            });
        }
        match row.address {
            AddressKind::NoAccess => {
                if row.ram_increment != 0 {
                    return Err(OwnerError::NonzeroIncrementWithoutAccessKind { cycle });
                }
            }
            AddressKind::RawZero => {}
            AddressKind::Remapped(address) => {
                let address_index = address as usize;
                if address_index >= self.address_domain {
                    return Err(OwnerError::AddressOutOfRange {
                        cycle,
                        address,
                        address_domain: self.address_domain,
                    });
                }
                let expected_increment = i128::from(row.post_value) - i128::from(row.pre_value);
                if row.ram_increment != expected_increment {
                    return Err(OwnerError::IncrementMismatch {
                        cycle,
                        expected: expected_increment,
                        got: row.ram_increment,
                    });
                }
                if self
                    .last_post
                    .get(address_index)
                    .copied()
                    .flatten()
                    .is_some_and(|previous| previous != row.pre_value)
                {
                    return Err(OwnerError::CheckpointDiscontinuity { cycle, address });
                }
                let destination =
                    self.last_post
                        .get_mut(address_index)
                        .ok_or(OwnerError::AddressOutOfRange {
                            cycle,
                            address,
                            address_domain: self.address_domain,
                        })?;
                *destination = Some(row.post_value);
                self.records.push(RamAccessRecord {
                    cycle: cycle_u32,
                    address,
                    pre_value: row.pre_value,
                    post_value: row.post_value,
                });
            }
        }
        if needs_increment_record {
            self.increment_cycles.push(cycle_u64);
            self.increments.push(row.ram_increment);
        }
        self.next_cycle = next_cycle;
        Ok(())
    }

    pub fn finish(self, final_memory: Vec<u64>) -> Result<RamCycleFamilyOwner, OwnerError> {
        if self.next_cycle != self.cycles {
            return Err(OwnerError::CycleCountMismatch {
                expected: self.cycles,
                got: self.next_cycle,
            });
        }
        if final_memory.len() != self.address_domain {
            return Err(OwnerError::FinalMemoryLength {
                expected: self.address_domain,
                got: final_memory.len(),
            });
        }
        for (address, expected) in self.last_post.iter().enumerate() {
            if let Some(expected) = expected {
                let got =
                    final_memory
                        .get(address)
                        .copied()
                        .ok_or(OwnerError::FinalMemoryLength {
                            expected: self.address_domain,
                            got: final_memory.len(),
                        })?;
                if got != *expected {
                    return Err(OwnerError::FinalMemoryMismatch {
                        address,
                        expected: *expected,
                        got,
                    });
                }
            }
        }

        let rw_topology = RamRwMergeTopology::build(
            self.config.log_t,
            &self.records,
            self.config.threadgroup_width,
        )?;
        let block_topology = RamBlockTopology::build(
            self.config.log_t,
            &self.records,
            &self.increment_cycles,
            self.config.threadgroup_width,
        )?;
        let final_memory = final_memory.into_boxed_slice();
        let records = self.records.into_boxed_slice();
        let increment_cycles = self.increment_cycles.into_boxed_slice();
        let increments = self.increments.into_boxed_slice();
        let fingerprint = owner_fingerprint(
            self.config,
            &records,
            &increment_cycles,
            &increments,
            &final_memory,
            &rw_topology,
            &block_topology,
        );
        let receipt = RamCycleFamilyReceipt {
            schema_version: RAM_CYCLE_FAMILY_SCHEMA_VERSION,
            source_generation: self.config.source_generation,
            log_t: self.config.log_t,
            log_k: self.config.log_k,
            cycles: self.cycles,
            address_domain: self.address_domain,
            threadgroup_width: self.config.threadgroup_width,
            access_count: records.len(),
            increment_count: increments.len(),
            rw_census: rw_topology.census().to_vec().into_boxed_slice(),
            block_census: block_topology.census().to_vec().into_boxed_slice(),
            fingerprint,
        };
        Ok(RamCycleFamilyOwner {
            receipt,
            records,
            increment_cycles,
            increments,
            final_memory,
            rw_topology,
            block_topology,
        })
    }
}

pub struct RamCycleFamilyReceipt {
    schema_version: u32,
    source_generation: u64,
    log_t: usize,
    log_k: usize,
    cycles: usize,
    address_domain: usize,
    threadgroup_width: usize,
    access_count: usize,
    increment_count: usize,
    rw_census: Box<[LevelCensus]>,
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
        threadgroup_width: usize,
        access_count: usize,
        increment_count: usize,
        fingerprint: u64,
    } }

    pub fn read_write_census(&self) -> &[LevelCensus] {
        &self.rw_census
    }

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
    rw_topology: RamRwMergeTopology,
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
            self.rw_topology.owned_heap_bytes() + self.block_topology.owned_heap_bytes(),
        );
        visitor.visit_simple(
            allocative::Key::new("receipt_census"),
            std::mem::size_of_val(self.receipt.rw_census.as_ref())
                + std::mem::size_of_val(self.receipt.block_census.as_ref()),
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

    pub fn read_write_topology(&self) -> &RamRwMergeTopology {
        &self.rw_topology
    }

    pub fn block_topology(&self) -> &RamBlockTopology {
        &self.block_topology
    }

    pub fn owned_heap_bytes(&self) -> usize {
        std::mem::size_of_val(self.records.as_ref())
            + std::mem::size_of_val(self.increment_cycles.as_ref())
            + std::mem::size_of_val(self.increments.as_ref())
            + std::mem::size_of_val(self.final_memory.as_ref())
            + self.rw_topology.owned_heap_bytes()
            + self.block_topology.owned_heap_bytes()
            + std::mem::size_of_val(self.receipt.rw_census.as_ref())
            + std::mem::size_of_val(self.receipt.block_census.as_ref())
    }

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
            || self.receipt.threadgroup_width == 0
            || self.rw_topology.log_t() != self.receipt.log_t
            || self.block_topology.log_t() != self.receipt.log_t
            || self.receipt.rw_census.as_ref() != self.rw_topology.census()
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
            threadgroup_width: self.receipt.threadgroup_width,
            max_sparse_records: self.records.len().max(self.increments.len()).max(1),
        };
        let fingerprint = owner_fingerprint(
            config,
            &self.records,
            &self.increment_cycles,
            &self.increments,
            &self.final_memory,
            &self.rw_topology,
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
    let rw_topology = RamRwMergeTopology::build(config.log_t, &records, config.threadgroup_width)?;
    let block_topology = RamBlockTopology::build(
        config.log_t,
        &records,
        &increment_cycles,
        config.threadgroup_width,
    )?;
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
        &rw_topology,
        &block_topology,
    );
    let receipt = RamCycleFamilyReceipt {
        schema_version: RAM_CYCLE_FAMILY_SCHEMA_VERSION,
        source_generation: config.source_generation,
        log_t: config.log_t,
        log_k: config.log_k,
        cycles,
        address_domain,
        threadgroup_width: config.threadgroup_width,
        access_count: records.len(),
        increment_count: increments.len(),
        rw_census: rw_topology.census().to_vec().into_boxed_slice(),
        block_census: block_topology.census().to_vec().into_boxed_slice(),
        fingerprint,
    };
    Ok(RamCycleFamilyOwner {
        receipt,
        records,
        increment_cycles,
        increments,
        final_memory,
        rw_topology,
        block_topology,
    })
}

fn owner_fingerprint(
    config: OwnerConfig,
    records: &[RamAccessRecord],
    increment_cycles: &[u64],
    increments: &[i128],
    final_memory: &[u64],
    rw_topology: &RamRwMergeTopology,
    block_topology: &RamBlockTopology,
) -> u64 {
    let mut state = 0x6a09_e667_f3bc_c909u64;
    for value in [
        u64::from(RAM_CYCLE_FAMILY_SCHEMA_VERSION),
        config.source_generation,
        config.log_t as u64,
        config.log_k as u64,
        config.threadgroup_width as u64,
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
    for level in rw_topology.census() {
        state = mix(state, level.entries());
        state = mix(state, level.groups());
        state = mix(state, level.tiles());
    }
    for level in block_topology.census() {
        state = mix(state, level.entries());
        state = mix(state, level.tiles());
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
    #[error("RAM owner threadgroup width must be nonzero")]
    ZeroThreadgroupWidth,
    #[error("RAM owner sparse capacity must be nonzero")]
    ZeroSparseCapacity,
    #[error("RAM owner expected {expected} cycles but received more")]
    TooManyCycles { expected: usize },
    #[error("RAM owner expected {expected} cycles but received {got}")]
    CycleCountMismatch { expected: usize, got: usize },
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
    #[error("RAM owner no-access row at cycle {cycle} has a nonzero increment")]
    NonzeroIncrementWithoutAccessKind { cycle: usize },
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
