use std::mem::size_of;

use thiserror::Error;

use super::owner::{OwnerError, RamAccessRecord, RamCycleFamilyOwner};
use super::topology::{
    BlockMerge, LevelCensus, RamRwMergeEvent, LEVEL_RANGE_BYTES, RAM_RW_GROUP_EVENT_BYTES,
};

const SOLINAS_FIELD_BYTES: u128 = 16;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoofRates {
    copy_bytes_per_second: u128,
    field_products_per_second: u128,
}

impl RoofRates {
    pub fn new(
        copy_bytes_per_second: u128,
        field_products_per_second: u128,
    ) -> Result<Self, AccountingError> {
        if copy_bytes_per_second == 0 {
            return Err(AccountingError::ZeroCopyRate);
        }
        if field_products_per_second == 0 {
            return Err(AccountingError::ZeroProductRate);
        }
        Ok(Self {
            copy_bytes_per_second,
            field_products_per_second,
        })
    }

    pub const fn copy_bytes_per_second(self) -> u128 {
        self.copy_bytes_per_second
    }

    pub const fn field_products_per_second(self) -> u128 {
        self.field_products_per_second
    }

    pub(crate) fn account(
        self,
        useful_field_products: u128,
        logical_bytes: u128,
    ) -> Result<RoofAccounting, AccountingError> {
        let compute_floor_ns =
            rate_floor_ns(useful_field_products, self.field_products_per_second)?;
        let traffic_floor_ns = rate_floor_ns(logical_bytes, self.copy_bytes_per_second)?;
        Ok(RoofAccounting {
            useful_field_products,
            logical_bytes,
            compute_floor_ns,
            traffic_floor_ns,
            lower_bound_ns: compute_floor_ns.max(traffic_floor_ns),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoofAccounting {
    useful_field_products: u128,
    logical_bytes: u128,
    compute_floor_ns: u128,
    traffic_floor_ns: u128,
    lower_bound_ns: u128,
}

impl RoofAccounting {
    pub const fn useful_field_products(self) -> u128 {
        self.useful_field_products
    }

    pub const fn logical_bytes(self) -> u128 {
        self.logical_bytes
    }

    pub const fn compute_floor_ns(self) -> u128 {
        self.compute_floor_ns
    }

    pub const fn traffic_floor_ns(self) -> u128 {
        self.traffic_floor_ns
    }

    pub const fn lower_bound_ns(self) -> u128 {
        self.lower_bound_ns
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the byte suffix makes every accounting unit explicit"
)]
pub struct OwnerByteAccounting {
    access_bytes: u128,
    increment_bytes: u128,
    final_memory_bytes: u128,
    read_write_event_bytes: u128,
    read_write_group_bytes: u128,
    block_leaf_bytes: u128,
    block_merge_bytes: u128,
    level_range_bytes: u128,
    topology_census_bytes: u128,
    receipt_census_bytes: u128,
    final_address_bytes: u128,
    logical_unique_bytes: u128,
    physical_allocation_bytes: u128,
}

impl OwnerByteAccounting {
    pub const fn access_bytes(self) -> u128 {
        self.access_bytes
    }

    pub const fn increment_bytes(self) -> u128 {
        self.increment_bytes
    }

    pub const fn final_memory_bytes(self) -> u128 {
        self.final_memory_bytes
    }

    pub const fn read_write_event_bytes(self) -> u128 {
        self.read_write_event_bytes
    }

    pub const fn read_write_group_bytes(self) -> u128 {
        self.read_write_group_bytes
    }

    pub const fn block_leaf_bytes(self) -> u128 {
        self.block_leaf_bytes
    }

    pub const fn block_merge_bytes(self) -> u128 {
        self.block_merge_bytes
    }

    pub const fn level_range_bytes(self) -> u128 {
        self.level_range_bytes
    }

    pub const fn topology_census_bytes(self) -> u128 {
        self.topology_census_bytes
    }

    pub const fn receipt_census_bytes(self) -> u128 {
        self.receipt_census_bytes
    }

    pub const fn final_address_bytes(self) -> u128 {
        self.final_address_bytes
    }

    /// Unique semantic payload, counting each census once.
    pub const fn logical_unique_bytes(self) -> u128 {
        self.logical_unique_bytes
    }

    /// Bytes in exact-length heap allocations, excluding allocator padding.
    pub const fn physical_allocation_bytes(self) -> u128 {
        self.physical_allocation_bytes
    }
}

pub fn owner_byte_accounting(
    owner: &RamCycleFamilyOwner,
) -> Result<OwnerByteAccounting, AccountingError> {
    owner.verify_integrity()?;
    let rw = owner.read_write_topology();
    let block = owner.block_topology();
    let receipt = owner.receipt();

    let access_bytes = allocation_bytes(receipt.access_count(), size_of::<RamAccessRecord>())?;
    let increment_cycle_bytes = allocation_bytes(receipt.increment_count(), size_of::<u64>())?;
    let increment_value_bytes = allocation_bytes(receipt.increment_count(), size_of::<i128>())?;
    let increment_bytes = checked_add(increment_cycle_bytes, increment_value_bytes)?;
    let final_memory_bytes = allocation_bytes(owner.final_memory().len(), size_of::<u64>())?;
    let read_write_event_bytes = allocation_bytes(rw.event_count(), size_of::<RamRwMergeEvent>())?;
    let read_write_group_bytes =
        allocation_bytes(rw.group_event_count(), RAM_RW_GROUP_EVENT_BYTES)?;
    let block_leaf_bytes = allocation_bytes(block.leaf_cycles().len(), size_of::<u32>())?;
    let block_merge_bytes = allocation_bytes(block.merge_count(), size_of::<BlockMerge>())?;
    let range_count = rw
        .event_range_count()
        .checked_add(rw.group_range_count())
        .and_then(|count| count.checked_add(block.range_count()))
        .ok_or(AccountingError::Overflow)?;
    let level_range_bytes = allocation_bytes(range_count, LEVEL_RANGE_BYTES)?;
    let topology_census_count = rw
        .census()
        .len()
        .checked_add(block.census().len())
        .ok_or(AccountingError::Overflow)?;
    let topology_census_bytes = allocation_bytes(topology_census_count, size_of::<LevelCensus>())?;
    let receipt_census_count = receipt
        .read_write_census()
        .len()
        .checked_add(receipt.block_census().len())
        .ok_or(AccountingError::Overflow)?;
    let receipt_census_bytes = allocation_bytes(receipt_census_count, size_of::<LevelCensus>())?;
    let final_address_bytes = allocation_bytes(rw.final_addresses().len(), size_of::<u32>())?;

    let unique_parts = [
        access_bytes,
        increment_bytes,
        final_memory_bytes,
        read_write_event_bytes,
        read_write_group_bytes,
        block_leaf_bytes,
        block_merge_bytes,
        level_range_bytes,
        topology_census_bytes,
        final_address_bytes,
    ];
    let logical_unique_bytes = checked_sum(unique_parts)?;
    let physical_allocation_bytes = checked_add(logical_unique_bytes, receipt_census_bytes)?;

    Ok(OwnerByteAccounting {
        access_bytes,
        increment_bytes,
        final_memory_bytes,
        read_write_event_bytes,
        read_write_group_bytes,
        block_leaf_bytes,
        block_merge_bytes,
        level_range_bytes,
        topology_census_bytes,
        receipt_census_bytes,
        final_address_bytes,
        logical_unique_bytes,
        physical_allocation_bytes,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadWriteAccounting {
    parent_entries: u128,
    parent_groups: u128,
    flat_products: u128,
    grouped_products: u128,
    cache_logical_bytes: u128,
    group_miss_logical_bytes: u128,
    flat_cache_roof: RoofAccounting,
    flat_group_miss_roof: RoofAccounting,
    grouped_cache_roof: RoofAccounting,
    grouped_group_miss_roof: RoofAccounting,
}

impl ReadWriteAccounting {
    pub const fn parent_entries(self) -> u128 {
        self.parent_entries
    }

    pub const fn parent_groups(self) -> u128 {
        self.parent_groups
    }

    pub const fn flat_products(self) -> u128 {
        self.flat_products
    }

    pub const fn grouped_products(self) -> u128 {
        self.grouped_products
    }

    pub const fn cache_logical_bytes(self) -> u128 {
        self.cache_logical_bytes
    }

    pub const fn group_miss_logical_bytes(self) -> u128 {
        self.group_miss_logical_bytes
    }

    pub const fn flat_cache_roof(self) -> RoofAccounting {
        self.flat_cache_roof
    }

    pub const fn flat_group_miss_roof(self) -> RoofAccounting {
        self.flat_group_miss_roof
    }

    pub const fn grouped_cache_roof(self) -> RoofAccounting {
        self.grouped_cache_roof
    }

    pub const fn grouped_group_miss_roof(self) -> RoofAccounting {
        self.grouped_group_miss_roof
    }
}

pub fn read_write_accounting(
    owner: &RamCycleFamilyOwner,
    rates: RoofRates,
) -> Result<ReadWriteAccounting, AccountingError> {
    owner.verify_integrity()?;
    let census = owner.read_write_topology().census();
    validate_census(census, owner.receipt().log_t())?;
    let leaves = census.first().ok_or(AccountingError::MissingCensus)?;
    let terminal = census.last().ok_or(AccountingError::MissingCensus)?;
    let parent_entries = checked_sum(
        census
            .iter()
            .skip(1)
            .map(|level| u128::from(level.entries())),
    )?;
    let parent_groups = checked_sum(
        census
            .iter()
            .skip(1)
            .map(|level| u128::from(level.groups())),
    )?;
    let middle_entries = checked_sum(
        census
            .iter()
            .skip(1)
            .take(census.len().saturating_sub(2))
            .map(|level| u128::from(level.entries())),
    )?;
    let prior_groups = checked_sum(
        census
            .iter()
            .take(census.len().saturating_sub(1))
            .map(|level| u128::from(level.groups())),
    )?;
    let flat_products = checked_add(
        checked_mul(8, parent_entries)?,
        checked_mul(2, parent_groups)?,
    )?;
    let grouped_products = checked_add(
        checked_mul(6, parent_entries)?,
        checked_mul(4, parent_groups)?,
    )?;

    let leaves_entries = u128::from(leaves.entries());
    let terminal_entries = u128::from(terminal.entries());
    let state_visits = checked_sum([
        checked_mul(2, leaves_entries)?,
        checked_mul(2, middle_entries)?,
        terminal_entries,
    ])?;
    let shared_traffic = checked_sum([
        checked_mul(72, leaves_entries)?,
        checked_mul(32, state_visits)?,
        checked_mul(64, parent_entries)?,
        checked_mul(128, parent_groups)?,
        checked_mul(32, prior_groups)?,
        checked_mul(32, owner.receipt().log_t() as u128)?,
        checked_mul(32, terminal_entries)?,
        16,
    ])?;
    let cache_logical_bytes = checked_add(shared_traffic, checked_mul(48, parent_groups)?)?;
    let group_miss_logical_bytes = checked_add(shared_traffic, checked_mul(48, parent_entries)?)?;

    Ok(ReadWriteAccounting {
        parent_entries,
        parent_groups,
        flat_products,
        grouped_products,
        cache_logical_bytes,
        group_miss_logical_bytes,
        flat_cache_roof: rates.account(flat_products, cache_logical_bytes)?,
        flat_group_miss_roof: rates.account(flat_products, group_miss_logical_bytes)?,
        grouped_cache_roof: rates.account(grouped_products, cache_logical_bytes)?,
        grouped_group_miss_roof: rates.account(grouped_products, group_miss_logical_bytes)?,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueCheckAccounting {
    union_nodes: u128,
    useful_field_products: u128,
    eq_address_bytes: u128,
    split_lt_bytes: u128,
    frontier_logical_bytes: u128,
    logical_bytes: u128,
    roof: RoofAccounting,
}

impl ValueCheckAccounting {
    pub const fn union_nodes(self) -> u128 {
        self.union_nodes
    }

    pub const fn useful_field_products(self) -> u128 {
        self.useful_field_products
    }

    pub const fn eq_address_bytes(self) -> u128 {
        self.eq_address_bytes
    }

    pub const fn split_lt_bytes(self) -> u128 {
        self.split_lt_bytes
    }

    pub const fn frontier_logical_bytes(self) -> u128 {
        self.frontier_logical_bytes
    }

    pub const fn logical_bytes(self) -> u128 {
        self.logical_bytes
    }

    pub const fn roof(self) -> RoofAccounting {
        self.roof
    }
}

pub fn value_check_accounting(
    owner: &RamCycleFamilyOwner,
    rates: RoofRates,
) -> Result<ValueCheckAccounting, AccountingError> {
    owner.verify_integrity()?;
    let census = owner.block_topology().census();
    validate_census(census, owner.receipt().log_t())?;
    let union_nodes = checked_sum(census.iter().map(|level| u128::from(level.entries())))?;
    let useful_field_products = checked_mul(12, union_nodes)?;
    let eq_address_bytes = checked_mul(
        SOLINAS_FIELD_BYTES,
        owner.receipt().address_domain() as u128,
    )?;
    let low_variables = owner.receipt().log_t() / 2;
    let high_variables = owner
        .receipt()
        .log_t()
        .checked_sub(low_variables)
        .ok_or(AccountingError::Overflow)?;
    let low_entries = domain_size_u128(low_variables)?;
    let high_entries = domain_size_u128(high_variables)?;
    let split_lt_entries = checked_add(low_entries, checked_mul(2, high_entries)?)?;
    let split_lt_bytes = checked_mul(SOLINAS_FIELD_BYTES, split_lt_entries)?;
    let frontier_logical_bytes = checked_mul(144, union_nodes)?;
    let logical_bytes = checked_sum([frontier_logical_bytes, eq_address_bytes, split_lt_bytes])?;
    Ok(ValueCheckAccounting {
        union_nodes,
        useful_field_products,
        eq_address_bytes,
        split_lt_bytes,
        frontier_logical_bytes,
        logical_bytes,
        roof: rates.account(useful_field_products, logical_bytes)?,
    })
}

fn validate_census(census: &[LevelCensus], log_t: usize) -> Result<(), AccountingError> {
    let expected = log_t.checked_add(1).ok_or(AccountingError::Overflow)?;
    if census.len() != expected {
        return Err(AccountingError::CensusLength {
            expected,
            got: census.len(),
        });
    }
    Ok(())
}

fn allocation_bytes(count: usize, element_size: usize) -> Result<u128, AccountingError> {
    checked_mul(count as u128, element_size as u128)
}

fn checked_add(left: u128, right: u128) -> Result<u128, AccountingError> {
    left.checked_add(right).ok_or(AccountingError::Overflow)
}

fn checked_mul(left: u128, right: u128) -> Result<u128, AccountingError> {
    left.checked_mul(right).ok_or(AccountingError::Overflow)
}

fn checked_sum<I>(values: I) -> Result<u128, AccountingError>
where
    I: IntoIterator<Item = u128>,
{
    values.into_iter().try_fold(0u128, checked_add)
}

fn rate_floor_ns(units: u128, rate_per_second: u128) -> Result<u128, AccountingError> {
    if rate_per_second == 0 {
        return Err(AccountingError::ZeroRate);
    }
    let numerator = checked_mul(units, NANOS_PER_SECOND)?;
    ceil_div(numerator, rate_per_second)
}

fn ceil_div(numerator: u128, denominator: u128) -> Result<u128, AccountingError> {
    if denominator == 0 {
        return Err(AccountingError::ZeroRate);
    }
    Ok(numerator.div_ceil(denominator))
}

fn domain_size_u128(log_size: usize) -> Result<u128, AccountingError> {
    let shift = u32::try_from(log_size).map_err(|_| AccountingError::Overflow)?;
    1u128.checked_shl(shift).ok_or(AccountingError::Overflow)
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AccountingError {
    #[error(transparent)]
    Owner(#[from] OwnerError),
    #[error("roof model copy rate must be nonzero")]
    ZeroCopyRate,
    #[error("roof model field-product rate must be nonzero")]
    ZeroProductRate,
    #[error("roof model rate must be nonzero")]
    ZeroRate,
    #[error("RAM topology census is missing")]
    MissingCensus,
    #[error("RAM topology census has {got} levels, expected {expected}")]
    CensusLength { expected: usize, got: usize },
    #[error("RAM accounting arithmetic overflowed")]
    Overflow,
}
