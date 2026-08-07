//! Compact producer records and checked Metal parameter blocks.

use core::mem::{align_of, size_of};

pub const RAM_RAF_SUCCESSOR_AKITA_OFFSET: u32 = 0xffff_a7f7;
pub const RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN: usize = 1 << 13;
pub const RAM_RAF_SUCCESSOR_INNER_LOG2: usize = 15;
pub const RAM_RAF_SUCCESSOR_INNER_LENGTH: usize = 1 << RAM_RAF_SUCCESSOR_INNER_LOG2;
pub const RAM_RAF_SUCCESSOR_TILE_ADDRESSES: usize = 1_376;
pub const RAM_RAF_SUCCESSOR_TILE_COUNT: usize = 6;
pub const RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS: usize = 5;
pub const RAM_RAF_SUCCESSOR_DIRECT_THREADS: usize = 256;
pub const RAM_RAF_SUCCESSOR_BUCKET_THREADS: usize = 1_024;
pub const RAM_RAF_SUCCESSOR_FINALIZE_THREADS: usize = 256;
pub const RAM_RAF_SUCCESSOR_MISSING_INDEX: u32 = u32::MAX;

pub const RAM_RAF_SUCCESSOR_DIRECT_PIPELINE: &str = "solinas_ram_raf_successor_direct";
pub const RAM_RAF_SUCCESSOR_BUCKET_PIPELINE: &str = "solinas_ram_raf_successor_bucketed";
pub const RAM_RAF_SUCCESSOR_FINALIZE_PIPELINE: &str = "solinas_ram_raf_successor_finalize";

const INNER_BITS: u32 = RAM_RAF_SUCCESSOR_INNER_LOG2 as u32;
const LOCAL_ADDRESS_BITS: u32 = 11;
const INNER_MASK: u32 = (1 << INNER_BITS) - 1;
const LOCAL_ADDRESS_MASK: u32 = (1 << LOCAL_ADDRESS_BITS) - 1;
const RESERVED_SHIFT: u32 = INNER_BITS + LOCAL_ADDRESS_BITS;

/// One non-sentinel access from the common RAM producer, ordered by cycle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafAccessRecord {
    cycle: u32,
    address: u32,
}

const _: [(); 8] = [(); size_of::<RamRafAccessRecord>()];
const _: [(); 4] = [(); align_of::<RamRafAccessRecord>()];

impl RamRafAccessRecord {
    pub const fn new(cycle: u32, address: u32) -> Self {
        Self { cycle, address }
    }

    pub const fn cycle(self) -> u32 {
        self.cycle
    }

    pub const fn address(self) -> u32 {
        self.address
    }
}

/// Member-local record packed inside one `(outer, tile)` bucket.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafBucketRecord(u32);

const _: [(); 4] = [(); size_of::<RamRafBucketRecord>()];
const _: [(); 4] = [(); align_of::<RamRafBucketRecord>()];

impl RamRafBucketRecord {
    pub fn new(inner: u32, local_address: u32) -> Result<Self, RamRafCompactError> {
        if inner >= RAM_RAF_SUCCESSOR_INNER_LENGTH as u32 {
            return Err(RamRafCompactError::InnerOutsideBlock { inner });
        }
        if local_address >= RAM_RAF_SUCCESSOR_TILE_ADDRESSES as u32
            || local_address > LOCAL_ADDRESS_MASK
        {
            return Err(RamRafCompactError::LocalAddressOutsideTile { local_address });
        }
        Ok(Self(inner | (local_address << INNER_BITS)))
    }

    pub const fn inner(self) -> u32 {
        self.0 & INNER_MASK
    }

    pub const fn local_address(self) -> u32 {
        (self.0 >> INNER_BITS) & LOCAL_ADDRESS_MASK
    }

    pub const fn reserved(self) -> u32 {
        self.0 >> RESERVED_SHIFT
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// A nonempty bucket. Descriptors are strictly ordered by `(outer, tile)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafBucketDescriptor {
    first_record: u32,
    record_count: u32,
    outer: u32,
    tile: u32,
}

const _: [(); 16] = [(); size_of::<RamRafBucketDescriptor>()];
const _: [(); 4] = [(); align_of::<RamRafBucketDescriptor>()];

impl RamRafBucketDescriptor {
    pub const fn first_record(self) -> u32 {
        self.first_record
    }

    pub const fn record_count(self) -> u32 {
        self.record_count
    }

    pub const fn outer(self) -> u32 {
        self.outer
    }

    pub const fn tile(self) -> u32 {
        self.tile
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RamRafBucketProjection {
    pub records: Vec<RamRafBucketRecord>,
    pub descriptors: Vec<RamRafBucketDescriptor>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafDirectParams {
    pub record_count: u32,
    pub rows: u32,
    pub addresses: u32,
    pub inner_length: u32,
    pub outer_length: u32,
    pub accumulator_words: u32,
    pub threads: u32,
    pub reserved: u32,
}

const _: [(); 32] = [(); size_of::<RamRafDirectParams>()];
const _: [(); 4] = [(); align_of::<RamRafDirectParams>()];

impl RamRafDirectParams {
    pub fn new(
        record_count: usize,
        rows: usize,
        addresses: usize,
    ) -> Result<Self, RamRafCompactError> {
        validate_geometry(rows, addresses)?;
        if record_count > rows || record_count > u32::MAX as usize {
            return Err(RamRafCompactError::TooManyRecords(record_count));
        }
        Ok(Self {
            record_count: record_count as u32,
            rows: rows as u32,
            addresses: addresses as u32,
            inner_length: RAM_RAF_SUCCESSOR_INNER_LENGTH as u32,
            outer_length: (rows / RAM_RAF_SUCCESSOR_INNER_LENGTH) as u32,
            accumulator_words: RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS as u32,
            threads: RAM_RAF_SUCCESSOR_DIRECT_THREADS as u32,
            reserved: 0,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafBucketedParams {
    pub descriptor_count: u32,
    pub record_count: u32,
    pub rows: u32,
    pub addresses: u32,
    pub inner_length: u32,
    pub outer_length: u32,
    pub tile_addresses: u32,
    pub tiles: u32,
    pub accumulator_words: u32,
    pub threads: u32,
    pub reserved: [u32; 2],
}

const _: [(); 48] = [(); size_of::<RamRafBucketedParams>()];
const _: [(); 4] = [(); align_of::<RamRafBucketedParams>()];

impl RamRafBucketedParams {
    pub fn new(
        projection: &RamRafBucketProjection,
        rows: usize,
        addresses: usize,
    ) -> Result<Self, RamRafCompactError> {
        validate_bucket_projection(projection, rows, addresses)?;
        let descriptor_count = u32::try_from(projection.descriptors.len())
            .map_err(|_| RamRafCompactError::TooManyDescriptors(projection.descriptors.len()))?;
        let record_count = u32::try_from(projection.records.len())
            .map_err(|_| RamRafCompactError::TooManyRecords(projection.records.len()))?;
        Ok(Self {
            descriptor_count,
            record_count,
            rows: rows as u32,
            addresses: addresses as u32,
            inner_length: RAM_RAF_SUCCESSOR_INNER_LENGTH as u32,
            outer_length: (rows / RAM_RAF_SUCCESSOR_INNER_LENGTH) as u32,
            tile_addresses: RAM_RAF_SUCCESSOR_TILE_ADDRESSES as u32,
            tiles: RAM_RAF_SUCCESSOR_TILE_COUNT as u32,
            accumulator_words: RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS as u32,
            threads: RAM_RAF_SUCCESSOR_BUCKET_THREADS as u32,
            reserved: [0; 2],
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafFinalizeParams {
    pub addresses: u32,
    pub accumulator_words: u32,
    pub threads: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<RamRafFinalizeParams>()];
const _: [(); 4] = [(); align_of::<RamRafFinalizeParams>()];

impl RamRafFinalizeParams {
    pub fn new(addresses: usize) -> Result<Self, RamRafCompactError> {
        if addresses != RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN {
            return Err(RamRafCompactError::InvalidAddressDomain(addresses));
        }
        Ok(Self {
            addresses: addresses as u32,
            accumulator_words: RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS as u32,
            threads: RAM_RAF_SUCCESSOR_FINALIZE_THREADS as u32,
            reserved: 0,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafStatus {
    pub flags: u32,
    pub invalid_records: u32,
    pub reserved: [u32; 2],
}

const _: [(); 16] = [(); size_of::<RamRafStatus>()];
const _: [(); 4] = [(); align_of::<RamRafStatus>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RamRafCompactError {
    InvalidRows(usize),
    InvalidAddressDomain(usize),
    TooManyRecords(usize),
    TooManyDescriptors(usize),
    RecordsNotStrictlyOrdered { index: usize },
    CycleOutsideDomain { cycle: u32, rows: usize },
    AddressOutsideDomain { address: u32, addresses: usize },
    InnerOutsideBlock { inner: u32 },
    LocalAddressOutsideTile { local_address: u32 },
    ProjectionLengthMismatch { expected: usize, got: usize },
    EmptyBucket { descriptor: usize },
    DescriptorOrder { descriptor: usize },
    DescriptorRange { descriptor: usize },
    DescriptorOuter { descriptor: usize, outer: u32 },
    DescriptorTile { descriptor: usize, tile: u32 },
    ReservedRecordBits { record: usize },
    ProjectionRecordOutsideBucket { record: usize },
    ArithmeticOverflow,
}

pub fn validate_access_records(
    records: &[RamRafAccessRecord],
    rows: usize,
    addresses: usize,
) -> Result<(), RamRafCompactError> {
    validate_geometry(rows, addresses)?;
    if records.len() > u32::MAX as usize {
        return Err(RamRafCompactError::TooManyRecords(records.len()));
    }
    let mut previous = None;
    for (index, record) in records.iter().copied().enumerate() {
        if previous.is_some_and(|cycle| record.cycle <= cycle) {
            return Err(RamRafCompactError::RecordsNotStrictlyOrdered { index });
        }
        if record.cycle as usize >= rows {
            return Err(RamRafCompactError::CycleOutsideDomain {
                cycle: record.cycle,
                rows,
            });
        }
        if record.address as usize >= addresses {
            return Err(RamRafCompactError::AddressOutsideDomain {
                address: record.address,
                addresses,
            });
        }
        previous = Some(record.cycle);
    }
    Ok(())
}

pub fn build_bucket_projection(
    records: &[RamRafAccessRecord],
    rows: usize,
    addresses: usize,
) -> Result<RamRafBucketProjection, RamRafCompactError> {
    validate_access_records(records, rows, addresses)?;
    let mut buckets: [Vec<RamRafBucketRecord>; RAM_RAF_SUCCESSOR_TILE_COUNT] =
        core::array::from_fn(|_| Vec::new());
    let mut projection = RamRafBucketProjection {
        records: Vec::with_capacity(records.len()),
        descriptors: Vec::with_capacity(
            records
                .len()
                .min(rows / RAM_RAF_SUCCESSOR_INNER_LENGTH * RAM_RAF_SUCCESSOR_TILE_COUNT),
        ),
    };
    let mut current_outer = None;
    for record in records.iter().copied() {
        let cycle = record.cycle as usize;
        let address = record.address as usize;
        let outer = cycle / RAM_RAF_SUCCESSOR_INNER_LENGTH;
        if current_outer != Some(outer) {
            if let Some(previous_outer) = current_outer {
                flush_outer(previous_outer, &mut buckets, &mut projection)?;
            }
            current_outer = Some(outer);
        }
        let inner = cycle % RAM_RAF_SUCCESSOR_INNER_LENGTH;
        let tile = address / RAM_RAF_SUCCESSOR_TILE_ADDRESSES;
        let local_address = address % RAM_RAF_SUCCESSOR_TILE_ADDRESSES;
        buckets[tile].push(RamRafBucketRecord::new(inner as u32, local_address as u32)?);
    }
    if let Some(outer) = current_outer {
        flush_outer(outer, &mut buckets, &mut projection)?;
    }
    validate_bucket_projection(&projection, rows, addresses)?;
    Ok(projection)
}

fn flush_outer(
    outer: usize,
    buckets: &mut [Vec<RamRafBucketRecord>; RAM_RAF_SUCCESSOR_TILE_COUNT],
    projection: &mut RamRafBucketProjection,
) -> Result<(), RamRafCompactError> {
    for (tile, entries) in buckets.iter_mut().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let first_record = u32::try_from(projection.records.len())
            .map_err(|_| RamRafCompactError::ArithmeticOverflow)?;
        let record_count =
            u32::try_from(entries.len()).map_err(|_| RamRafCompactError::ArithmeticOverflow)?;
        projection.descriptors.push(RamRafBucketDescriptor {
            first_record,
            record_count,
            outer: outer as u32,
            tile: tile as u32,
        });
        projection.records.extend(entries.iter().copied());
        entries.clear();
    }
    Ok(())
}

pub fn validate_bucket_projection(
    projection: &RamRafBucketProjection,
    rows: usize,
    addresses: usize,
) -> Result<(), RamRafCompactError> {
    validate_geometry(rows, addresses)?;
    let outer_length = rows / RAM_RAF_SUCCESSOR_INNER_LENGTH;
    let mut next_record = 0usize;
    let mut previous_key = None;
    for (descriptor_index, descriptor) in projection.descriptors.iter().copied().enumerate() {
        if descriptor.record_count == 0 {
            return Err(RamRafCompactError::EmptyBucket {
                descriptor: descriptor_index,
            });
        }
        if descriptor.outer as usize >= outer_length {
            return Err(RamRafCompactError::DescriptorOuter {
                descriptor: descriptor_index,
                outer: descriptor.outer,
            });
        }
        if descriptor.tile as usize >= RAM_RAF_SUCCESSOR_TILE_COUNT {
            return Err(RamRafCompactError::DescriptorTile {
                descriptor: descriptor_index,
                tile: descriptor.tile,
            });
        }
        let key = (descriptor.outer, descriptor.tile);
        if previous_key.is_some_and(|previous| key <= previous) {
            return Err(RamRafCompactError::DescriptorOrder {
                descriptor: descriptor_index,
            });
        }
        previous_key = Some(key);

        let first = descriptor.first_record as usize;
        let end = first
            .checked_add(descriptor.record_count as usize)
            .ok_or(RamRafCompactError::ArithmeticOverflow)?;
        if first != next_record || end > projection.records.len() {
            return Err(RamRafCompactError::DescriptorRange {
                descriptor: descriptor_index,
            });
        }
        let tile_start = descriptor.tile as usize * RAM_RAF_SUCCESSOR_TILE_ADDRESSES;
        let active = RAM_RAF_SUCCESSOR_TILE_ADDRESSES.min(addresses - tile_start);
        for (record_index, record) in projection.records[first..end].iter().copied().enumerate() {
            let absolute = first + record_index;
            if record.reserved() != 0 {
                return Err(RamRafCompactError::ReservedRecordBits { record: absolute });
            }
            if record.inner() as usize >= RAM_RAF_SUCCESSOR_INNER_LENGTH
                || record.local_address() as usize >= active
            {
                return Err(RamRafCompactError::ProjectionRecordOutsideBucket { record: absolute });
            }
        }
        next_record = end;
    }
    if next_record != projection.records.len() {
        return Err(RamRafCompactError::ProjectionLengthMismatch {
            expected: next_record,
            got: projection.records.len(),
        });
    }
    Ok(())
}

fn validate_geometry(rows: usize, addresses: usize) -> Result<(), RamRafCompactError> {
    if rows < RAM_RAF_SUCCESSOR_INNER_LENGTH
        || !rows.is_power_of_two()
        || !rows.is_multiple_of(RAM_RAF_SUCCESSOR_INNER_LENGTH)
        || rows > u32::MAX as usize
    {
        return Err(RamRafCompactError::InvalidRows(rows));
    }
    if addresses != RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN {
        return Err(RamRafCompactError::InvalidAddressDomain(addresses));
    }
    Ok(())
}
