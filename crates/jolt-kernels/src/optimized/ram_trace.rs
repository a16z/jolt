//! The shared RAM access columns: one typed trace walk serving every
//! RAM-family kernel in this backend, parked in the [`ProofSession`] so the
//! stage-2 kernel's walk is reused by stages 4 and 5.
//!
//! The columns are the sparse view of the `(K × T)` RAM grids: per cycle,
//! the remapped word address (or a no-access sentinel) plus the pre- and
//! post-access word values. `ra(k, j)` is 1 exactly at `(addresses[j], j)`;
//! `val(k, j)` walks from the initial state through the writes.

#[cfg(any(feature = "parallel", all(feature = "metal", target_os = "macos")))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "parallel")]
use std::sync::OnceLock;

use jolt_field::JoltField;
use jolt_witness::witnesses::{
    RamHammingWeight, RamInc, RamReadValue, RamWriteValue, RemappedRamAddress,
};
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle};
#[cfg(feature = "parallel")]
use jolt_witness::{RandomAccessRows, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::ram_access::{RamAccessRecord, RamAccessTape, MAX_RETAINED_RAM_ACCESSES};
use crate::{KernelError, ProofSession};

/// `addresses` sentinel for cycles with no (remappable) RAM access.
pub(crate) const NO_ACCESS: u32 = u32::MAX;

#[cfg(all(feature = "metal", target_os = "macos"))]
const MAX_RAM_RA_COMPACT_ACCESSES: usize = 1 << 23;

#[cfg(all(feature = "metal", target_os = "macos"))]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RamRaCompactRecord {
    cycle: u32,
    address: u32,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RamRaQRecord {
    x_hi: u32,
    address: u32,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
const _: [(); 8] = [(); std::mem::size_of::<RamRaCompactRecord>()];
#[cfg(all(feature = "metal", target_os = "macos"))]
const _: [(); 8] = [(); std::mem::size_of::<RamRaQRecord>()];

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamRaSparseLayout {
    q_offsets: Vec<u32>,
    q_records: Vec<RamRaQRecord>,
    h_offsets: Vec<u32>,
    h_records: Vec<RamRaCompactRecord>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamRaSparseLayout {
    fn build(log_t: usize, h_records: Option<Vec<RamRaCompactRecord>>) -> Option<Self> {
        let h_records = h_records?;
        let prefix_bits = log_t / 2;
        let prefix_elements = 1usize << prefix_bits;
        let suffix_elements = 1usize << (log_t - prefix_bits);
        let mut q_offsets = vec![0u32; prefix_elements + 1];
        let mut h_offsets = vec![0u32; suffix_elements + 1];
        for record in &h_records {
            let cycle = record.cycle as usize;
            q_offsets[(cycle & (prefix_elements - 1)) + 1] += 1;
            h_offsets[(cycle >> prefix_bits) + 1] += 1;
        }
        for index in 1..q_offsets.len() {
            q_offsets[index] += q_offsets[index - 1];
        }
        for index in 1..h_offsets.len() {
            h_offsets[index] += h_offsets[index - 1];
        }
        let mut cursors = q_offsets[..prefix_elements].to_vec();
        let mut q_records = vec![RamRaQRecord::default(); h_records.len()];
        for record in &h_records {
            let cycle = record.cycle as usize;
            let x_lo = cycle & (prefix_elements - 1);
            let destination = cursors[x_lo] as usize;
            q_records[destination] = RamRaQRecord {
                x_hi: (cycle >> prefix_bits) as u32,
                address: record.address,
            };
            cursors[x_lo] += 1;
        }
        Some(Self {
            q_offsets,
            q_records,
            h_offsets,
            h_records,
        })
    }

    pub(crate) fn q_offsets(&self) -> &[u32] {
        &self.q_offsets
    }

    pub(crate) fn q_records(&self) -> &[RamRaQRecord] {
        &self.q_records
    }

    pub(crate) fn h_offsets(&self) -> &[u32] {
        &self.h_offsets
    }

    pub(crate) fn h_records(&self) -> &[RamRaCompactRecord] {
        &self.h_records
    }
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
pub(crate) struct RamAccessBundle {
    pub(crate) address: RemappedRamAddress,
    pub(crate) pre_value: RamReadValue,
    pub(crate) post_value: RamWriteValue,
    pub(crate) ram_inc: RamInc,
    pub(crate) ram_hamming_weight: RamHammingWeight,
}

#[derive(Clone, Copy, Debug)]
enum AddressEncodingError {
    TooLarge,
    SentinelCollision,
    CycleTooLarge,
}

impl AddressEncodingError {
    const fn reason(self) -> &'static str {
        match self {
            Self::TooLarge => "optimized RAM kernels require remapped addresses below 2^32 - 1",
            Self::SentinelCollision => {
                "optimized RAM kernels reserve u32::MAX as the no-access sentinel"
            }
            Self::CycleTooLarge => "optimized RAM kernels require cycle indices below 2^32",
        }
    }

    fn into_kernel_error<F: JoltField>(self) -> KernelError<F> {
        KernelError::Unsupported {
            reason: self.reason(),
        }
    }
}

fn encode_address(address: Option<u64>) -> Result<u32, AddressEncodingError> {
    let Some(address) = address else {
        return Ok(NO_ACCESS);
    };
    let address = u32::try_from(address).map_err(|_| AddressEncodingError::TooLarge)?;
    if address == NO_ACCESS {
        return Err(AddressEncodingError::SentinelCollision);
    }
    Ok(address)
}

struct CollectRamAccessColumns {
    addresses: Vec<u32>,
    active_cycle_bound: usize,
    required_address_domain: usize,
    pre_values: Vec<u64>,
    post_values: Vec<u64>,
    ram_increment_cycles: Vec<u64>,
    ram_increments: Vec<i128>,
    access_count: usize,
    access_records: Option<Vec<RamAccessRecord>>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    ram_ra_records: Option<Vec<RamRaCompactRecord>>,
    increment_compatible: bool,
    ram_ra_compatible: bool,
    hamming_exact: bool,
    address_error: Option<AddressEncodingError>,
}

impl StreamConsumer for CollectRamAccessColumns {
    type Witness = RamAccessBundle;

    fn consume(&mut self, chunk: &[RamAccessBundle]) {
        for bundle in chunk {
            let address = encode_address(bundle.address.0).unwrap_or_else(|failure| {
                let _ = self.address_error.get_or_insert(failure);
                NO_ACCESS
            });
            self.addresses.push(address);
            self.pre_values.push(bundle.pre_value.0);
            self.post_values.push(bundle.post_value.0);
            let cycle = self.addresses.len() - 1;
            if let Some((cycle, increment)) = encode_ram_increment(cycle, bundle.ram_inc.0) {
                self.ram_increment_cycles.push(cycle);
                self.ram_increments.push(increment);
            }
            let delta = i128::from(bundle.post_value.0) - i128::from(bundle.pre_value.0);
            self.increment_compatible &=
                bundle.ram_inc.0 == delta && (address != NO_ACCESS || bundle.ram_inc.0 == 0);
            self.ram_ra_compatible &= !(bundle.ram_hamming_weight.0 && address == NO_ACCESS);
            self.hamming_exact &= bundle.ram_hamming_weight.0 == (address != NO_ACCESS);
            if address != NO_ACCESS {
                self.active_cycle_bound = cycle + 1;
                self.required_address_domain =
                    self.required_address_domain.max(address as usize + 1);
                self.access_count += 1;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if self
                    .ram_ra_records
                    .as_ref()
                    .is_some_and(|records| records.len() == MAX_RAM_RA_COMPACT_ACCESSES)
                {
                    self.ram_ra_records = None;
                } else if let Some(records) = &mut self.ram_ra_records {
                    match u32::try_from(cycle) {
                        Ok(cycle) => records.push(RamRaCompactRecord { cycle, address }),
                        Err(_) => {
                            let _ = self
                                .address_error
                                .get_or_insert(AddressEncodingError::CycleTooLarge);
                        }
                    }
                }
                if self
                    .access_records
                    .as_ref()
                    .is_some_and(|records| records.len() == MAX_RETAINED_RAM_ACCESSES)
                {
                    self.access_records = None;
                } else if let Some(records) = &mut self.access_records {
                    match u32::try_from(cycle) {
                        Ok(cycle) => records.push(RamAccessRecord {
                            cycle,
                            address,
                            pre_value: bundle.pre_value.0,
                            post_value: bundle.post_value.0,
                        }),
                        Err(_) => {
                            let _ = self
                                .address_error
                                .get_or_insert(AddressEncodingError::CycleTooLarge);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
enum CollectFailure {
    Witness(WitnessError),
    Address(AddressEncodingError),
}

struct SparseChunk {
    access_records: Vec<RamAccessRecord>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    ram_ra_records: Vec<RamRaCompactRecord>,
    activity_records: Vec<(u64, i128)>,
    active_cycle_bound: usize,
    required_address_domain: usize,
    increment_compatible: bool,
    ram_ra_compatible: bool,
    hamming_exact: bool,
}

/// Uninitialized final-column storage filled alongside another random-access
/// witness projection. The writer API keeps each chunk disjoint; sealing is
/// the only operation that turns the initialized planes into session state.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamAccessCollectionStorage {
    log_t: usize,
    cycles: usize,
    chunk_rows: usize,
    addresses: Vec<u32>,
    pre_values: Vec<u64>,
    post_values: Vec<u64>,
    sparse_chunks: Vec<Option<SparseChunk>>,
    access_count: AtomicUsize,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
struct RamReadWriteRecordChunk {
    records: Box<[RamAccessRecord]>,
    active_cycle_bound: usize,
    required_address_domain: usize,
    increment_compatible: bool,
    ram_ra_compatible: bool,
    hamming_exact: bool,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamReadWriteRecordCollectionStorage {
    log_t: usize,
    tile_log: usize,
    cycles: usize,
    chunk_rows: usize,
    addresses: Vec<u32>,
    chunks: Vec<Option<RamReadWriteRecordChunk>>,
    access_count: AtomicUsize,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamReadWriteRecordCollectionChunkWriter<'a> {
    base: usize,
    written: usize,
    addresses: &'a mut [std::mem::MaybeUninit<u32>],
    chunk: &'a mut Option<RamReadWriteRecordChunk>,
    access_count: &'a AtomicUsize,
    records: Vec<RamAccessRecord>,
    active_cycle_bound: usize,
    required_address_domain: usize,
    increment_compatible: bool,
    ram_ra_compatible: bool,
    hamming_exact: bool,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamReadWriteRecordChunks {
    rows: usize,
    chunks: Vec<AlignedRamReadWriteRecordArena>,
    worker_census: Vec<RamReadWriteRecordWorkerCensus>,
    address_count: usize,
    tile_log: usize,
    access_count: usize,
    compaction_wall_ns: u64,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamReadWriteRecordWorkerCensus {
    address_counts: Box<[u32]>,
    tile_counts: Box<[u32]>,
    initial_values: Vec<(u32, u64)>,
    increment_cycles: Vec<u64>,
    increments: Vec<i128>,
    accesses: usize,
    first_cycle: Option<u32>,
    last_cycle: Option<u32>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
const RAM_READ_WRITE_RECORD_ALIGNMENT: usize = 16 * 1024;

/// Page-aligned record and rank planes that can back borrowed Metal buffers.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct AlignedRamReadWriteRecordArena {
    ptr: core::ptr::NonNull<u8>,
    records: usize,
    record_allocation_bytes: usize,
    total_allocation_bytes: usize,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl AlignedRamReadWriteRecordArena {
    fn allocate(records: usize) -> Result<Self, RamAccessCollectionError> {
        let align_bytes = |bytes: usize| {
            bytes
                .max(1)
                .checked_add(RAM_READ_WRITE_RECORD_ALIGNMENT - 1)
                .map(|rounded| {
                    rounded / RAM_READ_WRITE_RECORD_ALIGNMENT * RAM_READ_WRITE_RECORD_ALIGNMENT
                })
                .ok_or_else(|| {
                    RamAccessCollectionError::new("aligned RAM record allocation overflowed")
                })
        };
        let record_bytes = records
            .checked_mul(std::mem::size_of::<RamAccessRecord>())
            .ok_or_else(|| RamAccessCollectionError::new("RAM record byte length overflowed"))?;
        let rank_bytes = records
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| RamAccessCollectionError::new("RAM rank byte length overflowed"))?;
        let record_allocation_bytes = align_bytes(record_bytes)?;
        let rank_allocation_bytes = align_bytes(rank_bytes)?;
        let total_allocation_bytes = record_allocation_bytes
            .checked_add(rank_allocation_bytes)
            .ok_or_else(|| RamAccessCollectionError::new("RAM arena byte length overflowed"))?;
        let layout = std::alloc::Layout::from_size_align(
            total_allocation_bytes,
            RAM_READ_WRITE_RECORD_ALIGNMENT,
        )
        .map_err(|_| RamAccessCollectionError::new("RAM arena allocation layout is invalid"))?;
        // SAFETY: `layout` has a nonzero page-aligned size.
        let raw = unsafe { std::alloc::alloc(layout) };
        let ptr = core::ptr::NonNull::new(raw)
            .ok_or_else(|| RamAccessCollectionError::new("RAM arena allocation failed"))?;
        Ok(Self {
            ptr,
            records,
            record_allocation_bytes,
            total_allocation_bytes,
        })
    }

    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the arena allocation uses RAM_READ_WRITE_RECORD_ALIGNMENT"
    )]
    fn records_mut(&mut self) -> &mut [std::mem::MaybeUninit<RamAccessRecord>] {
        // SAFETY: the first allocation region owns exactly `records` aligned
        // slots and construction has exclusive access.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr
                    .as_ptr()
                    .cast::<std::mem::MaybeUninit<RamAccessRecord>>(),
                self.records,
            )
        }
    }

    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the rank region begins at a RAM_READ_WRITE_RECORD_ALIGNMENT boundary"
    )]
    fn ranks_mut(&mut self) -> &mut [std::mem::MaybeUninit<u32>] {
        // SAFETY: the rank region begins at its page-aligned offset and owns
        // exactly `records` u32 slots during construction.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr
                    .as_ptr()
                    .add(self.record_allocation_bytes)
                    .cast::<std::mem::MaybeUninit<u32>>(),
                self.records,
            )
        }
    }

    fn finish(&mut self) {
        let record_bytes = self.records * std::mem::size_of::<RamAccessRecord>();
        let rank_bytes = self.records * std::mem::size_of::<u32>();
        // SAFETY: every logical slot was initialized by the compacting loop;
        // these writes initialize only the two padded allocation tails.
        unsafe {
            self.ptr
                .as_ptr()
                .add(record_bytes)
                .write_bytes(0, self.record_allocation_bytes - record_bytes);
            self.ptr
                .as_ptr()
                .add(self.record_allocation_bytes + rank_bytes)
                .write_bytes(
                    0,
                    self.total_allocation_bytes - self.record_allocation_bytes - rank_bytes,
                );
        }
    }

    pub(crate) fn records(&self) -> &[RamAccessRecord] {
        // SAFETY: construction initializes every logical record before the
        // arena is published and the owner keeps the allocation alive.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr().cast(), self.records) }
    }

    pub(crate) fn ranks(&self) -> &[u32] {
        // SAFETY: construction initializes every logical rank before the
        // arena is published and the owner keeps the allocation alive.
        unsafe {
            core::slice::from_raw_parts(
                self.ptr.as_ptr().add(self.record_allocation_bytes).cast(),
                self.records,
            )
        }
    }

    pub(crate) fn record_pointer(&self) -> *const RamAccessRecord {
        self.ptr.as_ptr().cast_const().cast()
    }

    pub(crate) fn rank_pointer(&self) -> *const u32 {
        // SAFETY: the checked allocation contains the rank region at this
        // page-aligned offset.
        unsafe {
            self.ptr
                .as_ptr()
                .add(self.record_allocation_bytes)
                .cast_const()
                .cast()
        }
    }

    pub(crate) const fn record_allocation_bytes(&self) -> usize {
        self.record_allocation_bytes
    }

    pub(crate) const fn rank_allocation_bytes(&self) -> usize {
        self.total_allocation_bytes - self.record_allocation_bytes
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Drop for AlignedRamReadWriteRecordArena {
    fn drop(&mut self) {
        // SAFETY: these are the exact size and alignment used by `allocate`.
        let layout = unsafe {
            std::alloc::Layout::from_size_align_unchecked(
                self.total_allocation_bytes,
                RAM_READ_WRITE_RECORD_ALIGNMENT,
            )
        };
        // SAFETY: `ptr` owns the live allocation described by `layout`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

// SAFETY: the owner publishes only immutable access after construction.
#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe impl Send for AlignedRamReadWriteRecordArena {}
// SAFETY: the owner publishes only immutable access after construction.
#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe impl Sync for AlignedRamReadWriteRecordArena {}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamReadWriteRecordCollection {
    columns: RamAccessColumns,
    records: RamReadWriteRecordChunks,
    tape: RamAccessTape,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamAccessCollectionChunkWriter<'a> {
    base: usize,
    written: usize,
    addresses: &'a mut [std::mem::MaybeUninit<u32>],
    pre_values: &'a mut [std::mem::MaybeUninit<u64>],
    post_values: &'a mut [std::mem::MaybeUninit<u64>],
    sparse_chunk: &'a mut Option<SparseChunk>,
    access_count: &'a AtomicUsize,
    access_records: Vec<RamAccessRecord>,
    ram_ra_records: Vec<RamRaCompactRecord>,
    activity_records: Vec<(u64, i128)>,
    active_cycle_bound: usize,
    required_address_domain: usize,
    increment_compatible: bool,
    ram_ra_compatible: bool,
    hamming_exact: bool,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) struct RamAccessCollection {
    columns: RamAccessColumns,
    values: RamAccessValues,
    activity: RamIncrementActivity,
    tape: RamAccessTape,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug)]
pub(crate) struct RamAccessCollectionError {
    reason: &'static str,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamAccessCollectionError {
    const fn new(reason: &'static str) -> Self {
        Self { reason }
    }

    pub(crate) const fn reason(&self) -> &'static str {
        self.reason
    }

    pub(crate) fn into_kernel_error<F: JoltField>(self) -> KernelError<F> {
        KernelError::InvariantViolation {
            reason: self.reason,
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl From<AddressEncodingError> for RamAccessCollectionError {
    fn from(error: AddressEncodingError) -> Self {
        Self::new(error.reason())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamReadWriteRecordCollectionStorage {
    pub(crate) fn new(
        cycles: usize,
        chunk_rows: usize,
        tile_log: usize,
    ) -> Result<Self, RamAccessCollectionError> {
        if !cycles.is_power_of_two() {
            return Err(RamAccessCollectionError::new(
                "RAM record collection requires a power-of-two cycle domain",
            ));
        }
        let log_t = cycles.ilog2() as usize;
        if log_t > u32::BITS as usize {
            return Err(RamAccessCollectionError::new(
                "RAM record collection requires at most 32 cycle variables",
            ));
        }
        if chunk_rows == 0 || tile_log > log_t {
            return Err(RamAccessCollectionError::new(
                "RAM record collection chunk geometry is invalid",
            ));
        }
        Ok(Self {
            log_t,
            tile_log,
            cycles,
            chunk_rows,
            addresses: Vec::with_capacity(cycles),
            chunks: std::iter::repeat_with(|| None)
                .take(cycles.div_ceil(chunk_rows))
                .collect(),
            access_count: AtomicUsize::new(0),
        })
    }

    pub(crate) fn with_chunk_writers<R, E>(
        &mut self,
        fill: impl FnOnce(&mut [RamReadWriteRecordCollectionChunkWriter<'_>]) -> Result<R, E>,
    ) -> Result<R, E> {
        debug_assert!(self.addresses.is_empty());
        let addresses = &mut self.addresses.spare_capacity_mut()[..self.cycles];
        let mut writers = addresses
            .chunks_mut(self.chunk_rows)
            .zip(self.chunks.iter_mut())
            .enumerate()
            .map(
                |(chunk, (addresses, output))| RamReadWriteRecordCollectionChunkWriter {
                    base: chunk * self.chunk_rows,
                    written: 0,
                    addresses,
                    chunk: output,
                    access_count: &self.access_count,
                    records: Vec::with_capacity(self.chunk_rows / 4),
                    active_cycle_bound: 0,
                    required_address_domain: 0,
                    increment_compatible: true,
                    ram_ra_compatible: true,
                    hamming_exact: true,
                },
            )
            .collect::<Vec<_>>();
        fill(&mut writers)
    }

    pub(crate) fn seal(mut self) -> Result<RamReadWriteRecordCollection, RamAccessCollectionError> {
        if self.chunks.iter().any(Option::is_none) {
            return Err(RamAccessCollectionError::new(
                "RAM record collection left an incomplete chunk",
            ));
        }
        // SAFETY: each chunk publishes only after initializing its complete,
        // disjoint address slice.
        unsafe {
            self.addresses.set_len(self.cycles);
        }
        let access_count = self.access_count.load(Ordering::Relaxed);
        let mut active_cycle_bound = 0;
        let mut required_address_domain = 0;
        let mut increment_compatible = true;
        let mut ram_ra_compatible = true;
        let mut hamming_exact = true;
        let mut observed_accesses = 0usize;
        let compaction_started = std::time::Instant::now();
        let mut chunks = Vec::with_capacity(self.chunks.len());
        for chunk in self.chunks {
            let chunk = chunk.ok_or_else(|| {
                RamAccessCollectionError::new("RAM record collection lost a completed chunk")
            })?;
            active_cycle_bound = active_cycle_bound.max(chunk.active_cycle_bound);
            required_address_domain = required_address_domain.max(chunk.required_address_domain);
            increment_compatible &= chunk.increment_compatible;
            ram_ra_compatible &= chunk.ram_ra_compatible;
            hamming_exact &= chunk.hamming_exact;
            observed_accesses = observed_accesses
                .checked_add(chunk.records.len())
                .ok_or_else(|| {
                    RamAccessCollectionError::new("RAM record count overflowed usize")
                })?;
            chunks.push(chunk);
        }
        if observed_accesses != access_count {
            return Err(RamAccessCollectionError::new(
                "RAM record collection access counts disagree",
            ));
        }
        #[cfg(feature = "parallel")]
        let worker_count = rayon::current_num_threads().min(chunks.len().max(1));
        #[cfg(not(feature = "parallel"))]
        let worker_count = 1;
        let chunks_per_worker = chunks.len().div_ceil(worker_count);
        let mut groups = std::iter::repeat_with(Vec::new)
            .take(worker_count)
            .collect::<Vec<Vec<RamReadWriteRecordChunk>>>();
        for (chunk, source) in chunks.into_iter().enumerate() {
            groups[chunk / chunks_per_worker].push(source);
        }
        let address_count = required_address_domain
            .checked_next_power_of_two()
            .ok_or_else(|| {
                RamAccessCollectionError::new("RAM record address domain overflowed usize")
            })?
            .max(1);
        let tile_log = self.tile_log;
        let tile_count = 1usize << (self.log_t - tile_log);
        let compact = |group: Vec<RamReadWriteRecordChunk>| {
            let records = group.iter().map(|chunk| chunk.records.len()).sum();
            let mut compacted = AlignedRamReadWriteRecordArena::allocate(records)?;
            let mut address_counts = vec![0u32; address_count];
            let mut tile_counts = vec![0u32; tile_count];
            let mut initial_values = Vec::new();
            let mut increment_cycles = Vec::with_capacity(records / 2);
            let mut increments = Vec::with_capacity(records / 2);
            let mut first_cycle = None;
            let mut last_cycle = None;
            let mut destination = 0usize;
            for chunk in group {
                for record in &chunk.records {
                    let address = record.address as usize;
                    let rank = address_counts[address];
                    if rank == 0 {
                        initial_values.push((record.address, record.pre_value));
                    }
                    address_counts[address] += 1;
                    tile_counts[(record.cycle as usize) >> tile_log] += 1;
                    let increment = i128::from(record.post_value) - i128::from(record.pre_value);
                    if increment != 0 {
                        increment_cycles.push(u64::from(record.cycle));
                        increments.push(increment);
                    }
                    let _ = first_cycle.get_or_insert(record.cycle);
                    last_cycle = Some(record.cycle);
                    let _ = compacted.records_mut()[destination].write(*record);
                    let _ = compacted.ranks_mut()[destination].write(rank);
                    destination += 1;
                }
            }
            if destination != records {
                return Err(RamAccessCollectionError::new(
                    "RAM arena compaction wrote the wrong record count",
                ));
            }
            compacted.finish();
            Ok((
                compacted,
                RamReadWriteRecordWorkerCensus {
                    address_counts: address_counts.into_boxed_slice(),
                    tile_counts: tile_counts.into_boxed_slice(),
                    initial_values,
                    increment_cycles,
                    increments,
                    accesses: records,
                    first_cycle,
                    last_cycle,
                },
            ))
        };
        #[cfg(feature = "parallel")]
        let compacted = groups
            .into_par_iter()
            .map(compact)
            .collect::<Result<Vec<_>, RamAccessCollectionError>>()?;
        #[cfg(not(feature = "parallel"))]
        let compacted = groups
            .into_iter()
            .map(compact)
            .collect::<Result<Vec<_>, RamAccessCollectionError>>()?;
        let (records, worker_census) = compacted.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        let compaction_wall_ns =
            u64::try_from(compaction_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        tracing::info!(
            target: "jolt::metal",
            records = access_count,
            record_bytes = access_count * std::mem::size_of::<RamAccessRecord>(),
            worker_arenas = records.len(),
            compaction_wall_ns,
            "compacted co-produced RAM records"
        );
        Ok(RamReadWriteRecordCollection {
            columns: RamAccessColumns {
                addresses: self.addresses,
                active_cycle_bound,
                required_address_domain,
                ram_ra_sparse: None,
            },
            records: RamReadWriteRecordChunks {
                rows: self.cycles,
                chunks: records,
                worker_census,
                address_count,
                tile_log,
                access_count,
                compaction_wall_ns,
            },
            tape: RamAccessTape::new(
                self.log_t,
                access_count,
                None,
                increment_compatible,
                ram_ra_compatible,
                hamming_exact,
            ),
        })
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamReadWriteRecordCollectionChunkWriter<'_> {
    pub(crate) fn len(&self) -> usize {
        self.addresses.len()
    }

    fn record(
        &mut self,
        cycle: usize,
        address: u32,
        bundle: RamAccessBundle,
    ) -> Result<(), RamAccessCollectionError> {
        let delta = i128::from(bundle.post_value.0) - i128::from(bundle.pre_value.0);
        self.increment_compatible &=
            bundle.ram_inc.0 == delta && (address != NO_ACCESS || bundle.ram_inc.0 == 0);
        self.ram_ra_compatible &= !(bundle.ram_hamming_weight.0 && address == NO_ACCESS);
        self.hamming_exact &= bundle.ram_hamming_weight.0 == (address != NO_ACCESS);
        if address == NO_ACCESS {
            return Ok(());
        }
        self.active_cycle_bound = cycle + 1;
        self.required_address_domain = self.required_address_domain.max(address as usize + 1);
        let cycle = u32::try_from(cycle)
            .map_err(|_| RamAccessCollectionError::from(AddressEncodingError::CycleTooLarge))?;
        self.records.push(RamAccessRecord {
            cycle,
            address,
            pre_value: bundle.pre_value.0,
            post_value: bundle.post_value.0,
        });
        Ok(())
    }

    pub(crate) fn push(&mut self, bundle: RamAccessBundle) -> Result<(), RamAccessCollectionError> {
        if self.written == self.len() {
            return Err(RamAccessCollectionError::new(
                "RAM record collection chunk received too many rows",
            ));
        }
        let address = encode_address(bundle.address.0)?;
        let offset = self.written;
        let _ = self.addresses[offset].write(address);
        self.record(self.base + offset, address, bundle)?;
        self.written += 1;
        Ok(())
    }

    pub(crate) fn fill_repeated(
        &mut self,
        bundle: RamAccessBundle,
        count: usize,
    ) -> Result<(), RamAccessCollectionError> {
        let end = self.written.checked_add(count).ok_or_else(|| {
            RamAccessCollectionError::new("RAM record collection row count overflowed")
        })?;
        if end > self.len() {
            return Err(RamAccessCollectionError::new(
                "RAM record collection padding exceeds its chunk",
            ));
        }
        let address = encode_address(bundle.address.0)?;
        self.addresses[self.written..end].fill(std::mem::MaybeUninit::new(address));
        if address == NO_ACCESS && bundle.ram_inc.0 == 0 {
            let delta = i128::from(bundle.post_value.0) - i128::from(bundle.pre_value.0);
            self.increment_compatible &= delta == 0;
            self.ram_ra_compatible &= !bundle.ram_hamming_weight.0;
            self.hamming_exact &= !bundle.ram_hamming_weight.0;
        } else {
            for offset in self.written..end {
                self.record(self.base + offset, address, bundle)?;
            }
        }
        self.written = end;
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), RamAccessCollectionError> {
        if self.written != self.len() {
            return Err(RamAccessCollectionError::new(
                "RAM record collection chunk was not fully initialized",
            ));
        }
        if self.chunk.is_some() {
            return Err(RamAccessCollectionError::new(
                "RAM record collection chunk was finished twice",
            ));
        }
        let records = std::mem::take(&mut self.records).into_boxed_slice();
        let _ = self
            .access_count
            .fetch_add(records.len(), Ordering::Relaxed);
        *self.chunk = Some(RamReadWriteRecordChunk {
            records,
            active_cycle_bound: self.active_cycle_bound,
            required_address_domain: self.required_address_domain,
            increment_compatible: self.increment_compatible,
            ram_ra_compatible: self.ram_ra_compatible,
            hamming_exact: self.hamming_exact,
        });
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamReadWriteRecordChunks {
    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn chunks(&self) -> &[AlignedRamReadWriteRecordArena] {
        &self.chunks
    }

    pub(crate) fn apply_initial_memory(
        &self,
        memory: &mut [u64],
    ) -> Result<(), RamAccessCollectionError> {
        if memory.len() != self.address_count {
            return Err(RamAccessCollectionError::new(
                "RAM record initial memory has the wrong address domain",
            ));
        }
        for worker in self.worker_census.iter().rev() {
            for &(address, value) in &worker.initial_values {
                memory[address as usize] = value;
            }
        }
        Ok(())
    }

    pub(crate) fn take_increment_chunks(&mut self) -> Vec<(Vec<u64>, Vec<i128>)> {
        self.worker_census
            .iter_mut()
            .map(|worker| {
                (
                    std::mem::take(&mut worker.increment_cycles),
                    std::mem::take(&mut worker.increments),
                )
            })
            .collect()
    }

    pub(crate) fn worker_census(&self) -> &[RamReadWriteRecordWorkerCensus] {
        &self.worker_census
    }

    pub(crate) const fn address_count(&self) -> usize {
        self.address_count
    }

    pub(crate) const fn tile_log(&self) -> usize {
        self.tile_log
    }

    pub(crate) const fn access_count(&self) -> usize {
        self.access_count
    }

    pub(crate) fn record_bytes(&self) -> usize {
        self.access_count * std::mem::size_of::<RamAccessRecord>()
    }

    pub(crate) const fn compaction_wall_ns(&self) -> u64 {
        self.compaction_wall_ns
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamReadWriteRecordWorkerCensus {
    pub(crate) fn address_counts(&self) -> &[u32] {
        &self.address_counts
    }

    pub(crate) fn tile_counts(&self) -> &[u32] {
        &self.tile_counts
    }

    pub(crate) const fn accesses(&self) -> usize {
        self.accesses
    }

    pub(crate) const fn first_cycle(&self) -> Option<u32> {
        self.first_cycle
    }

    pub(crate) const fn last_cycle(&self) -> Option<u32> {
        self.last_cycle
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamAccessCollectionStorage {
    pub(crate) fn new(cycles: usize, chunk_rows: usize) -> Result<Self, RamAccessCollectionError> {
        if !cycles.is_power_of_two() {
            return Err(RamAccessCollectionError::new(
                "RAM access collection requires a power-of-two cycle domain",
            ));
        }
        let log_t = cycles.ilog2() as usize;
        if log_t > u32::BITS as usize {
            return Err(RamAccessCollectionError::new(
                "RAM access collection requires at most 32 cycle variables",
            ));
        }
        if chunk_rows == 0 {
            return Err(RamAccessCollectionError::new(
                "RAM access collection chunk size must be nonzero",
            ));
        }
        Ok(Self {
            log_t,
            cycles,
            chunk_rows,
            addresses: Vec::with_capacity(cycles),
            pre_values: Vec::with_capacity(cycles),
            post_values: Vec::with_capacity(cycles),
            sparse_chunks: std::iter::repeat_with(|| None)
                .take(cycles.div_ceil(chunk_rows))
                .collect(),
            access_count: AtomicUsize::new(0),
        })
    }

    pub(crate) fn with_chunk_writers<R, E>(
        &mut self,
        fill: impl FnOnce(&mut [RamAccessCollectionChunkWriter<'_>]) -> Result<R, E>,
    ) -> Result<R, E> {
        debug_assert!(self.addresses.is_empty());
        debug_assert!(self.pre_values.is_empty());
        debug_assert!(self.post_values.is_empty());
        let addresses = &mut self.addresses.spare_capacity_mut()[..self.cycles];
        let pre_values = &mut self.pre_values.spare_capacity_mut()[..self.cycles];
        let post_values = &mut self.post_values.spare_capacity_mut()[..self.cycles];
        let mut writers = addresses
            .chunks_mut(self.chunk_rows)
            .zip(pre_values.chunks_mut(self.chunk_rows))
            .zip(post_values.chunks_mut(self.chunk_rows))
            .zip(self.sparse_chunks.iter_mut())
            .enumerate()
            .map(
                |(chunk, (((addresses, pre_values), post_values), sparse_chunk))| {
                    RamAccessCollectionChunkWriter {
                        base: chunk * self.chunk_rows,
                        written: 0,
                        addresses,
                        pre_values,
                        post_values,
                        sparse_chunk,
                        access_count: &self.access_count,
                        access_records: Vec::new(),
                        ram_ra_records: Vec::new(),
                        activity_records: Vec::new(),
                        active_cycle_bound: 0,
                        required_address_domain: 0,
                        increment_compatible: true,
                        ram_ra_compatible: true,
                        hamming_exact: true,
                    }
                },
            )
            .collect::<Vec<_>>();
        fill(&mut writers)
    }

    pub(crate) fn seal(mut self) -> Result<RamAccessCollection, RamAccessCollectionError> {
        if self.sparse_chunks.iter().any(Option::is_none) {
            return Err(RamAccessCollectionError::new(
                "RAM access collection left an incomplete chunk",
            ));
        }
        // SAFETY: a chunk publishes its SparseChunk only after initializing
        // every element in each of its three disjoint final-column slices.
        unsafe {
            self.addresses.set_len(self.cycles);
            self.pre_values.set_len(self.cycles);
            self.post_values.set_len(self.cycles);
        }
        let total_accesses = self.access_count.load(Ordering::Relaxed);
        let mut retained_records = (total_accesses <= MAX_RETAINED_RAM_ACCESSES)
            .then(|| Vec::with_capacity(total_accesses));
        let mut ram_ra_records = (total_accesses <= MAX_RAM_RA_COMPACT_ACCESSES)
            .then(|| Vec::with_capacity(total_accesses));
        let mut activity_records = Vec::new();
        let mut required_address_domain = 0;
        let mut increment_compatible = true;
        let mut ram_ra_compatible = true;
        let mut hamming_exact = true;
        let mut active_cycle_bound = 0;
        for chunk in self.sparse_chunks {
            let chunk = chunk.ok_or_else(|| {
                RamAccessCollectionError::new("RAM access collection lost a completed chunk")
            })?;
            required_address_domain = required_address_domain.max(chunk.required_address_domain);
            increment_compatible &= chunk.increment_compatible;
            ram_ra_compatible &= chunk.ram_ra_compatible;
            hamming_exact &= chunk.hamming_exact;
            active_cycle_bound = active_cycle_bound.max(chunk.active_cycle_bound);
            activity_records.extend(chunk.activity_records);
            if let Some(records) = &mut retained_records {
                records.extend(&chunk.access_records);
            }
            if let Some(records) = &mut ram_ra_records {
                records.extend(chunk.ram_ra_records);
            }
        }
        let (cycles, increments) = activity_records.into_iter().unzip();
        Ok(RamAccessCollection {
            columns: RamAccessColumns {
                addresses: self.addresses,
                active_cycle_bound,
                required_address_domain,
                ram_ra_sparse: RamRaSparseLayout::build(self.log_t, ram_ra_records),
            },
            values: RamAccessValues {
                pre_values: self.pre_values,
                post_values: self.post_values,
            },
            activity: RamIncrementActivity { cycles, increments },
            tape: RamAccessTape::new(
                self.log_t,
                total_accesses,
                retained_records,
                increment_compatible,
                ram_ra_compatible,
                hamming_exact,
            ),
        })
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamAccessCollectionChunkWriter<'_> {
    pub(crate) fn len(&self) -> usize {
        self.addresses.len()
    }

    fn record_sparse(
        &mut self,
        cycle: usize,
        address: u32,
        bundle: RamAccessBundle,
    ) -> Result<(), RamAccessCollectionError> {
        if let Some(record) = encode_ram_increment(cycle, bundle.ram_inc.0) {
            self.activity_records.push(record);
        }
        let delta = i128::from(bundle.post_value.0) - i128::from(bundle.pre_value.0);
        self.increment_compatible &=
            bundle.ram_inc.0 == delta && (address != NO_ACCESS || bundle.ram_inc.0 == 0);
        self.ram_ra_compatible &= !(bundle.ram_hamming_weight.0 && address == NO_ACCESS);
        self.hamming_exact &= bundle.ram_hamming_weight.0 == (address != NO_ACCESS);
        if address != NO_ACCESS {
            self.active_cycle_bound = self.active_cycle_bound.max(cycle + 1);
            self.required_address_domain = self.required_address_domain.max(address as usize + 1);
            let cycle = u32::try_from(cycle)
                .map_err(|_| RamAccessCollectionError::from(AddressEncodingError::CycleTooLarge))?;
            self.ram_ra_records
                .push(RamRaCompactRecord { cycle, address });
            self.access_records.push(RamAccessRecord {
                cycle,
                address,
                pre_value: bundle.pre_value.0,
                post_value: bundle.post_value.0,
            });
        }
        Ok(())
    }

    pub(crate) fn push(&mut self, bundle: RamAccessBundle) -> Result<(), RamAccessCollectionError> {
        if self.written == self.len() {
            return Err(RamAccessCollectionError::new(
                "RAM access collection chunk received too many rows",
            ));
        }
        let address = encode_address(bundle.address.0)?;
        let offset = self.written;
        let _ = self.addresses[offset].write(address);
        let _ = self.pre_values[offset].write(bundle.pre_value.0);
        let _ = self.post_values[offset].write(bundle.post_value.0);
        self.record_sparse(self.base + offset, address, bundle)?;
        self.written += 1;
        Ok(())
    }

    pub(crate) fn fill_repeated(
        &mut self,
        bundle: RamAccessBundle,
        count: usize,
    ) -> Result<(), RamAccessCollectionError> {
        let end = self.written.checked_add(count).ok_or_else(|| {
            RamAccessCollectionError::new("RAM access collection row count overflowed")
        })?;
        if end > self.len() {
            return Err(RamAccessCollectionError::new(
                "RAM access collection padding exceeds its chunk",
            ));
        }
        let address = encode_address(bundle.address.0)?;
        self.addresses[self.written..end].fill(std::mem::MaybeUninit::new(address));
        self.pre_values[self.written..end].fill(std::mem::MaybeUninit::new(bundle.pre_value.0));
        self.post_values[self.written..end].fill(std::mem::MaybeUninit::new(bundle.post_value.0));
        if address == NO_ACCESS && bundle.ram_inc.0 == 0 {
            let delta = i128::from(bundle.post_value.0) - i128::from(bundle.pre_value.0);
            self.increment_compatible &= delta == 0;
            self.ram_ra_compatible &= !bundle.ram_hamming_weight.0;
            self.hamming_exact &= !bundle.ram_hamming_weight.0;
        } else {
            for offset in self.written..end {
                self.record_sparse(self.base + offset, address, bundle)?;
            }
        }
        self.written = end;
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), RamAccessCollectionError> {
        if self.written != self.len() {
            return Err(RamAccessCollectionError::new(
                "RAM access collection chunk was not fully initialized",
            ));
        }
        if self.sparse_chunk.is_some() {
            return Err(RamAccessCollectionError::new(
                "RAM access collection chunk was finished twice",
            ));
        }
        let chunk_accesses = self.access_records.len();
        let preceding_accesses = self
            .access_count
            .fetch_add(chunk_accesses, Ordering::Relaxed);
        if preceding_accesses
            .checked_add(chunk_accesses)
            .is_none_or(|end| end > MAX_RETAINED_RAM_ACCESSES)
        {
            self.access_records.clear();
        }
        if preceding_accesses
            .checked_add(chunk_accesses)
            .is_none_or(|end| end > MAX_RAM_RA_COMPACT_ACCESSES)
        {
            self.ram_ra_records.clear();
        }
        *self.sparse_chunk = Some(SparseChunk {
            access_records: std::mem::take(&mut self.access_records),
            ram_ra_records: std::mem::take(&mut self.ram_ra_records),
            activity_records: std::mem::take(&mut self.activity_records),
            active_cycle_bound: self.active_cycle_bound,
            required_address_domain: self.required_address_domain,
            increment_compatible: self.increment_compatible,
            ram_ra_compatible: self.ram_ra_compatible,
            hamming_exact: self.hamming_exact,
        });
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamAccessCollection {
    pub(crate) fn validate_publish<F: JoltField>(
        &self,
        session: &ProofSession,
    ) -> Result<(), KernelError<F>> {
        if self.values.pre_values.len() != self.columns.addresses.len()
            || self.values.post_values.len() != self.columns.addresses.len()
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM access collection columns disagree on cycle count",
            });
        }
        if session.state::<Arc<RamAccessColumns>>().is_some()
            || session.state::<RamAccessValues>().is_some()
            || session.state::<Arc<RamIncrementActivity>>().is_some()
            || session.state::<RamAccessTape>().is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM access collection publication would replace resident state",
            });
        }
        Ok(())
    }

    pub(crate) fn publish<F: JoltField>(
        self,
        session: &mut ProofSession,
    ) -> Result<(), KernelError<F>> {
        self.validate_publish(session)?;
        session.park(Arc::new(self.columns));
        session.park(self.values);
        session.park(Arc::new(self.activity));
        session.park(self.tape);
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl RamReadWriteRecordCollection {
    pub(crate) fn validate_publish<F: JoltField>(
        &self,
        session: &ProofSession,
    ) -> Result<(), KernelError<F>> {
        if self.columns.addresses.len() != self.records.rows
            || self.tape.access_count() != self.records.access_count
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM record collection geometry disagrees",
            });
        }
        if session.state::<Arc<RamAccessColumns>>().is_some()
            || session.state::<RamAccessValues>().is_some()
            || session.state::<Arc<RamIncrementActivity>>().is_some()
            || session.state::<RamAccessTape>().is_some()
            || session.state::<RamReadWriteRecordChunks>().is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM record collection publication would replace resident state",
            });
        }
        Ok(())
    }

    pub(crate) fn publish<F: JoltField>(
        self,
        session: &mut ProofSession,
    ) -> Result<(), KernelError<F>> {
        self.validate_publish(session)?;
        session.park(Arc::new(self.columns));
        session.park(self.records);
        session.park(self.tape);
        Ok(())
    }
}

/// Column-major per-cycle RAM access data over the full padded cycle domain.
pub(crate) struct RamAccessColumns {
    /// Remapped word address per cycle; [`NO_ACCESS`] when the cycle makes no
    /// remappable RAM access (no-ops and address 0).
    pub addresses: Vec<u32>,
    #[cfg_attr(
        not(all(feature = "metal", target_os = "macos")),
        expect(
            dead_code,
            reason = "the active prefix certificate is consumed by Metal"
        )
    )]
    active_cycle_bound: usize,
    required_address_domain: usize,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    ram_ra_sparse: Option<RamRaSparseLayout>,
}

#[derive(Clone, Copy)]
pub(crate) struct ValidatedRamAccessAddresses<'a> {
    addresses: &'a [u32],
}

#[cfg_attr(
    not(any(test, all(feature = "metal", target_os = "macos"))),
    expect(
        dead_code,
        reason = "the checked slice is consumed by the Metal RAF upload"
    )
)]
impl ValidatedRamAccessAddresses<'_> {
    pub(crate) const fn as_slice(&self) -> &[u32] {
        self.addresses
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamAccessColumns {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("addresses"),
            crate::backend::vec_heap_bytes(&self.addresses),
        );
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(layout) = &self.ram_ra_sparse {
            visitor.visit_simple(
                allocative::Key::new("ram_ra_q_offsets"),
                crate::backend::vec_heap_bytes(&layout.q_offsets),
            );
            visitor.visit_simple(
                allocative::Key::new("ram_ra_q_records"),
                crate::backend::vec_heap_bytes(&layout.q_records),
            );
            visitor.visit_simple(
                allocative::Key::new("ram_ra_h_offsets"),
                crate::backend::vec_heap_bytes(&layout.h_offsets),
            );
            visitor.visit_simple(
                allocative::Key::new("ram_ra_h_records"),
                crate::backend::vec_heap_bytes(&layout.h_records),
            );
        }
        visitor.exit();
    }
}

/// RAM values have one final consumer in stage 4, so they are parked
/// separately from the address column and consumed there.
pub(crate) struct RamAccessValues {
    /// Pre-access word value per cycle (a read's value, a write's pre-value);
    /// 0 on no-access cycles.
    pub pre_values: Vec<u64>,
    /// Post-access word value per cycle (equals the pre-value for reads).
    pub post_values: Vec<u64>,
}

/// Sparse nonzero RAM deltas collected alongside the shared address column.
pub(crate) struct RamIncrementActivity {
    cycles: Vec<u64>,
    increments: Vec<i128>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamIncrementActivity {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("cycles"),
            crate::backend::vec_heap_bytes(&self.cycles),
        );
        visitor.visit_simple(
            allocative::Key::new("increments"),
            crate::backend::vec_heap_bytes(&self.increments),
        );
        visitor.exit();
    }
}

#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(
        dead_code,
        reason = "the sparse activity is consumed by the Metal RAM owner"
    )
)]
impl RamIncrementActivity {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn from_sorted_parts(cycles: Vec<u64>, increments: Vec<i128>) -> Self {
        debug_assert_eq!(cycles.len(), increments.len());
        debug_assert!(cycles.windows(2).all(|window| window[0] < window[1]));
        Self { cycles, increments }
    }

    pub(crate) fn len(&self) -> usize {
        self.cycles.len()
    }

    pub(crate) fn records(&self) -> impl ExactSizeIterator<Item = (usize, i128)> + '_ {
        self.cycles
            .iter()
            .copied()
            .zip(self.increments.iter().copied())
            .map(|(cycle, increment)| (cycle as usize, increment))
    }

    pub(crate) fn cycle_slice(&self) -> &[u64] {
        &self.cycles
    }

    pub(crate) fn increment_slice(&self) -> &[i128] {
        &self.increments
    }
}

fn encode_ram_increment(cycle: usize, increment: i128) -> Option<(u64, i128)> {
    if increment == 0 {
        return None;
    }
    let cycle = u64::try_from(cycle).ok()?;
    Some((cycle, increment))
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamAccessValues {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("pre_values"),
            crate::backend::vec_heap_bytes(&self.pre_values),
        );
        visitor.visit_simple(
            allocative::Key::new("post_values"),
            crate::backend::vec_heap_bytes(&self.post_values),
        );
        visitor.exit();
    }
}

impl RamAccessColumns {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn from_direct_addresses(
        addresses: Vec<u32>,
        active_cycle_bound: usize,
        required_address_domain: usize,
    ) -> Self {
        debug_assert!(active_cycle_bound <= addresses.len());
        Self {
            addresses,
            active_cycle_bound,
            required_address_domain,
            ram_ra_sparse: None,
        }
    }

    fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<(Self, RamAccessValues, RamIncrementActivity, RamAccessTape), KernelError<F>> {
        if log_t > u32::BITS as usize {
            return Err(KernelError::Unsupported {
                reason: "optimized RAM kernels require at most 32 cycle variables",
            });
        }
        let cycles = 1usize << log_t;
        #[cfg(feature = "parallel")]
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                return Self::collect_par(&access, cycles, log_t);
            }
        }

        const COLLECT_CHUNK: usize = 1 << 16;
        let mut consumers = (CollectRamAccessColumns {
            addresses: Vec::with_capacity(cycles),
            active_cycle_bound: 0,
            required_address_domain: 0,
            pre_values: Vec::with_capacity(cycles),
            post_values: Vec::with_capacity(cycles),
            ram_increment_cycles: Vec::new(),
            ram_increments: Vec::new(),
            access_count: 0,
            access_records: Some(Vec::new()),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            ram_ra_records: Some(Vec::new()),
            increment_compatible: true,
            ram_ra_compatible: true,
            hamming_exact: true,
            address_error: None,
        },);
        stream_witnesses(witness, 0..cycles, COLLECT_CHUNK, &mut consumers)?;
        let collected = consumers.0;
        if let Some(failure) = collected.address_error {
            return Err(failure.into_kernel_error());
        }
        Ok((
            Self {
                addresses: collected.addresses,
                active_cycle_bound: collected.active_cycle_bound,
                required_address_domain: collected.required_address_domain,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                ram_ra_sparse: RamRaSparseLayout::build(log_t, collected.ram_ra_records),
            },
            RamAccessValues {
                pre_values: collected.pre_values,
                post_values: collected.post_values,
            },
            RamIncrementActivity {
                cycles: collected.ram_increment_cycles,
                increments: collected.ram_increments,
            },
            RamAccessTape::new(
                log_t,
                collected.access_count,
                collected.access_records,
                collected.increment_compatible,
                collected.ram_ra_compatible,
                collected.hamming_exact,
            ),
        ))
    }

    /// Slice-backed traces scatter directly into the three final columns,
    /// avoiding a full-width `RamAccessBundle` vector at the collection peak.
    #[cfg(feature = "parallel")]
    fn collect_par<F: JoltField>(
        access: &RandomAccessRows,
        cycles: usize,
        log_t: usize,
    ) -> Result<(Self, RamAccessValues, RamIncrementActivity, RamAccessTape), KernelError<F>> {
        const CHUNK: usize = 1 << 12;
        let mut addresses = Vec::with_capacity(cycles);
        let mut pre_values = Vec::with_capacity(cycles);
        let mut post_values = Vec::with_capacity(cycles);
        let mut sparse_chunks = (0..cycles.div_ceil(CHUNK))
            .map(|_| OnceLock::new())
            .collect::<Vec<_>>();
        let access_count = AtomicUsize::new(0);
        let error = std::sync::Mutex::new(None);
        (
            addresses.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            pre_values.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
            post_values.spare_capacity_mut()[..cycles].par_chunks_mut(CHUNK),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_index, (addresses, pre_values, post_values))| {
                let base = chunk_index * CHUNK;
                let mut activity = Vec::new();
                let mut access_records = Vec::new();
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let mut ram_ra_records = Vec::new();
                let mut required_address_domain = 0;
                let mut active_cycle_bound = 0;
                let mut increment_compatible = true;
                let mut ram_ra_compatible = true;
                let mut hamming_exact = true;
                for offset in 0..addresses.len() {
                    let bundle = match access.window::<RamAccessBundle>(base + offset) {
                        Ok(bundle) => bundle,
                        Err(failure) => {
                            if let Ok(mut guard) = error.lock() {
                                let _ = guard.get_or_insert(CollectFailure::Witness(failure));
                            }
                            return;
                        }
                    };
                    let address = match encode_address(bundle.address.0) {
                        Ok(address) => address,
                        Err(failure) => {
                            if let Ok(mut guard) = error.lock() {
                                let _ = guard.get_or_insert(CollectFailure::Address(failure));
                            }
                            return;
                        }
                    };
                    let _ = addresses[offset].write(address);
                    let _ = pre_values[offset].write(bundle.pre_value.0);
                    let _ = post_values[offset].write(bundle.post_value.0);
                    let cycle = base + offset;
                    if let Some(record) = encode_ram_increment(cycle, bundle.ram_inc.0) {
                        activity.push(record);
                    }
                    let delta = i128::from(bundle.post_value.0) - i128::from(bundle.pre_value.0);
                    increment_compatible &= bundle.ram_inc.0 == delta
                        && (address != NO_ACCESS || bundle.ram_inc.0 == 0);
                    ram_ra_compatible &= !(bundle.ram_hamming_weight.0 && address == NO_ACCESS);
                    hamming_exact &= bundle.ram_hamming_weight.0 == (address != NO_ACCESS);
                    if address != NO_ACCESS {
                        active_cycle_bound = cycle + 1;
                        required_address_domain = required_address_domain.max(address as usize + 1);
                        let cycle = if let Ok(cycle) = u32::try_from(cycle) {
                            cycle
                        } else {
                            if let Ok(mut guard) = error.lock() {
                                let _ = guard.get_or_insert(CollectFailure::Address(
                                    AddressEncodingError::CycleTooLarge,
                                ));
                            }
                            return;
                        };
                        access_records.push(RamAccessRecord {
                            cycle,
                            address,
                            pre_value: bundle.pre_value.0,
                            post_value: bundle.post_value.0,
                        });
                        #[cfg(all(feature = "metal", target_os = "macos"))]
                        ram_ra_records.push(RamRaCompactRecord { cycle, address });
                    }
                }
                let chunk_accesses = access_records.len();
                let preceding_accesses = access_count.fetch_add(chunk_accesses, Ordering::Relaxed);
                if preceding_accesses
                    .checked_add(chunk_accesses)
                    .is_none_or(|end| end > MAX_RETAINED_RAM_ACCESSES)
                {
                    access_records = Vec::new();
                }
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if preceding_accesses
                    .checked_add(chunk_accesses)
                    .is_none_or(|end| end > MAX_RAM_RA_COMPACT_ACCESSES)
                {
                    ram_ra_records = Vec::new();
                }
                let _ = sparse_chunks[chunk_index].set(SparseChunk {
                    access_records,
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    ram_ra_records,
                    activity_records: activity,
                    active_cycle_bound,
                    required_address_domain,
                    increment_compatible,
                    ram_ra_compatible,
                    hamming_exact,
                });
            });
        #[expect(clippy::unwrap_used, reason = "no lock user can panic")]
        if let Some(failure) = error.into_inner().unwrap() {
            return Err(match failure {
                CollectFailure::Witness(failure) => failure.into(),
                CollectFailure::Address(failure) => failure.into_kernel_error(),
            });
        }
        // SAFETY: with no latched error, every worker initialized its entire
        // disjoint span in all three vectors.
        unsafe {
            addresses.set_len(cycles);
            pre_values.set_len(cycles);
            post_values.set_len(cycles);
        }
        let total_accesses = access_count.load(Ordering::Relaxed);
        let mut retained_records = (total_accesses <= MAX_RETAINED_RAM_ACCESSES)
            .then(|| Vec::with_capacity(total_accesses));
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let mut ram_ra_records = (total_accesses <= MAX_RAM_RA_COMPACT_ACCESSES)
            .then(|| Vec::with_capacity(total_accesses));
        let mut activity_records = Vec::new();
        let mut required_address_domain = 0;
        let mut increment_compatible = true;
        let mut ram_ra_compatible = true;
        let mut hamming_exact = true;
        let mut active_cycle_bound = 0;
        for chunk in sparse_chunks.drain(..) {
            let Some(chunk) = chunk.into_inner() else {
                return Err(KernelError::InvariantViolation {
                    reason: "parallel RAM collection left an incomplete chunk",
                });
            };
            required_address_domain = required_address_domain.max(chunk.required_address_domain);
            increment_compatible &= chunk.increment_compatible;
            ram_ra_compatible &= chunk.ram_ra_compatible;
            hamming_exact &= chunk.hamming_exact;
            active_cycle_bound = active_cycle_bound.max(chunk.active_cycle_bound);
            activity_records.extend(chunk.activity_records);
            if let Some(records) = &mut retained_records {
                records.extend(&chunk.access_records);
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if let Some(records) = &mut ram_ra_records {
                records.extend(chunk.ram_ra_records);
            }
        }
        let (cycles, increments) = activity_records.into_iter().unzip();
        Ok((
            Self {
                addresses,
                active_cycle_bound,
                required_address_domain,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                ram_ra_sparse: RamRaSparseLayout::build(log_t, ram_ra_records),
            },
            RamAccessValues {
                pre_values,
                post_values,
            },
            RamIncrementActivity { cycles, increments },
            RamAccessTape::new(
                log_t,
                total_accesses,
                retained_records,
                increment_compatible,
                ram_ra_compatible,
                hamming_exact,
            ),
        ))
    }

    /// The session-shared columns: collected on first request (whichever RAM
    /// kernel prepares first), cloned out as an [`Arc`] afterwards.
    #[expect(
        clippy::expect_used,
        reason = "the entry is parked by this function right above the read"
    )]
    pub fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Arc<Self>, KernelError<F>> {
        if session.state::<Arc<Self>>().is_none() {
            let (columns, values, activity, tape) = Self::collect(witness, log_t)?;
            debug_assert_eq!(activity.cycles.len(), activity.increments.len());
            let columns = Arc::new(columns);
            session.park(columns);
            session.park(values);
            session.park(Arc::new(activity));
            session.park(tape);
        }
        let columns = Arc::clone(
            session
                .state::<Arc<Self>>()
                .expect("RAM access columns parked above"),
        );
        if columns.addresses.len() != 1usize << log_t {
            return Err(KernelError::InvariantViolation {
                reason: "parked RAM access columns cover a different cycle domain than requested",
            });
        }
        Ok(columns)
    }

    /// Reclaims the value columns at their final consumer while leaving the
    /// shared address column available to later stages.
    pub fn shared_with_values<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<(Arc<Self>, RamAccessValues), KernelError<F>> {
        let columns = Self::shared(session, witness, log_t)?;
        let values = session
            .take::<RamAccessValues>()
            .ok_or(KernelError::InvariantViolation {
                reason: "RAM access value columns were already consumed",
            })?;
        Ok((columns, values))
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) const fn active_cycle_bound(&self) -> usize {
        self.active_cycle_bound
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) const fn ram_ra_sparse_layout(&self) -> Option<&RamRaSparseLayout> {
        self.ram_ra_sparse.as_ref()
    }

    /// Checks the collection-time maximum against the proof's `K`.
    pub fn validate_addresses<F: JoltField>(&self, ram_k: usize) -> Result<(), KernelError<F>> {
        if self.required_address_domain > ram_k {
            return Err(KernelError::InvariantViolation {
                reason: "RAM access address remapped beyond ram_K",
            });
        }
        Ok(())
    }

    #[cfg_attr(
        not(any(test, all(feature = "metal", target_os = "macos"))),
        expect(
            dead_code,
            reason = "the checked slice is consumed by the Metal RAF upload"
        )
    )]
    pub(crate) fn validated_addresses<F: JoltField>(
        &self,
        ram_k: usize,
    ) -> Result<ValidatedRamAccessAddresses<'_>, KernelError<F>> {
        self.validate_addresses(ram_k)?;
        Ok(ValidatedRamAccessAddresses {
            addresses: &self.addresses,
        })
    }

    /// The address-eq fold of the one-hot `ra` grid:
    /// `out[j] = Σ_k eq(r_address, k) · ra(k, j) = eq_address[addresses[j]]`
    /// (0 on no-access cycles). Reproduces `views::address_fold` of the dense
    /// grid without materializing it.
    pub fn fold_addresses<F: JoltField>(&self, eq_address: &[F]) -> Vec<F> {
        self.addresses
            .iter()
            .map(|&address| {
                if address == NO_ACCESS {
                    F::zero()
                } else {
                    eq_address[address as usize]
                }
            })
            .collect()
    }

    /// The cycle-eq fold of the one-hot `ra` grid:
    /// `out[k] = Σ_j eq(r_cycle, j) · ra(k, j) = Σ_{j : addresses[j] = k} eq_cycle[j]`.
    /// Reproduces `views::cycle_fold` of the dense grid without
    /// materializing it.
    pub fn fold_cycles<F: JoltField>(&self, eq_cycle: &[F], ram_k: usize) -> Vec<F> {
        let mut out = vec![F::zero(); ram_k];
        for (&address, &eq) in self.addresses.iter().zip(eq_cycle) {
            if address != NO_ACCESS {
                out[address as usize] += eq;
            }
        }
        out
    }

    /// Reconstruct the initial RAM state from the trace and the final-state
    /// oracle: an accessed address's initial value is its first access's
    /// pre-value; a never-accessed address's value never changes, so its
    /// final value IS its initial value.
    ///
    /// WARNING: this is the honest-prover data path — it relies on the trace
    /// being consistent with the final memory image (exactly what the RAM
    /// val/output sumchecks prove). A dishonest witness diverges here and
    /// fails the engine's round checks loudly.
    pub fn reconstruct_val_init<F: JoltField>(
        &self,
        pre_values: &[u64],
        val_final: Vec<F>,
    ) -> Vec<F> {
        debug_assert_eq!(self.addresses.len(), pre_values.len());
        let mut val_init = val_final;
        let mut seen = vec![false; val_init.len()];
        for (&address, &pre_value) in self.addresses.iter().zip(pre_values) {
            if address == NO_ACCESS {
                continue;
            }
            let address = address as usize;
            if !seen[address] {
                seen[address] = true;
                val_init[address] = F::from_u64(pre_value);
            }
        }
        val_init
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid fixtures")]
mod tests {
    use super::*;

    fn collector() -> CollectRamAccessColumns {
        CollectRamAccessColumns {
            addresses: Vec::new(),
            active_cycle_bound: 0,
            required_address_domain: 0,
            pre_values: Vec::new(),
            post_values: Vec::new(),
            ram_increment_cycles: Vec::new(),
            ram_increments: Vec::new(),
            access_count: 0,
            access_records: Some(Vec::new()),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            ram_ra_records: Some(Vec::new()),
            increment_compatible: true,
            ram_ra_compatible: true,
            hamming_exact: true,
            address_error: None,
        }
    }

    fn bundle(
        address: Option<u64>,
        pre: u64,
        post: u64,
        ram_inc: i128,
        hamming: bool,
    ) -> RamAccessBundle {
        RamAccessBundle {
            address: RemappedRamAddress(address),
            pre_value: RamReadValue(pre),
            post_value: RamWriteValue(post),
            ram_inc: RamInc(ram_inc),
            ram_hamming_weight: RamHammingWeight(hamming),
        }
    }

    fn feed(collector: &mut CollectRamAccessColumns, row: RamAccessBundle, count: usize) {
        const CHUNK: usize = 1 << 12;
        let rows = vec![row; CHUNK.min(count)];
        let mut remaining = count;
        while remaining != 0 {
            let take = remaining.min(rows.len());
            collector.consume(&rows[..take]);
            remaining -= take;
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn ram_ra_sparse_layout_preserves_coordinate_order() {
        let h_records = vec![
            RamRaCompactRecord {
                cycle: 1,
                address: 7,
            },
            RamRaCompactRecord {
                cycle: 3,
                address: 2,
            },
            RamRaCompactRecord {
                cycle: 4,
                address: 9,
            },
            RamRaCompactRecord {
                cycle: 9,
                address: 5,
            },
        ];
        let layout = RamRaSparseLayout::build(4, Some(h_records.clone())).unwrap();
        assert_eq!(layout.q_offsets(), &[0, 1, 3, 3, 4]);
        assert_eq!(
            layout.q_records(),
            &[
                RamRaQRecord {
                    x_hi: 1,
                    address: 9,
                },
                RamRaQRecord {
                    x_hi: 0,
                    address: 7,
                },
                RamRaQRecord {
                    x_hi: 2,
                    address: 5,
                },
                RamRaQRecord {
                    x_hi: 0,
                    address: 2,
                },
            ]
        );
        assert_eq!(layout.h_offsets(), &[0, 2, 3, 4, 4]);
        assert_eq!(layout.h_records(), h_records);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn co_produced_collection_matches_stream_collection() {
        let rows = [
            bundle(None, 0, 0, 0, false),
            bundle(Some(3), 7, 7, 0, true),
            bundle(Some(5), 9, 12, 3, true),
            bundle(None, 0, 0, 0, false),
            bundle(Some(3), 7, 2, -5, true),
            bundle(None, 0, 0, 0, false),
            bundle(None, 0, 0, 0, false),
            bundle(None, 0, 0, 0, false),
        ];
        let mut expected = collector();
        expected.consume(&rows);

        let mut storage = RamAccessCollectionStorage::new(rows.len(), 4).unwrap();
        storage
            .with_chunk_writers(|writers| -> Result<(), RamAccessCollectionError> {
                for (chunk, writer) in writers.iter_mut().enumerate() {
                    let start = chunk * 4;
                    if chunk == 1 {
                        writer.push(rows[start])?;
                        writer.fill_repeated(rows[start + 1], 3)?;
                    } else {
                        for &row in &rows[start..start + writer.len()] {
                            writer.push(row)?;
                        }
                    }
                    writer.finish()?;
                }
                Ok(())
            })
            .unwrap();
        let actual = storage.seal().unwrap();

        assert_eq!(actual.columns.addresses, expected.addresses);
        assert_eq!(
            actual.columns.active_cycle_bound,
            expected.active_cycle_bound
        );
        assert_eq!(actual.values.pre_values, expected.pre_values);
        assert_eq!(actual.values.post_values, expected.post_values);
        assert_eq!(actual.activity.cycles, expected.ram_increment_cycles);
        assert_eq!(actual.activity.increments, expected.ram_increments);
        assert_eq!(actual.tape.access_count(), expected.access_count);
        assert_eq!(actual.tape.records(), expected.access_records.as_deref());
        assert_eq!(
            actual.tape.increment_compatible(),
            expected.increment_compatible
        );
        assert_eq!(actual.tape.ram_ra_compatible(), expected.ram_ra_compatible);
        assert_eq!(actual.tape.hamming_exact(), expected.hamming_exact);
        assert_eq!(
            actual.columns.required_address_domain,
            expected.required_address_domain
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn co_produced_record_collection_matches_stream_collection() {
        let rows = [
            bundle(None, 0, 0, 0, false),
            bundle(Some(3), 7, 7, 0, true),
            bundle(Some(5), 9, 12, 3, true),
            bundle(None, 0, 0, 0, false),
            bundle(Some(3), 7, 2, -5, true),
            bundle(None, 0, 0, 0, false),
            bundle(None, 0, 0, 0, false),
            bundle(None, 0, 0, 0, false),
        ];
        let mut expected = collector();
        expected.consume(&rows);

        let mut storage = RamReadWriteRecordCollectionStorage::new(rows.len(), 4, 3).unwrap();
        storage
            .with_chunk_writers(|writers| -> Result<(), RamAccessCollectionError> {
                for (chunk, writer) in writers.iter_mut().enumerate() {
                    let start = chunk * 4;
                    if chunk == 1 {
                        writer.push(rows[start])?;
                        writer.fill_repeated(rows[start + 1], 3)?;
                    } else {
                        for &row in &rows[start..start + writer.len()] {
                            writer.push(row)?;
                        }
                    }
                    writer.finish()?;
                }
                Ok(())
            })
            .unwrap();
        let actual = storage.seal().unwrap();
        let records = actual
            .records
            .chunks
            .iter()
            .flat_map(AlignedRamReadWriteRecordArena::records)
            .copied()
            .collect::<Vec<_>>();

        assert_eq!(actual.columns.addresses, expected.addresses);
        assert_eq!(
            actual.columns.active_cycle_bound,
            expected.active_cycle_bound
        );
        assert_eq!(
            actual.columns.required_address_domain,
            expected.required_address_domain
        );
        assert_eq!(records, expected.access_records.unwrap());
        assert_eq!(actual.records.address_count(), 8);
        assert_eq!(actual.records.tile_log(), 3);
        let mut address_counts = vec![0u32; 8];
        let mut tile_counts = vec![0u32; 1];
        for census in actual.records.worker_census() {
            for (total, count) in address_counts.iter_mut().zip(census.address_counts()) {
                *total += count;
            }
            for (total, count) in tile_counts.iter_mut().zip(census.tile_counts()) {
                *total += count;
            }
        }
        assert_eq!(address_counts, &[0, 0, 0, 2, 0, 1, 0, 0]);
        assert_eq!(tile_counts, &[3]);
        assert_eq!(
            actual
                .records
                .worker_census()
                .iter()
                .map(|census| (census.accesses(), census.first_cycle(), census.last_cycle()))
                .collect::<Vec<_>>(),
            vec![(2, Some(1), Some(2)), (1, Some(4), Some(4))]
        );
        assert_eq!(actual.tape.access_count(), expected.access_count);
        assert_eq!(
            actual.tape.increment_compatible(),
            expected.increment_compatible
        );
        assert_eq!(actual.tape.ram_ra_compatible(), expected.ram_ra_compatible);
        assert_eq!(actual.tape.hamming_exact(), expected.hamming_exact);
        assert!(actual.tape.records().is_none());
    }

    #[test]
    fn certificates_distinguish_raw_zero_and_failed_remap() {
        let mut raw_zero = collector();
        raw_zero.consume(&[bundle(None, 2, 9, 7, false)]);
        assert!(raw_zero.ram_ra_compatible);
        assert!(raw_zero.hamming_exact);
        assert!(!raw_zero.increment_compatible);
        assert_eq!(raw_zero.access_count, 0);
        assert_eq!(raw_zero.ram_increment_cycles, vec![0]);
        assert_eq!(raw_zero.ram_increments, vec![7]);

        let mut failed_remap = collector();
        failed_remap.consume(&[bundle(None, 4, 4, 0, true)]);
        assert!(!failed_remap.ram_ra_compatible);
        assert!(!failed_remap.hamming_exact);
        assert!(failed_remap.increment_compatible);

        let mut mapped_zero = collector();
        mapped_zero.consume(&[bundle(Some(0), 5, 5, 0, true)]);
        assert!(mapped_zero.ram_ra_compatible);
        assert!(mapped_zero.hamming_exact);
        assert!(mapped_zero.increment_compatible);
        assert_eq!(mapped_zero.access_records.unwrap()[0].address, 0);

        let mut missing_hamming = collector();
        missing_hamming.consume(&[bundle(Some(0), 5, 5, 0, false)]);
        assert!(missing_hamming.ram_ra_compatible);
        assert!(!missing_hamming.hamming_exact);
    }

    #[test]
    fn sparse_retention_is_complete_at_cap_and_absent_above_it() {
        let row = bundle(Some(1), 0, 0, 0, true);
        let mut at_cap = collector();
        feed(&mut at_cap, row, MAX_RETAINED_RAM_ACCESSES);
        assert_eq!(at_cap.access_count, MAX_RETAINED_RAM_ACCESSES);
        assert_eq!(
            at_cap.access_records.as_ref().map(Vec::len),
            Some(MAX_RETAINED_RAM_ACCESSES)
        );

        let mut above_cap = collector();
        feed(&mut above_cap, row, MAX_RETAINED_RAM_ACCESSES + 1);
        assert_eq!(above_cap.access_count, MAX_RETAINED_RAM_ACCESSES + 1);
        assert!(above_cap.access_records.is_none());
    }

    #[test]
    fn address_domain_certificate_is_built_during_collection() {
        let mut collected = collector();
        collected.consume(&[
            bundle(None, 0, 0, 0, false),
            bundle(Some(7), 0, 0, 0, true),
            bundle(Some(3), 0, 0, 0, true),
        ]);
        assert_eq!(collected.active_cycle_bound, 3);
        assert_eq!(collected.required_address_domain, 8);

        let columns = RamAccessColumns {
            addresses: collected.addresses,
            active_cycle_bound: collected.active_cycle_bound,
            required_address_domain: collected.required_address_domain,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            ram_ra_sparse: RamRaSparseLayout::build(2, collected.ram_ra_records),
        };
        assert!(columns.validate_addresses::<jolt_field::Fr>(8).is_ok());
        assert!(columns.validate_addresses::<jolt_field::Fr>(7).is_err());
        assert_eq!(
            columns
                .validated_addresses::<jolt_field::Fr>(8)
                .unwrap()
                .as_slice(),
            &[NO_ACCESS, 7, 3]
        );
    }
}
