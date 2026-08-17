//! The device-resident co-production of the shared RAM access columns:
//! uninitialized storage filled chunk-by-chunk alongside the Stage-1 owner
//! projection, sealed into the same session state the streaming collector in
//! [`crate::optimized::ram_trace`] produces.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jolt_field::Field;

use crate::optimized::ram_trace::{
    encode_address, encode_ram_increment, AddressEncodingError, RamAccessBundle, RamAccessColumns,
    RamAccessValues, RamIncrementActivity, SparseChunk, NO_ACCESS,
};
use crate::ram_access::{RamAccessRecord, RamAccessTape, MAX_RETAINED_RAM_ACCESSES};
use crate::{KernelError, ProofSession};

/// Uninitialized final-column storage filled alongside another random-access
/// witness projection. The writer API keeps each chunk disjoint; sealing is
/// the only operation that turns the initialized planes into session state.
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

pub(crate) struct RamAccessCollectionChunkWriter<'a> {
    base: usize,
    written: usize,
    addresses: &'a mut [std::mem::MaybeUninit<u32>],
    pre_values: &'a mut [std::mem::MaybeUninit<u64>],
    post_values: &'a mut [std::mem::MaybeUninit<u64>],
    sparse_chunk: &'a mut Option<SparseChunk>,
    access_count: &'a AtomicUsize,
    access_records: Vec<RamAccessRecord>,
    activity_records: Vec<(u64, i128)>,
    required_address_domain: usize,
    increment_compatible: bool,
    ram_ra_compatible: bool,
    hamming_exact: bool,
}

pub(crate) struct RamAccessCollection {
    pub(crate) columns: RamAccessColumns,
    pub(crate) values: RamAccessValues,
    pub(crate) activity: RamIncrementActivity,
    pub(crate) tape: RamAccessTape,
}

#[derive(Debug)]
pub(crate) struct RamAccessCollectionError {
    reason: &'static str,
}

impl RamAccessCollectionError {
    const fn new(reason: &'static str) -> Self {
        Self { reason }
    }

    pub(crate) const fn reason(&self) -> &'static str {
        self.reason
    }

    pub(crate) fn into_kernel_error<F: Field>(self) -> KernelError<F> {
        KernelError::InvariantViolation {
            reason: self.reason,
        }
    }
}

impl From<AddressEncodingError> for RamAccessCollectionError {
    fn from(error: AddressEncodingError) -> Self {
        Self::new(error.reason())
    }
}

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
                        activity_records: Vec::new(),
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
        let mut activity_records = Vec::new();
        let mut required_address_domain = 0;
        let mut increment_compatible = true;
        let mut ram_ra_compatible = true;
        let mut hamming_exact = true;
        for chunk in self.sparse_chunks {
            let chunk = chunk.ok_or_else(|| {
                RamAccessCollectionError::new("RAM access collection lost a completed chunk")
            })?;
            required_address_domain = required_address_domain.max(chunk.required_address_domain);
            increment_compatible &= chunk.increment_compatible;
            ram_ra_compatible &= chunk.ram_ra_compatible;
            hamming_exact &= chunk.hamming_exact;
            activity_records.extend(chunk.activity_records);
            if let Some(records) = &mut retained_records {
                records.extend(chunk.access_records);
            }
        }
        let (cycles, increments) = activity_records.into_iter().unzip();
        Ok(RamAccessCollection {
            columns: RamAccessColumns {
                addresses: self.addresses,
                required_address_domain,
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
            self.required_address_domain = self.required_address_domain.max(address as usize + 1);
            self.access_records.push(RamAccessRecord {
                cycle: u32::try_from(cycle).map_err(|_| {
                    RamAccessCollectionError::from(AddressEncodingError::CycleTooLarge)
                })?,
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
        *self.sparse_chunk = Some(SparseChunk {
            access_records: std::mem::take(&mut self.access_records),
            activity_records: std::mem::take(&mut self.activity_records),
            required_address_domain: self.required_address_domain,
            increment_compatible: self.increment_compatible,
            ram_ra_compatible: self.ram_ra_compatible,
            hamming_exact: self.hamming_exact,
        });
        Ok(())
    }
}

impl RamAccessCollection {
    pub(crate) fn validate_publish<F: Field>(
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
            || session.state::<RamIncrementActivity>().is_some()
            || session.state::<RamAccessTape>().is_some()
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM access collection publication would replace resident state",
            });
        }
        Ok(())
    }

    pub(crate) fn publish<F: Field>(
        self,
        session: &mut ProofSession,
    ) -> Result<(), KernelError<F>> {
        self.validate_publish(session)?;
        session.park(Arc::new(self.columns));
        session.park(self.values);
        session.park(self.activity);
        session.park(self.tape);
        Ok(())
    }
}
