//! The shared RAM access columns: one typed trace walk serving every
//! RAM-family kernel in this backend, parked in the [`ProofSession`] so the
//! stage-2 kernel's walk is reused by stages 4 and 5.
//!
//! The columns are the sparse view of the `(K × T)` RAM grids: per cycle,
//! the remapped word address (or a no-access sentinel) plus the pre- and
//! post-access word values. `ra(k, j)` is 1 exactly at `(addresses[j], j)`;
//! `val(k, j)` walks from the initial state through the writes.

#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "parallel")]
use std::sync::OnceLock;

use jolt_field::Field;
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

#[derive(Clone, Copy, Debug, WitnessBundle)]
pub(crate) struct RamAccessBundle {
    pub(crate) address: RemappedRamAddress,
    pub(crate) pre_value: RamReadValue,
    pub(crate) post_value: RamWriteValue,
    pub(crate) ram_inc: RamInc,
    pub(crate) ram_hamming_weight: RamHammingWeight,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AddressEncodingError {
    TooLarge,
    SentinelCollision,
    CycleTooLarge,
}

impl AddressEncodingError {
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::TooLarge => "optimized RAM kernels require remapped addresses below 2^32 - 1",
            Self::SentinelCollision => {
                "optimized RAM kernels reserve u32::MAX as the no-access sentinel"
            }
            Self::CycleTooLarge => "optimized RAM kernels require cycle indices below 2^32",
        }
    }

    fn into_kernel_error<F: Field>(self) -> KernelError<F> {
        KernelError::Unsupported {
            reason: self.reason(),
        }
    }
}

pub(crate) fn encode_address(address: Option<u64>) -> Result<u32, AddressEncodingError> {
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
    required_address_domain: usize,
    pre_values: Vec<u64>,
    post_values: Vec<u64>,
    ram_increment_cycles: Vec<u64>,
    ram_increments: Vec<i128>,
    access_count: usize,
    access_records: Option<Vec<RamAccessRecord>>,
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
                self.required_address_domain =
                    self.required_address_domain.max(address as usize + 1);
                self.access_count += 1;
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

pub(crate) struct SparseChunk {
    pub(crate) access_records: Vec<RamAccessRecord>,
    pub(crate) activity_records: Vec<(u64, i128)>,
    pub(crate) required_address_domain: usize,
    pub(crate) increment_compatible: bool,
    pub(crate) ram_ra_compatible: bool,
    pub(crate) hamming_exact: bool,
}

/// Column-major per-cycle RAM access data over the full padded cycle domain.
pub(crate) struct RamAccessColumns {
    /// Remapped word address per cycle; [`NO_ACCESS`] when the cycle makes no
    /// remappable RAM access (no-ops and address 0).
    pub addresses: Vec<u32>,
    pub(crate) required_address_domain: usize,
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
    pub(crate) cycles: Vec<u64>,
    pub(crate) increments: Vec<i128>,
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
}

pub(crate) fn encode_ram_increment(cycle: usize, increment: i128) -> Option<(u64, i128)> {
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
    fn collect<F: Field>(
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
            required_address_domain: 0,
            pre_values: Vec::with_capacity(cycles),
            post_values: Vec::with_capacity(cycles),
            ram_increment_cycles: Vec::new(),
            ram_increments: Vec::new(),
            access_count: 0,
            access_records: Some(Vec::new()),
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
                required_address_domain: collected.required_address_domain,
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
    fn collect_par<F: Field>(
        access: &RandomAccessRows<'_>,
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
                let mut required_address_domain = 0;
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
                let _ = sparse_chunks[chunk_index].set(SparseChunk {
                    access_records,
                    activity_records: activity,
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
        let mut activity_records = Vec::new();
        let mut required_address_domain = 0;
        let mut increment_compatible = true;
        let mut ram_ra_compatible = true;
        let mut hamming_exact = true;
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
            activity_records.extend(chunk.activity_records);
            if let Some(records) = &mut retained_records {
                records.extend(chunk.access_records);
            }
        }
        let (cycles, increments) = activity_records.into_iter().unzip();
        Ok((
            Self {
                addresses,
                required_address_domain,
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
    pub fn shared<F: Field>(
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
            session.park(activity);
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
    pub fn shared_with_values<F: Field>(
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

    /// Checks the collection-time maximum against the proof's `K`.
    pub fn validate_addresses<F: Field>(&self, ram_k: usize) -> Result<(), KernelError<F>> {
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
    pub(crate) fn validated_addresses<F: Field>(
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
    pub fn fold_addresses<F: Field>(&self, eq_address: &[F]) -> Vec<F> {
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
    pub fn fold_cycles<F: Field>(&self, eq_cycle: &[F], ram_k: usize) -> Vec<F> {
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
    pub fn reconstruct_val_init<F: Field>(&self, pre_values: &[u64], val_final: Vec<F>) -> Vec<F> {
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::metal::offload::ram_trace::{RamAccessCollectionError, RamAccessCollectionStorage};

    fn collector() -> CollectRamAccessColumns {
        CollectRamAccessColumns {
            addresses: Vec::new(),
            required_address_domain: 0,
            pre_values: Vec::new(),
            post_values: Vec::new(),
            ram_increment_cycles: Vec::new(),
            ram_increments: Vec::new(),
            access_count: 0,
            access_records: Some(Vec::new()),
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
        assert_eq!(collected.required_address_domain, 8);

        let columns = RamAccessColumns {
            addresses: collected.addresses,
            required_address_domain: collected.required_address_domain,
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
