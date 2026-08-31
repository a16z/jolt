//! Checked ABI, analytical plan, and independent oracles for RAM RAF evaluation.

use std::mem::{align_of, size_of};

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use thiserror::Error;

mod runtime;

pub use runtime::{PendingRamRafSequence, RamRafAddressPlane, RamRafObservation, RamRafSequence};

pub const SOURCE: &str = include_str!("shader.metal");

pub const RAM_RAF_ADDRESS_DOMAIN: usize = 1 << 13;
pub const RAM_RAF_INNER_LOG2: usize = 15;
pub const RAM_RAF_INNER_LENGTH: usize = 1 << RAM_RAF_INNER_LOG2;
pub const RAM_RAF_TILE_ADDRESSES: usize = 1_376;
pub const RAM_RAF_TILE_COUNT: usize = 6;
pub const RAM_RAF_THREADS: usize = 1_024;
pub const RAM_RAF_SIMD_WIDTH: usize = 32;
pub const RAM_RAF_ACCUMULATOR_WORDS: usize = 5;
pub const RAM_RAF_NO_ACCESS: u32 = u32::MAX;
pub const RAM_RAF_DEFAULT_TRACE_CUTOFF: usize = 1 << 20;
pub const RAM_RAF_AKITA_OFFSET: u32 = 0xffff_a7f7;

pub const RAM_RAF_FOLD_PIPELINE: &str = "solinas_ram_raf_fold_tiles";
pub const RAM_RAF_FINALIZE_PIPELINE: &str = "solinas_ram_raf_finalize";
pub const RAM_RAF_SEGMENTED_COLD_PIPELINE: &str = "solinas_ram_raf_segmented_cold";
pub const RAM_RAF_SEGMENTED_BOUNDED_PIPELINE: &str = "solinas_ram_raf_segmented_bounded";
pub const RAM_RAF_SEGMENTED_HOT_CHUNK_PIPELINE: &str = "solinas_ram_raf_segmented_hot_chunk";
pub const RAM_RAF_SEGMENTED_HOT_FINALIZE_PIPELINE: &str = "solinas_ram_raf_segmented_hot_finalize";
pub const RAM_RAF_SEGMENTED_THREADS: usize = 256;
pub const RAM_RAF_SEGMENTED_THREADGROUP_BYTES: usize = 8 * FIELD_BYTES;

const FIELD_BYTES: usize = 16;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafAddress(u32);

impl RamRafAddress {
    pub const NO_ACCESS: Self = Self(RAM_RAF_NO_ACCESS);

    pub fn accessed(address: u32) -> Result<Self, RamRafError> {
        if address < RAM_RAF_ADDRESS_DOMAIN as u32 {
            Ok(Self(address))
        } else {
            Err(RamRafError::AddressOutsideDomain { address })
        }
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for RamRafAddress {
    type Error = RamRafError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == RAM_RAF_NO_ACCESS {
            Ok(Self::NO_ACCESS)
        } else {
            Self::accessed(value)
        }
    }
}

const _: [(); 4] = [(); size_of::<RamRafAddress>()];
const _: [(); 4] = [(); align_of::<RamRafAddress>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafFoldParams {
    pub rows: u32,
    pub addresses: u32,
    pub inner_length: u32,
    pub outer_length: u32,
    pub tile_addresses: u32,
    pub tiles: u32,
    pub accumulator_words: u32,
    pub no_access: u32,
}

const _: [(); 32] = [(); size_of::<RamRafFoldParams>()];
const _: [(); 4] = [(); align_of::<RamRafFoldParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RamRafSegmentedParams {
    pub rows: u32,
    pub addresses: u32,
    pub accesses: u32,
    pub inner_length: u32,
    pub outer_length: u32,
    pub cold_segment_threshold: u32,
    pub hot_message_chunk_size: u32,
    pub bounded_address_count: u32,
    pub hot_address_count: u32,
    pub hot_message_chunk_count: u32,
}

const _: [(); 40] = [(); size_of::<RamRafSegmentedParams>()];
const _: [(); 4] = [(); align_of::<RamRafSegmentedParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafCounters {
    pub nonzero_subtotals: u32,
    pub invalid_rows: u32,
    pub accessed_rows: u32,
    pub unsupported_dispatches: u32,
}

const _: [(); 16] = [(); size_of::<RamRafCounters>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafConfig {
    pub inner_log2: usize,
    pub tile_addresses: usize,
    pub threads: usize,
    pub trace_cutoff: usize,
}

impl Default for RamRafConfig {
    fn default() -> Self {
        Self {
            inner_log2: RAM_RAF_INNER_LOG2,
            tile_addresses: RAM_RAF_TILE_ADDRESSES,
            threads: RAM_RAF_THREADS,
            trace_cutoff: RAM_RAF_DEFAULT_TRACE_CUTOFF,
        }
    }
}

impl RamRafConfig {
    pub fn validate_metal(self, rows: usize, addresses: usize) -> Result<RamRafShape, RamRafError> {
        if self.inner_log2 != RAM_RAF_INNER_LOG2 {
            return Err(RamRafError::UnsupportedInnerLog2 {
                got: self.inner_log2,
            });
        }
        if !matches!(self.threads, 512 | RAM_RAF_THREADS) {
            return Err(RamRafError::InvalidThreads { got: self.threads });
        }
        if self.tile_addresses == 0 || !self.tile_addresses.is_multiple_of(RAM_RAF_SIMD_WIDTH) {
            return Err(RamRafError::InvalidTileWidth {
                got: self.tile_addresses,
            });
        }
        let dynamic_bytes = checked_product(
            "dynamic threadgroup bytes",
            self.tile_addresses,
            RAM_RAF_ACCUMULATOR_WORDS * size_of::<u32>(),
        )?;
        if dynamic_bytes > 32 * 1_024 {
            return Err(RamRafError::ThreadgroupMemory { dynamic_bytes });
        }
        if !self.trace_cutoff.is_power_of_two() || self.trace_cutoff < RAM_RAF_INNER_LENGTH {
            return Err(RamRafError::InvalidTraceCutoff {
                got: self.trace_cutoff,
            });
        }

        let shape = RamRafShape::new(rows, addresses, self.tile_addresses)?;
        if shape.tiles != RAM_RAF_TILE_COUNT {
            return Err(RamRafError::UnsupportedTileCount { got: shape.tiles });
        }
        Ok(shape)
    }

    pub fn fold_params(self, shape: RamRafShape) -> Result<RamRafFoldParams, RamRafError> {
        let checked = self.validate_metal(shape.rows, shape.addresses)?;
        if checked != shape {
            return Err(RamRafError::ShapeConfigMismatch);
        }
        Ok(RamRafFoldParams {
            rows: shader_count("rows", shape.rows)?,
            addresses: shader_count("addresses", shape.addresses)?,
            inner_length: shader_count("inner length", shape.inner_length)?,
            outer_length: shader_count("outer length", shape.outer_length)?,
            tile_addresses: shader_count("tile addresses", shape.tile_addresses)?,
            tiles: shader_count("tiles", shape.tiles)?,
            accumulator_words: RAM_RAF_ACCUMULATOR_WORDS as u32,
            no_access: RAM_RAF_NO_ACCESS,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafShape {
    rows: usize,
    addresses: usize,
    inner_length: usize,
    outer_length: usize,
    tile_addresses: usize,
    tiles: usize,
}

impl RamRafShape {
    fn new(rows: usize, addresses: usize, tile_addresses: usize) -> Result<Self, RamRafError> {
        validate_power_of_two("rows", rows)?;
        if rows < RAM_RAF_INNER_LENGTH || !rows.is_multiple_of(RAM_RAF_INNER_LENGTH) {
            return Err(RamRafError::RowsNotSplitCompatible { rows });
        }
        if addresses != RAM_RAF_ADDRESS_DOMAIN {
            return Err(RamRafError::UnsupportedAddressDomain { got: addresses });
        }
        let outer_length = rows / RAM_RAF_INNER_LENGTH;
        let tiles = addresses.div_ceil(tile_addresses);
        Ok(Self {
            rows,
            addresses,
            inner_length: RAM_RAF_INNER_LENGTH,
            outer_length,
            tile_addresses,
            tiles,
        })
    }

    copy_field_getters! { pub, {
        rows: usize,
        addresses: usize,
        inner_length: usize,
        outer_length: usize,
        tile_addresses: usize,
        tiles: usize,
    }}
}

/// Metadata proving that a shared address allocation was checked once by its producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedRamRafAddressPlane {
    rows: usize,
    byte_length: usize,
    addresses: usize,
    device_registry_id: u64,
    storage_id: usize,
}

impl ValidatedRamRafAddressPlane {
    pub fn new_after_content_validation(
        shape: RamRafShape,
        byte_length: usize,
        device_registry_id: u64,
        storage_id: usize,
    ) -> Result<Self, RamRafError> {
        let expected = checked_product("resident address bytes", shape.rows, size_of::<u32>())?;
        if byte_length != expected {
            return Err(RamRafError::ResidentByteLength {
                expected,
                got: byte_length,
            });
        }
        if storage_id == 0 {
            return Err(RamRafError::MissingStorageIdentity);
        }
        Ok(Self {
            rows: shape.rows,
            byte_length,
            addresses: shape.addresses,
            device_registry_id,
            storage_id,
        })
    }

    pub fn validate_consumer(
        self,
        shape: RamRafShape,
        device_registry_id: u64,
    ) -> Result<(), RamRafError> {
        if self.rows != shape.rows || self.addresses != shape.addresses {
            return Err(RamRafError::ResidentShapeMismatch);
        }
        if self.device_registry_id != device_registry_id {
            return Err(RamRafError::ResidentDeviceMismatch);
        }
        Ok(())
    }

    copy_field_getters! { pub, {
        byte_length: usize,
        storage_id: usize,
        device_registry_id: u64,
    }}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafStoragePlan {
    pub borrowed_address_bytes: usize,
    pub e_lo_bytes: usize,
    pub e_hi_bytes: usize,
    pub deferred_bytes: usize,
    pub canonical_bytes: usize,
    pub sequence_owned_bytes: usize,
    pub dynamic_threadgroup_bytes: usize,
}

impl RamRafStoragePlan {
    pub fn new(shape: RamRafShape) -> Result<Self, RamRafError> {
        let borrowed_address_bytes = checked_product("address bytes", shape.rows, 4)?;
        let e_lo_bytes = checked_product("E_lo bytes", shape.inner_length, FIELD_BYTES)?;
        let e_hi_bytes = checked_product("E_hi bytes", shape.outer_length, FIELD_BYTES)?;
        let deferred_bytes = checked_product(
            "deferred bytes",
            shape.addresses,
            RAM_RAF_ACCUMULATOR_WORDS * size_of::<u32>(),
        )?;
        let canonical_bytes = checked_product("canonical bytes", shape.addresses, FIELD_BYTES)?;
        let sequence_owned_bytes = [e_lo_bytes, e_hi_bytes, deferred_bytes, canonical_bytes]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                total.checked_add(bytes).ok_or(RamRafError::SizeOverflow {
                    label: "sequence-owned bytes",
                })
            })?;
        let dynamic_threadgroup_bytes = checked_product(
            "dynamic threadgroup bytes",
            shape.tile_addresses,
            RAM_RAF_ACCUMULATOR_WORDS * size_of::<u32>(),
        )?;
        Ok(Self {
            borrowed_address_bytes,
            e_lo_bytes,
            e_hi_bytes,
            deferred_bytes,
            canonical_bytes,
            sequence_owned_bytes,
            dynamic_threadgroup_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafDeviceLimits {
    pub thread_execution_width: usize,
    pub max_threads_per_threadgroup: usize,
    pub max_threadgroup_memory_bytes: usize,
    pub pipeline_static_threadgroup_bytes: usize,
    pub max_buffer_bytes: usize,
    pub recommended_working_set_bytes: usize,
}

impl RamRafDeviceLimits {
    pub fn validate(
        self,
        config: RamRafConfig,
        shape: RamRafShape,
    ) -> Result<RamRafStoragePlan, RamRafError> {
        if config.validate_metal(shape.rows, shape.addresses)? != shape {
            return Err(RamRafError::ShapeConfigMismatch);
        }
        if self.thread_execution_width != RAM_RAF_SIMD_WIDTH {
            return Err(RamRafError::ThreadExecutionWidth {
                got: self.thread_execution_width,
            });
        }
        if self.max_threads_per_threadgroup < config.threads {
            return Err(RamRafError::DeviceThreadLimit {
                required: config.threads,
                got: self.max_threads_per_threadgroup,
            });
        }
        let storage = RamRafStoragePlan::new(shape)?;
        let required_threadgroup_bytes = storage
            .dynamic_threadgroup_bytes
            .checked_add(self.pipeline_static_threadgroup_bytes)
            .ok_or(RamRafError::SizeOverflow {
                label: "total threadgroup bytes",
            })?;
        if required_threadgroup_bytes > self.max_threadgroup_memory_bytes {
            return Err(RamRafError::DeviceThreadgroupMemory {
                required: required_threadgroup_bytes,
                got: self.max_threadgroup_memory_bytes,
            });
        }
        let largest_buffer = storage
            .borrowed_address_bytes
            .max(storage.e_lo_bytes)
            .max(storage.e_hi_bytes)
            .max(storage.deferred_bytes)
            .max(storage.canonical_bytes);
        if largest_buffer > self.max_buffer_bytes {
            return Err(RamRafError::DeviceBufferLimit {
                required: largest_buffer,
                got: self.max_buffer_bytes,
            });
        }
        let peak = storage
            .borrowed_address_bytes
            .checked_add(storage.sequence_owned_bytes)
            .ok_or(RamRafError::SizeOverflow {
                label: "peak resident bytes",
            })?;
        if peak > self.recommended_working_set_bytes {
            return Err(RamRafError::RecommendedWorkingSet {
                required: peak,
                got: self.recommended_working_set_bytes,
            });
        }
        Ok(storage)
    }
}

/// Builds the two small big-endian equality tables; it never materializes `T` fields.
pub fn split_equality<F: Field>(point: &[F]) -> Result<(Vec<F>, Vec<F>), RamRafError> {
    if point.len() < RAM_RAF_INNER_LOG2 {
        return Err(RamRafError::PointTooShort { got: point.len() });
    }
    let split = point.len() - RAM_RAF_INNER_LOG2;
    Ok((
        EqPolynomial::evals(&point[split..], None),
        EqPolynomial::evals(&point[..split], None),
    ))
}

/// Dense independent oracle corresponding to `RamAccessColumns::fold_cycles`.
pub fn dense_pushforward_oracle<F: Field>(
    raw: &[u32],
    eq_cycle: &[F],
    addresses: usize,
) -> Result<Vec<F>, RamRafError> {
    if raw.len() != eq_cycle.len() {
        return Err(RamRafError::Length {
            label: "dense equality table",
            expected: raw.len(),
            got: eq_cycle.len(),
        });
    }
    let mut output = vec![F::zero(); addresses];
    for (&address, &weight) in raw.iter().zip(eq_cycle) {
        if address == RAM_RAF_NO_ACCESS {
            continue;
        }
        let address_index = address as usize;
        if address_index >= addresses {
            return Err(RamRafError::AddressOutsideDomain { address });
        }
        output[address_index] += weight;
    }
    Ok(output)
}

/// Split-equality oracle with a direct row loop, independent of address tiling.
pub fn split_pushforward_oracle<F: Field>(
    raw: &[u32],
    e_lo: &[F],
    e_hi: &[F],
    addresses: usize,
) -> Result<Vec<F>, RamRafError> {
    if e_lo.is_empty() || e_hi.is_empty() {
        return Err(RamRafError::EmptyEqualityTable);
    }
    let expected = checked_product("split equality rows", e_lo.len(), e_hi.len())?;
    if raw.len() != expected {
        return Err(RamRafError::Length {
            label: "address plane",
            expected,
            got: raw.len(),
        });
    }
    let mut output = vec![F::zero(); addresses];
    for (row, &address) in raw.iter().enumerate() {
        if address == RAM_RAF_NO_ACCESS {
            continue;
        }
        let address_index = address as usize;
        if address_index >= addresses {
            return Err(RamRafError::AddressOutsideDomain { address });
        }
        output[address_index] += e_lo[row % e_lo.len()] * e_hi[row / e_lo.len()];
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafQuadraticMessage<F> {
    pub at_zero: F,
    pub at_one: F,
    pub at_two: F,
    pub leading: F,
}

impl<F: Field> RamRafQuadraticMessage<F> {
    pub const fn evaluations(self) -> [F; 3] {
        [self.at_zero, self.at_one, self.at_two]
    }

    pub fn coefficients(self) -> [F; 3] {
        [
            self.at_zero,
            self.at_one - self.at_zero - self.leading,
            self.leading,
        ]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RamRafAffineTail<F> {
    ra: Vec<F>,
    base: F,
    step: F,
    rounds_bound: usize,
}

impl<F: Field> RamRafAffineTail<F> {
    pub fn new(ra: Vec<F>, lowest_address: u64) -> Result<Self, RamRafError> {
        if ra.is_empty() || !ra.len().is_power_of_two() {
            return Err(RamRafError::NonPowerOfTwoTail { got: ra.len() });
        }
        let last_offset = 8u64
            .checked_mul((ra.len() - 1) as u64)
            .ok_or(RamRafError::LowestAddressOverflow)?;
        let _ = lowest_address
            .checked_add(last_offset)
            .ok_or(RamRafError::LowestAddressOverflow)?;
        Ok(Self {
            ra,
            base: F::from_u64(lowest_address),
            step: F::from_u64(8),
            rounds_bound: 0,
        })
    }

    pub fn remaining_rounds(&self) -> usize {
        self.ra.len().ilog2() as usize
    }

    pub fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    #[cfg(feature = "allocative")]
    pub(crate) fn heap_bytes(&self) -> usize {
        self.ra.capacity() * size_of::<F>()
    }

    pub fn input_claim(&self) -> F {
        let mut claim = F::zero();
        let mut address = self.base;
        for &value in &self.ra {
            claim += address * value;
            address += self.step;
        }
        claim
    }

    pub fn message(&self, previous_claim: F) -> Result<RamRafQuadraticMessage<F>, RamRafError> {
        if self.ra.len() < 2 {
            return Err(RamRafError::TailFullyBound);
        }
        let mut at_zero = F::zero();
        let mut at_one = F::zero();
        let mut delta_sum = F::zero();
        let mut u_zero = self.base;
        let pair_step = self.step + self.step;
        for pair in self.ra.chunks_exact(2) {
            let r_zero = pair[0];
            let r_one = pair[1];
            at_zero += u_zero * r_zero;
            at_one += (u_zero + self.step) * r_one;
            delta_sum += r_one - r_zero;
            u_zero += pair_step;
        }
        if at_zero + at_one != previous_claim {
            return Err(RamRafError::RoundClaimMismatch);
        }
        let leading = self.step * delta_sum;
        let at_two = at_one + at_one - at_zero + leading + leading;
        Ok(RamRafQuadraticMessage {
            at_zero,
            at_one,
            at_two,
            leading,
        })
    }

    pub fn bind(&mut self, challenge: F) -> Result<(), RamRafError> {
        if self.ra.len() < 2 {
            return Err(RamRafError::TailFullyBound);
        }
        let half = self.ra.len() / 2;
        for pair_index in 0..half {
            let r_zero = self.ra[2 * pair_index];
            let delta = self.ra[2 * pair_index + 1] - r_zero;
            self.ra[pair_index] = r_zero + challenge * delta;
        }
        self.ra.truncate(half);
        self.base += self.step * challenge;
        self.step += self.step;
        self.rounds_bound += 1;
        Ok(())
    }

    pub fn output(self) -> Result<RamRafTailOutput<F>, RamRafError> {
        if self.ra.len() != 1 {
            return Err(RamRafError::TailNotFullyBound {
                remaining: self.remaining_rounds(),
            });
        }
        Ok(RamRafTailOutput {
            ram_ra: self.ra[0],
            unmap_address: self.base,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRafTailOutput<F> {
    pub ram_ra: F,
    pub unmap_address: F,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum RamRafError {
    #[error("{label} must be a nonzero power of two, got {got}")]
    NotPowerOfTwo { label: &'static str, got: usize },
    #[error("RAM RAF rows do not admit the fixed 2^15 equality split: {rows}")]
    RowsNotSplitCompatible { rows: usize },
    #[error("RAM RAF Metal specializes the address domain to 8192, got {got}")]
    UnsupportedAddressDomain { got: usize },
    #[error("RAM RAF Metal specializes inner_log2 to 15, got {got}")]
    UnsupportedInnerLog2 { got: usize },
    #[error("invalid RAM RAF thread count {got}")]
    InvalidThreads { got: usize },
    #[error("invalid RAM RAF address tile width {got}")]
    InvalidTileWidth { got: usize },
    #[error("RAM RAF address tiling must use six scans, got {got}")]
    UnsupportedTileCount { got: usize },
    #[error("RAM RAF dynamic threadgroup allocation is {dynamic_bytes} bytes")]
    ThreadgroupMemory { dynamic_bytes: usize },
    #[error("RAM RAF requires SIMD width 32, got {got}")]
    ThreadExecutionWidth { got: usize },
    #[error("RAM RAF requires {required} threads per group, device admits {got}")]
    DeviceThreadLimit { required: usize, got: usize },
    #[error("RAM RAF requires {required} threadgroup bytes, device admits {got}")]
    DeviceThreadgroupMemory { required: usize, got: usize },
    #[error("RAM RAF requires a {required}-byte buffer, device admits {got}")]
    DeviceBufferLimit { required: usize, got: usize },
    #[error("RAM RAF peak is {required} bytes, recommended working set is {got}")]
    RecommendedWorkingSet { required: usize, got: usize },
    #[error("invalid RAM RAF trace cutoff {got}")]
    InvalidTraceCutoff { got: usize },
    #[error("RAM RAF shape and config disagree")]
    ShapeConfigMismatch,
    #[error("RAM RAF address {address} is outside the 8192-entry domain")]
    AddressOutsideDomain { address: u32 },
    #[error("{label} length mismatch: expected {expected}, got {got}")]
    Length {
        label: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{label} overflowed usize")]
    SizeOverflow { label: &'static str },
    #[error("{label} does not fit the Metal u32 ABI: {value}")]
    ShaderCount { label: &'static str, value: usize },
    #[error("resident address bytes mismatch: expected {expected}, got {got}")]
    ResidentByteLength { expected: usize, got: usize },
    #[error("resident RAM address allocation has no storage identity")]
    MissingStorageIdentity,
    #[error("resident RAM address allocation has the wrong shape")]
    ResidentShapeMismatch,
    #[error("resident RAM address allocation belongs to another Metal device")]
    ResidentDeviceMismatch,
    #[error("equality point is shorter than 15 coordinates: {got}")]
    PointTooShort { got: usize },
    #[error("RAM RAF split equality tables must be nonempty")]
    EmptyEqualityTable,
    #[error("RAM RAF affine tail length must be a nonzero power of two, got {got}")]
    NonPowerOfTwoTail { got: usize },
    #[error("lowest RAM address overflows the affine unmap table")]
    LowestAddressOverflow,
    #[error("RAM RAF round message does not sum to the running claim")]
    RoundClaimMismatch,
    #[error("RAM RAF affine tail is already fully bound")]
    TailFullyBound,
    #[error("RAM RAF affine tail still has {remaining} rounds")]
    TailNotFullyBound { remaining: usize },
}

fn validate_power_of_two(label: &'static str, value: usize) -> Result<(), RamRafError> {
    if value == 0 || !value.is_power_of_two() {
        Err(RamRafError::NotPowerOfTwo { label, got: value })
    } else {
        Ok(())
    }
}

fn checked_product(label: &'static str, lhs: usize, rhs: usize) -> Result<usize, RamRafError> {
    lhs.checked_mul(rhs)
        .ok_or(RamRafError::SizeOverflow { label })
}

fn shader_count(label: &'static str, value: usize) -> Result<u32, RamRafError> {
    u32::try_from(value).map_err(|_| RamRafError::ShaderCount { label, value })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tail property test setup")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    fn direct_claim(ra: &[AkitaField], base: AkitaField, step: AkitaField) -> AkitaField {
        ra.iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (base + step * AkitaField::from_u64(index as u64)) * value)
            .sum()
    }

    fn direct_message(
        ra: &[AkitaField],
        base: AkitaField,
        step: AkitaField,
        x: AkitaField,
    ) -> AkitaField {
        ra.chunks_exact(2)
            .enumerate()
            .map(|(pair_index, pair)| {
                let index = AkitaField::from_u64((2 * pair_index) as u64);
                let address = base + step * (index + x);
                let value = pair[0] + x * (pair[1] - pair[0]);
                address * value
            })
            .sum()
    }

    #[test]
    fn affine_tail_matches_direct_polynomial_definition() {
        let values = (0..32)
            .map(|index| AkitaField::from_u64(11 + 17 * index))
            .collect::<Vec<_>>();
        let mut tail = RamRafAffineTail::new(values, 0x1000).unwrap();

        while tail.ra.len() > 1 {
            let claim = direct_claim(&tail.ra, tail.base, tail.step);
            assert_eq!(tail.input_claim(), claim);
            let message = tail.message(claim).unwrap();
            for (x, got) in message.evaluations().into_iter().enumerate() {
                assert_eq!(
                    got,
                    direct_message(
                        &tail.ra,
                        tail.base,
                        tail.step,
                        AkitaField::from_u64(x as u64),
                    )
                );
            }
            tail.bind(AkitaField::from_u64(3 + tail.rounds_bound() as u64))
                .unwrap();
        }
    }
}
