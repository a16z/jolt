//! Exact geometry, traffic, arithmetic, and admission arithmetic.

use core::mem::size_of;

use super::abi::{
    validate_bucket_projection, RamRafAccessRecord, RamRafBucketDescriptor, RamRafBucketProjection,
    RamRafBucketRecord, RamRafStatus, RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS,
    RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN, RAM_RAF_SUCCESSOR_INNER_LENGTH,
    RAM_RAF_SUCCESSOR_TILE_ADDRESSES, RAM_RAF_SUCCESSOR_TILE_COUNT,
};

pub const TARGET_LOG_T: u32 = 26;
pub const TARGET_ROWS: u64 = 1 << TARGET_LOG_T;
pub const TARGET_ADDRESSES: u64 = RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN as u64;

pub const FROZEN_CPU_ARTIFACT: &str =
    "benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json";
pub const FROZEN_CPU_ARTIFACT_SHA256: &str =
    "587e00a65bde003a7c3481f58b1ea047ed2c908b0e3d9808bbc6e894b2df";
pub const FROZEN_CPU_REVISION: &str = "5f520c21e338632aa0bf5936ceb02be6c22fa40f";
pub const FROZEN_CPU_SAMPLE_SELECTOR: &str =
    ".attribution_samples[].optimized.kernels[] | select(.kernel == \"RamRafEvaluation\") | .wall_ms";
pub const FROZEN_CPU_SAMPLES_NS: [u64; 5] =
    [76_520_166, 76_746_208, 73_944_962, 73_501_876, 74_870_252];
pub const FROZEN_CPU_MEDIAN_NS: u64 = 74_870_252;
pub const FIVE_X_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 5;
pub const EIGHT_X_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 8;

pub const RETAINED_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const RETAINED_FIELD_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const RETAINED_COMMAND_WALL_FLOOR_NS: u64 = 141_000;
pub const OBSERVED_AFFINE_TAIL_NO_FS_NS: u64 = 177_958;
pub const TARGET_FIBONACCI_EVIDENCE: &str =
    "crates/jolt-kernels/autoresearch/evidence/piop_log26_9de144572_diagnostic.json";
pub const TARGET_FIBONACCI_EVIDENCE_SHA256: &str =
    "9c38dc2b06261e70a47a1fb206bd6aa12c828f19b697f9c36420c70c1fbcd69d";
pub const TARGET_FIBONACCI_OBSERVED_NONZERO_SUBTOTALS: u64 = 76;
pub const RETAINED_RANDOM_EVIDENCE: &str =
    "crates/jolt-kernels/autoresearch/evidence/ram_raf_evaluation_log26_observed_9de144572.json";
pub const RETAINED_RANDOM_EVIDENCE_SHA256: &str =
    "9a26cec509ab11b8e7f33963ee157fc8cc4b00e153f5dab162fc17a6bb0ea6ff";

pub const HOST_SPARSE_SCREEN_MAX_ACCESSES: u64 = 1 << 15;
pub const DEVICE_DIRECT_SCREEN_MAX_ACCESSES: u64 = 1 << 20;
pub const DEVICE_DIRECT_MIN_SUBTOTAL_FRACTION_DENOMINATOR: u64 = 8;

const FIELD_BYTES: u128 = 16;
const ACCUMULATOR_BYTES: u128 = (RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS * size_of::<u32>()) as u128;
const REQUIRED_ATOMIC_RMW_BYTES: u128 = 2 * 4 * size_of::<u32>() as u128;
const GLOBAL_ATOMIC_RMW_BYTES: u128 = 2 * ACCUMULATOR_BYTES;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// The retained Fibonacci diagnostic observed 190 accesses and 76 nonzero
/// `(outer, address)` subtotals. Structural occupancy, descriptor count, and
/// cleared bucket slots were not recorded, so they use their access-count
/// maxima here.
pub const TARGET_FIBONACCI_TOPOLOGY: RamRafTopology = RamRafTopology {
    accesses: 190,
    occupied_subtotals: 190,
    nonempty_buckets: 190,
    bucket_slots: 190 * RAM_RAF_SUCCESSOR_TILE_ADDRESSES as u64,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    InvalidRows,
    InvalidAddressDomain,
    InvalidInnerLength,
    InvalidTileGeometry,
    InvalidTopology,
    InvalidCarryCensus,
    Overflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub rows: u64,
    pub addresses: u64,
    pub inner_length: u64,
    pub tile_addresses: u64,
}

impl Geometry {
    pub const fn target() -> Self {
        Self {
            rows: TARGET_ROWS,
            addresses: TARGET_ADDRESSES,
            inner_length: RAM_RAF_SUCCESSOR_INNER_LENGTH as u64,
            tile_addresses: RAM_RAF_SUCCESSOR_TILE_ADDRESSES as u64,
        }
    }

    pub fn validate(self) -> Result<Self, ModelError> {
        if self.rows == 0 || !self.rows.is_power_of_two() || self.rows > u32::MAX as u64 {
            return Err(ModelError::InvalidRows);
        }
        if self.addresses != TARGET_ADDRESSES {
            return Err(ModelError::InvalidAddressDomain);
        }
        if self.inner_length != RAM_RAF_SUCCESSOR_INNER_LENGTH as u64
            || self.rows < self.inner_length
            || !self.rows.is_multiple_of(self.inner_length)
        {
            return Err(ModelError::InvalidInnerLength);
        }
        if self.tile_addresses != RAM_RAF_SUCCESSOR_TILE_ADDRESSES as u64
            || self.addresses.div_ceil(self.tile_addresses) != RAM_RAF_SUCCESSOR_TILE_COUNT as u64
        {
            return Err(ModelError::InvalidTileGeometry);
        }
        Ok(self)
    }

    pub const fn outer_length(self) -> u64 {
        self.rows / self.inner_length
    }

    pub fn tiles(self) -> u64 {
        self.addresses.div_ceil(self.tile_addresses)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRafTopology {
    pub accesses: u64,
    /// Structurally occupied `(outer, address)` cells. A challenge-dependent
    /// subtotal may still cancel to zero.
    pub occupied_subtotals: u64,
    pub nonempty_buckets: u64,
    /// Sum of the active address capacities of all dispatched buckets.
    pub bucket_slots: u64,
}

impl RamRafTopology {
    pub fn from_bucket_projection(
        geometry: Geometry,
        projection: &RamRafBucketProjection,
    ) -> Result<Self, ModelError> {
        let geometry = geometry.validate()?;
        let rows = usize::try_from(geometry.rows).map_err(|_| ModelError::Overflow)?;
        let addresses = usize::try_from(geometry.addresses).map_err(|_| ModelError::Overflow)?;
        validate_bucket_projection(projection, rows, addresses)
            .map_err(|_| ModelError::InvalidTopology)?;
        let mut touched = vec![false; RAM_RAF_SUCCESSOR_TILE_ADDRESSES];
        let mut occupied_subtotals = 0u64;
        let mut bucket_slots = 0u64;
        for descriptor in projection.descriptors.iter().copied() {
            let tile_start = descriptor.tile() as usize * RAM_RAF_SUCCESSOR_TILE_ADDRESSES;
            let active = RAM_RAF_SUCCESSOR_TILE_ADDRESSES.min(addresses - tile_start);
            bucket_slots = bucket_slots
                .checked_add(active as u64)
                .ok_or(ModelError::Overflow)?;
            let first = descriptor.first_record() as usize;
            let end = first + descriptor.record_count() as usize;
            for record in projection.records[first..end].iter().copied() {
                let local = record.local_address() as usize;
                if !touched[local] {
                    touched[local] = true;
                    occupied_subtotals = occupied_subtotals
                        .checked_add(1)
                        .ok_or(ModelError::Overflow)?;
                }
            }
            touched[..active].fill(false);
        }
        Self {
            accesses: u64::try_from(projection.records.len()).map_err(|_| ModelError::Overflow)?,
            occupied_subtotals,
            nonempty_buckets: u64::try_from(projection.descriptors.len())
                .map_err(|_| ModelError::Overflow)?,
            bucket_slots,
        }
        .validate(geometry)
    }

    pub fn validate(self, geometry: Geometry) -> Result<Self, ModelError> {
        let geometry = geometry.validate()?;
        let max_subtotals = geometry
            .outer_length()
            .checked_mul(geometry.addresses)
            .ok_or(ModelError::Overflow)?;
        let max_buckets = geometry
            .outer_length()
            .checked_mul(geometry.tiles())
            .ok_or(ModelError::Overflow)?;
        let full_bucket_slots = self
            .nonempty_buckets
            .checked_mul(geometry.tile_addresses)
            .ok_or(ModelError::Overflow)?;
        let short_tile_slots = geometry
            .tiles()
            .checked_mul(geometry.tile_addresses)
            .and_then(|slots| slots.checked_sub(geometry.addresses))
            .ok_or(ModelError::Overflow)?;
        let short_bucket_deficit = full_bucket_slots.checked_sub(self.bucket_slots);
        let bucket_slots_valid = short_bucket_deficit.is_some_and(|deficit| {
            short_tile_slots != 0
                && deficit.is_multiple_of(short_tile_slots)
                && deficit / short_tile_slots <= self.nonempty_buckets.min(geometry.outer_length())
        });
        let empty = self.accesses == 0
            && self.occupied_subtotals == 0
            && self.nonempty_buckets == 0
            && self.bucket_slots == 0;
        let nonempty = self.accesses > 0
            && self.accesses <= geometry.rows
            && self.occupied_subtotals > 0
            && self.occupied_subtotals <= self.accesses.min(max_subtotals)
            && self.nonempty_buckets > 0
            && self.nonempty_buckets <= self.occupied_subtotals.min(max_buckets)
            && self.bucket_slots >= self.occupied_subtotals
            && self.bucket_slots <= max_subtotals
            && bucket_slots_valid;
        if !empty && !nonempty {
            return Err(ModelError::InvalidTopology);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RamRafExecutionLane {
    HostSparse,
    DeviceDirect,
    DeviceBucketed,
    RetainedDenseTiled,
    OptimizedCpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostSparseProjection {
    pub equality_products: u128,
    pub pushforward_products: u128,
    pub affine_tail_products: u128,
    pub total_products: u128,
    pub record_bytes: u128,
    pub equality_bytes: u128,
    pub output_bytes: u128,
    pub working_bytes: u128,
}

impl HostSparseProjection {
    pub fn new(geometry: Geometry, accesses: u64) -> Result<Self, ModelError> {
        let geometry = geometry.validate()?;
        if accesses > geometry.rows {
            return Err(ModelError::InvalidTopology);
        }
        let equality_products = u128::from(
            geometry
                .inner_length
                .checked_sub(1)
                .and_then(|lo| geometry.outer_length().checked_sub(1).map(|hi| lo + hi))
                .ok_or(ModelError::Overflow)?,
        );
        let affine_tail_products = checked_sum(&[
            checked_mul(u128::from(geometry.addresses - 1), 3)?,
            u128::from(geometry.addresses.ilog2()) * 2,
        ])?;
        let pushforward_products = u128::from(accesses);
        let total_products = checked_sum(&[
            equality_products,
            pushforward_products,
            affine_tail_products,
        ])?;
        let record_bytes = checked_mul(
            u128::from(accesses),
            size_of::<RamRafAccessRecord>() as u128,
        )?;
        let equality_bytes = checked_mul(
            u128::from(
                geometry
                    .inner_length
                    .checked_add(geometry.outer_length())
                    .ok_or(ModelError::Overflow)?,
            ),
            FIELD_BYTES,
        )?;
        let output_bytes = checked_mul(u128::from(geometry.addresses), FIELD_BYTES)?;
        let working_bytes = checked_sum(&[record_bytes, equality_bytes, output_bytes])?;
        Ok(Self {
            equality_products,
            pushforward_products,
            affine_tail_products,
            total_products,
            record_bytes,
            equality_bytes,
            output_bytes,
            working_bytes,
        })
    }
}

/// A pre-benchmark screen only. Promotion freezes measured crossovers for the
/// host, direct-device, bucketed-device, and retained dense lanes.
pub fn execution_screen(
    topology: RamRafTopology,
    bucket_projection_resident: bool,
    dense_plane_resident: bool,
) -> Result<RamRafExecutionLane, ModelError> {
    let topology = topology.validate(Geometry::target())?;
    if topology.accesses <= HOST_SPARSE_SCREEN_MAX_ACCESSES {
        return Ok(RamRafExecutionLane::HostSparse);
    }
    let direct_contention_screen = topology
        .occupied_subtotals
        .saturating_mul(DEVICE_DIRECT_MIN_SUBTOTAL_FRACTION_DENOMINATOR)
        >= topology.accesses;
    if topology.accesses <= DEVICE_DIRECT_SCREEN_MAX_ACCESSES && direct_contention_screen {
        return Ok(RamRafExecutionLane::DeviceDirect);
    }
    if bucket_projection_resident {
        return Ok(RamRafExecutionLane::DeviceBucketed);
    }
    if dense_plane_resident {
        return Ok(RamRafExecutionLane::RetainedDenseTiled);
    }
    Ok(RamRafExecutionLane::OptimizedCpu)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoragePlan {
    pub common_access_records: u128,
    pub bucket_records: u128,
    pub bucket_descriptors: u128,
    pub e_lo: u128,
    pub e_hi: u128,
    pub deferred: u128,
    pub canonical_output: u128,
    pub status: u128,
    pub sequence_owned: u128,
    pub direct_total_resident: u128,
    pub bucket_total_resident: u128,
    pub bucket_dynamic_threadgroup: u128,
}

impl StoragePlan {
    pub fn new(geometry: Geometry, topology: RamRafTopology) -> Result<Self, ModelError> {
        let geometry = geometry.validate()?;
        let topology = topology.validate(geometry)?;
        let common_access_records = checked_mul(
            u128::from(topology.accesses),
            size_of::<RamRafAccessRecord>() as u128,
        )?;
        let bucket_records = checked_mul(
            u128::from(topology.accesses),
            size_of::<RamRafBucketRecord>() as u128,
        )?;
        let bucket_descriptors = checked_mul(
            u128::from(topology.nonempty_buckets),
            size_of::<RamRafBucketDescriptor>() as u128,
        )?;
        let e_lo = checked_mul(u128::from(geometry.inner_length), FIELD_BYTES)?;
        let e_hi = checked_mul(u128::from(geometry.outer_length()), FIELD_BYTES)?;
        let deferred = checked_mul(u128::from(geometry.addresses), ACCUMULATOR_BYTES)?;
        let canonical_output = checked_mul(u128::from(geometry.addresses), FIELD_BYTES)?;
        let status = size_of::<RamRafStatus>() as u128;
        let sequence_owned = checked_sum(&[e_lo, e_hi, deferred, canonical_output, status])?;
        let direct_total_resident = checked_sum(&[common_access_records, sequence_owned])?;
        let bucket_total_resident = checked_sum(&[
            common_access_records,
            bucket_records,
            bucket_descriptors,
            sequence_owned,
        ])?;
        let bucket_dynamic_threadgroup =
            checked_mul(u128::from(geometry.tile_addresses), ACCUMULATOR_BYTES)?;
        Ok(Self {
            common_access_records,
            bucket_records,
            bucket_descriptors,
            e_lo,
            e_hi,
            deferred,
            canonical_output,
            status,
            sequence_owned,
            direct_total_resident,
            bucket_total_resident,
            bucket_dynamic_threadgroup,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnownRoof {
    Direct,
    Bucketed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoofProjection {
    pub lane: KnownRoof,
    pub products: u128,
    /// Mandatory bytes when no deferred add wraps above 128 bits.
    pub minimum_external_bytes: u128,
    /// Conservative bytes when every deferred add updates its fifth word.
    pub maximum_external_bytes: u128,
    pub equality_shader_logical_bytes: u128,
    pub global_atomic_operations_min: u128,
    pub global_atomic_operations_max: u128,
    pub threadgroup_atomic_operations_min: u128,
    pub threadgroup_atomic_operations_max: u128,
    pub threadgroup_internal_bytes_min: u128,
    pub threadgroup_internal_bytes_max: u128,
    pub compute_floor_ns: u64,
    pub minimum_traffic_floor_ns: u64,
    pub cached_max_traffic_floor_ns: u64,
    pub uncached_max_traffic_floor_ns: u64,
    pub cached_conservative_bottom_ns: u64,
    pub uncached_conservative_bottom_ns: u64,
    pub cached_conservative_eighty_percent_screen_ns: u64,
    pub uncached_conservative_eighty_percent_screen_ns: u64,
    /// Known terms only. Atomic issue, setup, and host Fiat-Shamir are absent.
    pub known_complete_no_fs_ns: u64,
}

impl RoofProjection {
    pub fn direct(geometry: Geometry, topology: RamRafTopology) -> Result<Self, ModelError> {
        let geometry = geometry.validate()?;
        let topology = topology.validate(geometry)?;
        let fixed = fixed_service_bytes(geometry)?;
        let record_bytes = checked_mul(
            u128::from(topology.accesses),
            size_of::<RamRafAccessRecord>() as u128,
        )?;
        let minimum_atomic_bytes =
            checked_mul(u128::from(topology.accesses), REQUIRED_ATOMIC_RMW_BYTES)?;
        let global_atomic_bytes =
            checked_mul(u128::from(topology.accesses), GLOBAL_ATOMIC_RMW_BYTES)?;
        let minimum_external_bytes = checked_sum(&[fixed, record_bytes, minimum_atomic_bytes])?;
        let maximum_external_bytes = checked_sum(&[fixed, record_bytes, global_atomic_bytes])?;
        let equality_shader_logical_bytes =
            checked_mul(u128::from(topology.accesses), 2 * FIELD_BYTES)?;
        finish_projection(
            KnownRoof::Direct,
            u128::from(topology.accesses),
            minimum_external_bytes,
            maximum_external_bytes,
            equality_shader_logical_bytes,
            0,
            0,
            0,
        )
    }

    pub fn bucketed(
        geometry: Geometry,
        topology: RamRafTopology,
        nonzero_subtotals: u64,
    ) -> Result<Self, ModelError> {
        let geometry = geometry.validate()?;
        let topology = topology.validate(geometry)?;
        if nonzero_subtotals > topology.occupied_subtotals {
            return Err(ModelError::InvalidTopology);
        }
        let fixed = fixed_service_bytes(geometry)?;
        let record_bytes = checked_mul(
            u128::from(topology.accesses),
            size_of::<RamRafBucketRecord>() as u128,
        )?;
        let descriptor_bytes = checked_mul(
            u128::from(topology.nonempty_buckets),
            size_of::<RamRafBucketDescriptor>() as u128,
        )?;
        let minimum_atomic_bytes =
            checked_mul(u128::from(nonzero_subtotals), REQUIRED_ATOMIC_RMW_BYTES)?;
        let global_atomic_bytes =
            checked_mul(u128::from(nonzero_subtotals), GLOBAL_ATOMIC_RMW_BYTES)?;
        let minimum_external_bytes =
            checked_sum(&[fixed, record_bytes, descriptor_bytes, minimum_atomic_bytes])?;
        let maximum_external_bytes =
            checked_sum(&[fixed, record_bytes, descriptor_bytes, global_atomic_bytes])?;
        let equality_shader_logical_bytes = checked_mul(
            u128::from(
                topology
                    .accesses
                    .checked_add(nonzero_subtotals)
                    .ok_or(ModelError::Overflow)?,
            ),
            FIELD_BYTES,
        )?;
        let local_units = topology
            .accesses
            .checked_add(topology.bucket_slots)
            .ok_or(ModelError::Overflow)?;
        let threadgroup_internal_bytes_min = checked_sum(&[
            checked_mul(u128::from(topology.bucket_slots), GLOBAL_ATOMIC_RMW_BYTES)?,
            checked_mul(u128::from(topology.accesses), REQUIRED_ATOMIC_RMW_BYTES)?,
        ])?;
        let threadgroup_internal_bytes_max =
            checked_mul(u128::from(local_units), GLOBAL_ATOMIC_RMW_BYTES)?;
        finish_projection(
            KnownRoof::Bucketed,
            u128::from(nonzero_subtotals),
            minimum_external_bytes,
            maximum_external_bytes,
            equality_shader_logical_bytes,
            u128::from(topology.accesses) * RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS as u128,
            threadgroup_internal_bytes_min,
            threadgroup_internal_bytes_max,
        )
    }

    /// Pre-execution upper bound that assumes every occupied subtotal is
    /// algebraically nonzero.
    pub fn bucketed_structural_upper_bound(
        geometry: Geometry,
        topology: RamRafTopology,
    ) -> Result<Self, ModelError> {
        Self::bucketed(geometry, topology, topology.occupied_subtotals)
    }

    /// Exact external traffic after counting fifth-word global atomics.
    pub fn exact_external_bytes(
        self,
        global_carries: u128,
        equality_tables_cache_resident: bool,
    ) -> Result<u128, ModelError> {
        let max_carries = self
            .global_atomic_operations_max
            .checked_sub(self.global_atomic_operations_min)
            .ok_or(ModelError::Overflow)?;
        if global_carries > max_carries {
            return Err(ModelError::InvalidCarryCensus);
        }
        let carry_bytes = checked_mul(global_carries, 2 * size_of::<u32>() as u128)?;
        let bytes = self
            .minimum_external_bytes
            .checked_add(carry_bytes)
            .ok_or(ModelError::Overflow)?;
        if equality_tables_cache_resident {
            Ok(bytes)
        } else {
            bytes
                .checked_add(self.equality_shader_logical_bytes)
                .ok_or(ModelError::Overflow)
        }
    }

    /// Exact threadgroup traffic after counting fifth-word local atomics.
    pub fn exact_threadgroup_internal_bytes(
        self,
        threadgroup_carries: u128,
    ) -> Result<u128, ModelError> {
        let max_carries = self
            .threadgroup_atomic_operations_max
            .checked_sub(self.threadgroup_atomic_operations_min)
            .ok_or(ModelError::Overflow)?;
        if threadgroup_carries > max_carries {
            return Err(ModelError::InvalidCarryCensus);
        }
        self.threadgroup_internal_bytes_min
            .checked_add(checked_mul(
                threadgroup_carries,
                2 * size_of::<u32>() as u128,
            )?)
            .ok_or(ModelError::Overflow)
    }

    pub fn frozen_cpu_speedup_over_known_terms(self) -> f64 {
        FROZEN_CPU_MEDIAN_NS as f64 / self.known_complete_no_fs_ns as f64
    }
}

fn finish_projection(
    lane: KnownRoof,
    products: u128,
    minimum_external_bytes: u128,
    maximum_external_bytes: u128,
    equality_shader_logical_bytes: u128,
    threadgroup_atomic_operations_max: u128,
    threadgroup_internal_bytes_min: u128,
    threadgroup_internal_bytes_max: u128,
) -> Result<RoofProjection, ModelError> {
    let compute_floor_ns = ceil_rate_ns(products, RETAINED_FIELD_PRODUCTS_PER_SECOND)?;
    let minimum_traffic_floor_ns =
        ceil_rate_ns(minimum_external_bytes, RETAINED_COPY_BYTES_PER_SECOND)?;
    let cached_max_traffic_floor_ns =
        ceil_rate_ns(maximum_external_bytes, RETAINED_COPY_BYTES_PER_SECOND)?;
    let uncached_bytes = maximum_external_bytes
        .checked_add(equality_shader_logical_bytes)
        .ok_or(ModelError::Overflow)?;
    let uncached_max_traffic_floor_ns =
        ceil_rate_ns(uncached_bytes, RETAINED_COPY_BYTES_PER_SECOND)?;
    let cached_conservative_bottom_ns = compute_floor_ns.max(cached_max_traffic_floor_ns);
    let uncached_conservative_bottom_ns = compute_floor_ns.max(uncached_max_traffic_floor_ns);
    let cached_conservative_eighty_percent_screen_ns =
        ceil_ratio(cached_conservative_bottom_ns, 5, 4)?;
    let uncached_conservative_eighty_percent_screen_ns =
        ceil_ratio(uncached_conservative_bottom_ns, 5, 4)?;
    let known_complete_no_fs_ns = RETAINED_COMMAND_WALL_FLOOR_NS
        .checked_add(cached_conservative_eighty_percent_screen_ns)
        .and_then(|value| value.checked_add(OBSERVED_AFFINE_TAIL_NO_FS_NS))
        .ok_or(ModelError::Overflow)?;
    let global_atomic_operations_min = products.checked_mul(4).ok_or(ModelError::Overflow)?;
    let global_atomic_operations_max = products.checked_mul(5).ok_or(ModelError::Overflow)?;
    let threadgroup_atomic_operations_min =
        threadgroup_atomic_operations_max / RAM_RAF_SUCCESSOR_ACCUMULATOR_WORDS as u128 * 4;
    Ok(RoofProjection {
        lane,
        products,
        minimum_external_bytes,
        maximum_external_bytes,
        equality_shader_logical_bytes,
        global_atomic_operations_min,
        global_atomic_operations_max,
        threadgroup_atomic_operations_min,
        threadgroup_atomic_operations_max,
        threadgroup_internal_bytes_min,
        threadgroup_internal_bytes_max,
        compute_floor_ns,
        minimum_traffic_floor_ns,
        cached_max_traffic_floor_ns,
        uncached_max_traffic_floor_ns,
        cached_conservative_bottom_ns,
        uncached_conservative_bottom_ns,
        cached_conservative_eighty_percent_screen_ns,
        uncached_conservative_eighty_percent_screen_ns,
        known_complete_no_fs_ns,
    })
}

fn fixed_service_bytes(geometry: Geometry) -> Result<u128, ModelError> {
    let deferred = checked_mul(u128::from(geometry.addresses), ACCUMULATOR_BYTES)?;
    let canonical = checked_mul(u128::from(geometry.addresses), FIELD_BYTES)?;
    let status = size_of::<RamRafStatus>() as u128;
    checked_sum(&[2 * deferred, 2 * canonical, 2 * status])
}

fn checked_mul(lhs: u128, rhs: u128) -> Result<u128, ModelError> {
    lhs.checked_mul(rhs).ok_or(ModelError::Overflow)
}

fn checked_sum(values: &[u128]) -> Result<u128, ModelError> {
    values.iter().try_fold(0u128, |sum, value| {
        sum.checked_add(*value).ok_or(ModelError::Overflow)
    })
}

fn ceil_rate_ns(units: u128, units_per_second: u64) -> Result<u64, ModelError> {
    let numerator = units
        .checked_mul(NANOS_PER_SECOND)
        .and_then(|value| value.checked_add(u128::from(units_per_second) - 1))
        .ok_or(ModelError::Overflow)?;
    u64::try_from(numerator / u128::from(units_per_second)).map_err(|_| ModelError::Overflow)
}

fn ceil_ratio(value: u64, numerator: u64, denominator: u64) -> Result<u64, ModelError> {
    let scaled = u128::from(value)
        .checked_mul(u128::from(numerator))
        .and_then(|value| value.checked_add(u128::from(denominator) - 1))
        .ok_or(ModelError::Overflow)?;
    u64::try_from(scaled / u128::from(denominator)).map_err(|_| ModelError::Overflow)
}
