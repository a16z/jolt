//! Checked plan and independent oracles for RAM RA claim reduction.

mod host;
mod runtime;

pub use host::*;
pub use runtime::*;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid fixtures")]
mod tests;

use std::mem::{align_of, size_of};

use jolt_field::Field;
use jolt_poly::EqPolynomial;
use thiserror::Error;

pub const SOURCE: &str = include_str!("shader.metal");

pub const RAM_RA_CLAIM_TERMS: usize = 3;
pub const RAM_RA_CLAIM_ADDRESS_DOMAIN: usize = 1 << 13;
pub const RAM_RA_CLAIM_NO_ACCESS: u32 = u32::MAX;
pub const RAM_RA_CLAIM_SIMD_WIDTH: usize = 32;
pub const RAM_RA_CLAIM_Q_PARTITIONS: usize = 8;
pub const RAM_RA_CLAIM_MAX_Q_PARTITIONS: usize = 16;
pub const RAM_RA_CLAIM_DEFAULT_TRACE_CUTOFF: usize = 1 << 26;
pub const RAM_RA_CLAIM_AKITA_OFFSET: u32 = 0xffff_a7f7;

pub const RAM_RA_CLAIM_TARGET_LOG_T: usize = 26;
pub const RAM_RA_CLAIM_TARGET_ROWS: usize = 1 << RAM_RA_CLAIM_TARGET_LOG_T;
pub const RAM_RA_CLAIM_TARGET_CPU_NS: u64 = 40_507_503;
pub const RAM_RA_CLAIM_TARGET_FIVE_X_NS: u64 = 8_101_500;
pub const RAM_RA_CLAIM_TARGET_ACCESSED_ROWS: usize = 22_000_000;
pub const RAM_RA_CLAIM_TARGET_FIXED_NS: u64 = 1_500_000;

/// Measured on the M4 Max full-width six-accumulator control, not this shader.
pub const M4_MAX_SIX_ACCUMULATOR_FULL_WIDTH_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
/// Measured on the M4 Max one-chain full-width probe matched to the compact gather.
pub const M4_MAX_ONE_CHAIN_FULL_WIDTH_PRODUCTS_PER_SECOND: u64 = 45_709_000_000;
/// Measured on the M4 Max large streaming-copy control.
pub const M4_MAX_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;

pub const BUILD_Q_PARTIALS_PIPELINE: &str = "solinas_ram_ra_claim_build_q_partials";
pub const BUILD_Q_PARTIALS_EXPLICIT_PIPELINE: &str =
    "solinas_ram_ra_claim_build_q_partials_explicit";
pub const BUILD_Q_PARTIALS_COMPACT_PIPELINE: &str = "solinas_ram_ra_claim_build_q_partials_compact";
pub const REDUCE_Q_PIPELINE: &str = "solinas_ram_ra_claim_reduce_q";
pub const GATHER_H_PIPELINE: &str = "solinas_ram_ra_claim_gather_h";
pub const GATHER_H_COMPACT_PIPELINE: &str = "solinas_ram_ra_claim_gather_h_compact";

pub const Q_BUILD_ADDRESSES_SLOT: u64 = 0;
pub const Q_BUILD_EQ_ADDRESS_SLOT: u64 = 1;
pub const Q_BUILD_EQ_HI_SLOT: u64 = 2;
pub const Q_BUILD_PARTIALS_SLOT: u64 = 3;
pub const Q_BUILD_COUNTERS_SLOT: u64 = 4;
pub const Q_BUILD_PARAMS_SLOT: u64 = 5;
pub const Q_COMPACT_BUILD_ENTRIES_SLOT: u64 = 0;
pub const Q_COMPACT_BUILD_OFFSETS_SLOT: u64 = 1;
pub const Q_COMPACT_BUILD_EQ_ADDRESS_SLOT: u64 = 2;
pub const Q_COMPACT_BUILD_EQ_HI_SLOT: u64 = 3;
pub const Q_COMPACT_BUILD_PARTIALS_SLOT: u64 = 4;
pub const Q_COMPACT_BUILD_COUNTERS_SLOT: u64 = 5;
pub const Q_COMPACT_BUILD_PARAMS_SLOT: u64 = 6;
pub const Q_REDUCE_PARTIALS_SLOT: u64 = 0;
pub const Q_REDUCE_OUTPUT_SLOT: u64 = 1;
pub const Q_REDUCE_COUNTERS_SLOT: u64 = 2;
pub const Q_REDUCE_PARAMS_SLOT: u64 = 3;
pub const H_COMPACT_ENTRIES_SLOT: u64 = 0;
pub const H_COMPACT_OFFSETS_SLOT: u64 = 1;
pub const H_COMPACT_EQ_ADDRESS_SLOT: u64 = 2;
pub const H_COMPACT_EQ_PREFIX_SLOT: u64 = 3;
pub const H_COMPACT_OUTPUT_SLOT: u64 = 4;
pub const H_COMPACT_COUNTERS_SLOT: u64 = 5;
pub const H_COMPACT_PARAMS_SLOT: u64 = 6;

const FIELD_BYTES: usize = 16;
const NANOSECONDS_PER_SECOND: u128 = 1_000_000_000;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimAddress(u32);

impl RamRaClaimAddress {
    pub const NO_ACCESS: Self = Self(RAM_RA_CLAIM_NO_ACCESS);

    pub fn accessed(address: u32) -> Result<Self, RamRaClaimError> {
        if address < RAM_RA_CLAIM_ADDRESS_DOMAIN as u32 {
            Ok(Self(address))
        } else {
            Err(RamRaClaimError::AddressOutsideDomain { address })
        }
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    pub const fn is_access(self) -> bool {
        self.0 != RAM_RA_CLAIM_NO_ACCESS
    }
}

impl TryFrom<u32> for RamRaClaimAddress {
    type Error = RamRaClaimError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == RAM_RA_CLAIM_NO_ACCESS {
            Ok(Self::NO_ACCESS)
        } else {
            Self::accessed(value)
        }
    }
}

const _: [(); 4] = [(); size_of::<RamRaClaimAddress>()];
const _: [(); 4] = [(); align_of::<RamRaClaimAddress>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRaClaimParams {
    pub rows: u32,
    pub address_limit: u32,
    pub prefix_length: u32,
    pub suffix_length: u32,
    pub terms: u32,
    pub no_access: u32,
    pub q_partitions: u32,
    pub threads: u32,
}

const _: [(); 32] = [(); size_of::<RamRaClaimParams>()];
const _: [(); 4] = [(); align_of::<RamRaClaimParams>()];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamRaClaimCounters {
    pub q_accessed_rows: u32,
    pub q_invalid_rows: u32,
    pub gather_invalid_rows: u32,
    pub unsupported_dispatches: u32,
}

const _: [(); 16] = [(); size_of::<RamRaClaimCounters>()];
const _: [(); 4] = [(); align_of::<RamRaClaimCounters>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimShape {
    rows: usize,
    prefix_bits: usize,
    suffix_bits: usize,
    prefix_length: usize,
    suffix_length: usize,
}

impl RamRaClaimShape {
    pub fn new(rows: usize, address_limit: usize) -> Result<Self, RamRaClaimError> {
        validate_power_of_two("rows", rows)?;
        if address_limit != RAM_RA_CLAIM_ADDRESS_DOMAIN {
            return Err(RamRaClaimError::UnsupportedAddressDomain { got: address_limit });
        }
        let log_t = rows.ilog2() as usize;
        let prefix_bits = log_t / 2;
        let suffix_bits = log_t - prefix_bits;
        let prefix_length = checked_pow2("prefix length", prefix_bits)?;
        let suffix_length = checked_pow2("suffix length", suffix_bits)?;
        if prefix_length < RAM_RA_CLAIM_SIMD_WIDTH
            || !prefix_length.is_multiple_of(RAM_RA_CLAIM_SIMD_WIDTH)
        {
            return Err(RamRaClaimError::PrefixTooSmall { prefix_length });
        }
        Ok(Self {
            rows,
            prefix_bits,
            suffix_bits,
            prefix_length,
            suffix_length,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn log_t(self) -> usize {
        self.prefix_bits + self.suffix_bits
    }

    pub const fn prefix_bits(self) -> usize {
        self.prefix_bits
    }

    pub const fn suffix_bits(self) -> usize {
        self.suffix_bits
    }

    pub const fn prefix_length(self) -> usize {
        self.prefix_length
    }

    pub const fn suffix_length(self) -> usize {
        self.suffix_length
    }

    pub fn params(self, config: RamRaClaimConfig) -> Result<RamRaClaimParams, RamRaClaimError> {
        config.validate_shape(self)?;
        Ok(RamRaClaimParams {
            rows: shader_count("rows", self.rows)?,
            address_limit: RAM_RA_CLAIM_ADDRESS_DOMAIN as u32,
            prefix_length: shader_count("prefix length", self.prefix_length)?,
            suffix_length: shader_count("suffix length", self.suffix_length)?,
            terms: RAM_RA_CLAIM_TERMS as u32,
            no_access: RAM_RA_CLAIM_NO_ACCESS,
            q_partitions: config.q_partitions as u32,
            threads: config.threads as u32,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RamRaClaimQAccumulator {
    Array,
    Explicit,
    #[default]
    Compact,
}

impl RamRaClaimQAccumulator {
    pub const fn pipeline(self) -> &'static str {
        match self {
            Self::Array => BUILD_Q_PARTIALS_PIPELINE,
            Self::Explicit => BUILD_Q_PARTIALS_EXPLICIT_PIPELINE,
            Self::Compact => BUILD_Q_PARTIALS_COMPACT_PIPELINE,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Array => "array",
            Self::Explicit => "explicit",
            Self::Compact => "compact",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimConfig {
    pub threads: usize,
    pub q_partitions: usize,
    pub trace_cutoff: usize,
    pub q_accumulator: RamRaClaimQAccumulator,
}

impl Default for RamRaClaimConfig {
    fn default() -> Self {
        Self {
            threads: RAM_RA_CLAIM_SIMD_WIDTH,
            q_partitions: RAM_RA_CLAIM_Q_PARTITIONS,
            trace_cutoff: RAM_RA_CLAIM_DEFAULT_TRACE_CUTOFF,
            q_accumulator: RamRaClaimQAccumulator::Compact,
        }
    }
}

impl RamRaClaimConfig {
    pub fn validate(self) -> Result<(), RamRaClaimError> {
        if self.threads != RAM_RA_CLAIM_SIMD_WIDTH {
            return Err(RamRaClaimError::UnsupportedThreads { got: self.threads });
        }
        if !self.q_partitions.is_power_of_two() || self.q_partitions > RAM_RA_CLAIM_MAX_Q_PARTITIONS
        {
            return Err(RamRaClaimError::UnsupportedQPartitions {
                got: self.q_partitions,
            });
        }
        if !self.trace_cutoff.is_power_of_two()
            || self.trace_cutoff < RAM_RA_CLAIM_SIMD_WIDTH.pow(2)
        {
            return Err(RamRaClaimError::InvalidTraceCutoff {
                got: self.trace_cutoff,
            });
        }
        Ok(())
    }

    fn validate_shape(self, shape: RamRaClaimShape) -> Result<(), RamRaClaimError> {
        self.validate()?;
        if !shape.suffix_length.is_multiple_of(self.q_partitions) {
            return Err(RamRaClaimError::QPartitionsDoNotDivideSuffix {
                partitions: self.q_partitions,
                suffix_length: shape.suffix_length,
            });
        }
        Ok(())
    }

    fn execution_from_accessed_rows(
        self,
        shape: RamRaClaimShape,
        accessed_rows: usize,
    ) -> Result<RamRaClaimExecution, RamRaClaimError> {
        self.validate_shape(shape)?;
        validate_accessed_rows(shape.rows, accessed_rows)?;
        if shape.rows < self.trace_cutoff {
            return Ok(RamRaClaimExecution::OptimizedCpu(
                RamRaClaimFallback::TraceBelowCutoff,
            ));
        }
        if !density_admitted(shape.rows, accessed_rows) {
            return Ok(RamRaClaimExecution::OptimizedCpu(
                RamRaClaimFallback::AccessDensity,
            ));
        }
        Ok(RamRaClaimExecution::MetalHybrid)
    }

    pub fn execution_for_validated_plane(
        self,
        shape: RamRaClaimShape,
        plane: ValidatedRamRaClaimAddressPlane,
        byte_length: usize,
        device_registry_id: u64,
        storage_id: usize,
    ) -> Result<RamRaClaimExecution, RamRaClaimError> {
        plane.validate_consumer(shape, byte_length, device_registry_id, storage_id)?;
        self.execution_from_accessed_rows(shape, plane.accessed_rows())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RamRaClaimExecution {
    OptimizedCpu(RamRaClaimFallback),
    MetalHybrid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RamRaClaimFallback {
    TraceBelowCutoff,
    AccessDensity,
}

pub const fn density_admitted(rows: usize, accessed_rows: usize) -> bool {
    (accessed_rows as u128) * (RAM_RA_CLAIM_TARGET_ROWS as u128)
        <= (rows as u128) * (RAM_RA_CLAIM_TARGET_ACCESSED_ROWS as u128)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedRamRaClaimAddressPlane {
    rows: usize,
    byte_length: usize,
    address_limit: usize,
    accessed_rows: usize,
    device_registry_id: u64,
    storage_id: usize,
}

impl ValidatedRamRaClaimAddressPlane {
    pub(crate) fn new_after_content_validation(
        shape: RamRaClaimShape,
        byte_length: usize,
        accessed_rows: usize,
        device_registry_id: u64,
        storage_id: usize,
    ) -> Result<Self, RamRaClaimError> {
        let expected = checked_product("resident address bytes", shape.rows, size_of::<u32>())?;
        if byte_length != expected {
            return Err(RamRaClaimError::ResidentByteLength {
                expected,
                got: byte_length,
            });
        }
        validate_accessed_rows(shape.rows, accessed_rows)?;
        if storage_id == 0 {
            return Err(RamRaClaimError::MissingStorageIdentity);
        }
        Ok(Self {
            rows: shape.rows,
            byte_length,
            address_limit: RAM_RA_CLAIM_ADDRESS_DOMAIN,
            accessed_rows,
            device_registry_id,
            storage_id,
        })
    }

    pub fn validate_consumer(
        self,
        shape: RamRaClaimShape,
        byte_length: usize,
        device_registry_id: u64,
        storage_id: usize,
    ) -> Result<(), RamRaClaimError> {
        if self.rows != shape.rows {
            return Err(RamRaClaimError::ResidentRows {
                expected: shape.rows,
                got: self.rows,
            });
        }
        if self.byte_length != byte_length {
            return Err(RamRaClaimError::ResidentByteLength {
                expected: byte_length,
                got: self.byte_length,
            });
        }
        if self.address_limit != RAM_RA_CLAIM_ADDRESS_DOMAIN {
            return Err(RamRaClaimError::UnsupportedAddressDomain {
                got: self.address_limit,
            });
        }
        if self.device_registry_id != device_registry_id {
            return Err(RamRaClaimError::ResidentDevice {
                expected: device_registry_id,
                got: self.device_registry_id,
            });
        }
        if self.storage_id != storage_id {
            return Err(RamRaClaimError::ResidentStorage {
                expected: storage_id,
                got: self.storage_id,
            });
        }
        Ok(())
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn byte_length(self) -> usize {
        self.byte_length
    }

    pub const fn accessed_rows(self) -> usize {
        self.accessed_rows
    }

    pub const fn device_registry_id(self) -> u64 {
        self.device_registry_id
    }

    pub const fn storage_id(self) -> usize {
        self.storage_id
    }

    pub fn validate_completed_dispatches(
        self,
        counters: RamRaClaimCounters,
    ) -> Result<(), RamRaClaimError> {
        if counters.unsupported_dispatches != 0 {
            return Err(RamRaClaimError::UnsupportedDispatches {
                got: counters.unsupported_dispatches,
            });
        }
        if counters.q_invalid_rows != 0 || counters.gather_invalid_rows != 0 {
            return Err(RamRaClaimError::InvalidShaderRows {
                q: counters.q_invalid_rows,
                gather: counters.gather_invalid_rows,
            });
        }
        if counters.q_accessed_rows as usize != self.accessed_rows {
            return Err(RamRaClaimError::AccessedRowAudit {
                expected: self.accessed_rows,
                got: counters.q_accessed_rows as usize,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimDispatch {
    pub threadgroups: usize,
    pub threads_per_threadgroup: usize,
    pub logical_outputs: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimStoragePlan {
    pub borrowed_address_bytes: usize,
    pub eq_address_bytes: usize,
    pub eq_hi_bytes: usize,
    pub p_bytes: usize,
    pub q_partial_bytes: usize,
    pub q_bytes: usize,
    pub eq_prefix_bytes: usize,
    pub h_bytes: usize,
    pub readback_bytes: usize,
}

impl RamRaClaimStoragePlan {
    pub fn new(shape: RamRaClaimShape, q_partitions: usize) -> Result<Self, RamRaClaimError> {
        if !q_partitions.is_power_of_two() || q_partitions > RAM_RA_CLAIM_MAX_Q_PARTITIONS {
            return Err(RamRaClaimError::UnsupportedQPartitions { got: q_partitions });
        }
        if !shape.suffix_length.is_multiple_of(q_partitions) {
            return Err(RamRaClaimError::QPartitionsDoNotDivideSuffix {
                partitions: q_partitions,
                suffix_length: shape.suffix_length,
            });
        }
        let field_bytes = |label, elements| checked_product(label, elements, FIELD_BYTES);
        let triple_prefix = checked_product(
            "three prefix tables",
            RAM_RA_CLAIM_TERMS,
            shape.prefix_length,
        )?;
        let triple_suffix = checked_product(
            "three suffix tables",
            RAM_RA_CLAIM_TERMS,
            shape.suffix_length,
        )?;
        let q_partial_elements =
            checked_product("Q partial elements", triple_prefix, q_partitions)?;
        let q_bytes = field_bytes("Q bytes", triple_prefix)?;
        let h_bytes = field_bytes("H-prime bytes", shape.suffix_length)?;
        Ok(Self {
            borrowed_address_bytes: checked_product(
                "borrowed address bytes",
                shape.rows,
                size_of::<u32>(),
            )?,
            eq_address_bytes: field_bytes("address equality bytes", RAM_RA_CLAIM_ADDRESS_DOMAIN)?,
            eq_hi_bytes: field_bytes("high equality bytes", triple_suffix)?,
            p_bytes: field_bytes("P bytes", triple_prefix)?,
            q_partial_bytes: field_bytes("Q partial bytes", q_partial_elements)?,
            q_bytes,
            eq_prefix_bytes: field_bytes("prefix equality bytes", shape.prefix_length)?,
            h_bytes,
            readback_bytes: q_bytes
                .checked_add(h_bytes)
                .ok_or(RamRaClaimError::SizeOverflow { label: "readback" })?,
        })
    }

    pub fn sequence_device_bytes(self) -> Result<usize, RamRaClaimError> {
        checked_sum(
            "sequence device bytes",
            &[
                self.eq_address_bytes,
                self.eq_hi_bytes,
                self.q_partial_bytes,
                self.q_bytes,
                self.eq_prefix_bytes,
                self.h_bytes,
            ],
        )
    }

    pub fn host_table_bytes(self) -> Result<usize, RamRaClaimError> {
        checked_sum(
            "host table bytes",
            &[
                self.eq_address_bytes,
                self.eq_hi_bytes,
                self.p_bytes,
                self.q_bytes,
                self.eq_prefix_bytes,
                self.h_bytes,
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimPlan {
    pub shape: RamRaClaimShape,
    pub q_dispatch: RamRaClaimDispatch,
    pub q_reduce_dispatch: RamRaClaimDispatch,
    pub gather_dispatch: RamRaClaimDispatch,
    pub storage: RamRaClaimStoragePlan,
    pub prefix_messages: usize,
    pub suffix_messages: usize,
    pub dispatches: usize,
    pub command_buffers: usize,
    pub completion_waits: usize,
}

impl RamRaClaimPlan {
    pub fn new(config: RamRaClaimConfig, shape: RamRaClaimShape) -> Result<Self, RamRaClaimError> {
        config.validate_shape(shape)?;
        let q_outputs = checked_product("Q outputs", RAM_RA_CLAIM_TERMS, shape.prefix_length)?;
        Ok(Self {
            shape,
            q_dispatch: RamRaClaimDispatch {
                threadgroups: checked_product(
                    "Q producer threadgroups",
                    shape.prefix_length / config.threads,
                    config.q_partitions,
                )?,
                threads_per_threadgroup: config.threads,
                logical_outputs: checked_product(
                    "Q partial outputs",
                    q_outputs,
                    config.q_partitions,
                )?,
            },
            q_reduce_dispatch: RamRaClaimDispatch {
                threadgroups: shape.prefix_length / config.threads,
                threads_per_threadgroup: config.threads,
                logical_outputs: q_outputs,
            },
            gather_dispatch: RamRaClaimDispatch {
                threadgroups: shape.suffix_length,
                threads_per_threadgroup: config.threads,
                logical_outputs: shape.suffix_length,
            },
            storage: RamRaClaimStoragePlan::new(shape, config.q_partitions)?,
            prefix_messages: shape.prefix_bits,
            suffix_messages: shape.suffix_bits,
            dispatches: 3,
            command_buffers: 2,
            completion_waits: 2,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimQStoragePlan {
    pub borrowed_address_bytes: usize,
    pub eq_address_bytes: usize,
    pub eq_hi_bytes: usize,
    pub q_partial_bytes: usize,
    pub q_bytes: usize,
    pub counter_bytes: usize,
    pub sequence_owned_bytes: usize,
    pub total_resident_bytes: usize,
    pub readback_bytes: usize,
}

impl RamRaClaimQStoragePlan {
    pub fn new(shape: RamRaClaimShape, q_partitions: usize) -> Result<Self, RamRaClaimError> {
        let full = RamRaClaimStoragePlan::new(shape, q_partitions)?;
        let counter_bytes = size_of::<RamRaClaimCounters>();
        let sequence_owned_bytes = checked_sum(
            "Q sequence-owned bytes",
            &[
                full.eq_address_bytes,
                full.eq_hi_bytes,
                full.q_partial_bytes,
                full.q_bytes,
                counter_bytes,
            ],
        )?;
        let total_resident_bytes = checked_sum(
            "Q sequence total resident bytes",
            &[full.borrowed_address_bytes, sequence_owned_bytes],
        )?;
        let readback_bytes = checked_sum("Q readback bytes", &[full.q_bytes, counter_bytes])?;
        Ok(Self {
            borrowed_address_bytes: full.borrowed_address_bytes,
            eq_address_bytes: full.eq_address_bytes,
            eq_hi_bytes: full.eq_hi_bytes,
            q_partial_bytes: full.q_partial_bytes,
            q_bytes: full.q_bytes,
            counter_bytes,
            sequence_owned_bytes,
            total_resident_bytes,
            readback_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRaClaimQPlan {
    pub config: RamRaClaimConfig,
    pub shape: RamRaClaimShape,
    pub params: RamRaClaimParams,
    pub producer_dispatch: RamRaClaimDispatch,
    pub reducer_dispatch: RamRaClaimDispatch,
    pub storage: RamRaClaimQStoragePlan,
    pub dispatches: usize,
    pub command_buffers: usize,
    pub completion_waits: usize,
}

impl RamRaClaimQPlan {
    pub fn new(config: RamRaClaimConfig, shape: RamRaClaimShape) -> Result<Self, RamRaClaimError> {
        let complete = RamRaClaimPlan::new(config, shape)?;
        Ok(Self {
            config,
            shape,
            params: shape.params(config)?,
            producer_dispatch: complete.q_dispatch,
            reducer_dispatch: complete.q_reduce_dispatch,
            storage: RamRaClaimQStoragePlan::new(shape, config.q_partitions)?,
            dispatches: 2,
            command_buffers: 1,
            completion_waits: 1,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RamRaClaimProjection {
    pub rows: usize,
    pub accessed_rows: usize,
    pub q_full_width_products: u64,
    pub gather_full_width_products: u64,
    pub half_width_products: u64,
    pub address_bytes_per_pass: u64,
    pub q_perfect_cache_bytes: u64,
    pub gather_perfect_cache_bytes: u64,
    pub q_lookup_logical_bytes: u64,
    pub gather_lookup_logical_bytes: u64,
    pub q_shader_logical_bytes: u64,
    pub gather_shader_logical_bytes: u64,
    pub q_product_floor_ns: u64,
    pub gather_product_floor_ns: u64,
    pub q_perfect_cache_traffic_floor_ns: u64,
    pub gather_perfect_cache_traffic_floor_ns: u64,
    pub q_no_cache_request_floor_ns: u64,
    pub gather_no_cache_request_floor_ns: u64,
    pub q_pursuit_ns: u64,
    pub gather_pursuit_ns: u64,
    pub q_no_cache_pursuit_ns: u64,
    pub gather_no_cache_pursuit_ns: u64,
    pub fixed_ns: u64,
    pub projected_complete_ns: u64,
    pub projected_no_cache_complete_ns: u64,
    pub target_speedup: Option<f64>,
    pub target_no_cache_speedup: Option<f64>,
}

impl RamRaClaimProjection {
    pub fn new(rows: usize, accessed_rows: usize) -> Result<Self, RamRaClaimError> {
        let shape = RamRaClaimShape::new(rows, RAM_RA_CLAIM_ADDRESS_DOMAIN)?;
        validate_accessed_rows(rows, accessed_rows)?;
        let storage = RamRaClaimStoragePlan::new(shape, RAM_RA_CLAIM_Q_PARTITIONS)?;
        let to_u64 = |label, value| {
            u64::try_from(value).map_err(|_| RamRaClaimError::SizeOverflow { label })
        };
        let q_products_host = checked_product("Q products", RAM_RA_CLAIM_TERMS, accessed_rows)?;
        let q_full_width_products = to_u64("Q products", q_products_host)?;
        let gather_full_width_products = to_u64("gather products", accessed_rows)?;
        let address_bytes = checked_product(
            "compact address entries per pass",
            accessed_rows,
            size_of::<u32>(),
        )?;
        let address_bytes_per_pass = to_u64("address bytes per pass", address_bytes)?;
        let q_offset_bytes = checked_product(
            "Q compact offsets",
            shape.prefix_length + 1,
            size_of::<u32>(),
        )?;
        let gather_offset_bytes = checked_product(
            "gather compact offsets",
            shape.suffix_length + 1,
            size_of::<u32>(),
        )?;
        let counter_bytes = size_of::<RamRaClaimCounters>();
        let q_nonlookup_bytes = checked_sum(
            "Q compact non-lookup bytes",
            &[
                address_bytes,
                q_offset_bytes,
                storage.q_partial_bytes,
                storage.q_partial_bytes,
                storage.q_bytes,
                counter_bytes,
            ],
        )?;
        let gather_nonlookup_bytes = checked_sum(
            "gather compact non-lookup bytes",
            &[
                address_bytes,
                gather_offset_bytes,
                storage.h_bytes,
                counter_bytes,
            ],
        )?;
        let q_perfect_cache_bytes_host = checked_sum(
            "Q perfect-cache bytes",
            &[
                q_nonlookup_bytes,
                storage.eq_address_bytes,
                storage.eq_hi_bytes,
            ],
        )?;
        let gather_perfect_cache_bytes_host = checked_sum(
            "gather perfect-cache bytes",
            &[
                gather_nonlookup_bytes,
                storage.eq_address_bytes,
                storage.eq_prefix_bytes,
            ],
        )?;
        let q_lookup_fields = checked_product(
            "Q logical lookup fields",
            accessed_rows,
            RAM_RA_CLAIM_TERMS + 1,
        )?;
        let q_lookup_logical_bytes_host =
            checked_product("Q logical lookup bytes", q_lookup_fields, FIELD_BYTES)?;
        let gather_lookup_fields =
            checked_product("gather logical lookup fields", accessed_rows, 2)?;
        let gather_lookup_logical_bytes_host = checked_product(
            "gather logical lookup bytes",
            gather_lookup_fields,
            FIELD_BYTES,
        )?;
        let q_shader_logical_bytes_host = q_nonlookup_bytes
            .checked_add(q_lookup_logical_bytes_host)
            .ok_or(RamRaClaimError::SizeOverflow {
                label: "Q shader logical bytes",
            })?;
        let gather_shader_logical_bytes_host = gather_nonlookup_bytes
            .checked_add(gather_lookup_logical_bytes_host)
            .ok_or(RamRaClaimError::SizeOverflow {
                label: "gather shader logical bytes",
            })?;

        let q_perfect_cache_bytes = to_u64("Q perfect-cache bytes", q_perfect_cache_bytes_host)?;
        let gather_perfect_cache_bytes = to_u64(
            "gather perfect-cache bytes",
            gather_perfect_cache_bytes_host,
        )?;
        let q_lookup_logical_bytes = to_u64("Q logical lookup bytes", q_lookup_logical_bytes_host)?;
        let gather_lookup_logical_bytes = to_u64(
            "gather logical lookup bytes",
            gather_lookup_logical_bytes_host,
        )?;
        let q_shader_logical_bytes = to_u64("Q shader logical bytes", q_shader_logical_bytes_host)?;
        let gather_shader_logical_bytes = to_u64(
            "gather shader logical bytes",
            gather_shader_logical_bytes_host,
        )?;

        let product_rate = M4_MAX_ONE_CHAIN_FULL_WIDTH_PRODUCTS_PER_SECOND;
        let q_product_floor_ns = rate_floor_ns(q_full_width_products, product_rate)?;
        let gather_product_floor_ns = rate_floor_ns(gather_full_width_products, product_rate)?;
        let q_perfect_cache_traffic_floor_ns =
            rate_floor_ns(q_perfect_cache_bytes, M4_MAX_COPY_BYTES_PER_SECOND)?;
        let gather_perfect_cache_traffic_floor_ns =
            rate_floor_ns(gather_perfect_cache_bytes, M4_MAX_COPY_BYTES_PER_SECOND)?;
        let q_no_cache_request_floor_ns =
            rate_floor_ns(q_shader_logical_bytes, M4_MAX_COPY_BYTES_PER_SECOND)?;
        let gather_no_cache_request_floor_ns =
            rate_floor_ns(gather_shader_logical_bytes, M4_MAX_COPY_BYTES_PER_SECOND)?;
        let q_pursuit_ns =
            eighty_percent_cap(q_product_floor_ns.max(q_perfect_cache_traffic_floor_ns))?;
        let gather_pursuit_ns =
            eighty_percent_cap(gather_product_floor_ns.max(gather_perfect_cache_traffic_floor_ns))?;
        let q_no_cache_pursuit_ns =
            eighty_percent_cap(q_product_floor_ns.max(q_no_cache_request_floor_ns))?;
        let gather_no_cache_pursuit_ns =
            eighty_percent_cap(gather_product_floor_ns.max(gather_no_cache_request_floor_ns))?;
        let projected_complete_ns = q_pursuit_ns
            .checked_add(gather_pursuit_ns)
            .and_then(|value| value.checked_add(RAM_RA_CLAIM_TARGET_FIXED_NS))
            .ok_or(RamRaClaimError::SizeOverflow {
                label: "projected complete time",
            })?;
        let projected_no_cache_complete_ns = q_no_cache_pursuit_ns
            .checked_add(gather_no_cache_pursuit_ns)
            .and_then(|value| value.checked_add(RAM_RA_CLAIM_TARGET_FIXED_NS))
            .ok_or(RamRaClaimError::SizeOverflow {
                label: "projected no-cache complete time",
            })?;
        let target_speedup = (rows == RAM_RA_CLAIM_TARGET_ROWS)
            .then(|| RAM_RA_CLAIM_TARGET_CPU_NS as f64 / projected_complete_ns as f64);
        let target_no_cache_speedup = (rows == RAM_RA_CLAIM_TARGET_ROWS)
            .then(|| RAM_RA_CLAIM_TARGET_CPU_NS as f64 / projected_no_cache_complete_ns as f64);
        Ok(Self {
            rows,
            accessed_rows,
            q_full_width_products,
            gather_full_width_products,
            half_width_products: 0,
            address_bytes_per_pass,
            q_perfect_cache_bytes,
            gather_perfect_cache_bytes,
            q_lookup_logical_bytes,
            gather_lookup_logical_bytes,
            q_shader_logical_bytes,
            gather_shader_logical_bytes,
            q_product_floor_ns,
            gather_product_floor_ns,
            q_perfect_cache_traffic_floor_ns,
            gather_perfect_cache_traffic_floor_ns,
            q_no_cache_request_floor_ns,
            gather_no_cache_request_floor_ns,
            q_pursuit_ns,
            gather_pursuit_ns,
            q_no_cache_pursuit_ns,
            gather_no_cache_pursuit_ns,
            fixed_ns: RAM_RA_CLAIM_TARGET_FIXED_NS,
            projected_complete_ns,
            projected_no_cache_complete_ns,
            target_speedup,
            target_no_cache_speedup,
        })
    }

    pub const fn clears_target_five_x_under_perfect_cache(self) -> bool {
        self.rows == RAM_RA_CLAIM_TARGET_ROWS
            && self.projected_complete_ns.saturating_mul(5) <= RAM_RA_CLAIM_TARGET_CPU_NS
    }

    pub const fn clears_target_five_x_without_lookup_cache(self) -> bool {
        self.rows == RAM_RA_CLAIM_TARGET_ROWS
            && self.projected_no_cache_complete_ns.saturating_mul(5) <= RAM_RA_CLAIM_TARGET_CPU_NS
    }
}

pub mod oracle {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct RamRaClaimOracleResult<F: Field> {
        pub input_claim: F,
        pub messages: Vec<[F; 2]>,
        pub ram_ra: F,
        pub derived_cycle_eq: [F; RAM_RA_CLAIM_TERMS],
        pub output_point: Vec<F>,
    }

    pub struct RamRaClaimOracleInputs<'a, F: Field> {
        pub addresses: &'a [u32],
        pub r_address: &'a [F],
        pub cycle_points: [&'a [F]; RAM_RA_CLAIM_TERMS],
        pub gamma: F,
    }

    pub fn dense<F: Field>(
        inputs: RamRaClaimOracleInputs<'_, F>,
        challenges: &[F],
    ) -> Result<RamRaClaimOracleResult<F>, RamRaClaimError> {
        let _ = validate_oracle_inputs(&inputs, challenges)?;
        let eq_address = EqPolynomial::evals(inputs.r_address, None);
        let mut h = folded_addresses(inputs.addresses, &eq_address)?;
        let mut eq_cycle = inputs
            .cycle_points
            .map(|point| EqPolynomial::evals(point, None));
        let gamma_powers = gamma_powers(inputs.gamma);
        let input_claim = h.iter().enumerate().fold(F::zero(), |sum, (j, &h_j)| {
            let e = (0..RAM_RA_CLAIM_TERMS)
                .fold(F::zero(), |acc, x| acc + gamma_powers[x] * eq_cycle[x][j]);
            sum + h_j * e
        });
        let mut messages = Vec::with_capacity(challenges.len());
        for &challenge in challenges {
            messages.push(dense_message(&h, &eq_cycle, &gamma_powers)?);
            bind_pairs(&mut h, challenge)?;
            for table in &mut eq_cycle {
                bind_pairs(table, challenge)?;
            }
        }
        let output_cycle = output_cycle_point(challenges);
        let derived_cycle_eq =
            core::array::from_fn(|x| EqPolynomial::<F>::mle(inputs.cycle_points[x], &output_cycle));
        Ok(RamRaClaimOracleResult {
            input_claim,
            messages,
            ram_ra: h[0],
            derived_cycle_eq,
            output_point: [inputs.r_address, output_cycle.as_slice()].concat(),
        })
    }

    pub fn split<F: Field>(
        inputs: RamRaClaimOracleInputs<'_, F>,
        challenges: &[F],
    ) -> Result<RamRaClaimOracleResult<F>, RamRaClaimError> {
        let shape = validate_oracle_inputs(&inputs, challenges)?;
        let eq_address = EqPolynomial::evals(inputs.r_address, None);
        let mut p = inputs
            .cycle_points
            .map(|point| EqPolynomial::evals(&point[shape.suffix_bits..], None));
        let mut eq_hi = inputs
            .cycle_points
            .map(|point| EqPolynomial::evals(&point[..shape.suffix_bits], None));
        let mut q = build_q(inputs.addresses, &eq_address, &eq_hi, shape.prefix_bits)?;
        let gamma_powers = gamma_powers(inputs.gamma);
        let input_claim = prefix_claim(&p, &q, &gamma_powers)?;
        let mut messages = Vec::with_capacity(challenges.len());
        for &challenge in &challenges[..shape.prefix_bits] {
            messages.push(prefix_message(&p, &q, &gamma_powers)?);
            for table in p.iter_mut().chain(q.iter_mut()) {
                bind_pairs(table, challenge)?;
            }
        }

        let r_prefix: Vec<F> = challenges[..shape.prefix_bits]
            .iter()
            .rev()
            .copied()
            .collect();
        let eq_prefix = EqPolynomial::evals(&r_prefix, None);
        let mut h = gather_h(
            inputs.addresses,
            &eq_address,
            &eq_prefix,
            shape.prefix_bits,
            shape.suffix_bits,
        )?;
        let scales: [F; RAM_RA_CLAIM_TERMS] = core::array::from_fn(|x| {
            EqPolynomial::<F>::mle(&inputs.cycle_points[x][shape.suffix_bits..], &r_prefix)
        });
        let coefficients = core::array::from_fn(|x| gamma_powers[x] * scales[x]);
        for &challenge in &challenges[shape.prefix_bits..] {
            messages.push(suffix_message(&h, &eq_hi, &coefficients)?);
            bind_pairs(&mut h, challenge)?;
            for table in &mut eq_hi {
                bind_pairs(table, challenge)?;
            }
        }
        let output_cycle = output_cycle_point(challenges);
        let derived_cycle_eq = core::array::from_fn(|x| scales[x] * eq_hi[x][0]);
        Ok(RamRaClaimOracleResult {
            input_claim,
            messages,
            ram_ra: h[0],
            derived_cycle_eq,
            output_point: [inputs.r_address, output_cycle.as_slice()].concat(),
        })
    }

    pub fn check_parity<F: Field>(
        dense: &RamRaClaimOracleResult<F>,
        split: &RamRaClaimOracleResult<F>,
    ) -> Result<(), RamRaClaimError> {
        if dense.input_claim != split.input_claim {
            return Err(RamRaClaimError::OracleInputClaimDrift);
        }
        if dense.messages.len() != split.messages.len() {
            return Err(RamRaClaimError::OracleRoundCountDrift {
                dense: dense.messages.len(),
                split: split.messages.len(),
            });
        }
        for (round, (lhs, rhs)) in dense.messages.iter().zip(&split.messages).enumerate() {
            if lhs != rhs {
                return Err(RamRaClaimError::OracleMessageDrift { round });
            }
        }
        if dense.ram_ra != split.ram_ra {
            return Err(RamRaClaimError::OracleOutputDrift);
        }
        if dense.output_point != split.output_point {
            return Err(RamRaClaimError::OracleOutputPointDrift);
        }
        for term in 0..RAM_RA_CLAIM_TERMS {
            if dense.derived_cycle_eq[term] != split.derived_cycle_eq[term] {
                return Err(RamRaClaimError::OracleDerivedDrift { term });
            }
        }
        Ok(())
    }

    pub fn build_q<F: Field>(
        addresses: &[u32],
        eq_address: &[F],
        eq_hi: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        prefix_bits: usize,
    ) -> Result<[Vec<F>; RAM_RA_CLAIM_TERMS], RamRaClaimError> {
        if eq_address.len() != RAM_RA_CLAIM_ADDRESS_DOMAIN {
            return Err(RamRaClaimError::TableLength {
                table: "eq_address",
                expected: RAM_RA_CLAIM_ADDRESS_DOMAIN,
                got: eq_address.len(),
            });
        }
        let prefix_length = checked_pow2("oracle prefix length", prefix_bits)?;
        if !addresses.len().is_multiple_of(prefix_length) {
            return Err(RamRaClaimError::RowsNotSplitCompatible {
                rows: addresses.len(),
                prefix_length,
            });
        }
        let suffix_length = addresses.len() / prefix_length;
        for table in eq_hi {
            if table.len() != suffix_length {
                return Err(RamRaClaimError::TableLength {
                    table: "eq_hi",
                    expected: suffix_length,
                    got: table.len(),
                });
            }
        }
        let mut q = core::array::from_fn(|_| vec![F::zero(); prefix_length]);
        for (j, &raw) in addresses.iter().enumerate() {
            let address = RamRaClaimAddress::try_from(raw)?;
            if !address.is_access() {
                continue;
            }
            let lo = j & (prefix_length - 1);
            let hi = j >> prefix_bits;
            let h = eq_address[address.raw() as usize];
            for x in 0..RAM_RA_CLAIM_TERMS {
                q[x][lo] += h * eq_hi[x][hi];
            }
        }
        Ok(q)
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RamRaClaimQPartials<F: Field> {
        pub values: [Vec<F>; RAM_RA_CLAIM_TERMS],
        pub partitions: usize,
        pub prefix_length: usize,
    }

    pub fn build_q_partials<F: Field>(
        addresses: &[u32],
        eq_address: &[F],
        eq_hi: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        prefix_bits: usize,
        partitions: usize,
    ) -> Result<RamRaClaimQPartials<F>, RamRaClaimError> {
        if partitions != RAM_RA_CLAIM_Q_PARTITIONS {
            return Err(RamRaClaimError::UnsupportedQPartitions { got: partitions });
        }
        let prefix_length = checked_pow2("oracle prefix length", prefix_bits)?;
        if !addresses.len().is_multiple_of(prefix_length) {
            return Err(RamRaClaimError::RowsNotSplitCompatible {
                rows: addresses.len(),
                prefix_length,
            });
        }
        let suffix_length = addresses.len() / prefix_length;
        if !suffix_length.is_multiple_of(partitions) {
            return Err(RamRaClaimError::QPartitionsDoNotDivideSuffix {
                partitions,
                suffix_length,
            });
        }
        if eq_address.len() != RAM_RA_CLAIM_ADDRESS_DOMAIN {
            return Err(RamRaClaimError::TableLength {
                table: "eq_address",
                expected: RAM_RA_CLAIM_ADDRESS_DOMAIN,
                got: eq_address.len(),
            });
        }
        for table in eq_hi {
            if table.len() != suffix_length {
                return Err(RamRaClaimError::TableLength {
                    table: "eq_hi",
                    expected: suffix_length,
                    got: table.len(),
                });
            }
        }

        let partial_length = checked_product("oracle Q partial length", partitions, prefix_length)?;
        let mut values = core::array::from_fn(|_| vec![F::zero(); partial_length]);
        let high_per_partition = suffix_length / partitions;
        for partition in 0..partitions {
            let high_start = partition * high_per_partition;
            let high_end = high_start + high_per_partition;
            for (high_offset, _) in eq_hi[0][high_start..high_end].iter().enumerate() {
                let high = high_start + high_offset;
                for low in 0..prefix_length {
                    let row = high * prefix_length + low;
                    let address = RamRaClaimAddress::try_from(addresses[row])?;
                    if !address.is_access() {
                        continue;
                    }
                    let h = eq_address[address.raw() as usize];
                    let output = partition * prefix_length + low;
                    for term in 0..RAM_RA_CLAIM_TERMS {
                        values[term][output] += h * eq_hi[term][high];
                    }
                }
            }
        }
        Ok(RamRaClaimQPartials {
            values,
            partitions,
            prefix_length,
        })
    }

    pub fn reduce_q_partials<F: Field>(
        partials: &RamRaClaimQPartials<F>,
    ) -> Result<[Vec<F>; RAM_RA_CLAIM_TERMS], RamRaClaimError> {
        if partials.partitions != RAM_RA_CLAIM_Q_PARTITIONS {
            return Err(RamRaClaimError::UnsupportedQPartitions {
                got: partials.partitions,
            });
        }
        let expected = checked_product(
            "oracle Q partial length",
            partials.partitions,
            partials.prefix_length,
        )?;
        for table in &partials.values {
            if table.len() != expected {
                return Err(RamRaClaimError::TableLength {
                    table: "Q partial",
                    expected,
                    got: table.len(),
                });
            }
        }
        Ok(core::array::from_fn(|term| {
            (0..partials.prefix_length)
                .map(|low| {
                    (0..partials.partitions).fold(F::zero(), |sum, partition| {
                        sum + partials.values[term][partition * partials.prefix_length + low]
                    })
                })
                .collect()
        }))
    }

    pub fn gather_h<F: Field>(
        addresses: &[u32],
        eq_address: &[F],
        eq_prefix: &[F],
        prefix_bits: usize,
        suffix_bits: usize,
    ) -> Result<Vec<F>, RamRaClaimError> {
        let prefix_length = checked_pow2("oracle prefix length", prefix_bits)?;
        let suffix_length = checked_pow2("oracle suffix length", suffix_bits)?;
        let expected_rows = checked_product("oracle split rows", prefix_length, suffix_length)?;
        if addresses.len() != expected_rows {
            return Err(RamRaClaimError::RowsNotSplitCompatible {
                rows: addresses.len(),
                prefix_length,
            });
        }
        if eq_address.len() != RAM_RA_CLAIM_ADDRESS_DOMAIN {
            return Err(RamRaClaimError::TableLength {
                table: "eq_address",
                expected: RAM_RA_CLAIM_ADDRESS_DOMAIN,
                got: eq_address.len(),
            });
        }
        if eq_prefix.len() != prefix_length {
            return Err(RamRaClaimError::TableLength {
                table: "eq_prefix",
                expected: prefix_length,
                got: eq_prefix.len(),
            });
        }
        let mut h = vec![F::zero(); suffix_length];
        for (j, &raw) in addresses.iter().enumerate() {
            let address = RamRaClaimAddress::try_from(raw)?;
            if !address.is_access() {
                continue;
            }
            let lo = j & (prefix_length - 1);
            let hi = j >> prefix_bits;
            h[hi] += eq_address[address.raw() as usize] * eq_prefix[lo];
        }
        Ok(h)
    }

    fn validate_oracle_inputs<F: Field>(
        inputs: &RamRaClaimOracleInputs<'_, F>,
        challenges: &[F],
    ) -> Result<RamRaClaimShape, RamRaClaimError> {
        if inputs.r_address.len() != RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize {
            return Err(RamRaClaimError::PointLength {
                point: "address point",
                expected: RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize,
                got: inputs.r_address.len(),
            });
        }
        let shape = RamRaClaimShape::new(inputs.addresses.len(), RAM_RA_CLAIM_ADDRESS_DOMAIN)?;
        if challenges.len() != shape.log_t() {
            return Err(RamRaClaimError::PointLength {
                point: "sumcheck challenges",
                expected: shape.log_t(),
                got: challenges.len(),
            });
        }
        for point in inputs.cycle_points {
            if point.len() != shape.log_t() {
                return Err(RamRaClaimError::PointLength {
                    point: "cycle point",
                    expected: shape.log_t(),
                    got: point.len(),
                });
            }
        }
        Ok(shape)
    }

    fn folded_addresses<F: Field>(
        addresses: &[u32],
        eq_address: &[F],
    ) -> Result<Vec<F>, RamRaClaimError> {
        addresses
            .iter()
            .map(|&raw| {
                let address = RamRaClaimAddress::try_from(raw)?;
                Ok(if address.is_access() {
                    eq_address[address.raw() as usize]
                } else {
                    F::zero()
                })
            })
            .collect()
    }

    fn dense_message<F: Field>(
        h: &[F],
        eq_cycle: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        gamma_powers: &[F; RAM_RA_CLAIM_TERMS],
    ) -> Result<[F; 2], RamRaClaimError> {
        validate_message_tables(h, eq_cycle)?;
        let mut evals = [F::zero(); 2];
        for y in 0..h.len() / 2 {
            let h_0 = h[2 * y];
            let h_1 = h[2 * y + 1];
            let h_2 = h_1 + h_1 - h_0;
            let mut e_0 = F::zero();
            let mut e_2 = F::zero();
            for x in 0..RAM_RA_CLAIM_TERMS {
                let x_0 = eq_cycle[x][2 * y];
                let x_1 = eq_cycle[x][2 * y + 1];
                e_0 += gamma_powers[x] * x_0;
                e_2 += gamma_powers[x] * (x_1 + x_1 - x_0);
            }
            evals[0] += h_0 * e_0;
            evals[1] += h_2 * e_2;
        }
        Ok(evals)
    }

    fn prefix_claim<F: Field>(
        p: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        q: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        gamma_powers: &[F; RAM_RA_CLAIM_TERMS],
    ) -> Result<F, RamRaClaimError> {
        validate_pq(p, q)?;
        Ok((0..RAM_RA_CLAIM_TERMS).fold(F::zero(), |claim, x| {
            claim
                + gamma_powers[x]
                    * p[x]
                        .iter()
                        .zip(&q[x])
                        .fold(F::zero(), |sum, (&p, &q)| sum + p * q)
        }))
    }

    fn prefix_message<F: Field>(
        p: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        q: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        gamma_powers: &[F; RAM_RA_CLAIM_TERMS],
    ) -> Result<[F; 2], RamRaClaimError> {
        validate_pq(p, q)?;
        let mut evals = [F::zero(); 2];
        for x in 0..RAM_RA_CLAIM_TERMS {
            for y in 0..p[x].len() / 2 {
                let p_0 = p[x][2 * y];
                let p_1 = p[x][2 * y + 1];
                let q_0 = q[x][2 * y];
                let q_1 = q[x][2 * y + 1];
                evals[0] += gamma_powers[x] * p_0 * q_0;
                evals[1] += gamma_powers[x] * (p_1 + p_1 - p_0) * (q_1 + q_1 - q_0);
            }
        }
        Ok(evals)
    }

    fn suffix_message<F: Field>(
        h: &[F],
        eq_hi: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        coefficients: &[F; RAM_RA_CLAIM_TERMS],
    ) -> Result<[F; 2], RamRaClaimError> {
        validate_message_tables(h, eq_hi)?;
        let mut evals = [F::zero(); 2];
        for y in 0..h.len() / 2 {
            let h_0 = h[2 * y];
            let h_1 = h[2 * y + 1];
            let mut e_0 = F::zero();
            let mut e_2 = F::zero();
            for x in 0..RAM_RA_CLAIM_TERMS {
                let x_0 = eq_hi[x][2 * y];
                let x_1 = eq_hi[x][2 * y + 1];
                e_0 += coefficients[x] * x_0;
                e_2 += coefficients[x] * (x_1 + x_1 - x_0);
            }
            evals[0] += h_0 * e_0;
            evals[1] += (h_1 + h_1 - h_0) * e_2;
        }
        Ok(evals)
    }

    fn validate_message_tables<F: Field>(
        h: &[F],
        tables: &[Vec<F>; RAM_RA_CLAIM_TERMS],
    ) -> Result<(), RamRaClaimError> {
        if h.len() < 2 || !h.len().is_power_of_two() {
            return Err(RamRaClaimError::OracleState { length: h.len() });
        }
        for table in tables {
            if table.len() != h.len() {
                return Err(RamRaClaimError::TableLength {
                    table: "message term",
                    expected: h.len(),
                    got: table.len(),
                });
            }
        }
        Ok(())
    }

    fn validate_pq<F: Field>(
        p: &[Vec<F>; RAM_RA_CLAIM_TERMS],
        q: &[Vec<F>; RAM_RA_CLAIM_TERMS],
    ) -> Result<(), RamRaClaimError> {
        let length = p[0].len();
        if length == 0 || !length.is_power_of_two() {
            return Err(RamRaClaimError::OracleState { length });
        }
        for table in p.iter().chain(q.iter()) {
            if table.len() != length {
                return Err(RamRaClaimError::TableLength {
                    table: "prefix term",
                    expected: length,
                    got: table.len(),
                });
            }
        }
        Ok(())
    }

    fn bind_pairs<F: Field>(table: &mut Vec<F>, challenge: F) -> Result<(), RamRaClaimError> {
        if table.len() < 2 || !table.len().is_power_of_two() {
            return Err(RamRaClaimError::OracleState {
                length: table.len(),
            });
        }
        let half = table.len() / 2;
        for y in 0..half {
            let even = table[2 * y];
            table[y] = even + challenge * (table[2 * y + 1] - even);
        }
        table.truncate(half);
        Ok(())
    }

    fn gamma_powers<F: Field>(gamma: F) -> [F; RAM_RA_CLAIM_TERMS] {
        [F::one(), gamma, gamma * gamma]
    }

    fn output_cycle_point<F: Field>(challenges: &[F]) -> Vec<F> {
        challenges.iter().rev().copied().collect()
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RamRaClaimError {
    #[error("{label} must be a nonzero power of two, got {got}")]
    NotPowerOfTwo { label: &'static str, got: usize },
    #[error("only the 8192-address production domain is admitted, got {got}")]
    UnsupportedAddressDomain { got: usize },
    #[error("RAM address {address} is outside the production domain")]
    AddressOutsideDomain { address: u32 },
    #[error("prefix length {prefix_length} cannot supply a full SIMDgroup")]
    PrefixTooSmall { prefix_length: usize },
    #[error("only one 32-thread SIMDgroup is admitted, got {got} threads")]
    UnsupportedThreads { got: usize },
    #[error("only eight disjoint Q partitions are admitted, got {got}")]
    UnsupportedQPartitions { got: usize },
    #[error("{partitions} Q partitions do not divide suffix length {suffix_length}")]
    QPartitionsDoNotDivideSuffix {
        partitions: usize,
        suffix_length: usize,
    },
    #[error("invalid trace cutoff {got}")]
    InvalidTraceCutoff { got: usize },
    #[error("accessed-row count {accessed_rows} exceeds row count {rows}")]
    AccessedRows { rows: usize, accessed_rows: usize },
    #[error("resident address byte length mismatch: expected {expected}, got {got}")]
    ResidentByteLength { expected: usize, got: usize },
    #[error("resident row count mismatch: expected {expected}, got {got}")]
    ResidentRows { expected: usize, got: usize },
    #[error("resident device mismatch: expected {expected}, got {got}")]
    ResidentDevice { expected: u64, got: u64 },
    #[error("resident storage mismatch: expected {expected}, got {got}")]
    ResidentStorage { expected: usize, got: usize },
    #[error("resident address plane has no storage identity")]
    MissingStorageIdentity,
    #[error("shader rejected {got} RAM RA claim-reduction dispatches")]
    UnsupportedDispatches { got: u32 },
    #[error("shader observed invalid RAM addresses: Q={q}, gather={gather}")]
    InvalidShaderRows { q: u32, gather: u32 },
    #[error("shader accessed-row audit mismatch: expected {expected}, got {got}")]
    AccessedRowAudit { expected: usize, got: usize },
    #[error("{label} overflows the host or shader count domain")]
    SizeOverflow { label: &'static str },
    #[error("{point} has {got} coordinates, expected {expected}")]
    PointLength {
        point: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{table} has {got} elements, expected {expected}")]
    TableLength {
        table: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{rows} rows are incompatible with prefix length {prefix_length}")]
    RowsNotSplitCompatible { rows: usize, prefix_length: usize },
    #[error("oracle table length {length} is not an active power-of-two round")]
    OracleState { length: usize },
    #[error("dense and split input claims differ")]
    OracleInputClaimDrift,
    #[error("dense oracle has {dense} rounds but split oracle has {split}")]
    OracleRoundCountDrift { dense: usize, split: usize },
    #[error("dense and split messages differ at round {round}")]
    OracleMessageDrift { round: usize },
    #[error("dense and split RAM RA outputs differ")]
    OracleOutputDrift,
    #[error("dense and split output points differ")]
    OracleOutputPointDrift,
    #[error("dense and split derived cycle equality differs for term {term}")]
    OracleDerivedDrift { term: usize },
}

fn validate_power_of_two(label: &'static str, got: usize) -> Result<(), RamRaClaimError> {
    if got == 0 || !got.is_power_of_two() {
        Err(RamRaClaimError::NotPowerOfTwo { label, got })
    } else {
        Ok(())
    }
}

fn validate_accessed_rows(rows: usize, accessed_rows: usize) -> Result<(), RamRaClaimError> {
    if accessed_rows > rows {
        Err(RamRaClaimError::AccessedRows {
            rows,
            accessed_rows,
        })
    } else {
        Ok(())
    }
}

fn checked_product(label: &'static str, lhs: usize, rhs: usize) -> Result<usize, RamRaClaimError> {
    lhs.checked_mul(rhs)
        .ok_or(RamRaClaimError::SizeOverflow { label })
}

fn checked_pow2(label: &'static str, bits: usize) -> Result<usize, RamRaClaimError> {
    let shift = u32::try_from(bits).map_err(|_| RamRaClaimError::SizeOverflow { label })?;
    1usize
        .checked_shl(shift)
        .ok_or(RamRaClaimError::SizeOverflow { label })
}

fn checked_sum(label: &'static str, values: &[usize]) -> Result<usize, RamRaClaimError> {
    values.iter().try_fold(0usize, |sum, value| {
        sum.checked_add(*value)
            .ok_or(RamRaClaimError::SizeOverflow { label })
    })
}

fn shader_count(label: &'static str, value: usize) -> Result<u32, RamRaClaimError> {
    u32::try_from(value).map_err(|_| RamRaClaimError::SizeOverflow { label })
}

fn rate_floor_ns(work: u64, rate: u64) -> Result<u64, RamRaClaimError> {
    let numerator = u128::from(work).checked_mul(NANOSECONDS_PER_SECOND).ok_or(
        RamRaClaimError::SizeOverflow {
            label: "rate numerator",
        },
    )?;
    let value = numerator.div_ceil(u128::from(rate));
    u64::try_from(value).map_err(|_| RamRaClaimError::SizeOverflow {
        label: "rate floor",
    })
}

fn eighty_percent_cap(floor_ns: u64) -> Result<u64, RamRaClaimError> {
    let value = u128::from(floor_ns)
        .checked_mul(5)
        .ok_or(RamRaClaimError::SizeOverflow {
            label: "80-percent cap",
        })?
        .div_ceil(4);
    u64::try_from(value).map_err(|_| RamRaClaimError::SizeOverflow {
        label: "80-percent cap",
    })
}
