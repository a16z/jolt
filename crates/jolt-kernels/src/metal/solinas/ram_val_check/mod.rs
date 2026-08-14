//! Native-row ABI, checked launch plan, and host oracle for RAM value-check.

use std::{
    mem::{align_of, size_of},
    slice,
    time::Duration,
};

use jolt_field::{AkitaField, Field};
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, ComputePipelineState,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const RAM_VAL_CHECK_MESSAGE_COLUMNS: usize = 3;
pub const RAM_VAL_CHECK_SIMD_WIDTH: usize = 32;
pub const RAM_VAL_CHECK_NO_ACCESS: u32 = u32::MAX;
pub const RAM_VAL_CHECK_DEFAULT_CPU_TAIL_ELEMENTS: usize = 1 << 16;
pub const RAM_VAL_CHECK_TARGET_CPU_NS: u64 = 234_656_875;
pub const RAM_VAL_CHECK_FIVE_X_GATE_NS: u64 = 46_931_375;

pub(crate) const FIRST_MESSAGE_PIPELINE: &str = "solinas_ram_val_check_first_message";
pub(crate) const NATIVE_TRANSITION_PIPELINE: &str = "solinas_ram_val_check_native_transition";
pub(crate) const DENSE_TRANSITION_PIPELINE: &str = "solinas_ram_val_check_dense_transition";
pub(crate) const REDUCTION_PIPELINE: &str = "solinas_ram_val_check_reduce3";

const FP128_BYTES: usize = 16;
const FLAG_INCREMENT_NONNEGATIVE: u32 = 1;
const VALID_FLAGS: u32 = FLAG_INCREMENT_NONNEGATIVE;

/// One cycle of the transcript-independent RAM value-check input.
///
/// Rust and Metal both use a 16-byte stride. The second word packs the address
/// in its low half and flags in its high half.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValCheckNativeRow {
    increment_magnitude: u64,
    address: u32,
    flags: u32,
}

#[derive(Clone)]
pub struct RamValCheckRows {
    buffer: Buffer,
    len: usize,
    address_domain: usize,
    device_registry_id: u64,
}

impl RamValCheckRows {
    copy_field_getters! { pub, {
        len: usize,
        address_domain: usize,
        device_registry_id: u64,
    } }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocation_identity(&self) -> usize {
        self.buffer.as_ptr() as usize
    }

    pub const fn resident_bytes(&self) -> usize {
        self.len * size_of::<RamValCheckNativeRow>()
    }

    fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

const _: [(); 16] = [(); size_of::<RamValCheckNativeRow>()];
const _: [(); 8] = [(); align_of::<RamValCheckNativeRow>()];

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RamValCheckRowError {
    #[error("RAM value-check increment {0} has magnitude larger than u64::MAX")]
    IncrementOutOfRange(i128),
    #[error("RAM value-check reserves u32::MAX as the no-access address")]
    ReservedNoAccessAddress,
    #[error("RAM value-check row has reserved flag bits {0:#x}")]
    ReservedFlags(u32),
    #[error("RAM value-check row encodes negative zero")]
    NegativeZero,
    #[error("RAM value-check row has a nonzero increment without a RAM access")]
    MissingAddressForIncrement,
    #[error("RAM value-check address {address} is outside the address domain of size {domain}")]
    AddressOutOfDomain { address: u32, domain: usize },
}

impl RamValCheckNativeRow {
    pub fn new(address: Option<u32>, increment: i128) -> Result<Self, RamValCheckRowError> {
        let address = match address {
            Some(RAM_VAL_CHECK_NO_ACCESS) => {
                return Err(RamValCheckRowError::ReservedNoAccessAddress)
            }
            Some(address) => address,
            None => RAM_VAL_CHECK_NO_ACCESS,
        };
        let magnitude = increment.unsigned_abs();
        if magnitude > u64::MAX as u128 {
            return Err(RamValCheckRowError::IncrementOutOfRange(increment));
        }
        let row = Self {
            increment_magnitude: magnitude as u64,
            address,
            flags: if increment >= 0 {
                FLAG_INCREMENT_NONNEGATIVE
            } else {
                0
            },
        };
        row.validate()?;
        Ok(row)
    }

    pub fn try_from_words(words: [u64; 2]) -> Result<Self, RamValCheckRowError> {
        let row = Self {
            increment_magnitude: words[0],
            address: words[1] as u32,
            flags: (words[1] >> 32) as u32,
        };
        row.validate()?;
        Ok(row)
    }

    pub const fn words(self) -> [u64; 2] {
        [
            self.increment_magnitude,
            self.address as u64 | ((self.flags as u64) << 32),
        ]
    }

    pub const fn address(self) -> Option<u32> {
        if self.address == RAM_VAL_CHECK_NO_ACCESS {
            None
        } else {
            Some(self.address)
        }
    }

    pub const fn increment(self) -> i128 {
        if self.increment_nonnegative() {
            self.increment_magnitude as i128
        } else {
            -(self.increment_magnitude as i128)
        }
    }

    copy_field_getters! { pub, { increment_magnitude: u64 } }

    pub const fn increment_nonnegative(self) -> bool {
        self.flags & FLAG_INCREMENT_NONNEGATIVE != 0
    }

    pub fn increment_field<F: Field>(self) -> F {
        F::from_i128(self.increment())
    }

    pub fn ram_ra<F: Field>(self, eq_address: &[F]) -> Result<F, RamValCheckRowError> {
        self.validate_address_domain(eq_address.len())?;
        Ok(match self.address() {
            Some(address) => eq_address[address as usize],
            None => F::zero(),
        })
    }

    pub fn validate(self) -> Result<(), RamValCheckRowError> {
        let reserved = self.flags & !VALID_FLAGS;
        if reserved != 0 {
            return Err(RamValCheckRowError::ReservedFlags(reserved));
        }
        if self.increment_magnitude == 0 && !self.increment_nonnegative() {
            return Err(RamValCheckRowError::NegativeZero);
        }
        if self.increment_magnitude != 0 && self.address().is_none() {
            return Err(RamValCheckRowError::MissingAddressForIncrement);
        }
        Ok(())
    }

    pub fn validate_address_domain(self, domain: usize) -> Result<(), RamValCheckRowError> {
        self.validate()?;
        if let Some(address) = self.address() {
            if address as usize >= domain {
                return Err(RamValCheckRowError::AddressOutOfDomain { address, domain });
            }
        }
        Ok(())
    }
}

impl Default for RamValCheckNativeRow {
    fn default() -> Self {
        Self {
            increment_magnitude: 0,
            address: RAM_VAL_CHECK_NO_ACCESS,
            flags: FLAG_INCREMENT_NONNEGATIVE,
        }
    }
}

/// The two fully materialized tables retained after the native bind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamValCheckDenseRow<F> {
    pub increment: F,
    pub ram_ra: F,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValCheckConfig {
    pub first_message_threads: usize,
    pub native_transition_threads: usize,
    pub dense_transition_threads: usize,
    pub cpu_tail_elements: usize,
}

impl Default for RamValCheckConfig {
    fn default() -> Self {
        Self {
            first_message_threads: 32,
            native_transition_threads: 32,
            dense_transition_threads: 32,
            cpu_tail_elements: RAM_VAL_CHECK_DEFAULT_CPU_TAIL_ELEMENTS,
        }
    }
}

impl RamValCheckConfig {
    pub fn validate(self) -> Result<Self, RamValCheckShapeError> {
        for (phase, width) in [
            ("first message", self.first_message_threads),
            ("native transition", self.native_transition_threads),
            ("dense transition", self.dense_transition_threads),
        ] {
            if width == 0 || !width.is_multiple_of(RAM_VAL_CHECK_SIMD_WIDTH) {
                return Err(RamValCheckShapeError::InvalidThreadgroupWidth { phase, width });
            }
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RamValCheckShapeError {
    #[error("RAM value-check needs a power-of-two cycle count of at least four, got {0}")]
    InvalidCycles(usize),
    #[error("RAM value-check address domain must be a nonzero power of two below 2^32, got {0}")]
    InvalidAddressDomain(usize),
    #[error("RAM value-check plan needs {expected} cycles, but resident rows have {got}")]
    PlanCycleMismatch { expected: usize, got: usize },
    #[error("RAM value-check plan needs address domain {expected}, but resident rows use {got}")]
    PlanAddressDomainMismatch { expected: usize, got: usize },
    #[error(
        "RAM value-check CPU cutoff must be a power of two smaller than {cycles}, got {cutoff}"
    )]
    InvalidCpuCutoff { cycles: usize, cutoff: usize },
    #[error(
        "RAM value-check GPU prefix binds {gpu_binds} variables but the split LT low half has only {low_bits} variables"
    )]
    GpuPrefixCrossesLtSplit { gpu_binds: usize, low_bits: usize },
    #[error(
        "RAM value-check {phase} threadgroup width must be a nonzero multiple of 32, got {width}"
    )]
    InvalidThreadgroupWidth { phase: &'static str, width: usize },
    #[error("RAM value-check dense transition index {index} is outside 0..{count}")]
    InvalidDenseTransition { index: usize, count: usize },
    #[error(
        "RAM value-check factorization has {elements} elements, high={high_blocks}, low={lt_lo_length}"
    )]
    FactorizationShape {
        elements: usize,
        high_blocks: usize,
        lt_lo_length: usize,
    },
    #[error("RAM value-check {name} table has length {got}, expected {expected}")]
    StorageLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("RAM value-check {name} must have a nonzero power-of-two length, got {length}")]
    InvalidTableLength { name: &'static str, length: usize },
    #[error("RAM value-check row {index} is invalid: {source}")]
    InvalidRow {
        index: usize,
        #[source]
        source: RamValCheckRowError,
    },
    #[error("RAM value-check reduction supports exactly three columns, got {0}")]
    InvalidReductionColumns(usize),
    #[error("RAM value-check reduction needs at least one input")]
    EmptyReduction,
    #[error("RAM value-check {name} element count exceeds its 32-bit shader index")]
    ShaderIndexOverflow { name: &'static str },
    #[error("RAM value-check {name} byte length overflows host indexing")]
    ByteLengthOverflow { name: &'static str },
}

/// The immutable round schedule for one resident GPU prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValCheckPlan {
    log_t: usize,
    log_k: usize,
    cycles: usize,
    address_domain: usize,
    low_bits: usize,
    high_bits: usize,
    lt_lo_length: usize,
    high_blocks: usize,
    gpu_bind_rounds: usize,
    dense_transition_rounds: usize,
    cpu_tail_elements: usize,
    config: RamValCheckConfig,
}

impl RamValCheckPlan {
    pub fn new(
        log_t: usize,
        log_k: usize,
        config: RamValCheckConfig,
    ) -> Result<Self, RamValCheckShapeError> {
        let config = config.validate()?;
        let cycles = checked_power_of_two("cycle domain", log_t)?;
        if cycles < 4 {
            return Err(RamValCheckShapeError::InvalidCycles(cycles));
        }
        let address_domain = checked_power_of_two("address domain", log_k)?;
        if address_domain > u32::MAX as usize {
            return Err(RamValCheckShapeError::InvalidAddressDomain(address_domain));
        }
        let _ = shader_count("cycle domain", cycles)?;
        let _ = shader_count("address domain", address_domain)?;

        let cpu_tail_elements = config.cpu_tail_elements;
        if cpu_tail_elements == 0
            || !cpu_tail_elements.is_power_of_two()
            || cpu_tail_elements >= cycles
        {
            return Err(RamValCheckShapeError::InvalidCpuCutoff {
                cycles,
                cutoff: cpu_tail_elements,
            });
        }
        let gpu_bind_rounds = log_t - cpu_tail_elements.ilog2() as usize;
        let low_bits = log_t / 2;
        let high_bits = log_t - low_bits;
        if gpu_bind_rounds >= low_bits {
            return Err(RamValCheckShapeError::GpuPrefixCrossesLtSplit {
                gpu_binds: gpu_bind_rounds,
                low_bits,
            });
        }
        let lt_lo_length = checked_power_of_two("LT low table", low_bits)?;
        let high_blocks = checked_power_of_two("LT high table", high_bits)?;
        let _ = RamValCheckMessageParams::new(cycles, high_blocks, lt_lo_length)?;

        Ok(Self {
            log_t,
            log_k,
            cycles,
            address_domain,
            low_bits,
            high_bits,
            lt_lo_length,
            high_blocks,
            gpu_bind_rounds,
            dense_transition_rounds: gpu_bind_rounds - 1,
            cpu_tail_elements,
            config,
        })
    }

    copy_field_getters! { pub, {
        log_t: usize,
        log_k: usize,
        cycles: usize,
        address_domain: usize,
        low_bits: usize,
        high_bits: usize,
        initial_lt_lo_length => lt_lo_length: usize,
        high_blocks: usize,
        gpu_bind_rounds: usize,
        dense_transition_rounds: usize,
        cpu_tail_elements: usize,
        config: RamValCheckConfig,
    } }

    pub const fn gpu_message_rounds(self) -> usize {
        self.gpu_bind_rounds + 1
    }

    pub const fn lt_lo_length_at_handoff(self) -> usize {
        self.lt_lo_length >> self.gpu_bind_rounds
    }

    pub(crate) fn first_message_params(
        self,
    ) -> Result<RamValCheckMessageParams, RamValCheckShapeError> {
        RamValCheckMessageParams::new(self.cycles, self.high_blocks, self.lt_lo_length)
    }

    pub(crate) fn native_transition_params(
        self,
    ) -> Result<RamValCheckMessageParams, RamValCheckShapeError> {
        RamValCheckMessageParams::new(self.cycles / 2, self.high_blocks, self.lt_lo_length / 2)
    }

    pub(crate) fn dense_transition_params(
        self,
        index: usize,
    ) -> Result<RamValCheckMessageParams, RamValCheckShapeError> {
        if index >= self.dense_transition_rounds {
            return Err(RamValCheckShapeError::InvalidDenseTransition {
                index,
                count: self.dense_transition_rounds,
            });
        }
        let bound_rounds = index + 2;
        RamValCheckMessageParams::new(
            self.cycles >> bound_rounds,
            self.high_blocks,
            self.lt_lo_length >> bound_rounds,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RamValCheckMessageParams {
    pub(crate) message_elements: u32,
    pub(crate) high_blocks: u32,
    pub(crate) lt_lo_length: u32,
    pub(crate) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RamValCheckReductionParams {
    pub(crate) input_count: u32,
    pub(crate) output_count: u32,
    pub(crate) columns: u32,
    pub(crate) _reserved: u32,
}

const _: [(); 16] = [(); size_of::<RamValCheckMessageParams>()];
const _: [(); 16] = [(); size_of::<RamValCheckReductionParams>()];

impl RamValCheckMessageParams {
    pub(crate) fn new(
        message_elements: usize,
        high_blocks: usize,
        lt_lo_length: usize,
    ) -> Result<Self, RamValCheckShapeError> {
        if message_elements < 2
            || !message_elements.is_power_of_two()
            || high_blocks == 0
            || !high_blocks.is_power_of_two()
            || lt_lo_length < 2
            || !lt_lo_length.is_power_of_two()
            || high_blocks.checked_mul(lt_lo_length) != Some(message_elements)
        {
            return Err(RamValCheckShapeError::FactorizationShape {
                elements: message_elements,
                high_blocks,
                lt_lo_length,
            });
        }
        let partial_fields = checked_product(
            "message partials",
            RAM_VAL_CHECK_MESSAGE_COLUMNS,
            high_blocks,
        )?;
        let _ = shader_count("message partials", partial_fields)?;
        Ok(Self {
            message_elements: shader_count("message elements", message_elements)?,
            high_blocks: shader_count("high blocks", high_blocks)?,
            lt_lo_length: shader_count("LT low table", lt_lo_length)?,
            _reserved: 0,
        })
    }
}

impl RamValCheckReductionParams {
    pub(crate) fn new(input_count: usize, columns: usize) -> Result<Self, RamValCheckShapeError> {
        if columns != RAM_VAL_CHECK_MESSAGE_COLUMNS {
            return Err(RamValCheckShapeError::InvalidReductionColumns(columns));
        }
        if input_count == 0 {
            return Err(RamValCheckShapeError::EmptyReduction);
        }
        let output_count = input_count.div_ceil(RAM_VAL_CHECK_SIMD_WIDTH);
        for count in [input_count, output_count] {
            let fields = checked_product("reduction buffer", columns, count)?;
            let _ = shader_count("reduction buffer", fields)?;
        }
        Ok(Self {
            input_count: shader_count("reduction input", input_count)?,
            output_count: shader_count("reduction output", output_count)?,
            columns: shader_count("reduction columns", columns)?,
            _reserved: 0,
        })
    }
}

/// Exact capacities for the two arenas and all factorized message tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamValCheckStorageLayout {
    native_row_bytes: usize,
    dense_a_fields: usize,
    dense_b_fields: usize,
    address_fields: usize,
    split_lt_fields: usize,
    partial_fields_per_buffer: usize,
    workspace_bytes: usize,
    resident_bytes: usize,
    tail_handoff_bytes: usize,
}

impl RamValCheckStorageLayout {
    pub fn new(plan: RamValCheckPlan) -> Result<Self, RamValCheckShapeError> {
        let native_row_bytes = checked_product(
            "native rows",
            plan.cycles,
            size_of::<RamValCheckNativeRow>(),
        )?;
        let dense_a_fields = plan.cycles;
        let dense_b_fields = plan.cycles / 2;
        let address_fields = plan.address_domain;
        let high_table_fields = checked_product("split LT tables", 2, plan.high_blocks)?;
        let split_lt_fields = plan.lt_lo_length.checked_add(high_table_fields).ok_or(
            RamValCheckShapeError::ByteLengthOverflow {
                name: "split LT tables",
            },
        )?;
        let partial_fields_per_buffer = checked_product(
            "partial buffer",
            RAM_VAL_CHECK_MESSAGE_COLUMNS,
            plan.high_blocks,
        )?;
        for (name, fields) in [
            ("dense arena A", dense_a_fields),
            ("dense arena B", dense_b_fields),
            ("address equality table", address_fields),
            ("split LT tables", split_lt_fields),
            ("partial buffer", partial_fields_per_buffer),
        ] {
            let _ = shader_count(name, fields)?;
        }
        let workspace_fields = [
            dense_a_fields,
            dense_b_fields,
            address_fields,
            split_lt_fields,
            partial_fields_per_buffer,
            partial_fields_per_buffer,
        ]
        .into_iter()
        .try_fold(0usize, |sum, fields| sum.checked_add(fields))
        .ok_or(RamValCheckShapeError::ByteLengthOverflow { name: "workspace" })?;
        let workspace_bytes = checked_product("workspace", workspace_fields, FP128_BYTES)?;
        let resident_bytes = native_row_bytes.checked_add(workspace_bytes).ok_or(
            RamValCheckShapeError::ByteLengthOverflow {
                name: "resident set",
            },
        )?;
        let tail_handoff_bytes =
            checked_product("CPU tail handoff", plan.cpu_tail_elements, 2 * FP128_BYTES)?;

        Ok(Self {
            native_row_bytes,
            dense_a_fields,
            dense_b_fields,
            address_fields,
            split_lt_fields,
            partial_fields_per_buffer,
            workspace_bytes,
            resident_bytes,
            tail_handoff_bytes,
        })
    }

    copy_field_getters! { pub, {
        native_row_bytes: usize,
        dense_a_fields: usize,
        dense_b_fields: usize,
        address_fields: usize,
        split_lt_fields: usize,
        partial_fields_per_buffer: usize,
        workspace_bytes: usize,
        resident_bytes: usize,
        tail_handoff_bytes: usize,
    } }
}

struct RamValCheckBuffers {
    eq_address: Buffer,
    lt_lo: Buffer,
    lt_hi: Buffer,
    eq_hi: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct RamValCheckSequence {
    context: SolinasMetal,
    first_message_pipeline: ComputePipelineState,
    native_transition_pipeline: ComputePipelineState,
    dense_transition_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    first_message_limits: PipelineLimits,
    native_transition_limits: PipelineLimits,
    dense_transition_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    rows: RamValCheckRows,
    buffers: RamValCheckBuffers,
    plan: RamValCheckPlan,
    layout: RamValCheckStorageLayout,
    first_message_done: bool,
    gpu_binds: usize,
    source_in_a: bool,
    gpu_active_time: Duration,
}

impl SolinasMetal {
    pub fn prepare_ram_val_check_rows(
        &self,
        rows: &[RamValCheckNativeRow],
        address_domain: usize,
    ) -> Result<RamValCheckRows, MetalError> {
        if rows.len() < 4 || !rows.len().is_power_of_two() {
            return Err(RamValCheckShapeError::InvalidCycles(rows.len()).into());
        }
        if address_domain == 0
            || !address_domain.is_power_of_two()
            || address_domain > u32::MAX as usize
        {
            return Err(RamValCheckShapeError::InvalidAddressDomain(address_domain).into());
        }
        let _ = shader_count("cycle domain", rows.len())?;
        for (index, row) in rows.iter().copied().enumerate() {
            row.validate_address_domain(address_domain)
                .map_err(|source| RamValCheckShapeError::InvalidRow { index, source })?;
        }
        let bytes = checked_product("native rows", rows.len(), size_of::<RamValCheckNativeRow>())?;
        let bytes = u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(rows.len()))?;
        self.validate_buffer_length(bytes)?;
        self.validate_additional_working_set(bytes)?;
        Ok(RamValCheckRows {
            buffer: buffer_from_slice(&self.device, rows),
            len: rows.len(),
            address_domain,
            device_registry_id: self.device_registry_id(),
        })
    }

    pub fn prepare_ram_val_check_sequence(
        &self,
        rows: RamValCheckRows,
        eq_address: &[AkitaField],
        lt_lo: &[AkitaField],
        lt_hi: &[AkitaField],
        eq_hi: &[AkitaField],
        plan: RamValCheckPlan,
    ) -> Result<RamValCheckSequence, MetalError> {
        if rows.device_registry_id() != self.device_registry_id() {
            return Err(MetalError::RamValCheckRowsDevice {
                expected: self.device_registry_id(),
                got: rows.device_registry_id(),
            });
        }
        if rows.len() != plan.cycles() {
            return Err(RamValCheckShapeError::PlanCycleMismatch {
                expected: plan.cycles(),
                got: rows.len(),
            }
            .into());
        }
        if rows.address_domain() != plan.address_domain() {
            return Err(RamValCheckShapeError::PlanAddressDomainMismatch {
                expected: plan.address_domain(),
                got: rows.address_domain(),
            }
            .into());
        }
        for (name, expected, got) in [
            ("address equality", plan.address_domain(), eq_address.len()),
            ("LT low", plan.initial_lt_lo_length(), lt_lo.len()),
            ("LT high", plan.high_blocks(), lt_hi.len()),
            ("EQ high", plan.high_blocks(), eq_hi.len()),
        ] {
            if got != expected {
                return Err(RamValCheckShapeError::StorageLength {
                    name,
                    expected,
                    got,
                }
                .into());
            }
        }

        let layout = RamValCheckStorageLayout::new(plan)?;
        let workspace_bytes = u64::try_from(layout.workspace_bytes())
            .map_err(|_| MetalError::InputTooLong(layout.workspace_bytes()))?;
        self.validate_additional_working_set(workspace_bytes)?;

        let first_message_pipeline = self.compile_named_pipeline(FIRST_MESSAGE_PIPELINE)?;
        let native_transition_pipeline = self.compile_named_pipeline(NATIVE_TRANSITION_PIPELINE)?;
        let dense_transition_pipeline = self.compile_named_pipeline(DENSE_TRANSITION_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let first_message_limits = Self::limits(&first_message_pipeline);
        let native_transition_limits = Self::limits(&native_transition_pipeline);
        let dense_transition_limits = Self::limits(&dense_transition_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (FIRST_MESSAGE_PIPELINE, first_message_limits),
            (NATIVE_TRANSITION_PIPELINE, native_transition_limits),
            (DENSE_TRANSITION_PIPELINE, dense_transition_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != RAM_VAL_CHECK_SIMD_WIDTH {
                return Err(MetalError::UnsupportedRamValCheckExecutionWidth {
                    pipeline,
                    expected: RAM_VAL_CHECK_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let config = plan.config();
        let first_message_threads = Self::resolve_threadgroup_width(
            Some(config.first_message_threads),
            first_message_limits,
        )?;
        let native_transition_threads = Self::resolve_threadgroup_width(
            Some(config.native_transition_threads),
            native_transition_limits,
        )?;
        let dense_transition_threads = Self::resolve_threadgroup_width(
            Some(config.dense_transition_threads),
            dense_transition_limits,
        )?;
        for (phase, threads) in [
            ("first message", first_message_threads),
            ("native transition", native_transition_threads),
            ("dense transition", dense_transition_threads),
        ] {
            let requested = ram_val_check_threadgroup_bytes(threads)?;
            let maximum = self.device.max_threadgroup_memory_length();
            if requested > maximum {
                return Err(MetalError::RamValCheckThreadgroupMemory {
                    phase,
                    requested,
                    maximum,
                });
            }
        }

        let eq_address = encode_fields(self, "RAM value-check address equality", eq_address)?;
        let lt_lo = encode_fields(self, "RAM value-check LT low", lt_lo)?;
        let lt_hi = encode_fields(self, "RAM value-check LT high", lt_hi)?;
        let eq_hi = encode_fields(self, "RAM value-check EQ high", eq_hi)?;

        Ok(RamValCheckSequence {
            context: self.clone(),
            first_message_pipeline,
            native_transition_pipeline,
            dense_transition_pipeline,
            reduction_pipeline,
            first_message_limits,
            native_transition_limits,
            dense_transition_limits,
            reduction_limits,
            rows,
            buffers: RamValCheckBuffers {
                eq_address: buffer_from_slice(&self.device, &eq_address),
                lt_lo: buffer_from_slice(&self.device, &lt_lo),
                lt_hi: buffer_from_slice(&self.device, &lt_hi),
                eq_hi: buffer_from_slice(&self.device, &eq_hi),
                dense_a: self.new_ram_val_check_field_buffer(layout.dense_a_fields())?,
                dense_b: self.new_ram_val_check_field_buffer(layout.dense_b_fields())?,
                partial_a: self
                    .new_ram_val_check_field_buffer(layout.partial_fields_per_buffer())?,
                partial_b: self
                    .new_ram_val_check_field_buffer(layout.partial_fields_per_buffer())?,
            },
            plan,
            layout,
            first_message_done: false,
            gpu_binds: 0,
            source_in_a: true,
            gpu_active_time: Duration::ZERO,
        })
    }

    fn new_ram_val_check_field_buffer(&self, fields: usize) -> Result<Buffer, MetalError> {
        let bytes = field_bytes(fields)?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl RamValCheckSequence {
    pub fn message(&mut self) -> Result<[AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], MetalError> {
        self.message_timed().map(|(message, _)| message)
    }

    pub fn message_timed(
        &mut self,
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        if self.first_message_done || self.gpu_binds != 0 {
            return Err(MetalError::InvalidRamValCheckState(
                "first message already consumed",
            ));
        }
        let result = self.execute_first_message()?;
        self.first_message_done = true;
        self.gpu_active_time += result.1;
        Ok(result)
    }

    #[doc(hidden)]
    pub fn replay_first_message_timed(
        &self,
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        self.execute_first_message()
    }

    #[doc(hidden)]
    pub fn restart_message_timed(
        &mut self,
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        self.first_message_done = false;
        self.gpu_binds = 0;
        self.source_in_a = true;
        self.gpu_active_time = Duration::ZERO;
        self.message_timed()
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        bound_lt_lo: &[AkitaField],
    ) -> Result<[AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], MetalError> {
        self.bind_and_message_timed(challenge, bound_lt_lo)
            .map(|(message, _)| message)
    }

    pub fn bind_and_message_timed(
        &mut self,
        challenge: AkitaField,
        bound_lt_lo: &[AkitaField],
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        let result = self.execute_current_bind_and_message(challenge, bound_lt_lo)?;
        self.gpu_binds += 1;
        if self.gpu_binds > 1 {
            self.source_in_a = !self.source_in_a;
        }
        self.gpu_active_time += result.1;
        Ok(result)
    }

    #[doc(hidden)]
    pub fn replay_current_bind_and_message_timed(
        &self,
        challenge: AkitaField,
        bound_lt_lo: &[AkitaField],
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        self.execute_current_bind_and_message(challenge, bound_lt_lo)
    }

    fn execute_first_message(
        &self,
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        let params = self.plan.first_message_params()?;
        let threads = self.plan.config().first_message_threads;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.first_message_pipeline);
            encoder.set_buffer(0, Some(self.rows.buffer()), 0);
            encoder.set_buffer(1, Some(&self.buffers.eq_address), 0);
            encoder.set_buffer(2, Some(&self.buffers.lt_lo), 0);
            encoder.set_buffer(3, Some(&self.buffers.lt_hi), 0);
            encoder.set_buffer(4, Some(&self.buffers.eq_hi), 0);
            encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 6, &params);
            encoder.set_threadgroup_memory_length(0, ram_val_check_threadgroup_bytes(threads)?);
            dispatch_ram_val_check(encoder, self.plan.high_blocks(), threads);
            let final_in_a = encode_ram_val_check_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                self.plan.high_blocks(),
            )?;
            encoder.end_encoding();
            finish_ram_val_check_command(
                &self.context,
                command_buffer,
                self.final_partial_buffer(final_in_a),
            )
        })
    }

    fn execute_current_bind_and_message(
        &self,
        challenge: AkitaField,
        bound_lt_lo: &[AkitaField],
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        if !self.first_message_done {
            return Err(MetalError::InvalidRamValCheckState(
                "first message must precede binding",
            ));
        }
        if self.gpu_binds >= self.plan.gpu_bind_rounds() {
            return Err(MetalError::InvalidRamValCheckState(
                "GPU prefix already reached the CPU handoff",
            ));
        }
        let expected_lt_lo = self.plan.initial_lt_lo_length() >> (self.gpu_binds + 1);
        if bound_lt_lo.len() != expected_lt_lo {
            return Err(RamValCheckShapeError::StorageLength {
                name: "bound LT low",
                expected: expected_lt_lo,
                got: bound_lt_lo.len(),
            }
            .into());
        }
        write_ram_val_check_fields(
            &self.buffers.lt_lo,
            self.plan.initial_lt_lo_length(),
            bound_lt_lo,
        )?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("RAM value-check challenge", &[challenge])?;
        if self.gpu_binds == 0 {
            self.execute_native_transition(challenge)
        } else {
            self.execute_dense_transition(challenge, self.gpu_binds - 1)
        }
    }

    fn execute_native_transition(
        &self,
        challenge: Fp128,
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        let params = self.plan.native_transition_params()?;
        let threads = self.plan.config().native_transition_threads;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.native_transition_pipeline);
            encoder.set_buffer(0, Some(self.rows.buffer()), 0);
            encoder.set_buffer(1, Some(&self.buffers.eq_address), 0);
            encoder.set_buffer(2, Some(&self.buffers.lt_lo), 0);
            encoder.set_buffer(3, Some(&self.buffers.lt_hi), 0);
            encoder.set_buffer(4, Some(&self.buffers.eq_hi), 0);
            encoder.set_buffer(5, Some(&self.buffers.dense_a), 0);
            encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 7, &challenge);
            set_inline_bytes(encoder, 8, &params);
            encoder.set_threadgroup_memory_length(0, ram_val_check_threadgroup_bytes(threads)?);
            dispatch_ram_val_check(encoder, self.plan.high_blocks(), threads);
            let final_in_a = encode_ram_val_check_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                self.plan.high_blocks(),
            )?;
            encoder.end_encoding();
            finish_ram_val_check_command(
                &self.context,
                command_buffer,
                self.final_partial_buffer(final_in_a),
            )
        })
    }

    fn execute_dense_transition(
        &self,
        challenge: Fp128,
        transition_index: usize,
    ) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
        let params = self.plan.dense_transition_params(transition_index)?;
        let threads = self.plan.config().dense_transition_threads;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.dense_transition_pipeline);
            encoder.set_buffer(0, Some(self.source_dense_buffer()), 0);
            encoder.set_buffer(1, Some(self.destination_dense_buffer()), 0);
            encoder.set_buffer(2, Some(&self.buffers.lt_lo), 0);
            encoder.set_buffer(3, Some(&self.buffers.lt_hi), 0);
            encoder.set_buffer(4, Some(&self.buffers.eq_hi), 0);
            encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 6, &challenge);
            set_inline_bytes(encoder, 7, &params);
            encoder.set_threadgroup_memory_length(0, ram_val_check_threadgroup_bytes(threads)?);
            dispatch_ram_val_check(encoder, self.plan.high_blocks(), threads);
            let final_in_a = encode_ram_val_check_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                self.plan.high_blocks(),
            )?;
            encoder.end_encoding();
            finish_ram_val_check_command(
                &self.context,
                command_buffer,
                self.final_partial_buffer(final_in_a),
            )
        })
    }

    pub const fn current_elements(&self) -> usize {
        self.plan.cycles() >> self.gpu_binds
    }

    pub const fn current_lt_lo_length(&self) -> usize {
        self.plan.initial_lt_lo_length() >> self.gpu_binds
    }

    copy_field_getters! { pub, {
        gpu_binds: usize,
        storage_layout => layout: RamValCheckStorageLayout,
        plan: RamValCheckPlan,
        gpu_active_time: Duration,
        first_message_pipeline_limits => first_message_limits: PipelineLimits,
        native_transition_pipeline_limits => native_transition_limits: PipelineLimits,
        dense_transition_pipeline_limits => dense_transition_limits: PipelineLimits,
        reduction_pipeline_limits => reduction_limits: PipelineLimits,
    } }

    pub const fn at_cpu_handoff(&self) -> bool {
        self.gpu_binds == self.plan.gpu_bind_rounds()
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn resident_buffer_count(&self) -> usize {
        9
    }

    pub fn row_allocation_identity(&self) -> usize {
        self.rows.allocation_identity()
    }

    pub fn read_current_state(&self) -> Result<Vec<RamValCheckDenseRow<AkitaField>>, MetalError> {
        let mut output = vec![RamValCheckDenseRow::default(); self.current_elements()];
        self.read_current_state_into(&mut output)?;
        Ok(output)
    }

    pub fn read_current_state_into(
        &self,
        output: &mut [RamValCheckDenseRow<AkitaField>],
    ) -> Result<(), MetalError> {
        if self.gpu_binds == 0 {
            return Err(MetalError::InvalidRamValCheckState(
                "native rows have not been bound into dense state",
            ));
        }
        if output.len() != self.current_elements() {
            return Err(RamValCheckShapeError::StorageLength {
                name: "CPU tail output",
                expected: self.current_elements(),
                got: output.len(),
            }
            .into());
        }
        let fields = 2 * self.current_elements();
        // SAFETY: the selected shared buffer contains two fields for every
        // current row and its producing command completed before state moved.
        let values = unsafe {
            slice::from_raw_parts(
                self.source_dense_buffer().contents().cast::<Fp128>(),
                fields,
            )
        };
        self.context
            .validate_inputs("RAM value-check dense state", values)?;
        for (output, row) in output.iter_mut().zip(values.chunks_exact(2)) {
            *output = RamValCheckDenseRow {
                increment: row[0].into_jolt_field(),
                ram_ra: row[1].into_jolt_field(),
            };
        }
        Ok(())
    }

    fn source_dense_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.dense_a
        } else {
            &self.buffers.dense_b
        }
    }

    fn destination_dense_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.dense_b
        } else {
            &self.buffers.dense_a
        }
    }

    fn final_partial_buffer(&self, final_in_a: bool) -> &Buffer {
        if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        }
    }
}

fn dispatch_ram_val_check(
    encoder: &metal::ComputeCommandEncoderRef,
    high_blocks: usize,
    threads: usize,
) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: high_blocks as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads as u64,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_ram_val_check_reductions(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    partial_a: &Buffer,
    partial_b: &Buffer,
    mut input_count: usize,
) -> Result<bool, MetalError> {
    let mut input_a = true;
    while input_count > 1 {
        let params = RamValCheckReductionParams::new(input_count, RAM_VAL_CHECK_MESSAGE_COLUMNS)?;
        let output_count = params.output_count as usize;
        encoder.set_compute_pipeline_state(pipeline);
        let (input, output) = if input_a {
            (partial_a, partial_b)
        } else {
            (partial_b, partial_a)
        };
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        set_inline_bytes(encoder, 2, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: output_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: RAM_VAL_CHECK_SIMD_WIDTH as u64,
                height: 1,
                depth: 1,
            },
        );
        input_count = output_count;
        input_a = !input_a;
    }
    Ok(input_a)
}

fn finish_ram_val_check_command(
    context: &SolinasMetal,
    command_buffer: &metal::CommandBufferRef,
    output: &Buffer,
) -> Result<([AkitaField; RAM_VAL_CHECK_MESSAGE_COLUMNS], Duration), MetalError> {
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let status = command_buffer.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(MetalError::CommandFailed(status));
    }
    let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
    let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return Err(MetalError::InvalidGpuTimestamps { start, end });
    }
    // SAFETY: the completed reduction leaves three fields at the beginning of
    // the selected shared partial buffer.
    let values = unsafe {
        slice::from_raw_parts(
            output.contents().cast::<Fp128>(),
            RAM_VAL_CHECK_MESSAGE_COLUMNS,
        )
    };
    context.validate_inputs("RAM value-check message", values)?;
    Ok((
        std::array::from_fn(|index| values[index].into_jolt_field()),
        Duration::from_secs_f64(end - start),
    ))
}

fn ram_val_check_threadgroup_bytes(threads: usize) -> Result<u64, MetalError> {
    let simdgroups = threads / RAM_VAL_CHECK_SIMD_WIDTH;
    let bytes = checked_product(
        "threadgroup scratch",
        2 * RAM_VAL_CHECK_MESSAGE_COLUMNS * simdgroups,
        size_of::<Fp128>(),
    )?;
    u64::try_from(bytes).map_err(|_| MetalError::InputTooLong(bytes))
}

fn encode_fields(
    context: &SolinasMetal,
    name: &'static str,
    values: &[AkitaField],
) -> Result<Vec<Fp128>, MetalError> {
    let values = values
        .iter()
        .map(Fp128::from_jolt_field)
        .collect::<Vec<_>>();
    context.validate_inputs(name, &values)?;
    Ok(values)
}

fn write_ram_val_check_fields(
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(RamValCheckShapeError::StorageLength {
            name: "bound LT low",
            expected: capacity,
            got: values.len(),
        }
        .into());
    }
    // SAFETY: the buffer has `capacity` shared fields and the sequence waits
    // for every command before the host updates this prefix.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn field_bytes(fields: usize) -> Result<u64, MetalError> {
    fields
        .checked_mul(size_of::<Fp128>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(fields))
}

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
}

fn checked_power_of_two(
    name: &'static str,
    exponent: usize,
) -> Result<usize, RamValCheckShapeError> {
    let exponent =
        u32::try_from(exponent).map_err(|_| RamValCheckShapeError::ShaderIndexOverflow { name })?;
    1usize
        .checked_shl(exponent)
        .ok_or(RamValCheckShapeError::ShaderIndexOverflow { name })
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, RamValCheckShapeError> {
    u32::try_from(value).map_err(|_| RamValCheckShapeError::ShaderIndexOverflow { name })
}

fn checked_product(
    name: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize, RamValCheckShapeError> {
    lhs.checked_mul(rhs)
        .ok_or(RamValCheckShapeError::ByteLengthOverflow { name })
}

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod oracle {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct TransitionMessage<F> {
        pub state: Vec<RamValCheckDenseRow<F>>,
        pub evals: [F; RAM_VAL_CHECK_MESSAGE_COLUMNS],
    }

    pub fn first_message<F: Field>(
        rows: &[RamValCheckNativeRow],
        eq_address: &[F],
        lt_lo: &[F],
        lt_hi: &[F],
        eq_hi: &[F],
    ) -> Result<[F; RAM_VAL_CHECK_MESSAGE_COLUMNS], RamValCheckShapeError> {
        validate_native_rows(rows, eq_address)?;
        validate_message_shape(rows.len(), lt_lo, lt_hi, eq_hi)?;
        message_from_native(rows, eq_address, lt_lo, lt_hi, eq_hi)
    }

    pub fn native_bind_and_message<F: Field>(
        rows: &[RamValCheckNativeRow],
        eq_address: &[F],
        challenge: F,
        bound_lt_lo: &[F],
        lt_hi: &[F],
        eq_hi: &[F],
    ) -> Result<TransitionMessage<F>, RamValCheckShapeError> {
        validate_native_rows(rows, eq_address)?;
        if rows.len() < 4 || !rows.len().is_power_of_two() {
            return Err(RamValCheckShapeError::InvalidCycles(rows.len()));
        }
        validate_message_shape(rows.len() / 2, bound_lt_lo, lt_hi, eq_hi)?;

        let mut state = Vec::with_capacity(rows.len() / 2);
        for (pair_index, pair) in rows.chunks_exact(2).enumerate() {
            state.push(RamValCheckDenseRow {
                increment: bind(
                    pair[0].increment_field(),
                    pair[1].increment_field(),
                    challenge,
                ),
                ram_ra: bind(
                    pair[0]
                        .ram_ra(eq_address)
                        .map_err(|source| invalid_row(2 * pair_index, source))?,
                    pair[1]
                        .ram_ra(eq_address)
                        .map_err(|source| invalid_row(2 * pair_index + 1, source))?,
                    challenge,
                ),
            });
        }
        let evals = dense_message(&state, bound_lt_lo, lt_hi, eq_hi)?;
        Ok(TransitionMessage { state, evals })
    }

    pub fn dense_bind_and_message<F: Field>(
        source: &[RamValCheckDenseRow<F>],
        challenge: F,
        bound_lt_lo: &[F],
        lt_hi: &[F],
        eq_hi: &[F],
    ) -> Result<TransitionMessage<F>, RamValCheckShapeError> {
        if source.len() < 4 || !source.len().is_power_of_two() {
            return Err(RamValCheckShapeError::InvalidCycles(source.len()));
        }
        validate_message_shape(source.len() / 2, bound_lt_lo, lt_hi, eq_hi)?;

        let state = source
            .chunks_exact(2)
            .map(|pair| RamValCheckDenseRow {
                increment: bind(pair[0].increment, pair[1].increment, challenge),
                ram_ra: bind(pair[0].ram_ra, pair[1].ram_ra, challenge),
            })
            .collect::<Vec<_>>();
        let evals = dense_message(&state, bound_lt_lo, lt_hi, eq_hi)?;
        Ok(TransitionMessage { state, evals })
    }

    pub fn dense_message<F: Field>(
        state: &[RamValCheckDenseRow<F>],
        lt_lo: &[F],
        lt_hi: &[F],
        eq_hi: &[F],
    ) -> Result<[F; RAM_VAL_CHECK_MESSAGE_COLUMNS], RamValCheckShapeError> {
        validate_message_shape(state.len(), lt_lo, lt_hi, eq_hi)?;
        let mut evals = [F::zero(); RAM_VAL_CHECK_MESSAGE_COLUMNS];
        for pair_index in 0..state.len() / 2 {
            let low_index = 2 * pair_index;
            let high_index = low_index + 1;
            let high_block = low_index / lt_lo.len();
            let low_offset = low_index % lt_lo.len();
            for (sample, t) in [0_u64, 2, 3].into_iter().enumerate() {
                let t = F::from_u64(t);
                let increment = bind(state[low_index].increment, state[high_index].increment, t);
                let ram_ra = bind(state[low_index].ram_ra, state[high_index].ram_ra, t);
                let lt_low = bind(lt_lo[low_offset], lt_lo[low_offset + 1], t);
                let lt = lt_hi[high_block] + eq_hi[high_block] * lt_low;
                evals[sample] += increment * ram_ra * lt;
            }
        }
        Ok(evals)
    }

    fn message_from_native<F: Field>(
        rows: &[RamValCheckNativeRow],
        eq_address: &[F],
        lt_lo: &[F],
        lt_hi: &[F],
        eq_hi: &[F],
    ) -> Result<[F; RAM_VAL_CHECK_MESSAGE_COLUMNS], RamValCheckShapeError> {
        let mut evals = [F::zero(); RAM_VAL_CHECK_MESSAGE_COLUMNS];
        for pair_index in 0..rows.len() / 2 {
            let low_index = 2 * pair_index;
            let high_index = low_index + 1;
            let high_block = low_index / lt_lo.len();
            let low_offset = low_index % lt_lo.len();
            let low_ra = rows[low_index]
                .ram_ra(eq_address)
                .map_err(|source| invalid_row(low_index, source))?;
            let high_ra = rows[high_index]
                .ram_ra(eq_address)
                .map_err(|source| invalid_row(high_index, source))?;
            for (sample, t) in [0_u64, 2, 3].into_iter().enumerate() {
                let t = F::from_u64(t);
                let increment = bind(
                    rows[low_index].increment_field(),
                    rows[high_index].increment_field(),
                    t,
                );
                let ram_ra = bind(low_ra, high_ra, t);
                let lt_low = bind(lt_lo[low_offset], lt_lo[low_offset + 1], t);
                let lt = lt_hi[high_block] + eq_hi[high_block] * lt_low;
                evals[sample] += increment * ram_ra * lt;
            }
        }
        Ok(evals)
    }

    fn validate_native_rows<F: Field>(
        rows: &[RamValCheckNativeRow],
        eq_address: &[F],
    ) -> Result<(), RamValCheckShapeError> {
        validate_power_of_two_table("address equality", eq_address.len())?;
        for (index, row) in rows.iter().copied().enumerate() {
            row.validate_address_domain(eq_address.len())
                .map_err(|source| invalid_row(index, source))?;
        }
        Ok(())
    }

    fn validate_message_shape<F: Field>(
        elements: usize,
        lt_lo: &[F],
        lt_hi: &[F],
        eq_hi: &[F],
    ) -> Result<(), RamValCheckShapeError> {
        validate_power_of_two_table("LT low", lt_lo.len())?;
        validate_power_of_two_table("LT high", lt_hi.len())?;
        if lt_lo.len() < 2 {
            return Err(RamValCheckShapeError::FactorizationShape {
                elements,
                high_blocks: lt_hi.len(),
                lt_lo_length: lt_lo.len(),
            });
        }
        if eq_hi.len() != lt_hi.len() {
            return Err(RamValCheckShapeError::StorageLength {
                name: "EQ high",
                expected: lt_hi.len(),
                got: eq_hi.len(),
            });
        }
        let _ = RamValCheckMessageParams::new(elements, lt_hi.len(), lt_lo.len())?;
        Ok(())
    }

    fn validate_power_of_two_table(
        name: &'static str,
        length: usize,
    ) -> Result<(), RamValCheckShapeError> {
        if length == 0 || !length.is_power_of_two() {
            return Err(RamValCheckShapeError::InvalidTableLength { name, length });
        }
        Ok(())
    }

    fn invalid_row(index: usize, source: RamValCheckRowError) -> RamValCheckShapeError {
        RamValCheckShapeError::InvalidRow { index, source }
    }

    fn bind<F: Field>(low: F, high: F, challenge: F) -> F {
        low + challenge * (high - low)
    }
}
