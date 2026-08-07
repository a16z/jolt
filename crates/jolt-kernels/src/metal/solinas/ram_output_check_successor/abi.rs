//! Checked producer, buffer-range, and dispatch ABIs for the successor sketch.

use core::mem::{align_of, size_of};

pub const FIELD_BYTES: u64 = 16;
pub const NATIVE_WORD_BYTES: u64 = 8;
pub const STATUS_BYTES: u64 = 4;
pub const SIMD_WIDTH: u32 = 32;

pub const TARGET_ADDRESSES: u32 = 1 << 13;
pub const TARGET_BLOCK_ELEMENTS: u32 = 1 << 10;
pub const TARGET_BLOCKS: u32 = 8;
pub const TARGET_CHUNKS_PER_BLOCK: u32 = 8;
pub const TARGET_THREADS: u32 = 128;
pub const TARGET_PARTIALS: u32 = TARGET_BLOCKS * TARGET_CHUNKS_PER_BLOCK;
pub const TARGET_WEIGHTS: u32 = TARGET_BLOCK_ELEMENTS;
pub const TARGET_CHALLENGES: u32 = 10;

pub const STATUS_UNSUPPORTED: u32 = 1 << 0;
pub const STATUS_INVALID_RANGE: u32 = 1 << 1;
pub const STATUS_NONCANONICAL_INPUT: u32 = 1 << 2;

pub const HOST_WEIGHT_PARTIAL_PIPELINE: &str = "solinas_ram_output_successor_partials_host_weights";
pub const DEVICE_WEIGHT_PARTIAL_PIPELINE: &str =
    "solinas_ram_output_successor_partials_device_weights";
pub const REDUCE_PIPELINE: &str = "solinas_ram_output_successor_reduce8";

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightMode {
    HostTable = 0,
    DeviceChallenges = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamOutputSuccessorParams {
    pub addresses: u32,
    pub block_elements: u32,
    pub blocks: u32,
    pub chunks_per_block: u32,
    pub threads: u32,
    pub weight_mode: u32,
    pub reserved: [u32; 2],
}

const _: [(); 32] = [(); size_of::<RamOutputSuccessorParams>()];
const _: [(); 4] = [(); align_of::<RamOutputSuccessorParams>()];

impl RamOutputSuccessorParams {
    pub const fn target(mode: WeightMode) -> Self {
        Self {
            addresses: TARGET_ADDRESSES,
            block_elements: TARGET_BLOCK_ELEMENTS,
            blocks: TARGET_BLOCKS,
            chunks_per_block: TARGET_CHUNKS_PER_BLOCK,
            threads: TARGET_THREADS,
            weight_mode: mode as u32,
            reserved: [0; 2],
        }
    }

    pub fn validate(self) -> Result<Self, AbiError> {
        let known_mode = self.weight_mode == WeightMode::HostTable as u32
            || self.weight_mode == WeightMode::DeviceChallenges as u32;
        if self.addresses != TARGET_ADDRESSES
            || self.block_elements != TARGET_BLOCK_ELEMENTS
            || self.blocks != TARGET_BLOCKS
            || self.chunks_per_block != TARGET_CHUNKS_PER_BLOCK
            || self.threads != TARGET_THREADS
            || !known_mode
            || self.reserved[0] != 0
            || self.reserved[1] != 0
            || self.blocks.checked_mul(self.block_elements) != Some(self.addresses)
            || self.chunks_per_block.checked_mul(self.threads) != Some(self.block_elements)
        {
            return Err(AbiError::InvalidParams);
        }
        Ok(self)
    }

    pub const fn coefficient_elements(self) -> u32 {
        if self.weight_mode == WeightMode::HostTable as u32 {
            TARGET_WEIGHTS
        } else {
            TARGET_CHALLENGES
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamOutputReductionParams {
    pub input_count: u32,
    pub blocks: u32,
    pub chunks_per_block: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<RamOutputReductionParams>()];
const _: [(); 4] = [(); align_of::<RamOutputReductionParams>()];

impl RamOutputReductionParams {
    pub const fn target() -> Self {
        Self {
            input_count: TARGET_PARTIALS,
            blocks: TARGET_BLOCKS,
            chunks_per_block: TARGET_CHUNKS_PER_BLOCK,
            reserved: 0,
        }
    }

    pub fn validate(self) -> Result<Self, AbiError> {
        if self.input_count != TARGET_PARTIALS
            || self.blocks != TARGET_BLOCKS
            || self.chunks_per_block != TARGET_CHUNKS_PER_BLOCK
            || self.reserved != 0
            || self.blocks.checked_mul(self.chunks_per_block) != Some(self.input_count)
        {
            return Err(AbiError::InvalidReductionParams);
        }
        Ok(self)
    }
}

/// A byte range within one host-tracked Metal allocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BufferRange {
    pub storage_id: u64,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

impl BufferRange {
    pub fn end(self) -> Option<u64> {
        self.offset_bytes.checked_add(self.length_bytes)
    }

    pub fn contains(self, required_bytes: u64, alignment: u64) -> bool {
        self.storage_id != 0
            && alignment != 0
            && self.offset_bytes.is_multiple_of(alignment)
            && self.length_bytes >= required_bytes
            && self.end().is_some()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResidentRamFinalMetadata {
    pub range: BufferRange,
    pub device_registry_id: u64,
    pub allocation_identity: u64,
    pub elements: u32,
    pub stride_bytes: u32,
    pub public_io_certified: bool,
    pub host_readable: bool,
}

impl ResidentRamFinalMetadata {
    pub fn validate(self, expected_device_registry_id: u64) -> Result<Self, AbiError> {
        if self.device_registry_id == 0
            || self.device_registry_id != expected_device_registry_id
            || self.allocation_identity == 0
            || self.elements != TARGET_ADDRESSES
            || self.stride_bytes != NATIVE_WORD_BYTES as u32
            || !self.public_io_certified
            || !self.host_readable
            || !self.range.contains(
                TARGET_ADDRESSES as u64 * NATIVE_WORD_BYTES,
                NATIVE_WORD_BYTES,
            )
        {
            return Err(AbiError::InvalidResidentSource);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RamOutputSuccessorRanges {
    pub source: BufferRange,
    pub coefficients: BufferRange,
    pub partials: BufferRange,
    pub output: BufferRange,
    pub status: BufferRange,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchShape {
    pub threadgroups: u32,
    pub threads_per_threadgroup: u32,
    pub initial_status: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbiError {
    InvalidParams,
    InvalidReductionParams,
    InvalidResidentSource,
    InvalidDispatch,
    MissingRange(&'static str),
    MisalignedRange(&'static str),
    ShortRange(&'static str),
    RangeOverflow(&'static str),
    OverlappingRanges {
        left: &'static str,
        right: &'static str,
    },
}

impl RamOutputSuccessorRanges {
    pub fn validate_partials(
        self,
        params: RamOutputSuccessorParams,
        dispatch: DispatchShape,
    ) -> Result<Self, AbiError> {
        let params = params.validate()?;
        if dispatch.threadgroups != TARGET_PARTIALS
            || dispatch.threads_per_threadgroup != TARGET_THREADS
            || dispatch.initial_status != 0
        {
            return Err(AbiError::InvalidDispatch);
        }
        let coefficient_bytes = params.coefficient_elements() as u64 * FIELD_BYTES;
        validate_range(
            "source",
            self.source,
            TARGET_ADDRESSES as u64 * NATIVE_WORD_BYTES,
            NATIVE_WORD_BYTES,
        )?;
        validate_range(
            "coefficients",
            self.coefficients,
            coefficient_bytes,
            FIELD_BYTES,
        )?;
        validate_range(
            "partials",
            self.partials,
            TARGET_PARTIALS as u64 * FIELD_BYTES,
            FIELD_BYTES,
        )?;
        validate_range("status", self.status, STATUS_BYTES, STATUS_BYTES)?;
        validate_nonoverlap(&[
            ("source", self.source),
            ("coefficients", self.coefficients),
            ("partials", self.partials),
            ("status", self.status),
        ])?;
        Ok(self)
    }

    pub fn validate_device_reduction(
        self,
        params: RamOutputReductionParams,
        dispatch: DispatchShape,
    ) -> Result<Self, AbiError> {
        let _ = params.validate()?;
        if dispatch.threadgroups != 1
            || dispatch.threads_per_threadgroup != SIMD_WIDTH
            || dispatch.initial_status != 0
        {
            return Err(AbiError::InvalidDispatch);
        }
        validate_range(
            "partials",
            self.partials,
            TARGET_PARTIALS as u64 * FIELD_BYTES,
            FIELD_BYTES,
        )?;
        validate_range(
            "output",
            self.output,
            TARGET_BLOCKS as u64 * FIELD_BYTES,
            FIELD_BYTES,
        )?;
        validate_range("status", self.status, STATUS_BYTES, STATUS_BYTES)?;
        validate_nonoverlap(&[
            ("partials", self.partials),
            ("output", self.output),
            ("status", self.status),
        ])?;
        Ok(self)
    }
}

fn validate_range(
    name: &'static str,
    range: BufferRange,
    required_bytes: u64,
    alignment: u64,
) -> Result<(), AbiError> {
    if range.storage_id == 0 {
        return Err(AbiError::MissingRange(name));
    }
    if !range.offset_bytes.is_multiple_of(alignment) {
        return Err(AbiError::MisalignedRange(name));
    }
    if range.length_bytes < required_bytes {
        return Err(AbiError::ShortRange(name));
    }
    if range.end().is_none() {
        return Err(AbiError::RangeOverflow(name));
    }
    Ok(())
}

fn validate_nonoverlap(ranges: &[(&'static str, BufferRange)]) -> Result<(), AbiError> {
    let mut left = 0;
    while left < ranges.len() {
        let mut right = left + 1;
        while right < ranges.len() {
            let (left_name, left_range) = ranges[left];
            let (right_name, right_range) = ranges[right];
            if left_range.storage_id == right_range.storage_id {
                let Some(left_end) = left_range.end() else {
                    return Err(AbiError::RangeOverflow(left_name));
                };
                let Some(right_end) = right_range.end() else {
                    return Err(AbiError::RangeOverflow(right_name));
                };
                if left_range.offset_bytes < right_end && right_range.offset_bytes < left_end {
                    return Err(AbiError::OverlappingRanges {
                        left: left_name,
                        right: right_name,
                    });
                }
            }
            right += 1;
        }
        left += 1;
    }
    Ok(())
}
