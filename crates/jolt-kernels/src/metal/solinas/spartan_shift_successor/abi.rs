//! Checked geometry and producer leases for the Spartan shift successor.

use core::mem::{align_of, size_of};

pub const FIELD_BYTES: u64 = 16;
pub const SIMD_WIDTH: usize = 32;
pub const OUTER_COMPONENT_TABLES: usize = 8;
pub const PRODUCT_COMPONENT_TABLES: usize = 2;
pub const PREFIX_Q_TABLES: usize = 4;
pub const MIDPOINT_RESIDUAL_TABLES: usize = 4;
pub const MIDPOINT_FULL_TABLES: usize = 5;
pub const FLAG_PLANES: usize = 3;
pub const FLAG_ROWS_PER_WORD: usize = 32;

pub const OUTER_NUMERIC_PIPELINE: &str = "solinas_spartan_shift_successor_outer_numeric";
pub const OUTER_FLAGS_PIPELINE: &str = "solinas_spartan_shift_successor_outer_flags";
pub const PRODUCT_FLAGS_PIPELINE: &str = "solinas_spartan_shift_successor_product_flags";
pub const REDUCE_PARTIALS_PIPELINE: &str = "solinas_spartan_shift_successor_reduce_partials";
pub const FOLD_RESIDUAL_PIPELINE: &str = "solinas_spartan_shift_successor_fold_residual";
pub const FOLD_FULL_PIPELINE: &str = "solinas_spartan_shift_successor_fold_full";

/// Three current-cycle flag masks for 32 consecutive rows.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpartanShiftSuccessorFlagWord {
    pub is_virtual: u32,
    pub is_first_in_sequence: u32,
    pub is_noop: u32,
}

const _: [(); 12] = [(); size_of::<SpartanShiftSuccessorFlagWord>()];
const _: [(); 4] = [(); align_of::<SpartanShiftSuccessorFlagWord>()];

/// Common parameters for the component-producing entry points.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpartanShiftSuccessorPartialParams {
    pub rows: u32,
    pub prefix_elements: u32,
    pub suffix_elements: u32,
    pub high_tile_elements: u32,
    pub high_tiles: u32,
    pub output_columns: u32,
    pub reserved: [u32; 2],
}

const _: [(); 32] = [(); size_of::<SpartanShiftSuccessorPartialParams>()];
const _: [(); 4] = [(); align_of::<SpartanShiftSuccessorPartialParams>()];

/// Parameters for reducing column-major high-tile partials.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpartanShiftSuccessorReductionParams {
    pub prefix_elements: u32,
    pub high_tiles: u32,
    pub columns: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<SpartanShiftSuccessorReductionParams>()];
const _: [(); 4] = [(); align_of::<SpartanShiftSuccessorReductionParams>()];

/// Parameters for either midpoint fold.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpartanShiftSuccessorFoldParams {
    pub rows: u32,
    pub prefix_elements: u32,
    pub suffix_elements: u32,
    pub reserved: u32,
}

const _: [(); 16] = [(); size_of::<SpartanShiftSuccessorFoldParams>()];
const _: [(); 4] = [(); align_of::<SpartanShiftSuccessorFoldParams>()];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarrierProducer {
    Stage1Outer,
    Stage2Product,
    Stage3InstructionInput,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentBufferDescriptor {
    pub storage_id: u64,
    pub byte_len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialCarrierHeader {
    pub producer: CarrierProducer,
    pub witness_generation: u64,
    pub device_registry_id: u64,
    pub rows: usize,
    pub table_elements: usize,
    pub point_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialCarrierLease<const TABLES: usize> {
    pub header: PartialCarrierHeader,
    pub tables: [ResidentBufferDescriptor; TABLES],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MidpointUpcLease {
    pub producer: CarrierProducer,
    pub witness_generation: u64,
    pub device_registry_id: u64,
    pub rows: usize,
    pub table_elements: usize,
    pub ordered_challenge_digest: [u8; 32],
    pub table: ResidentBufferDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualPlaneLease {
    pub witness_generation: u64,
    pub device_registry_id: u64,
    pub rows: usize,
    pub pc: ResidentBufferDescriptor,
    pub flags: ResidentBufferDescriptor,
    pub exact_current_flags: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpartanShiftSuccessorGeometry {
    pub rows: usize,
    pub log_t: usize,
    pub prefix_vars: usize,
    pub suffix_vars: usize,
    pub prefix_elements: usize,
    pub suffix_elements: usize,
    pub flag_words: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpartanShiftSuccessorAbiError {
    InvalidRows,
    InvalidHighTile,
    InvalidOutputColumns,
    InvalidThreadgroupWidth,
    ShaderIndexOverflow,
    ArithmeticOverflow,
    WrongProducer,
    WrongWitnessGeneration,
    WrongDevice,
    WrongPointDigest,
    WrongChallengeDigest,
    WrongTableElements,
    WrongBufferLength,
    MissingBufferIdentity,
    DuplicateBufferIdentity,
    UncertifiedCurrentFlags,
}

impl SpartanShiftSuccessorGeometry {
    pub fn new(rows: usize) -> Result<Self, SpartanShiftSuccessorAbiError> {
        if rows < 2 || !rows.is_power_of_two() || rows > u32::MAX as usize {
            return Err(SpartanShiftSuccessorAbiError::InvalidRows);
        }
        let log_t = rows.ilog2() as usize;
        let suffix_vars = log_t / 2;
        let prefix_vars = log_t - suffix_vars;
        let prefix_elements = 1usize
            .checked_shl(prefix_vars as u32)
            .ok_or(SpartanShiftSuccessorAbiError::ArithmeticOverflow)?;
        let suffix_elements = 1usize
            .checked_shl(suffix_vars as u32)
            .ok_or(SpartanShiftSuccessorAbiError::ArithmeticOverflow)?;
        Ok(Self {
            rows,
            log_t,
            prefix_vars,
            suffix_vars,
            prefix_elements,
            suffix_elements,
            flag_words: rows.div_ceil(FLAG_ROWS_PER_WORD),
        })
    }

    pub fn outer_partial_params(
        self,
        high_tile_elements: usize,
    ) -> Result<SpartanShiftSuccessorPartialParams, SpartanShiftSuccessorAbiError> {
        self.partial_params(high_tile_elements, PREFIX_Q_TABLES)
    }

    pub fn product_partial_params(
        self,
        high_tile_elements: usize,
    ) -> Result<SpartanShiftSuccessorPartialParams, SpartanShiftSuccessorAbiError> {
        self.partial_params(high_tile_elements, PRODUCT_COMPONENT_TABLES)
    }

    fn partial_params(
        self,
        high_tile_elements: usize,
        output_columns: usize,
    ) -> Result<SpartanShiftSuccessorPartialParams, SpartanShiftSuccessorAbiError> {
        if high_tile_elements == 0
            || !high_tile_elements.is_power_of_two()
            || !self.suffix_elements.is_multiple_of(high_tile_elements)
        {
            return Err(SpartanShiftSuccessorAbiError::InvalidHighTile);
        }
        if output_columns != PRODUCT_COMPONENT_TABLES && output_columns != PREFIX_Q_TABLES {
            return Err(SpartanShiftSuccessorAbiError::InvalidOutputColumns);
        }
        let high_tiles = self.suffix_elements / high_tile_elements;
        let output_elements = self
            .prefix_elements
            .checked_mul(high_tiles)
            .and_then(|elements| elements.checked_mul(output_columns))
            .ok_or(SpartanShiftSuccessorAbiError::ArithmeticOverflow)?;
        checked_u32(output_elements)?;
        Ok(SpartanShiftSuccessorPartialParams {
            rows: checked_u32(self.rows)?,
            prefix_elements: checked_u32(self.prefix_elements)?,
            suffix_elements: checked_u32(self.suffix_elements)?,
            high_tile_elements: checked_u32(high_tile_elements)?,
            high_tiles: checked_u32(high_tiles)?,
            output_columns: checked_u32(output_columns)?,
            reserved: [0; 2],
        })
    }

    pub fn reduction_params(
        self,
        high_tile_elements: usize,
        columns: usize,
    ) -> Result<SpartanShiftSuccessorReductionParams, SpartanShiftSuccessorAbiError> {
        let partial = self.partial_params(high_tile_elements, columns)?;
        Ok(SpartanShiftSuccessorReductionParams {
            prefix_elements: partial.prefix_elements,
            high_tiles: partial.high_tiles,
            columns: partial.output_columns,
            reserved: 0,
        })
    }

    pub fn fold_params(
        self,
    ) -> Result<SpartanShiftSuccessorFoldParams, SpartanShiftSuccessorAbiError> {
        Ok(SpartanShiftSuccessorFoldParams {
            rows: checked_u32(self.rows)?,
            prefix_elements: checked_u32(self.prefix_elements)?,
            suffix_elements: checked_u32(self.suffix_elements)?,
            reserved: 0,
        })
    }

    pub fn fold_threadgroup_bytes(
        threads: usize,
        output_columns: usize,
    ) -> Result<u64, SpartanShiftSuccessorAbiError> {
        if !(SIMD_WIDTH..=1024).contains(&threads) || !threads.is_multiple_of(SIMD_WIDTH) {
            return Err(SpartanShiftSuccessorAbiError::InvalidThreadgroupWidth);
        }
        if output_columns != MIDPOINT_RESIDUAL_TABLES && output_columns != MIDPOINT_FULL_TABLES {
            return Err(SpartanShiftSuccessorAbiError::InvalidOutputColumns);
        }
        let simdgroups = threads / SIMD_WIDTH;
        checked_bytes(
            simdgroups
                .checked_mul(output_columns)
                .ok_or(SpartanShiftSuccessorAbiError::ArithmeticOverflow)?,
            FIELD_BYTES,
        )
    }

    pub fn table_bytes(self) -> Result<u64, SpartanShiftSuccessorAbiError> {
        checked_bytes(self.prefix_elements, FIELD_BYTES)
    }

    pub fn dense_table_bytes(self) -> Result<u64, SpartanShiftSuccessorAbiError> {
        checked_bytes(self.suffix_elements, FIELD_BYTES)
    }

    pub fn pc_bytes(self) -> Result<u64, SpartanShiftSuccessorAbiError> {
        checked_bytes(self.rows, size_of::<u64>() as u64)
    }

    pub fn flag_bytes(self) -> Result<u64, SpartanShiftSuccessorAbiError> {
        checked_bytes(
            self.flag_words,
            size_of::<SpartanShiftSuccessorFlagWord>() as u64,
        )
    }
}

impl<const TABLES: usize> PartialCarrierLease<TABLES> {
    pub fn validate(
        self,
        geometry: SpartanShiftSuccessorGeometry,
        expected_producer: CarrierProducer,
        expected_generation: u64,
        expected_device: u64,
        expected_point_digest: [u8; 32],
    ) -> Result<Self, SpartanShiftSuccessorAbiError> {
        if self.header.producer != expected_producer {
            return Err(SpartanShiftSuccessorAbiError::WrongProducer);
        }
        if self.header.witness_generation != expected_generation {
            return Err(SpartanShiftSuccessorAbiError::WrongWitnessGeneration);
        }
        if self.header.device_registry_id != expected_device {
            return Err(SpartanShiftSuccessorAbiError::WrongDevice);
        }
        if self.header.point_digest != expected_point_digest {
            return Err(SpartanShiftSuccessorAbiError::WrongPointDigest);
        }
        if self.header.rows != geometry.rows
            || self.header.table_elements != geometry.prefix_elements
        {
            return Err(SpartanShiftSuccessorAbiError::WrongTableElements);
        }
        validate_buffers(&self.tables, geometry.table_bytes()?)?;
        Ok(self)
    }
}

impl MidpointUpcLease {
    pub fn validate(
        self,
        geometry: SpartanShiftSuccessorGeometry,
        expected_generation: u64,
        expected_device: u64,
        expected_challenge_digest: [u8; 32],
    ) -> Result<Self, SpartanShiftSuccessorAbiError> {
        if self.producer != CarrierProducer::Stage3InstructionInput {
            return Err(SpartanShiftSuccessorAbiError::WrongProducer);
        }
        if self.witness_generation != expected_generation {
            return Err(SpartanShiftSuccessorAbiError::WrongWitnessGeneration);
        }
        if self.device_registry_id != expected_device {
            return Err(SpartanShiftSuccessorAbiError::WrongDevice);
        }
        if self.ordered_challenge_digest != expected_challenge_digest {
            return Err(SpartanShiftSuccessorAbiError::WrongChallengeDigest);
        }
        if self.rows != geometry.rows || self.table_elements != geometry.suffix_elements {
            return Err(SpartanShiftSuccessorAbiError::WrongTableElements);
        }
        validate_buffers(&[self.table], geometry.dense_table_bytes()?)?;
        Ok(self)
    }
}

impl ResidualPlaneLease {
    pub fn validate(
        self,
        geometry: SpartanShiftSuccessorGeometry,
        expected_generation: u64,
        expected_device: u64,
    ) -> Result<Self, SpartanShiftSuccessorAbiError> {
        if self.witness_generation != expected_generation {
            return Err(SpartanShiftSuccessorAbiError::WrongWitnessGeneration);
        }
        if self.device_registry_id != expected_device {
            return Err(SpartanShiftSuccessorAbiError::WrongDevice);
        }
        if self.rows != geometry.rows {
            return Err(SpartanShiftSuccessorAbiError::WrongTableElements);
        }
        if !self.exact_current_flags {
            return Err(SpartanShiftSuccessorAbiError::UncertifiedCurrentFlags);
        }
        validate_buffers(&[self.pc], geometry.pc_bytes()?)?;
        validate_buffers(&[self.flags], geometry.flag_bytes()?)?;
        if self.pc.storage_id == self.flags.storage_id {
            return Err(SpartanShiftSuccessorAbiError::DuplicateBufferIdentity);
        }
        Ok(self)
    }
}

fn validate_buffers<const N: usize>(
    buffers: &[ResidentBufferDescriptor; N],
    expected_bytes: u64,
) -> Result<(), SpartanShiftSuccessorAbiError> {
    for (index, buffer) in buffers.iter().enumerate() {
        if buffer.storage_id == 0 {
            return Err(SpartanShiftSuccessorAbiError::MissingBufferIdentity);
        }
        if buffer.byte_len != expected_bytes {
            return Err(SpartanShiftSuccessorAbiError::WrongBufferLength);
        }
        if buffers[..index]
            .iter()
            .any(|previous| previous.storage_id == buffer.storage_id)
        {
            return Err(SpartanShiftSuccessorAbiError::DuplicateBufferIdentity);
        }
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, SpartanShiftSuccessorAbiError> {
    u32::try_from(value).map_err(|_| SpartanShiftSuccessorAbiError::ShaderIndexOverflow)
}

fn checked_bytes(
    elements: usize,
    bytes_per_element: u64,
) -> Result<u64, SpartanShiftSuccessorAbiError> {
    u64::try_from(elements)
        .ok()
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or(SpartanShiftSuccessorAbiError::ArithmeticOverflow)
}
