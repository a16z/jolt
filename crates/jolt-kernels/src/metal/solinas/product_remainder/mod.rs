use std::{
    mem::{self, align_of, size_of},
    slice,
    time::Duration,
};

use jolt_field::{AkitaField, Field};
use metal::{
    foreign_types::ForeignType, objc::rc::autoreleasepool, Buffer, CommandBuffer, CommandQueue,
    ComputePipelineState, MTLResourceOptions, MTLSize, NSRange,
};
use thiserror::Error;

use crate::optimized::spartan_product::SpartanProductRow;

use super::product_uniskip::{
    ProductUniskipBlockParams, ProductUniskipExtendedNodes,
    BLOCKS_PIPELINE as PRODUCT_UNISKIP_PIPELINE, PRODUCT_UNISKIP_EXTENDED_NODES,
    PRODUCT_UNISKIP_SIMD_WIDTH, STAGE1_BLOCKS_PIPELINE as PRODUCT_UNISKIP_STAGE1_PIPELINE,
};
use super::{
    buffer_from_slice, completed_command_gpu_time, encode_column_reductions, set_inline_bytes,
    spartan_outer_uniskip_successor_row_bytes, validate_completed_command, Fp128,
    InstructionInputRow, MetalError, SolinasMetal,
};

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const PRODUCT_REMAINDER_ROW_WORDS: usize = 5;
pub const PRODUCT_REMAINDER_MESSAGE_COLUMNS: usize = 2;
pub const PRODUCT_REMAINDER_OPENINGS: usize = 8;
pub const PRODUCT_REMAINDER_SIMD_WIDTH: usize = 32;

pub(crate) const MATERIALIZE_PIPELINE: &str = "solinas_product_remainder_materialize_message";
pub(crate) const MATERIALIZE_STAGE1_PIPELINE: &str =
    "solinas_product_remainder_materialize_stage1_message";
pub(crate) const TRANSITION_PIPELINE: &str = "solinas_product_remainder_bind_and_message";
const IN_PLACE_BIND_PIPELINE: &str = "solinas_product_remainder_bind_range";
const IN_PLACE_COPY_PIPELINE: &str = "solinas_product_remainder_copy_prefix";
const IN_PLACE_MESSAGE_PIPELINE: &str = "solinas_product_remainder_bound_message";
pub(crate) const OPENINGS_PIPELINE: &str = "solinas_product_remainder_openings";
pub(crate) const OPENINGS_STAGE1_PIPELINE: &str = "solinas_product_remainder_stage1_openings";
const TERMINAL_CACHE_OPENINGS_PIPELINE: &str =
    "solinas_product_instruction_terminal_cache_openings";
pub(crate) const REDUCTION_PIPELINE: &str = "solinas_product_remainder_reduce";

const ROW_LEFT_INPUT: usize = 0;
const ROW_RIGHT_MAGNITUDE_LOW: usize = 1;
const ROW_RIGHT_MAGNITUDE_HIGH: usize = 2;
const ROW_LOOKUP_OUTPUT: usize = 3;
const ROW_FLAGS: usize = 4;

const FLAG_RIGHT_NONNEGATIVE: u32 = 0;
const FLAG_JUMP: u32 = 1;
const FLAG_WRITE_LOOKUP: u32 = 2;
const FLAG_BRANCH: u32 = 3;
const FLAG_NEXT_IS_NOOP: u32 = 4;
const FLAG_VIRTUAL: u32 = 5;
const VALID_FLAGS: u64 = (1 << 6) - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductRemainderSequenceConfig {
    pub uniskip_threads_per_threadgroup: Option<usize>,
    pub materialize_threads_per_threadgroup: Option<usize>,
    pub transition_threads_per_threadgroup: Option<usize>,
    pub openings_threads_per_threadgroup: Option<usize>,
    pub prime_workspace: bool,
    pub async_state_b_fill: bool,
}

impl Default for ProductRemainderSequenceConfig {
    fn default() -> Self {
        Self {
            uniskip_threads_per_threadgroup: Some(64),
            materialize_threads_per_threadgroup: Some(128),
            transition_threads_per_threadgroup: Some(64),
            openings_threads_per_threadgroup: Some(128),
            prime_workspace: true,
            async_state_b_fill: false,
        }
    }
}

/// The product-remainder witness columns needed after product uni-skip.
///
/// The right input uses a 128-bit magnitude and a nonnegative flag. The final
/// word holds that flag followed by jump, write-lookup, branch, next-is-noop,
/// and virtual-instruction flags. Rust and Metal both use a 40-byte stride.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductRemainderRow {
    words: [u64; PRODUCT_REMAINDER_ROW_WORDS],
}

const _: [(); 40] = [(); size_of::<ProductRemainderRow>()];
const _: [(); 8] = [(); align_of::<ProductRemainderRow>()];

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ProductRemainderRowError {
    #[error("product remainder row has reserved flag bits {0:#x}")]
    ReservedFlags(u64),
    #[error(
        "product remainder right-input encoding has magnitude {magnitude} with nonnegative={nonnegative}"
    )]
    InvalidRightInputEncoding { magnitude: u128, nonnegative: bool },
}

impl ProductRemainderRow {
    #[expect(
        clippy::too_many_arguments,
        reason = "the arguments are the eight product-remainder witness columns"
    )]
    pub const fn new(
        left_instruction_input: u64,
        right_instruction_input: i128,
        jump: bool,
        write_lookup_output_to_rd: bool,
        lookup_output: u64,
        branch: bool,
        next_is_noop: bool,
        virtual_instruction: bool,
    ) -> Self {
        let magnitude = right_instruction_input.unsigned_abs();
        let flags = ((right_instruction_input >= 0) as u64) << FLAG_RIGHT_NONNEGATIVE
            | (jump as u64) << FLAG_JUMP
            | (write_lookup_output_to_rd as u64) << FLAG_WRITE_LOOKUP
            | (branch as u64) << FLAG_BRANCH
            | (next_is_noop as u64) << FLAG_NEXT_IS_NOOP
            | (virtual_instruction as u64) << FLAG_VIRTUAL;
        Self {
            words: [
                left_instruction_input,
                magnitude as u64,
                (magnitude >> 64) as u64,
                lookup_output,
                flags,
            ],
        }
    }

    pub fn try_from_words(
        words: [u64; PRODUCT_REMAINDER_ROW_WORDS],
    ) -> Result<Self, ProductRemainderRowError> {
        let row = Self { words };
        row.validate()?;
        Ok(row)
    }

    copy_field_getters! { pub, { words: [u64; PRODUCT_REMAINDER_ROW_WORDS] }}

    pub const fn left_instruction_input(self) -> u64 {
        self.words[ROW_LEFT_INPUT]
    }

    pub const fn lookup_output(self) -> u64 {
        self.words[ROW_LOOKUP_OUTPUT]
    }

    pub const fn jump(self) -> bool {
        self.flag(FLAG_JUMP)
    }

    pub const fn write_lookup_output_to_rd(self) -> bool {
        self.flag(FLAG_WRITE_LOOKUP)
    }

    pub const fn branch(self) -> bool {
        self.flag(FLAG_BRANCH)
    }

    pub const fn next_is_noop(self) -> bool {
        self.flag(FLAG_NEXT_IS_NOOP)
    }

    pub const fn virtual_instruction(self) -> bool {
        self.flag(FLAG_VIRTUAL)
    }

    pub const fn right_instruction_input(self) -> i128 {
        let magnitude = self.right_magnitude();
        if self.flag(FLAG_RIGHT_NONNEGATIVE) {
            magnitude as i128
        } else if magnitude == 1u128 << 127 {
            i128::MIN
        } else {
            -(magnitude as i128)
        }
    }

    pub fn validate(self) -> Result<(), ProductRemainderRowError> {
        let reserved = self.words[ROW_FLAGS] & !VALID_FLAGS;
        if reserved != 0 {
            return Err(ProductRemainderRowError::ReservedFlags(reserved));
        }

        let magnitude = self.right_magnitude();
        let nonnegative = self.flag(FLAG_RIGHT_NONNEGATIVE);
        let valid = if nonnegative {
            magnitude <= i128::MAX as u128
        } else {
            magnitude != 0 && magnitude <= 1u128 << 127
        };
        if !valid {
            return Err(ProductRemainderRowError::InvalidRightInputEncoding {
                magnitude,
                nonnegative,
            });
        }
        Ok(())
    }

    /// Returns the eight opening columns in protocol order.
    pub fn fields<F: Field>(self) -> [F; PRODUCT_REMAINDER_OPENINGS] {
        [
            F::from_u64(self.left_instruction_input()),
            F::from_i128(self.right_instruction_input()),
            F::from_bool(self.jump()),
            F::from_bool(self.write_lookup_output_to_rd()),
            F::from_u64(self.lookup_output()),
            F::from_bool(self.branch()),
            F::from_bool(self.next_is_noop()),
            F::from_bool(self.virtual_instruction()),
        ]
    }

    /// Evaluates the left and right linear factors at the uni-skip challenge.
    pub fn relation_values<F: Field>(self, weights: &[F; 3]) -> (F, F) {
        let left = weights[0] * F::from_u64(self.left_instruction_input())
            + weights[1] * F::from_u64(self.lookup_output())
            + weights[2] * F::from_bool(self.jump());
        let right = weights[0] * F::from_i128(self.right_instruction_input())
            + weights[1] * F::from_bool(self.branch())
            + weights[2] * F::from_bool(!self.next_is_noop());
        (left, right)
    }

    const fn right_magnitude(self) -> u128 {
        self.words[ROW_RIGHT_MAGNITUDE_LOW] as u128
            | ((self.words[ROW_RIGHT_MAGNITUDE_HIGH] as u128) << 64)
    }

    const fn flag(self, bit: u32) -> bool {
        ((self.words[ROW_FLAGS] >> bit) & 1) != 0
    }
}

impl From<&SpartanProductRow> for ProductRemainderRow {
    fn from(row: &SpartanProductRow) -> Self {
        Self::new(
            row.left_instruction_input.0,
            row.right_instruction_input.0,
            row.jump_flag.0,
            row.write_lookup_output_to_rd.0,
            row.lookup_output.0,
            row.branch_flag.0,
            row.next_is_noop.0,
            row.virtual_instruction.0,
        )
    }
}

impl Default for ProductRemainderRow {
    fn default() -> Self {
        Self::new(0, 0, false, false, 0, false, false, false)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductRemainderSourceKind {
    Packed,
    SpartanStage1,
}

impl ProductRemainderSourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Packed => "packed",
            Self::SpartanStage1 => "spartan_stage1",
        }
    }
}

#[derive(Clone)]
enum ProductRemainderSource {
    Packed(Buffer),
    SpartanStage1 {
        compact: Buffer,
        residual: Buffer,
        generation: u64,
    },
}

#[derive(Clone)]
pub struct ProductRemainderRows {
    source: ProductRemainderSource,
    len: usize,
    device_registry_id: u64,
}

impl ProductRemainderRows {
    copy_field_getters! { pub, {
        len: usize,
        device_registry_id: u64,
    }}

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocation_identity(&self) -> usize {
        match &self.source {
            ProductRemainderSource::Packed(buffer) => buffer.as_ptr() as usize,
            ProductRemainderSource::SpartanStage1 { compact, .. } => compact.as_ptr() as usize,
        }
    }

    pub fn allocation_identities(&self) -> Vec<usize> {
        match &self.source {
            ProductRemainderSource::Packed(buffer) => vec![buffer.as_ptr() as usize],
            ProductRemainderSource::SpartanStage1 {
                compact, residual, ..
            } => vec![compact.as_ptr() as usize, residual.as_ptr() as usize],
        }
    }

    pub const fn source_kind(&self) -> ProductRemainderSourceKind {
        match self.source {
            ProductRemainderSource::Packed(_) => ProductRemainderSourceKind::Packed,
            ProductRemainderSource::SpartanStage1 { .. } => {
                ProductRemainderSourceKind::SpartanStage1
            }
        }
    }

    pub const fn source_generation(&self) -> Option<u64> {
        match self.source {
            ProductRemainderSource::Packed(_) => None,
            ProductRemainderSource::SpartanStage1 { generation, .. } => Some(generation),
        }
    }

    pub const fn resident_bytes(&self) -> usize {
        match self.source {
            ProductRemainderSource::Packed(_) => self.len * size_of::<ProductRemainderRow>(),
            ProductRemainderSource::SpartanStage1 { .. } => 0,
        }
    }

    pub(crate) fn packed_buffer(&self) -> Option<&Buffer> {
        match &self.source {
            ProductRemainderSource::Packed(buffer) => Some(buffer),
            ProductRemainderSource::SpartanStage1 { .. } => None,
        }
    }

    pub(crate) fn stage1_buffers(&self) -> Option<(&Buffer, &Buffer)> {
        match &self.source {
            ProductRemainderSource::Packed(_) => None,
            ProductRemainderSource::SpartanStage1 {
                compact, residual, ..
            } => Some((compact, residual)),
        }
    }

    pub(crate) fn from_spartan_stage1(
        compact: Buffer,
        residual: Buffer,
        len: usize,
        device_registry_id: u64,
        generation: u64,
    ) -> Result<Self, MetalError> {
        let compact_bytes = len
            .checked_mul(size_of::<InstructionInputRow>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(len))?;
        let residual_bytes = spartan_outer_uniskip_successor_row_bytes(len)?;
        if len < 2
            || !len.is_power_of_two()
            || generation == 0
            || compact.length() != compact_bytes
            || residual.length() != residual_bytes
            || compact.as_ptr() == residual.as_ptr()
        {
            return Err(MetalError::InvalidProductRemainderState(
                "the Spartan Stage-1 product source has invalid provenance or shape",
            ));
        }
        Ok(Self {
            source: ProductRemainderSource::SpartanStage1 {
                compact,
                residual,
                generation,
            },
            len,
            device_registry_id,
        })
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ProductRemainderRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if self.source_kind() == ProductRemainderSourceKind::Packed {
            visitor.visit_simple(
                allocative::Key::new("device_rows"),
                self.len * size_of::<ProductRemainderRow>(),
            );
        }
        visitor.exit();
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ProductRemainderShapeError {
    #[error("product remainder needs a power-of-two row count of at least two, got {0}")]
    InvalidRows(usize),
    #[error(
        "product remainder transition needs a power-of-two source count of at least four, got {0}"
    )]
    InvalidTransitionSource(usize),
    #[error(
        "product remainder {phase} weights have e_in={e_in}, e_out={e_out}; expected product {expected}"
    )]
    WeightShape {
        phase: &'static str,
        expected: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error("product remainder {name} storage has length {got}, expected {expected}")]
    StorageLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("product remainder {name} element count exceeds its 32-bit shader index")]
    ShaderIndexOverflow { name: &'static str },
    #[error("product remainder {name} byte length overflows host indexing")]
    ByteLengthOverflow { name: &'static str },
    #[error("product remainder reduction supports 2 or 8 columns, got {0}")]
    InvalidReductionColumns(usize),
    #[error("product remainder reduction needs at least one input")]
    EmptyReduction,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProductRemainderPhaseParams {
    pub(crate) source_elements: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProductRemainderOpeningParams {
    pub(crate) rows: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProductRemainderReductionParams {
    pub(crate) input_count: u32,
    pub(crate) output_count: u32,
    pub(crate) columns: u32,
    pub(crate) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::metal::solinas) struct ProductInstructionTerminalCacheParams {
    pub(in crate::metal::solinas) rows: u32,
    pub(in crate::metal::solinas) e_in_length: u32,
    pub(in crate::metal::solinas) e_out_length: u32,
    pub(in crate::metal::solinas) exceptions_per_group: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::metal::solinas) struct ProductInstructionTerminalCacheStats {
    pub(in crate::metal::solinas) capacity_bytes: u64,
    pub(in crate::metal::solinas) dense_bytes: u64,
    pub(in crate::metal::solinas) exception_count: usize,
    pub(in crate::metal::solinas) overflow_groups: usize,
    pub(in crate::metal::solinas) active_logical_bytes: u64,
}

pub(in crate::metal::solinas) struct ProductInstructionTerminalCacheOpenings {
    pub(in crate::metal::solinas) values: [AkitaField; PRODUCT_REMAINDER_OPENINGS],
    pub(in crate::metal::solinas) gpu_active: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProductInstructionTerminalCacheLayout {
    rows: usize,
    groups: usize,
    rows_per_group: usize,
    lookup_low_offset: usize,
    left_lookup_low_offset: usize,
    flags_offset: usize,
    counts_offset: usize,
    exceptions_offset: usize,
    exceptions_per_group: usize,
    capacity_bytes: usize,
}

struct ProductInstructionTerminalCache {
    layout: ProductInstructionTerminalCacheLayout,
    openings_pipeline: ComputePipelineState,
    stats: Option<ProductInstructionTerminalCacheStats>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProductRemainderBindRangeParams {
    source_offset: u32,
    destination_offset: u32,
    output_start: u32,
    output_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::metal::solinas) struct InPlaceBindRange {
    pub(in crate::metal::solinas) start: usize,
    pub(in crate::metal::solinas) count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::metal::solinas) struct InPlaceBindSchedule {
    pub(in crate::metal::solinas) prefix: usize,
    pub(in crate::metal::solinas) direct: Vec<InPlaceBindRange>,
}

const _: [(); 16] = [(); size_of::<ProductRemainderPhaseParams>()];
const _: [(); 16] = [(); size_of::<ProductRemainderOpeningParams>()];
const _: [(); 16] = [(); size_of::<ProductRemainderReductionParams>()];
const _: [(); 16] = [(); size_of::<ProductRemainderBindRangeParams>()];
const _: [(); 16] = [(); size_of::<ProductInstructionTerminalCacheParams>()];

impl ProductInstructionTerminalCacheLayout {
    fn new(
        rows: usize,
        initial_e_in_length: usize,
        groups: usize,
        capacity_bytes: usize,
    ) -> Result<Self, ProductRemainderShapeError> {
        let rows_per_group = initial_e_in_length.checked_mul(2).ok_or(
            ProductRemainderShapeError::ByteLengthOverflow {
                name: "terminal cache rows per group",
            },
        )?;
        if groups == 0
            || rows_per_group == 0
            || groups.checked_mul(rows_per_group) != Some(rows)
            || rows_per_group >= (1usize << 31)
        {
            return Err(ProductRemainderShapeError::WeightShape {
                phase: "terminal cache",
                expected: rows,
                e_in: initial_e_in_length,
                e_out: groups,
            });
        }
        let lookup_low_offset = 0;
        let left_lookup_low_offset = rows.checked_mul(size_of::<u32>()).ok_or(
            ProductRemainderShapeError::ByteLengthOverflow {
                name: "terminal cache lookup low",
            },
        )?;
        let flags_offset = left_lookup_low_offset
            .checked_add(rows.checked_mul(size_of::<u32>()).ok_or(
                ProductRemainderShapeError::ByteLengthOverflow {
                    name: "terminal cache left-lookup low",
                },
            )?)
            .ok_or(ProductRemainderShapeError::ByteLengthOverflow {
                name: "terminal cache flags offset",
            })?;
        let counts_offset = flags_offset.checked_add(rows).ok_or(
            ProductRemainderShapeError::ByteLengthOverflow {
                name: "terminal cache counts offset",
            },
        )?;
        if !counts_offset.is_multiple_of(align_of::<u32>()) {
            return Err(ProductRemainderShapeError::ByteLengthOverflow {
                name: "terminal cache counts alignment",
            });
        }
        let counts_end = counts_offset
            .checked_add(groups.checked_mul(size_of::<u32>()).ok_or(
                ProductRemainderShapeError::ByteLengthOverflow {
                    name: "terminal cache counts",
                },
            )?)
            .ok_or(ProductRemainderShapeError::ByteLengthOverflow {
                name: "terminal cache counts end",
            })?;
        let exceptions_offset =
            counts_end
                .checked_add(7)
                .ok_or(ProductRemainderShapeError::ByteLengthOverflow {
                    name: "terminal cache exception alignment",
                })?
                & !7;
        let exception_bytes = capacity_bytes.checked_sub(exceptions_offset).ok_or(
            ProductRemainderShapeError::StorageLength {
                name: "terminal cache",
                expected: exceptions_offset,
                got: capacity_bytes,
            },
        )?;
        let exceptions_per_group = exception_bytes / groups / size_of::<u64>();
        if exceptions_per_group == 0 || exceptions_per_group >= (1usize << 31) {
            return Err(ProductRemainderShapeError::ByteLengthOverflow {
                name: "terminal cache exception capacity",
            });
        }
        Ok(Self {
            rows,
            groups,
            rows_per_group,
            lookup_low_offset,
            left_lookup_low_offset,
            flags_offset,
            counts_offset,
            exceptions_offset,
            exceptions_per_group,
            capacity_bytes,
        })
    }

    fn params(
        self,
        rows: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<ProductInstructionTerminalCacheParams, MetalError> {
        Ok(ProductInstructionTerminalCacheParams {
            rows: u32::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?,
            e_in_length: u32::try_from(e_in_length)
                .map_err(|_| MetalError::InputTooLong(e_in_length))?,
            e_out_length: u32::try_from(e_out_length)
                .map_err(|_| MetalError::InputTooLong(e_out_length))?,
            exceptions_per_group: u32::try_from(self.exceptions_per_group)
                .map_err(|_| MetalError::InputTooLong(self.exceptions_per_group))?,
        })
    }

    fn dense_bytes(self) -> usize {
        self.exceptions_offset
    }
}

impl ProductRemainderBindRangeParams {
    fn new(
        source_offset: usize,
        destination_offset: usize,
        range: InPlaceBindRange,
    ) -> Result<Self, ProductRemainderShapeError> {
        Ok(Self {
            source_offset: shader_count("in-place source offset", source_offset)?,
            destination_offset: shader_count("in-place destination offset", destination_offset)?,
            output_start: shader_count("in-place output start", range.start)?,
            output_count: shader_count("in-place output count", range.count)?,
        })
    }
}

pub(in crate::metal::solinas) fn in_place_bind_schedule(
    bound_elements: usize,
    scratch_capacity: usize,
) -> InPlaceBindSchedule {
    let prefix = bound_elements.min(scratch_capacity);
    let mut direct = Vec::new();
    let mut start = prefix;
    while start < bound_elements {
        let end = start.saturating_mul(2).min(bound_elements);
        direct.push(InPlaceBindRange {
            start,
            count: end - start,
        });
        start = end;
    }
    InPlaceBindSchedule { prefix, direct }
}

impl ProductRemainderPhaseParams {
    pub(crate) fn materialize(
        rows: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, ProductRemainderShapeError> {
        validate_rows(rows)?;
        validate_weight_shape("materialize", rows / 2, e_in_length, e_out_length)?;
        validate_partial_index(PRODUCT_REMAINDER_MESSAGE_COLUMNS, e_out_length)?;
        Self::new(rows, e_in_length, e_out_length)
    }

    pub(crate) fn transition(
        source_elements: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, ProductRemainderShapeError> {
        if source_elements < 4 || !source_elements.is_power_of_two() {
            return Err(ProductRemainderShapeError::InvalidTransitionSource(
                source_elements,
            ));
        }
        validate_rows(source_elements)?;
        validate_weight_shape("transition", source_elements / 4, e_in_length, e_out_length)?;
        validate_partial_index(PRODUCT_REMAINDER_MESSAGE_COLUMNS, e_out_length)?;
        Self::new(source_elements, e_in_length, e_out_length)
    }

    fn new(
        source_elements: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, ProductRemainderShapeError> {
        Ok(Self {
            source_elements: shader_count("source state", source_elements)?,
            e_in_length: shader_count("e_in", e_in_length)?,
            e_out_length: shader_count("e_out", e_out_length)?,
            _reserved: 0,
        })
    }
}

impl ProductRemainderOpeningParams {
    pub(crate) fn new(
        rows: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, ProductRemainderShapeError> {
        validate_rows(rows)?;
        validate_weight_shape("openings", rows, e_in_length, e_out_length)?;
        validate_partial_index(PRODUCT_REMAINDER_OPENINGS, e_out_length)?;
        Ok(Self {
            rows: shader_count("rows", rows)?,
            e_in_length: shader_count("e_in", e_in_length)?,
            e_out_length: shader_count("e_out", e_out_length)?,
            _reserved: 0,
        })
    }
}

impl ProductRemainderReductionParams {
    pub(crate) fn new(
        input_count: usize,
        columns: usize,
    ) -> Result<Self, ProductRemainderShapeError> {
        if !matches!(
            columns,
            PRODUCT_REMAINDER_MESSAGE_COLUMNS | PRODUCT_REMAINDER_OPENINGS
        ) {
            return Err(ProductRemainderShapeError::InvalidReductionColumns(columns));
        }
        if input_count == 0 {
            return Err(ProductRemainderShapeError::EmptyReduction);
        }
        let output_count = input_count.div_ceil(PRODUCT_REMAINDER_SIMD_WIDTH);
        validate_partial_index(columns, input_count)?;
        validate_partial_index(columns, output_count)?;
        Ok(Self {
            input_count: shader_count("reduction input", input_count)?,
            output_count: shader_count("reduction output", output_count)?,
            columns: shader_count("reduction columns", columns)?,
            _reserved: 0,
        })
    }
}

/// Buffer capacities for a resident product-remainder sequence.
///
/// State A contains two full row-count tables. State B contains the two
/// half-sized tables written by the first bind. Later rounds fit in the same
/// buffers. Both partial buffers use the eight-opening width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductRemainderStorageLayout {
    rows: usize,
    row_bytes: usize,
    state_a_fields: usize,
    state_b_fields: usize,
    e_in_fields: usize,
    e_out_fields: usize,
    partial_fields: usize,
    workspace_bytes: usize,
    resident_bytes: usize,
}

impl ProductRemainderStorageLayout {
    pub fn new(
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
    ) -> Result<Self, ProductRemainderShapeError> {
        validate_rows(rows)?;
        let state_a_fields = checked_product("state A", 2, rows)?;
        let state_b_fields = rows;
        let partial_fields =
            checked_product("partial buffer", PRODUCT_REMAINDER_OPENINGS, e_out_capacity)?;
        for (name, fields) in [
            ("state A", state_a_fields),
            ("state B", state_b_fields),
            ("e_in", e_in_capacity),
            ("e_out", e_out_capacity),
            ("partial buffer", partial_fields),
        ] {
            let _ = shader_count(name, fields)?;
        }
        let covered = e_in_capacity.checked_mul(e_out_capacity).ok_or(
            ProductRemainderShapeError::ByteLengthOverflow {
                name: "weight capacity",
            },
        )?;
        if e_in_capacity == 0 || e_out_capacity == 0 || covered < rows {
            return Err(ProductRemainderShapeError::WeightShape {
                phase: "storage capacity",
                expected: rows,
                e_in: e_in_capacity,
                e_out: e_out_capacity,
            });
        }

        let workspace_fields = [
            3,
            state_a_fields,
            state_b_fields,
            e_in_capacity,
            e_out_capacity,
            partial_fields,
            partial_fields,
        ]
        .into_iter()
        .try_fold(0usize, |sum, fields| sum.checked_add(fields))
        .ok_or(ProductRemainderShapeError::ByteLengthOverflow { name: "workspace" })?;
        let workspace_bytes =
            checked_product("workspace", workspace_fields, size_of::<super::Fp128>())?;
        let row_bytes = checked_product("rows", rows, size_of::<ProductRemainderRow>())?;
        let resident_bytes = workspace_bytes.checked_add(row_bytes).ok_or(
            ProductRemainderShapeError::ByteLengthOverflow {
                name: "resident set",
            },
        )?;

        Ok(Self {
            rows,
            row_bytes,
            state_a_fields,
            state_b_fields,
            e_in_fields: e_in_capacity,
            e_out_fields: e_out_capacity,
            partial_fields,
            workspace_bytes,
            resident_bytes,
        })
    }

    copy_field_getters! { pub, {
        rows: usize,
        row_bytes: usize,
        state_a_fields: usize,
        state_b_fields: usize,
        e_in_fields: usize,
        e_out_fields: usize,
        partial_fields: usize,
        workspace_bytes: usize,
        resident_bytes: usize,
    }}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProductRemainderPhase {
    Ready,
    Materialized,
    TransitionRetired,
}

struct ProductRemainderBuffers {
    rows: ProductRemainderRows,
    lagrange: Buffer,
    state_a: Buffer,
    state_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub struct ProductRemainderSequence {
    context: SolinasMetal,
    uniskip_pipeline: ComputePipelineState,
    materialize_pipeline: ComputePipelineState,
    transition_pipeline: ComputePipelineState,
    in_place_bind_pipeline: ComputePipelineState,
    in_place_copy_pipeline: ComputePipelineState,
    in_place_message_pipeline: ComputePipelineState,
    openings_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    buffers: ProductRemainderBuffers,
    state_a_borrowed: bool,
    layout: ProductRemainderStorageLayout,
    uniskip_threads_per_threadgroup: usize,
    materialize_threads_per_threadgroup: usize,
    transition_threads_per_threadgroup: usize,
    openings_threads_per_threadgroup: usize,
    prime_workspace: bool,
    state_b_fill_queue: Option<CommandQueue>,
    terminal_cache: Option<ProductInstructionTerminalCache>,
    current_elements: usize,
    source_in_a: bool,
    phase: ProductRemainderPhase,
}

struct ProductRemainderInitialMessageCommand {
    command_buffer: CommandBuffer,
    state_b_fill: Option<CommandBuffer>,
    output: Buffer,
    sequence_identity: usize,
}

#[must_use = "a submitted product-remainder message must be joined"]
pub(crate) struct PendingProductRemainderInitialMessage {
    sequence: Option<ProductRemainderSequence>,
    command: Option<ProductRemainderInitialMessageCommand>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for PendingProductRemainderInitialMessage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        visitor.exit();
    }
}

impl Drop for PendingProductRemainderInitialMessage {
    fn drop(&mut self) {
        if let Some(command) = &self.command {
            command.command_buffer.wait_until_completed();
            if let Some(state_b_fill) = &command.state_b_fill {
                state_b_fill.wait_until_completed();
            }
        }
    }
}

impl PendingProductRemainderInitialMessage {
    pub(crate) fn join(
        mut self,
    ) -> Result<
        (
            ProductRemainderSequence,
            [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
            Duration,
        ),
        MetalError,
    > {
        let mut sequence = self
            .sequence
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "the pending first message lost its resident sequence",
            ))?;
        if !sequence.is_ready() {
            return Err(MetalError::InvalidProductRemainderState(
                "the pending first message lost its ready sequence state",
            ));
        }
        let command = self
            .command
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "the pending first message lost its command buffer",
            ))?;
        let (message, gpu_active) = sequence.complete_initial_message(command)?;
        Ok((sequence, message, gpu_active))
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for ProductRemainderSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("rows"), &self.buffers.rows);
        visitor.visit_simple(
            allocative::Key::new("device_workspace"),
            self.layout.workspace_bytes(),
        );
        visitor.exit();
    }
}

impl SolinasMetal {
    pub fn prepare_product_remainder_rows(
        &self,
        rows: &[ProductRemainderRow],
    ) -> Result<ProductRemainderRows, MetalError> {
        validate_rows(rows.len())?;
        for (index, row) in rows.iter().copied().enumerate() {
            row.validate()
                .map_err(|source| MetalError::InvalidProductRemainderRow { index, source })?;
        }
        let row_bytes = checked_product("rows", rows.len(), size_of::<ProductRemainderRow>())?;
        let row_bytes =
            u64::try_from(row_bytes).map_err(|_| MetalError::InputTooLong(rows.len()))?;
        self.validate_buffer_length(row_bytes)?;
        self.validate_additional_working_set(row_bytes)?;
        Ok(ProductRemainderRows {
            source: ProductRemainderSource::Packed(buffer_from_slice(&self.device, rows)),
            len: rows.len(),
            device_registry_id: self.device_registry_id(),
        })
    }

    pub fn prepare_product_remainder_sequence(
        &self,
        rows: &[ProductRemainderRow],
        lagrange_weights: [AkitaField; 3],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: ProductRemainderSequenceConfig,
    ) -> Result<ProductRemainderSequence, MetalError> {
        let rows = self.prepare_product_remainder_rows(rows)?;
        self.prepare_product_remainder_sequence_with_rows(
            rows,
            lagrange_weights,
            e_in_capacity,
            e_out_capacity,
            config,
        )
    }

    pub fn prepare_product_remainder_sequence_with_rows(
        &self,
        rows: ProductRemainderRows,
        lagrange_weights: [AkitaField; 3],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: ProductRemainderSequenceConfig,
    ) -> Result<ProductRemainderSequence, MetalError> {
        self.prepare_product_remainder_sequence_with_rows_and_state_a(
            rows,
            lagrange_weights,
            e_in_capacity,
            e_out_capacity,
            config,
            None,
        )
    }

    pub(crate) fn prepare_product_remainder_sequence_with_rows_and_state_a(
        &self,
        rows: ProductRemainderRows,
        lagrange_weights: [AkitaField; 3],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: ProductRemainderSequenceConfig,
        state_a: Option<Buffer>,
    ) -> Result<ProductRemainderSequence, MetalError> {
        if rows.device_registry_id() != self.device_registry_id() {
            return Err(MetalError::ProductRemainderRowsDevice {
                expected: self.device_registry_id(),
                got: rows.device_registry_id(),
            });
        }
        let row_count = rows.len();
        let layout = ProductRemainderStorageLayout::new(row_count, e_in_capacity, e_out_capacity)?;
        let state_a_bytes = layout
            .state_a_fields()
            .checked_mul(size_of::<Fp128>())
            .ok_or(MetalError::InputTooLong(layout.state_a_fields()))?;
        if let Some(state_a) = &state_a {
            let expected = u64::try_from(state_a_bytes)
                .map_err(|_| MetalError::InputTooLong(state_a_bytes))?;
            if state_a.length() != expected {
                return Err(MetalError::InvalidProductRemainderState(
                    "borrowed product state A has the wrong length",
                ));
            }
        }
        let workspace_bytes = layout
            .workspace_bytes()
            .checked_sub(usize::from(state_a.is_some()) * state_a_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(layout.workspace_bytes()))?;
        self.validate_additional_working_set(workspace_bytes)?;

        let stage1_source = rows.source_kind() == ProductRemainderSourceKind::SpartanStage1;
        let uniskip_pipeline_name = if stage1_source {
            PRODUCT_UNISKIP_STAGE1_PIPELINE
        } else {
            PRODUCT_UNISKIP_PIPELINE
        };
        let materialize_pipeline_name = if stage1_source {
            MATERIALIZE_STAGE1_PIPELINE
        } else {
            MATERIALIZE_PIPELINE
        };
        let openings_pipeline_name = if stage1_source {
            OPENINGS_STAGE1_PIPELINE
        } else {
            OPENINGS_PIPELINE
        };
        let uniskip_pipeline = self.compile_named_pipeline(uniskip_pipeline_name)?;
        let materialize_pipeline = self.compile_named_pipeline(materialize_pipeline_name)?;
        let transition_pipeline = self.compile_named_pipeline(TRANSITION_PIPELINE)?;
        let in_place_bind_pipeline = self.compile_named_pipeline(IN_PLACE_BIND_PIPELINE)?;
        let in_place_copy_pipeline = self.compile_named_pipeline(IN_PLACE_COPY_PIPELINE)?;
        let in_place_message_pipeline = self.compile_named_pipeline(IN_PLACE_MESSAGE_PIPELINE)?;
        let openings_pipeline = self.compile_named_pipeline(openings_pipeline_name)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let uniskip_limits = Self::limits(&uniskip_pipeline);
        let materialize_limits = Self::limits(&materialize_pipeline);
        let transition_limits = Self::limits(&transition_pipeline);
        let in_place_bind_limits = Self::limits(&in_place_bind_pipeline);
        let in_place_copy_limits = Self::limits(&in_place_copy_pipeline);
        let in_place_message_limits = Self::limits(&in_place_message_pipeline);
        let openings_limits = Self::limits(&openings_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        if uniskip_limits.thread_execution_width != PRODUCT_UNISKIP_SIMD_WIDTH {
            return Err(MetalError::UnsupportedProductUniskipExecutionWidth {
                pipeline: PRODUCT_UNISKIP_PIPELINE,
                expected: PRODUCT_UNISKIP_SIMD_WIDTH,
                got: uniskip_limits.thread_execution_width,
            });
        }
        for (pipeline, limits) in [
            (materialize_pipeline_name, materialize_limits),
            (TRANSITION_PIPELINE, transition_limits),
            (IN_PLACE_BIND_PIPELINE, in_place_bind_limits),
            (IN_PLACE_COPY_PIPELINE, in_place_copy_limits),
            (IN_PLACE_MESSAGE_PIPELINE, in_place_message_limits),
            (openings_pipeline_name, openings_limits),
            (REDUCTION_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != PRODUCT_REMAINDER_SIMD_WIDTH {
                return Err(MetalError::UnsupportedProductRemainderExecutionWidth {
                    pipeline,
                    expected: PRODUCT_REMAINDER_SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }

        let uniskip_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.uniskip_threads_per_threadgroup,
            uniskip_limits,
        )?;
        let materialize_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.materialize_threads_per_threadgroup,
            materialize_limits,
        )?;
        let transition_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.transition_threads_per_threadgroup,
            transition_limits,
        )?;
        for limits in [
            in_place_bind_limits,
            in_place_copy_limits,
            in_place_message_limits,
        ] {
            let resolved =
                Self::resolve_threadgroup_width(Some(transition_threads_per_threadgroup), limits)?;
            if resolved != transition_threads_per_threadgroup {
                return Err(MetalError::InvalidProductRemainderState(
                    "in-place transition threadgroup width differs from the dense path",
                ));
            }
        }
        let openings_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.openings_threads_per_threadgroup,
            openings_limits,
        )?;
        let lagrange = lagrange_weights
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        self.validate_inputs("product remainder lagrange weights", &lagrange)?;

        let state_a_borrowed = state_a.is_some();
        let state_a = match state_a {
            Some(state_a) => state_a,
            None => self.new_product_remainder_buffer(layout.state_a_fields())?,
        };
        let state_b = self.new_product_remainder_buffer(layout.state_b_fields())?;
        if state_a.as_ptr() == state_b.as_ptr() {
            return Err(MetalError::InvalidProductRemainderState(
                "product state buffers alias",
            ));
        }

        Ok(ProductRemainderSequence {
            context: self.clone(),
            uniskip_pipeline,
            materialize_pipeline,
            transition_pipeline,
            in_place_bind_pipeline,
            in_place_copy_pipeline,
            in_place_message_pipeline,
            openings_pipeline,
            reduction_pipeline,
            buffers: ProductRemainderBuffers {
                rows,
                lagrange: buffer_from_slice(&self.device, &lagrange),
                state_a,
                state_b,
                e_in: self.new_product_remainder_buffer(layout.e_in_fields())?,
                e_out: self.new_product_remainder_buffer(layout.e_out_fields())?,
                partial_a: self.new_product_remainder_buffer(layout.partial_fields())?,
                partial_b: self.new_product_remainder_buffer(layout.partial_fields())?,
            },
            state_a_borrowed,
            layout,
            uniskip_threads_per_threadgroup,
            materialize_threads_per_threadgroup,
            transition_threads_per_threadgroup,
            openings_threads_per_threadgroup,
            prime_workspace: config.prime_workspace,
            state_b_fill_queue: config
                .async_state_b_fill
                .then(|| self.device.new_command_queue()),
            terminal_cache: None,
            current_elements: row_count,
            source_in_a: true,
            phase: ProductRemainderPhase::Ready,
        })
    }

    fn new_product_remainder_buffer(&self, fields: usize) -> Result<Buffer, MetalError> {
        let bytes = fields
            .checked_mul(size_of::<Fp128>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(fields))?;
        self.validate_buffer_length(bytes)?;
        Ok(self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
    }
}

impl ProductRemainderSequence {
    pub(in crate::metal::solinas) fn encode_joint_stage1_materialize(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<ProductRemainderPhaseParams, MetalError> {
        if self.phase != ProductRemainderPhase::Ready {
            return Err(MetalError::InvalidProductRemainderState(
                "joint materialization requires a ready product sequence",
            ));
        }
        let params =
            ProductRemainderPhaseParams::materialize(self.layout.rows(), e_in.len(), e_out.len())?;
        let Some((compact, residual)) = self.buffers.rows.stage1_buffers() else {
            return Err(MetalError::InvalidProductRemainderState(
                "joint materialization requires resident Stage-1 rows",
            ));
        };
        self.write_weights(e_in, e_out)?;
        encoder.set_buffer(0, Some(compact), 0);
        encoder.set_buffer(1, Some(residual), 0);
        encoder.set_buffer(2, Some(&self.buffers.lagrange), 0);
        encoder.set_buffer(4, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(5, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(6, Some(&self.buffers.state_a), 0);
        encoder.set_buffer(8, Some(&self.buffers.partial_a), 0);
        Ok(params)
    }

    pub(in crate::metal::solinas) fn encode_joint_initial_reductions(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        e_out_length: usize,
    ) -> Result<Buffer, MetalError> {
        let final_in_a = encode_product_remainder_reductions(
            encoder,
            &self.reduction_pipeline,
            &self.buffers.partial_a,
            &self.buffers.partial_b,
            e_out_length,
            PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        )?;
        Ok(if final_in_a {
            self.buffers.partial_a.clone()
        } else {
            self.buffers.partial_b.clone()
        })
    }

    pub(in crate::metal::solinas) fn complete_joint_materialize(
        &mut self,
    ) -> Result<(), MetalError> {
        if self.phase != ProductRemainderPhase::Ready {
            return Err(MetalError::InvalidProductRemainderState(
                "joint product state was materialized more than once",
            ));
        }
        self.current_elements = self.layout.rows();
        self.source_in_a = true;
        self.phase = ProductRemainderPhase::Materialized;
        Ok(())
    }

    pub(in crate::metal::solinas) fn encode_joint_transition(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<Buffer, MetalError> {
        if self.phase != ProductRemainderPhase::Materialized || self.current_elements < 4 {
            return Err(MetalError::InvalidProductRemainderState(
                "a joint transition needs a materialized product state",
            ));
        }
        if !self.source_in_a {
            return Err(MetalError::InvalidProductRemainderState(
                "joint in-place product state left state A",
            ));
        }
        let params = ProductRemainderPhaseParams::transition(
            self.current_elements,
            e_in.len(),
            e_out.len(),
        )?;
        self.write_weights(e_in, e_out)?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("joint product remainder challenge", &[challenge])?;
        let state = &self.buffers.state_a;
        let scratch = &self.buffers.partial_b;
        let bound_elements = self.current_elements / 2;
        let schedule = in_place_bind_schedule(bound_elements, self.layout.partial_fields());
        if schedule.prefix == 0 {
            return Err(MetalError::InvalidProductRemainderState(
                "joint in-place product scratch is empty",
            ));
        }

        encoder.set_compute_pipeline_state(&self.in_place_bind_pipeline);
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(scratch), 0);
        set_inline_bytes(encoder, 2, &challenge);
        let prefix = InPlaceBindRange {
            start: 0,
            count: schedule.prefix,
        };
        let prefix_params = ProductRemainderBindRangeParams::new(0, 0, prefix)?;
        set_inline_bytes(encoder, 3, &prefix_params);
        dispatch_product_remainder_linear(
            encoder,
            prefix.count,
            self.transition_threads_per_threadgroup,
        );
        encoder.memory_barrier_with_resources(&[&**state, &**scratch]);

        encoder.set_compute_pipeline_state(&self.in_place_copy_pipeline);
        encoder.set_buffer(0, Some(scratch), 0);
        encoder.set_buffer(1, Some(state), 0);
        let prefix_count = shader_count("in-place prefix", schedule.prefix)?;
        set_inline_bytes(encoder, 2, &prefix_count);
        dispatch_product_remainder_linear(
            encoder,
            schedule.prefix,
            self.transition_threads_per_threadgroup,
        );
        encoder.memory_barrier_with_resources(&[&**state, &**scratch]);

        encoder.set_compute_pipeline_state(&self.in_place_bind_pipeline);
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(state), 0);
        set_inline_bytes(encoder, 2, &challenge);
        for range in schedule.direct {
            let range_params = ProductRemainderBindRangeParams::new(0, 0, range)?;
            set_inline_bytes(encoder, 3, &range_params);
            dispatch_product_remainder_linear(
                encoder,
                range.count,
                self.transition_threads_per_threadgroup,
            );
            encoder.memory_barrier_with_resources(&[&**state]);
        }

        let right = InPlaceBindRange {
            start: 0,
            count: bound_elements,
        };
        let right_params =
            ProductRemainderBindRangeParams::new(self.current_elements, bound_elements, right)?;
        set_inline_bytes(encoder, 3, &right_params);
        dispatch_product_remainder_linear(
            encoder,
            right.count,
            self.transition_threads_per_threadgroup,
        );
        encoder.memory_barrier_with_resources(&[&**state]);

        encoder.set_compute_pipeline_state(&self.in_place_message_pipeline);
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
        encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
        encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
        set_inline_bytes(encoder, 4, &params);
        encoder.set_threadgroup_memory_length(
            0,
            product_remainder_threadgroup_bytes(
                PRODUCT_REMAINDER_MESSAGE_COLUMNS,
                self.transition_threads_per_threadgroup,
            ) as u64,
        );
        dispatch_product_remainder_blocks(
            encoder,
            e_out.len(),
            self.transition_threads_per_threadgroup,
        );
        let final_in_a = encode_product_remainder_reductions(
            encoder,
            &self.reduction_pipeline,
            &self.buffers.partial_a,
            &self.buffers.partial_b,
            e_out.len(),
            PRODUCT_REMAINDER_MESSAGE_COLUMNS,
        )?;
        Ok(if final_in_a {
            self.buffers.partial_a.clone()
        } else {
            self.buffers.partial_b.clone()
        })
    }

    pub(in crate::metal::solinas) fn complete_joint_transition(
        &mut self,
    ) -> Result<(), MetalError> {
        if self.phase != ProductRemainderPhase::Materialized || self.current_elements < 4 {
            return Err(MetalError::InvalidProductRemainderState(
                "joint product transition completed in the wrong phase",
            ));
        }
        self.current_elements /= 2;
        Ok(())
    }

    pub(in crate::metal::solinas) const fn joint_materialize_threads_per_threadgroup(
        &self,
    ) -> usize {
        self.materialize_threads_per_threadgroup
    }

    pub(in crate::metal::solinas) fn joint_stage1_allocation_identities(
        &self,
    ) -> Option<[usize; 2]> {
        let (compact, residual) = self.buffers.rows.stage1_buffers()?;
        Some([compact.as_ptr() as usize, residual.as_ptr() as usize])
    }

    pub(in crate::metal::solinas) const fn context(&self) -> &SolinasMetal {
        &self.context
    }

    pub(in crate::metal::solinas) fn enable_joint_terminal_cache(
        &mut self,
        initial_e_in_length: usize,
        e_out_length: usize,
    ) -> Result<(), MetalError> {
        if !self.is_ready()
            || self.terminal_cache.is_some()
            || self.buffers.rows.source_kind() != ProductRemainderSourceKind::SpartanStage1
        {
            return Err(MetalError::InvalidProductRemainderState(
                "joint terminal cache requires a ready Stage-1 product sequence",
            ));
        }
        let capacity_bytes = usize::try_from(self.buffers.state_b.length())
            .map_err(|_| MetalError::InputTooLong(usize::MAX))?;
        let layout = ProductInstructionTerminalCacheLayout::new(
            self.layout.rows(),
            initial_e_in_length,
            e_out_length,
            capacity_bytes,
        )?;
        let openings_pipeline = self
            .context
            .compile_named_pipeline(TERMINAL_CACHE_OPENINGS_PIPELINE)?;
        let limits = SolinasMetal::limits(&openings_pipeline);
        if limits.thread_execution_width != PRODUCT_REMAINDER_SIMD_WIDTH {
            return Err(MetalError::UnsupportedProductRemainderExecutionWidth {
                pipeline: TERMINAL_CACHE_OPENINGS_PIPELINE,
                expected: PRODUCT_REMAINDER_SIMD_WIDTH,
                got: limits.thread_execution_width,
            });
        }
        let resolved = SolinasMetal::resolve_threadgroup_width(
            Some(self.openings_threads_per_threadgroup),
            limits,
        )?;
        if resolved != self.openings_threads_per_threadgroup {
            return Err(MetalError::InvalidProductRemainderState(
                "terminal cache opening threadgroup width differs from Product openings",
            ));
        }
        self.state_b_fill_queue = None;
        self.terminal_cache = Some(ProductInstructionTerminalCache {
            layout,
            openings_pipeline,
            stats: None,
        });
        Ok(())
    }

    pub(in crate::metal::solinas) fn encode_joint_terminal_cache(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Result<Option<ProductInstructionTerminalCacheParams>, MetalError> {
        let Some(cache) = &self.terminal_cache else {
            return Ok(None);
        };
        let layout = cache.layout;
        encoder.set_buffer(
            10,
            Some(&self.buffers.state_b),
            layout.lookup_low_offset as u64,
        );
        encoder.set_buffer(
            11,
            Some(&self.buffers.state_b),
            layout.left_lookup_low_offset as u64,
        );
        encoder.set_buffer(12, Some(&self.buffers.state_b), layout.flags_offset as u64);
        encoder.set_buffer(13, Some(&self.buffers.state_b), layout.counts_offset as u64);
        encoder.set_buffer(
            14,
            Some(&self.buffers.state_b),
            layout.exceptions_offset as u64,
        );
        layout
            .params(layout.rows, layout.rows_per_group / 2, layout.groups)
            .map(Some)
    }

    pub(in crate::metal::solinas) fn complete_joint_terminal_cache(
        &mut self,
    ) -> Result<Option<ProductInstructionTerminalCacheStats>, MetalError> {
        let Some(cache) = &self.terminal_cache else {
            return Ok(None);
        };
        let layout = cache.layout;
        let counts = unsafe {
            // SAFETY: the joint materialization command has completed and wrote
            // one counter for every cache equality block; layout construction
            // validates the byte offset's u32 alignment.
            slice::from_raw_parts(
                self.buffers
                    .state_b
                    .contents()
                    .cast::<u32>()
                    .add(layout.counts_offset / size_of::<u32>()),
                layout.groups,
            )
        };
        let mut exception_count = 0usize;
        let mut overflow_groups = 0usize;
        for &encoded in counts {
            overflow_groups += usize::from(encoded & (1 << 31) != 0);
            exception_count = exception_count
                .checked_add((encoded & !(1 << 31)) as usize)
                .ok_or(MetalError::InputTooLong(usize::MAX))?;
        }
        let dense_bytes = u64::try_from(layout.dense_bytes())
            .map_err(|_| MetalError::InputTooLong(layout.dense_bytes()))?;
        let exception_bytes = exception_count
            .checked_mul(size_of::<u64>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(exception_count))?;
        let stats = ProductInstructionTerminalCacheStats {
            capacity_bytes: u64::try_from(layout.capacity_bytes)
                .map_err(|_| MetalError::InputTooLong(layout.capacity_bytes))?,
            dense_bytes,
            exception_count,
            overflow_groups,
            active_logical_bytes: dense_bytes
                .checked_add(exception_bytes)
                .ok_or(MetalError::InputTooLong(exception_count))?,
        };
        if overflow_groups == 0 {
            if let Some(cache) = &mut self.terminal_cache {
                cache.stats = Some(stats);
            }
        } else {
            let _ = self.release_terminal_cache_buffer()?;
        }
        Ok(Some(stats))
    }

    #[cfg(test)]
    pub(in crate::metal::solinas) const fn joint_state_b_buffer(&self) -> &Buffer {
        &self.buffers.state_b
    }

    pub(in crate::metal::solinas) fn release_joint_alternate(&mut self) -> Result<u64, MetalError> {
        if !self.is_ready() || self.terminal_cache.is_some() {
            return Err(MetalError::InvalidProductRemainderState(
                "joint alternate release requires a ready sequence",
            ));
        }
        let expected_bytes = self
            .layout
            .state_b_fields()
            .checked_mul(size_of::<Fp128>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InputTooLong(self.layout.state_b_fields()))?;
        if self.buffers.state_b.length() != expected_bytes {
            return Err(MetalError::InvalidProductRemainderState(
                "joint product alternate has already been released or has the wrong size",
            ));
        }
        let tombstone = self
            .context
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        let alternate = mem::replace(&mut self.buffers.state_b, tombstone);
        self.state_b_fill_queue = None;
        let released_bytes = alternate.length();
        drop(alternate);
        Ok(released_bytes)
    }

    fn release_terminal_cache_buffer(&mut self) -> Result<u64, MetalError> {
        let cache = self
            .terminal_cache
            .take()
            .ok_or(MetalError::InvalidProductRemainderState(
                "terminal cache was released without an active layout",
            ))?;
        let expected = u64::try_from(cache.layout.capacity_bytes)
            .map_err(|_| MetalError::InputTooLong(cache.layout.capacity_bytes))?;
        if self.buffers.state_b.length() != expected {
            return Err(MetalError::InvalidProductRemainderState(
                "terminal cache buffer has the wrong capacity",
            ));
        }
        let tombstone = self
            .context
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        let cache_buffer = mem::replace(&mut self.buffers.state_b, tombstone);
        let bytes = cache_buffer.length();
        drop(cache_buffer);
        Ok(bytes)
    }

    pub(in crate::metal::solinas) fn submit_state_b_fill(&self) -> Option<CommandBuffer> {
        self.state_b_fill_queue.as_ref().map(|queue| {
            let command = queue.new_command_buffer().to_owned();
            let encoder = command.new_blit_command_encoder();
            encoder.fill_buffer(
                &self.buffers.state_b,
                NSRange::new(0, self.buffers.state_b.length()),
                0,
            );
            encoder.end_encoding();
            command.commit();
            command
        })
    }

    pub(crate) fn prime_workspace(&self) -> Result<(), MetalError> {
        if !self.is_ready() {
            return Err(MetalError::InvalidProductRemainderState(
                "workspace priming requires a ready sequence",
            ));
        }
        if !self.prime_workspace {
            return Ok(());
        }
        let state_a_identity = self.buffers.state_a.as_ptr() as usize;
        let state_b_identity = self.buffers.state_b.as_ptr() as usize;
        if state_a_identity == state_b_identity {
            return Err(MetalError::InvalidProductRemainderState(
                "workspace state buffers alias",
            ));
        }
        let buffers = if self.state_a_borrowed {
            [&self.buffers.state_b, &self.buffers.state_b][..1].to_vec()
        } else {
            [&self.buffers.state_a, &self.buffers.state_b].to_vec()
        };
        let command_buffer = self.context.queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_blit_command_encoder();
            for buffer in &buffers {
                encoder.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        validate_completed_command(command_buffer)
    }

    pub(crate) fn share_outer_state_b(&self) -> Result<Buffer, MetalError> {
        if !self.is_ready() {
            return Err(MetalError::InvalidProductRemainderState(
                "Outer state-B sharing requires a ready sequence",
            ));
        }
        Ok(self.buffers.state_b.clone())
    }

    pub(crate) fn set_lagrange_weights(
        &mut self,
        weights: [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS + 1],
    ) -> Result<(), MetalError> {
        write_product_remainder_fields(
            &self.buffers.lagrange,
            PRODUCT_REMAINDER_MESSAGE_COLUMNS + 1,
            &weights,
            "Lagrange weights",
        )?;
        Ok(())
    }

    pub fn uniskip_message_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(ProductUniskipExtendedNodes<AkitaField>, Duration), MetalError> {
        if !self.is_ready() {
            return Err(MetalError::InvalidProductRemainderState(
                "product uni-skip requires a ready resident sequence",
            ));
        }
        let params = ProductUniskipBlockParams::new(self.layout.rows(), e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        let (values, active_time) = autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.uniskip_pipeline);
            if let Some((compact, residual)) = self.buffers.rows.stage1_buffers() {
                encoder.set_buffer(0, Some(compact), 0);
                encoder.set_buffer(1, Some(residual), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 5, &params);
            } else {
                let rows = self.buffers.rows.packed_buffer().ok_or(
                    MetalError::InvalidProductRemainderState(
                        "packed product uni-skip lost its row buffer",
                    ),
                )?;
                encoder.set_buffer(0, Some(rows), 0);
                encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 4, &params);
            }
            encoder.set_threadgroup_memory_length(
                0,
                product_remainder_threadgroup_bytes(
                    PRODUCT_UNISKIP_EXTENDED_NODES,
                    self.uniskip_threads_per_threadgroup,
                ) as u64,
            );
            dispatch_product_remainder_blocks(
                encoder,
                e_out.len(),
                self.uniskip_threads_per_threadgroup,
            );
            let final_in_a = encode_product_remainder_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                PRODUCT_UNISKIP_EXTENDED_NODES,
            )?;
            encoder.end_encoding();
            finish_product_remainder_command::<PRODUCT_UNISKIP_EXTENDED_NODES>(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                "product uni-skip endpoints",
            )
        })?;
        Ok((
            ProductUniskipExtendedNodes {
                minus_two: values[0],
                plus_two: values[1],
            },
            active_time,
        ))
    }

    pub fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], MetalError> {
        self.message_timed(e_in, e_out).map(|(message, _)| message)
    }

    pub fn message_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        if self.phase != ProductRemainderPhase::Ready {
            return Err(MetalError::InvalidProductRemainderState(
                "the materialization message was already emitted",
            ));
        }
        let (message, active_time) = self.execute_materialize_message(e_in, e_out)?;
        self.phase = ProductRemainderPhase::Materialized;
        Ok((message, active_time))
    }

    /// Materializes the initial protocol state in the existing buffers.
    #[doc(hidden)]
    pub fn restart_message_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        let command = self.submit_materialize_message_command(e_in, e_out)?;
        self.complete_initial_message(command)
    }

    pub(crate) fn submit_initial_message(
        self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<PendingProductRemainderInitialMessage, MetalError> {
        if !self.is_ready() {
            return Err(MetalError::InvalidProductRemainderState(
                "product-remainder prefetch requires a ready sequence",
            ));
        }
        let command = self.submit_materialize_message_command(e_in, e_out)?;
        Ok(PendingProductRemainderInitialMessage {
            sequence: Some(self),
            command: Some(command),
        })
    }

    fn complete_initial_message(
        &mut self,
        command: ProductRemainderInitialMessageCommand,
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        if command.sequence_identity != self.buffers.state_a.as_ptr() as usize {
            return Err(MetalError::InvalidProductRemainderState(
                "the pending first message belongs to a different product sequence",
            ));
        }
        command.command_buffer.wait_until_completed();
        let mut gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        if let Some(state_b_fill) = &command.state_b_fill {
            state_b_fill.wait_until_completed();
            gpu_active += completed_command_gpu_time(state_b_fill)?;
        }
        // SAFETY: completion makes the two reduced fields at the front of the
        // selected shared output buffer visible to the host.
        let values = unsafe {
            slice::from_raw_parts(
                command.output.contents().cast::<Fp128>(),
                PRODUCT_REMAINDER_MESSAGE_COLUMNS,
            )
        };
        self.context
            .validate_inputs("product remainder first message", values)?;
        let message = std::array::from_fn(|index| values[index].into_jolt_field());
        self.current_elements = self.layout.rows();
        self.source_in_a = true;
        self.phase = ProductRemainderPhase::Materialized;
        Ok((message, gpu_active))
    }

    fn execute_materialize_message(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        let command = self.submit_materialize_message_command(e_in, e_out)?;
        command.command_buffer.wait_until_completed();
        let mut gpu_active = completed_command_gpu_time(&command.command_buffer)?;
        if let Some(state_b_fill) = &command.state_b_fill {
            state_b_fill.wait_until_completed();
            gpu_active += completed_command_gpu_time(state_b_fill)?;
        }
        // SAFETY: the completed reduction leaves two fields at the front of
        // the selected shared output buffer.
        let values = unsafe {
            slice::from_raw_parts(
                command.output.contents().cast::<Fp128>(),
                PRODUCT_REMAINDER_MESSAGE_COLUMNS,
            )
        };
        self.context
            .validate_inputs("product remainder first message", values)?;
        Ok((
            std::array::from_fn(|index| values[index].into_jolt_field()),
            gpu_active,
        ))
    }

    fn submit_materialize_message_command(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<ProductRemainderInitialMessageCommand, MetalError> {
        let params =
            ProductRemainderPhaseParams::materialize(self.layout.rows(), e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer().to_owned();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.materialize_pipeline);
            if let Some((compact, residual)) = self.buffers.rows.stage1_buffers() {
                encoder.set_buffer(0, Some(compact), 0);
                encoder.set_buffer(1, Some(residual), 0);
                encoder.set_buffer(2, Some(&self.buffers.lagrange), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(5, Some(&self.buffers.state_a), 0);
                encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 7, &params);
            } else {
                let rows = self.buffers.rows.packed_buffer().ok_or(
                    MetalError::InvalidProductRemainderState(
                        "packed product materialization lost its row buffer",
                    ),
                )?;
                encoder.set_buffer(0, Some(rows), 0);
                encoder.set_buffer(1, Some(&self.buffers.lagrange), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(4, Some(&self.buffers.state_a), 0);
                encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 6, &params);
            }
            encoder.set_threadgroup_memory_length(
                0,
                product_remainder_threadgroup_bytes(
                    PRODUCT_REMAINDER_MESSAGE_COLUMNS,
                    self.materialize_threads_per_threadgroup,
                ) as u64,
            );
            dispatch_product_remainder_blocks(
                encoder,
                e_out.len(),
                self.materialize_threads_per_threadgroup,
            );
            let final_in_a = encode_product_remainder_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                PRODUCT_REMAINDER_MESSAGE_COLUMNS,
            )?;
            encoder.end_encoding();
            let output = if final_in_a {
                self.buffers.partial_a.clone()
            } else {
                self.buffers.partial_b.clone()
            };
            command_buffer.commit();
            let state_b_fill = self.submit_state_b_fill();
            Ok(ProductRemainderInitialMessageCommand {
                command_buffer,
                state_b_fill,
                output,
                sequence_identity: self.buffers.state_a.as_ptr() as usize,
            })
        })
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], MetalError> {
        self.bind_and_message_timed(challenge, e_in, e_out)
            .map(|(message, _)| message)
    }

    pub fn bind_and_message_timed(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        let (message, active_time) =
            self.execute_current_bind_and_message(challenge, e_in, e_out)?;
        self.current_elements /= 2;
        self.source_in_a = !self.source_in_a;
        Ok((message, active_time))
    }

    fn execute_current_bind_and_message(
        &self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        if self.phase != ProductRemainderPhase::Materialized {
            return Err(MetalError::InvalidProductRemainderState(
                "the first message must be emitted before a transition",
            ));
        }
        let params = ProductRemainderPhaseParams::transition(
            self.current_elements,
            e_in.len(),
            e_out.len(),
        )?;
        self.write_weights(e_in, e_out)?;
        let challenge = Fp128::from_jolt_field(&challenge);
        self.context
            .validate_inputs("product remainder challenge", &[challenge])?;
        let source = self.source_buffer();
        let destination = self.destination_buffer();
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.transition_pipeline);
            encoder.set_buffer(0, Some(source), 0);
            encoder.set_buffer(1, Some(destination), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 5, &challenge);
            set_inline_bytes(encoder, 6, &params);
            encoder.set_threadgroup_memory_length(
                0,
                product_remainder_threadgroup_bytes(
                    PRODUCT_REMAINDER_MESSAGE_COLUMNS,
                    self.transition_threads_per_threadgroup,
                ) as u64,
            );
            dispatch_product_remainder_blocks(
                encoder,
                e_out.len(),
                self.transition_threads_per_threadgroup,
            );
            let final_in_a = encode_product_remainder_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                PRODUCT_REMAINDER_MESSAGE_COLUMNS,
            )?;
            encoder.end_encoding();
            finish_product_remainder_command::<PRODUCT_REMAINDER_MESSAGE_COLUMNS>(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                "product remainder transition message",
            )
        })
    }

    pub fn openings(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; PRODUCT_REMAINDER_OPENINGS], MetalError> {
        self.openings_timed(e_in, e_out)
            .map(|(openings, _)| openings)
    }

    pub fn openings_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_OPENINGS], Duration), MetalError> {
        if self.phase != ProductRemainderPhase::Materialized || self.current_elements != 2 {
            return Err(MetalError::InvalidProductRemainderState(
                "openings require every message round to be completed",
            ));
        }
        let (openings, active_time) = self.execute_openings(e_in, e_out)?;
        Ok((openings, active_time))
    }

    pub(crate) fn openings_after_cpu_tail_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_OPENINGS], Duration), MetalError> {
        if !matches!(
            self.phase,
            ProductRemainderPhase::Materialized | ProductRemainderPhase::TransitionRetired
        ) {
            return Err(MetalError::InvalidProductRemainderState(
                "CPU-tail openings require an unfinished resident sequence",
            ));
        }
        let (openings, active_time) = self.execute_openings(e_in, e_out)?;
        Ok((openings, active_time))
    }

    pub(in crate::metal::solinas) fn terminal_cache_openings_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<Option<ProductInstructionTerminalCacheOpenings>, MetalError> {
        let Some(cache) = &self.terminal_cache else {
            return Ok(None);
        };
        if !matches!(
            self.phase,
            ProductRemainderPhase::Materialized | ProductRemainderPhase::TransitionRetired
        ) || self.current_elements <= 2
        {
            return Err(MetalError::InvalidProductRemainderState(
                "terminal cache openings require an unfinished Product sequence",
            ));
        }
        let layout = cache.layout;
        let groups_per_cache = layout
            .rows_per_group
            .checked_div(e_in.len())
            .filter(|&ratio| {
                ratio != 0
                    && layout.groups.checked_mul(ratio) == Some(e_out.len())
                    && e_in.len().checked_mul(e_out.len()) == Some(layout.rows)
            });
        if groups_per_cache.is_none() {
            return Err(MetalError::InvalidProductRemainderState(
                "terminal cache opening weights disagree with its equality blocks",
            ));
        }
        let _ = cache.stats.ok_or(MetalError::InvalidProductRemainderState(
            "terminal cache openings were requested before construction completed",
        ))?;
        let pipeline = cache.openings_pipeline.clone();
        let params = layout.params(layout.rows_per_group, e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        let (values, gpu_active) = autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(
                0,
                Some(&self.buffers.state_b),
                layout.lookup_low_offset as u64,
            );
            encoder.set_buffer(
                1,
                Some(&self.buffers.state_b),
                layout.left_lookup_low_offset as u64,
            );
            encoder.set_buffer(2, Some(&self.buffers.state_b), layout.flags_offset as u64);
            encoder.set_buffer(3, Some(&self.buffers.state_b), layout.counts_offset as u64);
            encoder.set_buffer(
                4,
                Some(&self.buffers.state_b),
                layout.exceptions_offset as u64,
            );
            encoder.set_buffer(5, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(6, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(7, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 8, &params);
            encoder.set_threadgroup_memory_length(
                0,
                product_remainder_threadgroup_bytes(
                    PRODUCT_REMAINDER_OPENINGS,
                    self.openings_threads_per_threadgroup,
                ) as u64,
            );
            dispatch_product_remainder_blocks(
                encoder,
                e_out.len(),
                self.openings_threads_per_threadgroup,
            );
            let final_in_a = encode_product_remainder_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                PRODUCT_REMAINDER_OPENINGS,
            )?;
            encoder.end_encoding();
            finish_product_remainder_command::<PRODUCT_REMAINDER_OPENINGS>(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                "product instruction terminal cache openings",
            )
        })?;
        let _ = self.release_terminal_cache_buffer()?;
        Ok(Some(ProductInstructionTerminalCacheOpenings {
            values,
            gpu_active,
        }))
    }

    fn execute_openings(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_OPENINGS], Duration), MetalError> {
        let params =
            ProductRemainderOpeningParams::new(self.layout.rows(), e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.openings_pipeline);
            if let Some((compact, residual)) = self.buffers.rows.stage1_buffers() {
                encoder.set_buffer(0, Some(compact), 0);
                encoder.set_buffer(1, Some(residual), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 5, &params);
            } else {
                let rows = self.buffers.rows.packed_buffer().ok_or(
                    MetalError::InvalidProductRemainderState(
                        "packed product openings lost its row buffer",
                    ),
                )?;
                encoder.set_buffer(0, Some(rows), 0);
                encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 4, &params);
            }
            encoder.set_threadgroup_memory_length(
                0,
                product_remainder_threadgroup_bytes(
                    PRODUCT_REMAINDER_OPENINGS,
                    self.openings_threads_per_threadgroup,
                ) as u64,
            );
            dispatch_product_remainder_blocks(
                encoder,
                e_out.len(),
                self.openings_threads_per_threadgroup,
            );
            let final_in_a = encode_product_remainder_reductions(
                encoder,
                &self.reduction_pipeline,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                PRODUCT_REMAINDER_OPENINGS,
            )?;
            encoder.end_encoding();
            finish_product_remainder_command::<PRODUCT_REMAINDER_OPENINGS>(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                "product remainder openings",
            )
        })
    }

    copy_field_getters! { pub, {
        current_elements: usize,
        storage_layout => layout: ProductRemainderStorageLayout,
    }}

    pub fn resident_buffer_count(&self) -> usize {
        self.buffers.rows.allocation_identities().len() + 7
    }

    pub fn row_allocation_identity(&self) -> usize {
        self.buffers.rows.allocation_identity()
    }

    pub(crate) fn device_registry_id(&self) -> u64 {
        self.context.device_registry_id()
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.phase == ProductRemainderPhase::Ready
            && self.current_elements == self.layout.rows()
            && self.source_in_a
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    #[doc(hidden)]
    pub fn read_current_state(&self) -> Result<(Vec<AkitaField>, Vec<AkitaField>), MetalError> {
        if self.phase != ProductRemainderPhase::Materialized {
            return Err(MetalError::InvalidProductRemainderState(
                "the product state is not materialized",
            ));
        }
        let state = self.source_buffer();
        // SAFETY: the active buffer stores two `current_elements` tables and
        // all preceding commands have completed.
        let values = unsafe {
            slice::from_raw_parts(state.contents().cast::<Fp128>(), 2 * self.current_elements)
        };
        self.context
            .validate_inputs("product remainder resident state", values)?;
        let (left, right) = values.split_at(self.current_elements);
        Ok((
            left.iter().copied().map(Fp128::into_jolt_field).collect(),
            right.iter().copied().map(Fp128::into_jolt_field).collect(),
        ))
    }

    fn write_weights(&self, e_in: &[AkitaField], e_out: &[AkitaField]) -> Result<(), MetalError> {
        write_product_remainder_fields(
            &self.buffers.e_in,
            self.layout.e_in_fields(),
            e_in,
            "e_in",
        )?;
        write_product_remainder_fields(
            &self.buffers.e_out,
            self.layout.e_out_fields(),
            e_out,
            "e_out",
        )
    }

    pub(crate) fn retire_transition_state_after_cpu_tail_copy(
        &mut self,
    ) -> Result<usize, MetalError> {
        if self.phase != ProductRemainderPhase::Materialized || self.current_elements <= 2 {
            return Err(MetalError::InvalidProductRemainderState(
                "transition retirement requires an active CPU tail",
            ));
        }
        let tombstone = self
            .context
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        let state_a = mem::replace(&mut self.buffers.state_a, tombstone.clone());
        let mut retired_bytes = state_a.length();
        if self.terminal_cache.is_none() {
            let state_b = mem::replace(&mut self.buffers.state_b, tombstone);
            retired_bytes = retired_bytes.saturating_add(state_b.length());
        }
        let retired_bytes =
            usize::try_from(retired_bytes).map_err(|_| MetalError::InputTooLong(usize::MAX))?;
        self.phase = ProductRemainderPhase::TransitionRetired;
        Ok(retired_bytes)
    }

    fn source_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.state_a
        } else {
            &self.buffers.state_b
        }
    }

    fn destination_buffer(&self) -> &Buffer {
        if self.source_in_a {
            &self.buffers.state_b
        } else {
            &self.buffers.state_a
        }
    }
}

fn product_remainder_threadgroup_bytes(columns: usize, threads: usize) -> usize {
    columns * (threads / PRODUCT_REMAINDER_SIMD_WIDTH) * size_of::<Fp128>()
}

fn dispatch_product_remainder_blocks(
    encoder: &metal::ComputeCommandEncoderRef,
    blocks: usize,
    threads_per_threadgroup: usize,
) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: blocks as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_threadgroup as u64,
            height: 1,
            depth: 1,
        },
    );
}

fn dispatch_product_remainder_linear(
    encoder: &metal::ComputeCommandEncoderRef,
    elements: usize,
    threads_per_threadgroup: usize,
) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: elements.div_ceil(threads_per_threadgroup) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_threadgroup as u64,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_product_remainder_reductions(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    partial_a: &Buffer,
    partial_b: &Buffer,
    input_count: usize,
    columns: usize,
) -> Result<bool, MetalError> {
    let _ = ProductRemainderReductionParams::new(input_count, columns)?;
    encode_column_reductions(
        encoder,
        pipeline,
        partial_a,
        partial_b,
        input_count,
        columns,
        PRODUCT_REMAINDER_SIMD_WIDTH,
    )
}

fn finish_product_remainder_command<const COLUMNS: usize>(
    context: &SolinasMetal,
    command_buffer: &metal::CommandBufferRef,
    output: &Buffer,
    label: &'static str,
) -> Result<([AkitaField; COLUMNS], Duration), MetalError> {
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let gpu_active = completed_command_gpu_time(command_buffer)?;
    // SAFETY: the completed recursive reduction leaves `COLUMNS` fields at
    // the front of the selected shared buffer.
    let values = unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), COLUMNS) };
    context.validate_inputs(label, values)?;
    Ok((
        std::array::from_fn(|index| values[index].into_jolt_field()),
        gpu_active,
    ))
}

fn write_product_remainder_fields(
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
    name: &'static str,
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(ProductRemainderShapeError::StorageLength {
            name,
            expected: capacity,
            got: values.len(),
        }
        .into());
    }
    // SAFETY: the shared buffer holds `capacity` fields and no device command
    // is active while the host writes the next split-equality prefix.
    let output = unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), capacity) };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn validate_rows(rows: usize) -> Result<(), ProductRemainderShapeError> {
    if rows < 2 || !rows.is_power_of_two() {
        return Err(ProductRemainderShapeError::InvalidRows(rows));
    }
    let state_fields = rows
        .checked_mul(2)
        .ok_or(ProductRemainderShapeError::ShaderIndexOverflow { name: "state A" })?;
    let _ = shader_count("state A", state_fields)?;
    Ok(())
}

fn validate_weight_shape(
    phase: &'static str,
    expected: usize,
    e_in: usize,
    e_out: usize,
) -> Result<(), ProductRemainderShapeError> {
    let covered = e_in.checked_mul(e_out);
    if e_in == 0 || e_out == 0 || covered != Some(expected) {
        return Err(ProductRemainderShapeError::WeightShape {
            phase,
            expected,
            e_in,
            e_out,
        });
    }
    Ok(())
}

fn validate_partial_index(
    columns: usize,
    fields_per_column: usize,
) -> Result<(), ProductRemainderShapeError> {
    let fields = checked_product("partial buffer", columns, fields_per_column)?;
    let _ = shader_count("partial buffer", fields)?;
    Ok(())
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, ProductRemainderShapeError> {
    u32::try_from(value).map_err(|_| ProductRemainderShapeError::ShaderIndexOverflow { name })
}

fn checked_product(
    name: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize, ProductRemainderShapeError> {
    lhs.checked_mul(rhs)
        .ok_or(ProductRemainderShapeError::ByteLengthOverflow { name })
}

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod reference {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct MaterializedMessage<F> {
        pub state: Vec<F>,
        pub endpoints: [F; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct TransitionMessage<F> {
        pub state: Vec<F>,
        pub endpoints: [F; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
    }

    pub fn materialize_message<F: Field>(
        rows: &[ProductRemainderRow],
        lagrange_weights: [F; 3],
        e_in: &[F],
        e_out: &[F],
    ) -> Result<MaterializedMessage<F>, ProductRemainderShapeError> {
        let _ = ProductRemainderPhaseParams::materialize(rows.len(), e_in.len(), e_out.len())?;
        let mut state = vec![F::zero(); 2 * rows.len()];
        let mut endpoints = [F::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = [F::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let pair = x_out * e_in.len() + x_in;
                let low_index = 2 * pair;
                let high_index = low_index + 1;
                let (left_low, right_low) = rows[low_index].relation_values(&lagrange_weights);
                let (left_high, right_high) = rows[high_index].relation_values(&lagrange_weights);
                state[low_index] = left_low;
                state[high_index] = left_high;
                state[rows.len() + low_index] = right_low;
                state[rows.len() + high_index] = right_high;
                inner[0] += inner_weight * (left_low * right_low);
                inner[1] += inner_weight * ((left_high - left_low) * (right_high - right_low));
            }
            endpoints[0] += outer_weight * inner[0];
            endpoints[1] += outer_weight * inner[1];
        }
        Ok(MaterializedMessage { state, endpoints })
    }

    pub fn bind_and_message<F: Field>(
        state: &[F],
        source_elements: usize,
        challenge: F,
        e_in: &[F],
        e_out: &[F],
    ) -> Result<TransitionMessage<F>, ProductRemainderShapeError> {
        let _ = ProductRemainderPhaseParams::transition(source_elements, e_in.len(), e_out.len())?;
        let expected = 2 * source_elements;
        if state.len() != expected {
            return Err(ProductRemainderShapeError::StorageLength {
                name: "source state",
                expected,
                got: state.len(),
            });
        }

        let bound_elements = source_elements / 2;
        let mut bound = vec![F::zero(); 2 * bound_elements];
        let mut endpoints = [F::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            let mut inner = [F::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let pair = x_out * e_in.len() + x_in;
                let source = 4 * pair;
                let destination = 2 * pair;
                let left_0 = bind(state[source], state[source + 1], challenge);
                let left_1 = bind(state[source + 2], state[source + 3], challenge);
                let right_base = source_elements;
                let right_0 = bind(
                    state[right_base + source],
                    state[right_base + source + 1],
                    challenge,
                );
                let right_1 = bind(
                    state[right_base + source + 2],
                    state[right_base + source + 3],
                    challenge,
                );
                bound[destination] = left_0;
                bound[destination + 1] = left_1;
                bound[bound_elements + destination] = right_0;
                bound[bound_elements + destination + 1] = right_1;
                inner[0] += inner_weight * (left_0 * right_0);
                inner[1] += inner_weight * ((left_1 - left_0) * (right_1 - right_0));
            }
            endpoints[0] += outer_weight * inner[0];
            endpoints[1] += outer_weight * inner[1];
        }
        Ok(TransitionMessage {
            state: bound,
            endpoints,
        })
    }

    pub fn openings<F: Field>(
        rows: &[ProductRemainderRow],
        e_in: &[F],
        e_out: &[F],
    ) -> Result<[F; PRODUCT_REMAINDER_OPENINGS], ProductRemainderShapeError> {
        let _ = ProductRemainderOpeningParams::new(rows.len(), e_in.len(), e_out.len())?;
        let mut sums = [F::zero(); PRODUCT_REMAINDER_OPENINGS];
        for (x_out, &outer_weight) in e_out.iter().enumerate() {
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let row = rows[x_out * e_in.len() + x_in];
                let weight = outer_weight * inner_weight;
                for (sum, value) in sums.iter_mut().zip(row.fields::<F>()) {
                    *sum += weight * value;
                }
            }
        }
        Ok(sums)
    }

    fn bind<F: Field>(low: F, high: F, challenge: F) -> F {
        low + challenge * (high - low)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests use a fixed valid storage shape"
)]
mod tests {
    use super::super::product_uniskip::evaluate_product_uniskip_extensions_cpu;
    use super::super::SpartanOuterUniskipRow;
    use super::*;

    fn native_width_boundary_row(index: usize) -> ProductRemainderRow {
        let u32_max = u64::from(u32::MAX);
        let unsigned = [
            0,
            u32_max,
            u32_max + 1,
            u64::MAX,
            1,
            1 << 31,
            1 << 32,
            u64::MAX - 1,
            17,
        ];
        let signed = [
            0,
            i128::from(u32::MAX),
            i128::from(u32::MAX) + 1,
            -i128::from(u32::MAX),
            -(i128::from(u32::MAX) + 1),
            i128::from(u64::MAX),
            -i128::from(u64::MAX),
            i128::MIN,
            i128::MAX,
        ];
        ProductRemainderRow::new(
            unsigned[index % unsigned.len()],
            signed[index % signed.len()],
            index.is_multiple_of(3),
            index.is_multiple_of(5),
            unsigned[(index + 3) % unsigned.len()],
            index.is_multiple_of(7),
            index.is_multiple_of(11),
            index.is_multiple_of(13),
        )
    }

    #[test]
    fn row_abi_and_word_order_are_fixed() {
        assert_eq!(size_of::<ProductRemainderRow>(), 40);
        assert_eq!(align_of::<ProductRemainderRow>(), 8);

        let row = ProductRemainderRow::new(u64::MAX, i128::MIN, true, true, 7, true, true, true);
        assert_eq!(row.words(), [u64::MAX, 0, 1 << 63, 7, 0b11_1110]);
        assert_eq!(row.left_instruction_input(), u64::MAX);
        assert_eq!(row.right_instruction_input(), i128::MIN);
        assert_eq!(row.lookup_output(), 7);
        assert!(row.jump());
        assert!(row.write_lookup_output_to_rd());
        assert!(row.branch());
        assert!(row.next_is_noop());
        assert!(row.virtual_instruction());
        assert_eq!(row.validate(), Ok(()));
    }

    #[test]
    fn right_input_signed_magnitude_round_trips() {
        for value in [i128::MIN, -1, 0, 1, i128::MAX] {
            let row = ProductRemainderRow::new(11, value, false, false, 12, false, false, false);
            assert_eq!(row.right_instruction_input(), value);
            assert_eq!(ProductRemainderRow::try_from_words(row.words()), Ok(row));
        }
    }

    #[test]
    fn ready_sequence_reuses_resident_rows_for_product_uniskip() {
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let row_count = 1usize << 10;
        let rows = (0..row_count)
            .map(native_width_boundary_row)
            .collect::<Vec<_>>();
        let e_in = (0..32)
            .map(|index| AkitaField::from_u64((37 * index + 19) as u64))
            .collect::<Vec<_>>();
        let e_out = (0..32)
            .map(|index| AkitaField::from_u64((43 * index + 23) as u64))
            .collect::<Vec<_>>();
        let expected = evaluate_product_uniskip_extensions_cpu(&rows, &e_in, &e_out)
            .expect("CPU product uni-skip should be well-shaped");
        let mut sequence = context
            .prepare_product_remainder_sequence(
                &rows,
                [AkitaField::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS + 1],
                e_in.len(),
                e_out.len(),
                ProductRemainderSequenceConfig::default(),
            )
            .expect("resident product sequence should prepare");
        let row_storage_id = sequence.row_allocation_identity();
        assert!(sequence.is_ready());

        let (actual, _) = sequence
            .uniskip_message_timed(&e_in, &e_out)
            .expect("resident product uni-skip should execute");
        assert_eq!(actual, expected);
        assert!(sequence.is_ready());
        assert_eq!(sequence.row_allocation_identity(), row_storage_id);
        assert_eq!(sequence.round_device_buffer_allocations(), 0);
    }

    #[test]
    fn submitted_initial_message_matches_the_independent_materialization_oracle() {
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let row_count = 1usize << 10;
        let rows = (0..row_count)
            .map(native_width_boundary_row)
            .collect::<Vec<_>>();
        let e_in = (0..16)
            .map(|index| AkitaField::from_u64((37 * index + 19) as u64))
            .collect::<Vec<_>>();
        let opening_e_in = (0..32)
            .map(|index| AkitaField::from_u64((37 * index + 19) as u64))
            .collect::<Vec<_>>();
        let e_out = (0..32)
            .map(|index| AkitaField::from_u64((43 * index + 23) as u64))
            .collect::<Vec<_>>();
        let lagrange = [
            AkitaField::from_u64(5),
            AkitaField::from_u64(7),
            AkitaField::from_u64(11),
        ];
        let expected = reference::materialize_message(&rows, lagrange, &e_in, &e_out)
            .expect("the independent materialization should be well-shaped");
        let sequence = context
            .prepare_product_remainder_sequence(
                &rows,
                lagrange,
                32,
                32,
                ProductRemainderSequenceConfig::default(),
            )
            .expect("resident product sequence should prepare");
        let pending = sequence
            .submit_initial_message(&e_in, &e_out)
            .expect("the materialization should submit");
        let (mut sequence, actual, gpu_active) =
            pending.join().expect("the materialization should join");
        let (left, right) = sequence
            .read_current_state()
            .expect("the materialized state should be readable");

        assert_eq!(actual, expected.endpoints);
        assert_eq!([left, right].concat(), expected.state);
        assert_eq!(
            sequence
                .openings_after_cpu_tail_timed(&opening_e_in, &e_out)
                .expect("the source openings should execute")
                .0,
            reference::openings(&rows, &opening_e_in, &e_out)
                .expect("the independent openings should be well-shaped")
        );
        assert!(gpu_active > Duration::ZERO);
        assert_eq!(sequence.current_elements(), row_count);
        assert_eq!(sequence.round_device_buffer_allocations(), 0);

        let round_1_challenge = AkitaField::from_u64(101);
        let expected_round_1 = reference::bind_and_message(
            &expected.state,
            row_count,
            round_1_challenge,
            &e_in,
            &e_out[..16],
        )
        .unwrap();
        assert_eq!(
            sequence
                .bind_and_message(round_1_challenge, &e_in, &e_out[..16])
                .unwrap(),
            expected_round_1.endpoints
        );
        let (left, right) = sequence.read_current_state().unwrap();
        assert_eq!([left, right].concat(), expected_round_1.state);

        let round_2_challenge = AkitaField::from_u64(103);
        let expected_round_2 = reference::bind_and_message(
            &expected_round_1.state,
            row_count / 2,
            round_2_challenge,
            &e_in[..8],
            &e_out[..16],
        )
        .unwrap();
        assert_eq!(
            sequence
                .bind_and_message(round_2_challenge, &e_in[..8], &e_out[..16])
                .unwrap(),
            expected_round_2.endpoints
        );
        let (left, right) = sequence.read_current_state().unwrap();
        assert_eq!([left, right].concat(), expected_round_2.state);
    }

    #[test]
    fn stage1_source_matches_packed_product_rows() {
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let row_count = 1usize << 8;
        let packed = (0..row_count)
            .map(native_width_boundary_row)
            .collect::<Vec<_>>();
        let stage1 = packed
            .iter()
            .map(|row| {
                let mut words = [0u64; 20];
                let right = row.right_instruction_input().unsigned_abs();
                words[0] = row.left_instruction_input();
                words[1] = right as u64;
                words[2] = (right >> 64) as u64;
                words[18] = row.lookup_output();
                words[19] = u64::from(row.jump()) << 5
                    | u64::from(row.virtual_instruction()) << 9
                    | u64::from(row.write_lookup_output_to_rd()) << 14
                    | u64::from(row.right_instruction_input() >= 0) << 17
                    | u64::from(row.branch()) << 25
                    | u64::from(row.next_is_noop()) << 26;
                SpartanOuterUniskipRow::from_words(words)
            })
            .collect::<Vec<_>>();
        let stage1 = context
            .prepare_spartan_outer_uniskip_rows(&stage1)
            .expect("Stage-1 rows should prepare");
        let resident = stage1
            .share_product_remainder_rows()
            .expect("Stage-1 rows should expose a product view");
        assert!(resident.source_generation().is_some());
        assert_eq!(
            resident.source_kind(),
            ProductRemainderSourceKind::SpartanStage1
        );

        let lagrange = [
            AkitaField::from_u64(5),
            AkitaField::from_u64(7),
            AkitaField::from_u64(11),
        ];
        let e_in = (0..16)
            .map(|index| AkitaField::from_u64((37 * index + 19) as u64))
            .collect::<Vec<_>>();
        let e_out = (0..16)
            .map(|index| AkitaField::from_u64((43 * index + 23) as u64))
            .collect::<Vec<_>>();
        let materialize_e_out = &e_out[..e_out.len() / 2];
        let mut expected = context
            .prepare_product_remainder_sequence(
                &packed,
                lagrange,
                e_in.len(),
                e_out.len(),
                ProductRemainderSequenceConfig::default(),
            )
            .expect("packed product sequence should prepare");
        let mut actual = context
            .prepare_product_remainder_sequence_with_rows(
                resident,
                lagrange,
                e_in.len(),
                e_out.len(),
                ProductRemainderSequenceConfig::default(),
            )
            .expect("Stage-1 product sequence should prepare");

        assert!(expected.is_ready());
        assert!(actual.is_ready());
        assert_eq!(
            actual.uniskip_message_timed(&e_in, &e_out).unwrap().0,
            expected.uniskip_message_timed(&e_in, &e_out).unwrap().0
        );
        assert_eq!(
            actual
                .restart_message_timed(&e_in, materialize_e_out)
                .unwrap()
                .0,
            expected
                .restart_message_timed(&e_in, materialize_e_out)
                .unwrap()
                .0
        );
        let expected_openings = reference::openings(&packed, &e_in, &e_out).unwrap();
        let layout = actual.storage_layout();
        let expected_retired_bytes =
            (layout.state_a_fields() + layout.state_b_fields()) * size_of::<Fp128>();
        assert_eq!(
            actual
                .retire_transition_state_after_cpu_tail_copy()
                .unwrap(),
            expected_retired_bytes
        );
        assert!(actual.read_current_state().is_err());
        assert_eq!(
            actual
                .openings_after_cpu_tail_timed(&e_in, &e_out)
                .unwrap()
                .0,
            expected_openings
        );
    }

    #[test]
    fn row_validation_rejects_noncanonical_encodings() {
        assert_eq!(
            ProductRemainderRow::try_from_words([0, 0, 0, 0, 1 << 6]),
            Err(ProductRemainderRowError::ReservedFlags(1 << 6))
        );
        assert_eq!(
            ProductRemainderRow::try_from_words([0, 0, 1 << 63, 0, 1]),
            Err(ProductRemainderRowError::InvalidRightInputEncoding {
                magnitude: 1u128 << 127,
                nonnegative: true,
            })
        );
        assert_eq!(
            ProductRemainderRow::try_from_words([0, 0, 0, 0, 0]),
            Err(ProductRemainderRowError::InvalidRightInputEncoding {
                magnitude: 0,
                nonnegative: false,
            })
        );
        assert_eq!(
            ProductRemainderRow::try_from_words([0, 1, 1 << 63, 0, 0]),
            Err(ProductRemainderRowError::InvalidRightInputEncoding {
                magnitude: (1u128 << 127) + 1,
                nonnegative: false,
            })
        );
    }

    #[test]
    fn in_place_schedule_covers_each_output_with_race_free_intervals() {
        for log_bound in 1..=30 {
            let bound_elements = 1usize << log_bound;
            for scratch_capacity in [1, 3, 17, 81_920, bound_elements] {
                let schedule = in_place_bind_schedule(bound_elements, scratch_capacity);
                assert_eq!(schedule.prefix, bound_elements.min(scratch_capacity));
                let mut cursor = schedule.prefix;
                for range in schedule.direct {
                    assert_eq!(range.start, cursor);
                    let end = range.start + range.count;
                    assert!(end <= 2 * range.start);
                    cursor = end;
                }
                assert_eq!(cursor, bound_elements);
            }
        }

        for bound_elements in [2, 4, 8, 16, 32, 64, 128, 256, 512] {
            for scratch_capacity in [1, 3, 17, 81] {
                let source = (0..2 * bound_elements).collect::<Vec<_>>();
                let expected = (0..bound_elements)
                    .map(|index| source[2 * index] + 3 * source[2 * index + 1])
                    .collect::<Vec<_>>();
                let schedule = in_place_bind_schedule(bound_elements, scratch_capacity);
                let mut state = source;
                let scratch = (0..schedule.prefix)
                    .map(|index| state[2 * index] + 3 * state[2 * index + 1])
                    .collect::<Vec<_>>();
                state[..schedule.prefix].copy_from_slice(&scratch);
                for range in schedule.direct {
                    for index in range.start..range.start + range.count {
                        state[index] = state[2 * index] + 3 * state[2 * index + 1];
                    }
                }
                assert_eq!(&state[..bound_elements], expected);
            }
        }
    }

    #[test]
    fn target_scale_storage_layout_is_pinned() {
        let rows = 1usize << 26;
        let split_capacity = 1usize << 13;
        let layout = ProductRemainderStorageLayout::new(rows, split_capacity, split_capacity)
            .expect("the target-scale layout is valid");
        let expected_workspace_fields =
            3 + 3 * rows + 2 * split_capacity + 2 * PRODUCT_REMAINDER_OPENINGS * split_capacity;
        let expected_workspace_bytes = expected_workspace_fields * 16;

        assert_eq!(layout.rows(), rows);
        assert_eq!(layout.row_bytes(), rows * size_of::<ProductRemainderRow>());
        assert_eq!(layout.state_a_fields(), 2 * rows);
        assert_eq!(layout.state_b_fields(), rows);
        assert_eq!(layout.e_in_fields(), split_capacity);
        assert_eq!(layout.e_out_fields(), split_capacity);
        assert_eq!(
            layout.partial_fields(),
            PRODUCT_REMAINDER_OPENINGS * split_capacity
        );
        assert_eq!(layout.workspace_bytes(), expected_workspace_bytes);
        assert_eq!(
            layout.resident_bytes(),
            expected_workspace_bytes + rows * size_of::<ProductRemainderRow>()
        );
    }
}
