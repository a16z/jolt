use std::{
    mem::{align_of, size_of},
    slice,
    time::Duration,
};

use jolt_field::{AkitaField, Field};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use thiserror::Error;

use crate::optimized::spartan_product::SpartanProductRow;

use super::{
    buffer_from_slice, command_buffer_timestamp, Fp128, MetalError, PipelineLimits, SolinasMetal,
};

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub const PRODUCT_REMAINDER_ROW_WORDS: usize = 5;
pub const PRODUCT_REMAINDER_MESSAGE_COLUMNS: usize = 2;
pub const PRODUCT_REMAINDER_OPENINGS: usize = 8;
pub const PRODUCT_REMAINDER_SIMD_WIDTH: usize = 32;

pub(crate) const MATERIALIZE_PIPELINE: &str = "solinas_product_remainder_materialize_message";
pub(crate) const TRANSITION_PIPELINE: &str = "solinas_product_remainder_bind_and_message";
pub(crate) const OPENINGS_PIPELINE: &str = "solinas_product_remainder_openings";
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
    pub materialize_threads_per_threadgroup: Option<usize>,
    pub transition_threads_per_threadgroup: Option<usize>,
    pub openings_threads_per_threadgroup: Option<usize>,
}

impl Default for ProductRemainderSequenceConfig {
    fn default() -> Self {
        Self {
            materialize_threads_per_threadgroup: Some(128),
            transition_threads_per_threadgroup: Some(64),
            openings_threads_per_threadgroup: Some(128),
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

    pub const fn words(self) -> [u64; PRODUCT_REMAINDER_ROW_WORDS] {
        self.words
    }

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

const _: [(); 16] = [(); size_of::<ProductRemainderPhaseParams>()];
const _: [(); 16] = [(); size_of::<ProductRemainderOpeningParams>()];
const _: [(); 16] = [(); size_of::<ProductRemainderReductionParams>()];

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

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }

    pub const fn state_a_fields(self) -> usize {
        self.state_a_fields
    }

    pub const fn state_b_fields(self) -> usize {
        self.state_b_fields
    }

    pub const fn e_in_fields(self) -> usize {
        self.e_in_fields
    }

    pub const fn e_out_fields(self) -> usize {
        self.e_out_fields
    }

    pub const fn partial_fields(self) -> usize {
        self.partial_fields
    }

    pub const fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }

    pub const fn resident_bytes(self) -> usize {
        self.resident_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProductRemainderPhase {
    Raw,
    Materialized,
}

struct ProductRemainderBuffers {
    rows: Buffer,
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
    materialize_pipeline: ComputePipelineState,
    transition_pipeline: ComputePipelineState,
    openings_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    materialize_limits: PipelineLimits,
    transition_limits: PipelineLimits,
    openings_limits: PipelineLimits,
    reduction_limits: PipelineLimits,
    buffers: ProductRemainderBuffers,
    layout: ProductRemainderStorageLayout,
    materialize_threads_per_threadgroup: usize,
    transition_threads_per_threadgroup: usize,
    openings_threads_per_threadgroup: usize,
    current_elements: usize,
    source_in_a: bool,
    phase: ProductRemainderPhase,
    gpu_active_time: Duration,
}

impl SolinasMetal {
    pub fn prepare_product_remainder_sequence(
        &self,
        rows: &[ProductRemainderRow],
        lagrange_weights: [AkitaField; 3],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: ProductRemainderSequenceConfig,
    ) -> Result<ProductRemainderSequence, MetalError> {
        let layout = ProductRemainderStorageLayout::new(rows.len(), e_in_capacity, e_out_capacity)?;
        let resident_bytes = u64::try_from(layout.resident_bytes())
            .map_err(|_| MetalError::InputTooLong(layout.resident_bytes()))?;
        self.validate_additional_working_set(resident_bytes)?;
        let row_bytes = u64::try_from(layout.row_bytes())
            .map_err(|_| MetalError::InputTooLong(layout.row_bytes()))?;
        self.validate_buffer_length(row_bytes)?;
        for (index, row) in rows.iter().copied().enumerate() {
            row.validate()
                .map_err(|source| MetalError::InvalidProductRemainderRow { index, source })?;
        }

        let materialize_pipeline = self.compile_named_pipeline(MATERIALIZE_PIPELINE)?;
        let transition_pipeline = self.compile_named_pipeline(TRANSITION_PIPELINE)?;
        let openings_pipeline = self.compile_named_pipeline(OPENINGS_PIPELINE)?;
        let reduction_pipeline = self.compile_named_pipeline(REDUCTION_PIPELINE)?;
        let materialize_limits = Self::limits(&materialize_pipeline);
        let transition_limits = Self::limits(&transition_pipeline);
        let openings_limits = Self::limits(&openings_pipeline);
        let reduction_limits = Self::limits(&reduction_pipeline);
        for (pipeline, limits) in [
            (MATERIALIZE_PIPELINE, materialize_limits),
            (TRANSITION_PIPELINE, transition_limits),
            (OPENINGS_PIPELINE, openings_limits),
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

        let materialize_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.materialize_threads_per_threadgroup,
            materialize_limits,
        )?;
        let transition_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.transition_threads_per_threadgroup,
            transition_limits,
        )?;
        let openings_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.openings_threads_per_threadgroup,
            openings_limits,
        )?;
        let lagrange = lagrange_weights
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        self.validate_inputs("product remainder lagrange weights", &lagrange)?;

        Ok(ProductRemainderSequence {
            context: self.clone(),
            materialize_pipeline,
            transition_pipeline,
            openings_pipeline,
            reduction_pipeline,
            materialize_limits,
            transition_limits,
            openings_limits,
            reduction_limits,
            buffers: ProductRemainderBuffers {
                rows: buffer_from_slice(&self.device, rows),
                lagrange: buffer_from_slice(&self.device, &lagrange),
                state_a: self.new_product_remainder_buffer(layout.state_a_fields())?,
                state_b: self.new_product_remainder_buffer(layout.state_b_fields())?,
                e_in: self.new_product_remainder_buffer(layout.e_in_fields())?,
                e_out: self.new_product_remainder_buffer(layout.e_out_fields())?,
                partial_a: self.new_product_remainder_buffer(layout.partial_fields())?,
                partial_b: self.new_product_remainder_buffer(layout.partial_fields())?,
            },
            layout,
            materialize_threads_per_threadgroup,
            transition_threads_per_threadgroup,
            openings_threads_per_threadgroup,
            current_elements: rows.len(),
            source_in_a: true,
            phase: ProductRemainderPhase::Raw,
            gpu_active_time: Duration::ZERO,
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
        if self.phase != ProductRemainderPhase::Raw {
            return Err(MetalError::InvalidProductRemainderState(
                "the materialization message was already emitted",
            ));
        }
        let (message, active_time) = self.execute_materialize_message(e_in, e_out)?;
        self.phase = ProductRemainderPhase::Materialized;
        self.gpu_active_time += active_time;
        Ok((message, active_time))
    }

    /// Replays materialization without advancing the resident sequence.
    #[doc(hidden)]
    pub fn replay_materialize_message_timed(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        if self.phase != ProductRemainderPhase::Materialized
            || self.current_elements != self.layout.rows()
            || !self.source_in_a
        {
            return Err(MetalError::InvalidProductRemainderState(
                "materialization replay requires the unbound resident state",
            ));
        }
        self.execute_materialize_message(e_in, e_out)
    }

    /// Rebuilds the initial state in the existing buffers.
    #[doc(hidden)]
    pub fn restart_message_timed(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        let (message, active_time) = self.execute_materialize_message(e_in, e_out)?;
        self.current_elements = self.layout.rows();
        self.source_in_a = true;
        self.phase = ProductRemainderPhase::Materialized;
        self.gpu_active_time += active_time;
        Ok((message, active_time))
    }

    fn execute_materialize_message(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        let params =
            ProductRemainderPhaseParams::materialize(self.layout.rows(), e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.materialize_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.rows), 0);
            encoder.set_buffer(1, Some(&self.buffers.lagrange), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(3, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(4, Some(&self.buffers.state_a), 0);
            encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 6, &params);
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
            finish_product_remainder_command::<PRODUCT_REMAINDER_MESSAGE_COLUMNS>(
                &self.context,
                command_buffer,
                if final_in_a {
                    &self.buffers.partial_a
                } else {
                    &self.buffers.partial_b
                },
                "product remainder first message",
            )
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
        self.gpu_active_time += active_time;
        Ok((message, active_time))
    }

    /// Replays the current transition without advancing the resident sequence.
    #[doc(hidden)]
    pub fn replay_current_bind_and_message_timed(
        &self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS], Duration), MetalError> {
        self.execute_current_bind_and_message(challenge, e_in, e_out)
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
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.transition_pipeline);
            encoder.set_buffer(0, Some(self.source_buffer()), 0);
            encoder.set_buffer(1, Some(self.destination_buffer()), 0);
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
        let (openings, active_time) = self.execute_openings(e_in, e_out)?;
        self.gpu_active_time += active_time;
        Ok((openings, active_time))
    }

    /// Replays the opening scan without changing sequence telemetry.
    #[doc(hidden)]
    pub fn replay_openings_timed(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_OPENINGS], Duration), MetalError> {
        self.execute_openings(e_in, e_out)
    }

    fn execute_openings(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<([AkitaField; PRODUCT_REMAINDER_OPENINGS], Duration), MetalError> {
        if self.phase != ProductRemainderPhase::Materialized || self.current_elements != 2 {
            return Err(MetalError::InvalidProductRemainderState(
                "openings require every message round to be completed",
            ));
        }
        let params =
            ProductRemainderOpeningParams::new(self.layout.rows(), e_in.len(), e_out.len())?;
        self.write_weights(e_in, e_out)?;
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.openings_pipeline);
            encoder.set_buffer(0, Some(&self.buffers.rows), 0);
            encoder.set_buffer(1, Some(&self.buffers.e_in), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_out), 0);
            encoder.set_buffer(3, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 4, &params);
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

    pub const fn current_elements(&self) -> usize {
        self.current_elements
    }

    pub const fn storage_layout(&self) -> ProductRemainderStorageLayout {
        self.layout
    }

    pub const fn resident_buffer_count(&self) -> usize {
        8
    }

    pub const fn round_device_buffer_allocations(&self) -> usize {
        0
    }

    pub const fn gpu_active_time(&self) -> Duration {
        self.gpu_active_time
    }

    pub const fn materialize_pipeline_limits(&self) -> PipelineLimits {
        self.materialize_limits
    }

    pub const fn transition_pipeline_limits(&self) -> PipelineLimits {
        self.transition_limits
    }

    pub const fn openings_pipeline_limits(&self) -> PipelineLimits {
        self.openings_limits
    }

    pub const fn reduction_pipeline_limits(&self) -> PipelineLimits {
        self.reduction_limits
    }

    #[cfg(feature = "test-utils")]
    pub fn read_current_state(&self) -> Result<(Vec<AkitaField>, Vec<AkitaField>), MetalError> {
        if self.phase != ProductRemainderPhase::Materialized {
            return Err(MetalError::InvalidProductRemainderState(
                "the product state is not materialized",
            ));
        }
        let values = unsafe {
            // SAFETY: the active buffer stores two `current_elements` tables
            // and all preceding commands have completed.
            slice::from_raw_parts(
                self.source_buffer().contents().cast::<Fp128>(),
                2 * self.current_elements,
            )
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

fn encode_product_remainder_reductions(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    partial_a: &Buffer,
    partial_b: &Buffer,
    mut input_count: usize,
    columns: usize,
) -> Result<bool, MetalError> {
    let mut input_a = true;
    while input_count > 1 {
        let params = ProductRemainderReductionParams::new(input_count, columns)?;
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
                width: PRODUCT_REMAINDER_SIMD_WIDTH as u64,
                height: 1,
                depth: 1,
            },
        );
        input_count = output_count;
        input_a = !input_a;
    }
    Ok(input_a)
}

fn finish_product_remainder_command<const COLUMNS: usize>(
    context: &SolinasMetal,
    command_buffer: &metal::CommandBufferRef,
    output: &Buffer,
    label: &'static str,
) -> Result<([AkitaField; COLUMNS], Duration), MetalError> {
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
    // SAFETY: the completed recursive reduction leaves `COLUMNS` fields at
    // the front of the selected shared buffer.
    let values = unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), COLUMNS) };
    context.validate_inputs(label, values)?;
    Ok((
        std::array::from_fn(|index| values[index].into_jolt_field()),
        Duration::from_secs_f64(end - start),
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

fn set_inline_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast::<std::ffi::c_void>(),
    );
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
#[expect(clippy::expect_used, reason = "tests use a fixed valid storage shape")]
mod tests {
    use super::*;

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
