//! Checked geometry, dispatch, storage, and roofline models.

use core::mem::{align_of, size_of};

use thiserror::Error;

use super::abi::{
    InstructionClaimOpeningMode, InstructionClaimRightInput, InstructionClaimRightLookup,
    INSTRUCTION_CLAIM_ALIASED_OPENINGS, INSTRUCTION_CLAIM_ALL_OPENINGS,
    INSTRUCTION_CLAIM_CORE_OPENINGS, INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
    INSTRUCTION_CLAIM_NONTRIVIAL_GAMMA_POWERS, INSTRUCTION_CLAIM_SIMD_WIDTH,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimKernelConfig {
    pub materialize_threads_per_threadgroup: usize,
    pub transition_threads_per_threadgroup: usize,
    pub opening_threads_per_threadgroup: usize,
}

impl Default for InstructionClaimKernelConfig {
    fn default() -> Self {
        Self {
            materialize_threads_per_threadgroup: 128,
            transition_threads_per_threadgroup: 64,
            opening_threads_per_threadgroup: 128,
        }
    }
}

impl InstructionClaimKernelConfig {
    pub fn validate(self) -> Result<Self, InstructionClaimShapeError> {
        for (phase, width) in [
            ("materialize", self.materialize_threads_per_threadgroup),
            ("transition", self.transition_threads_per_threadgroup),
            ("opening", self.opening_threads_per_threadgroup),
        ] {
            if width == 0 || !width.is_multiple_of(INSTRUCTION_CLAIM_SIMD_WIDTH) {
                return Err(InstructionClaimShapeError::InvalidThreadgroupWidth { phase, width });
            }
        }
        Ok(self)
    }

    pub fn materialize_threadgroup_bytes(self) -> Result<usize, InstructionClaimShapeError> {
        let config = self.validate()?;
        threadgroup_bytes(
            "materialize",
            INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            config.materialize_threads_per_threadgroup,
        )
    }

    pub fn transition_threadgroup_bytes(self) -> Result<usize, InstructionClaimShapeError> {
        let config = self.validate()?;
        threadgroup_bytes(
            "transition",
            INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            config.transition_threads_per_threadgroup,
        )
    }

    pub fn opening_threadgroup_bytes(
        self,
        columns: usize,
    ) -> Result<usize, InstructionClaimShapeError> {
        let config = self.validate()?;
        if !matches!(
            columns,
            INSTRUCTION_CLAIM_ALIASED_OPENINGS
                | INSTRUCTION_CLAIM_CORE_OPENINGS
                | INSTRUCTION_CLAIM_ALL_OPENINGS
        ) {
            return Err(InstructionClaimShapeError::InvalidOpeningColumns(columns));
        }
        threadgroup_bytes("opening", columns, config.opening_threads_per_threadgroup)
    }
}

/// Exact split-equality lengths for one message or opening scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimWeightGeometry {
    pub(crate) e_in_length: usize,
    pub(crate) e_out_length: usize,
}

impl InstructionClaimWeightGeometry {
    pub const fn e_in_length(self) -> usize {
        self.e_in_length
    }

    pub const fn e_out_length(self) -> usize {
        self.e_out_length
    }
}

/// One message's state and equality-table geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimMessageGeometry {
    round: usize,
    source_elements: usize,
    state_elements: usize,
    weights: InstructionClaimWeightGeometry,
}

impl InstructionClaimMessageGeometry {
    pub const fn round(self) -> usize {
        self.round
    }

    pub const fn source_elements(self) -> usize {
        self.source_elements
    }

    pub const fn state_elements(self) -> usize {
        self.state_elements
    }

    pub const fn weights(self) -> InstructionClaimWeightGeometry {
        self.weights
    }
}

/// Low-to-high Gruen geometry fixed by the trace row count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimGeometry {
    rows: usize,
    log_t: usize,
    split_bits: usize,
}

impl InstructionClaimGeometry {
    pub fn new(rows: usize) -> Result<Self, InstructionClaimShapeError> {
        validate_rows(rows)?;
        let log_t = rows.trailing_zeros() as usize;
        Ok(Self {
            rows,
            log_t,
            split_bits: log_t / 2,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub fn message(
        self,
        round: usize,
    ) -> Result<InstructionClaimMessageGeometry, InstructionClaimShapeError> {
        if round >= self.log_t {
            return Err(InstructionClaimShapeError::InvalidMessageRound {
                round,
                rounds: self.log_t,
            });
        }

        let state_elements = self.rows >> round;
        let source_elements = if round == 0 {
            self.rows
        } else {
            self.rows >> (round - 1)
        };
        let head_bits = self.log_t - round - 1;
        let out_bits = head_bits.min(self.split_bits);
        let in_bits = head_bits - out_bits;
        Ok(InstructionClaimMessageGeometry {
            round,
            source_elements,
            state_elements,
            weights: InstructionClaimWeightGeometry {
                e_in_length: 1usize << in_bits,
                e_out_length: 1usize << out_bits,
            },
        })
    }

    pub const fn opening(self) -> InstructionClaimWeightGeometry {
        InstructionClaimWeightGeometry {
            e_in_length: 1usize << (self.log_t - self.split_bits),
            e_out_length: 1usize << self.split_bits,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum InstructionClaimShapeError {
    #[error("instruction claim reduction needs a power-of-two row count of at least two, got {0}")]
    InvalidRows(usize),
    #[error("instruction claim reduction message round {round} is outside 0..{rounds}")]
    InvalidMessageRound { round: usize, rounds: usize },
    #[error("instruction claim reduction transition cannot produce round zero")]
    InvalidTransitionRound,
    #[error(
        "instruction claim reduction transition needs a power-of-two source count of at least four, got {0}"
    )]
    InvalidTransitionSource(usize),
    #[error(
        "instruction claim reduction {phase} weights have e_in={e_in}, e_out={e_out}; expected product {expected}"
    )]
    WeightShape {
        phase: &'static str,
        expected: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error(
        "instruction claim reduction {phase} weights have e_in={e_in}, e_out={e_out}; expected e_in={expected_e_in}, e_out={expected_e_out}"
    )]
    WeightLayout {
        phase: &'static str,
        expected_e_in: usize,
        expected_e_out: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error(
        "instruction claim reduction weight capacity has e_in={e_in}, e_out={e_out}; needs at least e_in={minimum_e_in}, e_out={minimum_e_out}"
    )]
    WeightCapacity {
        minimum_e_in: usize,
        minimum_e_out: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error("instruction claim reduction {name} plane has length {got}, expected {expected}")]
    OperandPlaneLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("instruction claim reduction {name} storage has length {got}, expected {expected}")]
    StorageLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("instruction claim reduction {name} element count exceeds its 32-bit shader index")]
    ShaderIndexOverflow { name: &'static str },
    #[error("instruction claim reduction {name} byte length overflows host indexing")]
    ByteLengthOverflow { name: &'static str },
    #[error("instruction claim reduction {name} dispatch count overflows host indexing")]
    DispatchCountOverflow { name: &'static str },
    #[error(
        "instruction claim reduction needs a {required}-byte buffer, device maximum is {maximum}"
    )]
    BufferLengthLimit { required: usize, maximum: usize },
    #[error("instruction claim reduction reduction supports 2, 4, or 5 columns, got {0}")]
    InvalidReductionColumns(usize),
    #[error("instruction claim reduction opening supports 2, 4, or 5 columns, got {0}")]
    InvalidOpeningColumns(usize),
    #[error("instruction claim reduction reduction needs at least one input")]
    EmptyReduction,
    #[error(
        "instruction claim reduction {phase} threadgroup width must be a nonzero multiple of 32, got {width}"
    )]
    InvalidThreadgroupWidth { phase: &'static str, width: usize },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionClaimPhaseParams {
    pub(crate) source_elements: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionClaimOpeningParams {
    pub(crate) rows: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) columns: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstructionClaimReductionParams {
    pub(crate) input_count: u32,
    pub(crate) output_count: u32,
    pub(crate) columns: u32,
    pub(crate) _reserved: u32,
}

const _: [(); 16] = [(); size_of::<InstructionClaimPhaseParams>()];
const _: [(); 16] = [(); size_of::<InstructionClaimOpeningParams>()];
const _: [(); 16] = [(); size_of::<InstructionClaimReductionParams>()];
const _: [(); 4] = [(); align_of::<InstructionClaimPhaseParams>()];
const _: [(); 4] = [(); align_of::<InstructionClaimOpeningParams>()];
const _: [(); 4] = [(); align_of::<InstructionClaimReductionParams>()];

impl InstructionClaimPhaseParams {
    pub(crate) fn materialize(
        geometry: InstructionClaimGeometry,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, InstructionClaimShapeError> {
        let message = geometry.message(0)?;
        validate_exact_weight_shape("materialize", message.weights(), e_in_length, e_out_length)?;
        validate_partial_index(INSTRUCTION_CLAIM_MESSAGE_COLUMNS, e_out_length)?;
        Self::new(geometry.rows(), e_in_length, e_out_length)
    }

    pub(crate) fn transition(
        geometry: InstructionClaimGeometry,
        round: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, InstructionClaimShapeError> {
        if round == 0 {
            return Err(InstructionClaimShapeError::InvalidTransitionRound);
        }
        let message = geometry.message(round)?;
        let source_elements = message.source_elements();
        if source_elements < 4 {
            return Err(InstructionClaimShapeError::InvalidTransitionSource(
                source_elements,
            ));
        }
        validate_exact_weight_shape("transition", message.weights(), e_in_length, e_out_length)?;
        validate_partial_index(INSTRUCTION_CLAIM_MESSAGE_COLUMNS, e_out_length)?;
        Self::new(source_elements, e_in_length, e_out_length)
    }

    fn new(
        source_elements: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, InstructionClaimShapeError> {
        Ok(Self {
            source_elements: shader_count("source state", source_elements)?,
            e_in_length: shader_count("e_in", e_in_length)?,
            e_out_length: shader_count("e_out", e_out_length)?,
            _reserved: 0,
        })
    }
}

impl InstructionClaimOpeningParams {
    pub(crate) fn new(
        geometry: InstructionClaimGeometry,
        e_in_length: usize,
        e_out_length: usize,
        mode: InstructionClaimOpeningMode,
    ) -> Result<Self, InstructionClaimShapeError> {
        Self::with_columns(
            geometry,
            e_in_length,
            e_out_length,
            mode.columns(),
            "openings",
        )
    }

    pub(crate) fn aliased(
        geometry: InstructionClaimGeometry,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, InstructionClaimShapeError> {
        Self::with_columns(
            geometry,
            e_in_length,
            e_out_length,
            INSTRUCTION_CLAIM_ALIASED_OPENINGS,
            "aliased openings",
        )
    }

    fn with_columns(
        geometry: InstructionClaimGeometry,
        e_in_length: usize,
        e_out_length: usize,
        columns: usize,
        phase: &'static str,
    ) -> Result<Self, InstructionClaimShapeError> {
        validate_exact_weight_shape(phase, geometry.opening(), e_in_length, e_out_length)?;
        validate_partial_index(columns, e_out_length)?;
        Ok(Self {
            rows: shader_count("opening rows", geometry.rows())?,
            e_in_length: shader_count("opening e_in", e_in_length)?,
            e_out_length: shader_count("opening e_out", e_out_length)?,
            columns: shader_count("opening columns", columns)?,
        })
    }
}

impl InstructionClaimReductionParams {
    pub(crate) fn new(
        input_count: usize,
        columns: usize,
    ) -> Result<Self, InstructionClaimShapeError> {
        if !matches!(
            columns,
            INSTRUCTION_CLAIM_MESSAGE_COLUMNS
                | INSTRUCTION_CLAIM_CORE_OPENINGS
                | INSTRUCTION_CLAIM_ALL_OPENINGS
        ) {
            return Err(InstructionClaimShapeError::InvalidReductionColumns(columns));
        }
        if input_count == 0 {
            return Err(InstructionClaimShapeError::EmptyReduction);
        }
        let output_count = input_count.div_ceil(INSTRUCTION_CLAIM_SIMD_WIDTH);
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

/// One recursive reduction dispatch over column-major partials.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimReductionPass {
    pub(crate) input_count: usize,
    pub(crate) output_count: usize,
    pub(crate) dispatched_threads: usize,
}

impl InstructionClaimReductionPass {
    pub const fn input_count(self) -> usize {
        self.input_count
    }

    pub const fn output_count(self) -> usize {
        self.output_count
    }

    pub const fn dispatched_threads(self) -> usize {
        self.dispatched_threads
    }
}

/// Checked recursive-reduction schedule. No dispatch is needed for one input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstructionClaimReductionPlan {
    columns: usize,
    passes: Vec<InstructionClaimReductionPass>,
}

impl InstructionClaimReductionPlan {
    pub fn new(input_count: usize, columns: usize) -> Result<Self, InstructionClaimShapeError> {
        let _ = InstructionClaimReductionParams::new(input_count, columns)?;
        let mut passes = Vec::new();
        let mut current = input_count;
        while current > 1 {
            let params = InstructionClaimReductionParams::new(current, columns)?;
            let output_count = params.output_count as usize;
            let dispatched_threads = output_count
                .checked_mul(INSTRUCTION_CLAIM_SIMD_WIDTH)
                .ok_or(InstructionClaimShapeError::DispatchCountOverflow { name: "reduction" })?;
            passes.push(InstructionClaimReductionPass {
                input_count: current,
                output_count,
                dispatched_threads,
            });
            current = output_count;
        }
        Ok(Self { columns, passes })
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub fn passes(&self) -> &[InstructionClaimReductionPass] {
        &self.passes
    }
}

/// Buffer capacities for the materialize and resident transition phases.
///
/// State A holds the full combined table, state B the first half-size bound
/// table, and later rounds alternate within those capacities. Each partial
/// buffer reserves all five opening columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimStorageLayout {
    rows: usize,
    lookup_output_bytes: usize,
    left_lookup_operand_bytes: usize,
    right_lookup_operand_bytes: usize,
    left_instruction_input_bytes: usize,
    right_input_bytes: usize,
    maximum_operand_plane_bytes: usize,
    maximum_buffer_bytes: usize,
    gamma_power_fields: usize,
    state_a_fields: usize,
    state_b_fields: usize,
    e_in_fields: usize,
    e_out_fields: usize,
    partial_fields: usize,
    workspace_bytes: usize,
    resident_bytes: usize,
}

impl InstructionClaimStorageLayout {
    pub fn new(
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
    ) -> Result<Self, InstructionClaimShapeError> {
        let geometry = InstructionClaimGeometry::new(rows)?;
        let opening = geometry.opening();
        if e_in_capacity < opening.e_in_length() || e_out_capacity < opening.e_out_length() {
            return Err(InstructionClaimShapeError::WeightCapacity {
                minimum_e_in: opening.e_in_length(),
                minimum_e_out: opening.e_out_length(),
                e_in: e_in_capacity,
                e_out: e_out_capacity,
            });
        }
        let state_a_fields = rows;
        let state_b_fields = rows / 2;
        let partial_fields = checked_product(
            "partial buffer",
            INSTRUCTION_CLAIM_ALL_OPENINGS,
            e_out_capacity,
        )?;
        for (name, fields) in [
            ("state A", state_a_fields),
            ("state B", state_b_fields),
            ("e_in", e_in_capacity),
            ("e_out", e_out_capacity),
            ("partial buffer", partial_fields),
        ] {
            let _ = shader_count(name, fields)?;
        }

        let workspace_fields = [
            INSTRUCTION_CLAIM_NONTRIVIAL_GAMMA_POWERS,
            state_a_fields,
            state_b_fields,
            e_in_capacity,
            e_out_capacity,
            partial_fields,
            partial_fields,
        ]
        .into_iter()
        .try_fold(0usize, |sum, fields| sum.checked_add(fields))
        .ok_or(InstructionClaimShapeError::ByteLengthOverflow { name: "workspace" })?;
        let workspace_bytes = checked_product(
            "workspace",
            workspace_fields,
            size_of::<super::super::Fp128>(),
        )?;
        let lookup_output_bytes = checked_product("lookup output", rows, size_of::<u64>())?;
        let left_lookup_operand_bytes =
            checked_product("left lookup operand", rows, size_of::<u64>())?;
        let right_lookup_operand_bytes = checked_product(
            "right lookup operand",
            rows,
            size_of::<InstructionClaimRightLookup>(),
        )?;
        let left_instruction_input_bytes =
            checked_product("left instruction input", rows, size_of::<u64>())?;
        let right_input_bytes = checked_product(
            "right-input rows",
            rows,
            size_of::<InstructionClaimRightInput>(),
        )?;
        let resident_bytes = workspace_bytes
            .checked_add(lookup_output_bytes)
            .and_then(|bytes| bytes.checked_add(left_lookup_operand_bytes))
            .and_then(|bytes| bytes.checked_add(right_lookup_operand_bytes))
            .and_then(|bytes| bytes.checked_add(left_instruction_input_bytes))
            .and_then(|bytes| bytes.checked_add(right_input_bytes))
            .ok_or(InstructionClaimShapeError::ByteLengthOverflow {
                name: "resident set",
            })?;
        let maximum_workspace_fields = state_a_fields
            .max(state_b_fields)
            .max(e_in_capacity)
            .max(e_out_capacity)
            .max(partial_fields)
            .max(INSTRUCTION_CLAIM_NONTRIVIAL_GAMMA_POWERS);
        let maximum_workspace_bytes = checked_product(
            "largest workspace buffer",
            maximum_workspace_fields,
            size_of::<super::super::Fp128>(),
        )?;
        let maximum_operand_plane_bytes = right_lookup_operand_bytes.max(right_input_bytes);

        Ok(Self {
            rows,
            lookup_output_bytes,
            left_lookup_operand_bytes,
            right_lookup_operand_bytes,
            left_instruction_input_bytes,
            right_input_bytes,
            maximum_operand_plane_bytes,
            maximum_buffer_bytes: maximum_operand_plane_bytes.max(maximum_workspace_bytes),
            gamma_power_fields: INSTRUCTION_CLAIM_NONTRIVIAL_GAMMA_POWERS,
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

    pub const fn lookup_output_bytes(self) -> usize {
        self.lookup_output_bytes
    }

    pub const fn left_lookup_operand_bytes(self) -> usize {
        self.left_lookup_operand_bytes
    }

    pub const fn right_lookup_operand_bytes(self) -> usize {
        self.right_lookup_operand_bytes
    }

    pub const fn left_instruction_input_bytes(self) -> usize {
        self.left_instruction_input_bytes
    }

    pub const fn right_input_bytes(self) -> usize {
        self.right_input_bytes
    }

    pub const fn maximum_operand_plane_bytes(self) -> usize {
        self.maximum_operand_plane_bytes
    }

    pub const fn maximum_buffer_bytes(self) -> usize {
        self.maximum_buffer_bytes
    }

    pub fn validate_max_buffer_length(
        self,
        maximum: usize,
    ) -> Result<Self, InstructionClaimShapeError> {
        if self.maximum_buffer_bytes > maximum {
            return Err(InstructionClaimShapeError::BufferLengthLimit {
                required: self.maximum_buffer_bytes,
                maximum,
            });
        }
        Ok(self)
    }

    pub const fn gamma_power_fields(self) -> usize {
        self.gamma_power_fields
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

pub(super) fn validate_rows(rows: usize) -> Result<(), InstructionClaimShapeError> {
    if rows < 2 || !rows.is_power_of_two() {
        return Err(InstructionClaimShapeError::InvalidRows(rows));
    }
    let _ = shader_count("rows", rows)?;
    Ok(())
}

fn validate_weight_shape(
    phase: &'static str,
    expected: usize,
    e_in: usize,
    e_out: usize,
) -> Result<(), InstructionClaimShapeError> {
    let covered = e_in.checked_mul(e_out);
    if e_in == 0 || e_out == 0 || covered != Some(expected) {
        return Err(InstructionClaimShapeError::WeightShape {
            phase,
            expected,
            e_in,
            e_out,
        });
    }
    Ok(())
}

fn validate_exact_weight_shape(
    phase: &'static str,
    expected: InstructionClaimWeightGeometry,
    e_in: usize,
    e_out: usize,
) -> Result<(), InstructionClaimShapeError> {
    validate_weight_shape(
        phase,
        expected.e_in_length() * expected.e_out_length(),
        e_in,
        e_out,
    )?;
    if e_in != expected.e_in_length() || e_out != expected.e_out_length() {
        return Err(InstructionClaimShapeError::WeightLayout {
            phase,
            expected_e_in: expected.e_in_length(),
            expected_e_out: expected.e_out_length(),
            e_in,
            e_out,
        });
    }
    Ok(())
}

fn validate_partial_index(
    columns: usize,
    fields_per_column: usize,
) -> Result<(), InstructionClaimShapeError> {
    let fields = checked_product("partial buffer", columns, fields_per_column)?;
    let _ = shader_count("partial buffer", fields)?;
    Ok(())
}

fn threadgroup_bytes(
    phase: &'static str,
    columns: usize,
    threads: usize,
) -> Result<usize, InstructionClaimShapeError> {
    if threads == 0 || !threads.is_multiple_of(INSTRUCTION_CLAIM_SIMD_WIDTH) {
        return Err(InstructionClaimShapeError::InvalidThreadgroupWidth {
            phase,
            width: threads,
        });
    }
    columns
        .checked_mul(threads / INSTRUCTION_CLAIM_SIMD_WIDTH)
        .and_then(|fields| fields.checked_mul(size_of::<super::super::Fp128>()))
        .ok_or(InstructionClaimShapeError::ByteLengthOverflow {
            name: "threadgroup memory",
        })
}

fn shader_count(name: &'static str, value: usize) -> Result<u32, InstructionClaimShapeError> {
    u32::try_from(value).map_err(|_| InstructionClaimShapeError::ShaderIndexOverflow { name })
}

pub(super) fn checked_product(
    name: &'static str,
    lhs: usize,
    rhs: usize,
) -> Result<usize, InstructionClaimShapeError> {
    lhs.checked_mul(rhs)
        .ok_or(InstructionClaimShapeError::ByteLengthOverflow { name })
}

pub const INSTRUCTION_CLAIM_TARGET_LOG_T: usize = 26;
pub const INSTRUCTION_CLAIM_TARGET_ROWS: usize = 1usize << INSTRUCTION_CLAIM_TARGET_LOG_T;
pub const INSTRUCTION_CLAIM_FIELD_BYTES: u128 = 16;
pub const INSTRUCTION_CLAIM_NATIVE_ROW_BYTES: u128 = 56;
pub const INSTRUCTION_CLAIM_PRODUCT_ROW_BYTES: u128 = 40;
pub const INSTRUCTION_CLAIM_UNIQUE_OPERAND_BYTES: u128 = 24;
pub const INSTRUCTION_CLAIM_CPU_MEDIAN_NS: u64 = 306_683_705;
pub const INSTRUCTION_CLAIM_REQUIRED_SPEEDUP: u64 = 5;
pub const INSTRUCTION_CLAIM_STRETCH_SPEEDUP: u64 = 8;
pub const INSTRUCTION_CLAIM_COPY_BYTES_PER_SECOND: u128 = 451_701_710_520;
pub const INSTRUCTION_CLAIM_MULTI_ACCUM_PRODUCTS_PER_SECOND: u128 = 32_690_000_000;
pub const INSTRUCTION_CLAIM_TRANSITION_PRODUCTS_PER_SECOND: u128 = 24_080_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InstructionClaimOpeningArchitecture {
    Aliased,
    CoreAndRecover,
    AllColumns,
}

impl InstructionClaimOpeningArchitecture {
    pub const fn columns(self) -> usize {
        match self {
            Self::Aliased => INSTRUCTION_CLAIM_ALIASED_OPENINGS,
            Self::CoreAndRecover => INSTRUCTION_CLAIM_CORE_OPENINGS,
            Self::AllColumns => INSTRUCTION_CLAIM_ALL_OPENINGS,
        }
    }

    pub const fn native_bytes_per_row(self) -> u128 {
        match self {
            Self::Aliased => INSTRUCTION_CLAIM_UNIQUE_OPERAND_BYTES,
            Self::CoreAndRecover => 40,
            Self::AllColumns => INSTRUCTION_CLAIM_NATIVE_ROW_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionClaimReductionWork {
    pub input_fields: u128,
    pub output_fields: u128,
    pub traffic_bytes: u128,
    pub useful_field_additions: u128,
    pub issued_field_add_lanes: u128,
}

fn reduction_work(input_count: usize, columns: usize) -> InstructionClaimReductionWork {
    let mut input = input_count as u128;
    let columns = columns as u128;
    let initial = input;
    let mut input_fields = 0;
    let mut output_fields = 0;
    let mut issued_lanes = 0;
    while input > 1 {
        let output = input.div_ceil(INSTRUCTION_CLAIM_SIMD_WIDTH as u128);
        input_fields += columns * input;
        output_fields += columns * output;
        issued_lanes += columns * output * INSTRUCTION_CLAIM_SIMD_WIDTH as u128;
        input = output;
    }
    InstructionClaimReductionWork {
        input_fields,
        output_fields,
        traffic_bytes: INSTRUCTION_CLAIM_FIELD_BYTES * (input_fields + output_fields),
        useful_field_additions: columns * initial.saturating_sub(1),
        issued_field_add_lanes: 5 * issued_lanes,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionClaimPhaseWork {
    pub useful_field_products: u128,
    pub issued_field_product_lanes: u128,
    pub compulsory_bytes: u128,
    pub shader_logical_equality_bytes: u128,
    pub cache_unique_equality_bytes: u128,
    pub partial_write_bytes: u128,
    pub reduction: InstructionClaimReductionWork,
}

impl InstructionClaimPhaseWork {
    pub const fn cache_optimistic_bytes(self) -> u128 {
        self.compulsory_bytes
            + self.cache_unique_equality_bytes
            + self.partial_write_bytes
            + self.reduction.traffic_bytes
    }

    pub const fn shader_logical_bytes(self) -> u128 {
        self.compulsory_bytes
            + self.shader_logical_equality_bytes
            + self.partial_write_bytes
            + self.reduction.traffic_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimWorkPlan {
    pub materialize: InstructionClaimPhaseWork,
    pub transitions: InstructionClaimPhaseWork,
    pub opening: InstructionClaimPhaseWork,
    pub opening_architecture: InstructionClaimOpeningArchitecture,
    /// Incremental bytes if ProductRemainder reads its 40-byte row in the same
    /// command and this member contributes only its 24-byte companion plus C.
    pub product_fused_materialize_incremental_bytes: u128,
    /// Standalone bytes when this member reuses ProductRemainder's 40-byte row
    /// but cannot fuse the two materializers.
    pub shared_row_standalone_materialize_bytes: u128,
    pub final_state_readback_bytes: u128,
}

impl InstructionClaimWorkPlan {
    pub fn new(
        geometry: InstructionClaimGeometry,
        opening_architecture: InstructionClaimOpeningArchitecture,
    ) -> Result<Self, InstructionClaimShapeError> {
        let rows = geometry.rows() as u128;
        let first = geometry.message(0)?;
        let first_weights = first.weights();
        let first_pairs =
            first_weights.e_in_length() as u128 * first_weights.e_out_length() as u128;
        let first_out = first_weights.e_out_length() as u128;
        let materialize = InstructionClaimPhaseWork {
            useful_field_products: 10 * first_pairs + 2 * first_out,
            issued_field_product_lanes: 10
                * round_up_to_simd(first_weights.e_in_length() as u128)
                * first_out
                + 32 * first_out,
            compulsory_bytes: (INSTRUCTION_CLAIM_NATIVE_ROW_BYTES + INSTRUCTION_CLAIM_FIELD_BYTES)
                * rows,
            shader_logical_equality_bytes: INSTRUCTION_CLAIM_FIELD_BYTES
                * (first_pairs + first_out),
            cache_unique_equality_bytes: INSTRUCTION_CLAIM_FIELD_BYTES
                * (first_weights.e_in_length() as u128 + first_out),
            partial_write_bytes: INSTRUCTION_CLAIM_FIELD_BYTES
                * INSTRUCTION_CLAIM_MESSAGE_COLUMNS as u128
                * first_out,
            reduction: reduction_work(
                first_weights.e_out_length(),
                INSTRUCTION_CLAIM_MESSAGE_COLUMNS,
            ),
        };

        let mut transitions = InstructionClaimPhaseWork::default();
        for round in 1..geometry.log_t() {
            let message = geometry.message(round)?;
            let weights = message.weights();
            let pairs = weights.e_in_length() as u128 * weights.e_out_length() as u128;
            let out = weights.e_out_length() as u128;
            let issued_pairs = round_up_to_simd(weights.e_in_length() as u128) * out;
            transitions.useful_field_products += 4 * pairs + 2 * out;
            transitions.issued_field_product_lanes += 4 * issued_pairs + 32 * out;
            transitions.compulsory_bytes += 96 * pairs;
            transitions.shader_logical_equality_bytes +=
                INSTRUCTION_CLAIM_FIELD_BYTES * (pairs + out);
            transitions.cache_unique_equality_bytes +=
                INSTRUCTION_CLAIM_FIELD_BYTES * (weights.e_in_length() as u128 + out);
            transitions.partial_write_bytes +=
                INSTRUCTION_CLAIM_FIELD_BYTES * INSTRUCTION_CLAIM_MESSAGE_COLUMNS as u128 * out;
            let reduction =
                reduction_work(weights.e_out_length(), INSTRUCTION_CLAIM_MESSAGE_COLUMNS);
            transitions.reduction.input_fields += reduction.input_fields;
            transitions.reduction.output_fields += reduction.output_fields;
            transitions.reduction.traffic_bytes += reduction.traffic_bytes;
            transitions.reduction.useful_field_additions += reduction.useful_field_additions;
            transitions.reduction.issued_field_add_lanes += reduction.issued_field_add_lanes;
        }

        let opening_weights = geometry.opening();
        let opening_out = opening_weights.e_out_length() as u128;
        let columns = opening_architecture.columns();
        let opening = InstructionClaimPhaseWork {
            useful_field_products: columns as u128 * (rows + opening_out),
            issued_field_product_lanes: columns as u128
                * round_up_to_simd(opening_weights.e_in_length() as u128)
                * opening_out
                + 32 * opening_out,
            compulsory_bytes: opening_architecture.native_bytes_per_row() * rows,
            shader_logical_equality_bytes: INSTRUCTION_CLAIM_FIELD_BYTES * (rows + opening_out),
            cache_unique_equality_bytes: INSTRUCTION_CLAIM_FIELD_BYTES
                * (opening_weights.e_in_length() as u128 + opening_out),
            partial_write_bytes: INSTRUCTION_CLAIM_FIELD_BYTES * columns as u128 * opening_out,
            reduction: reduction_work(opening_weights.e_out_length(), columns),
        };

        Ok(Self {
            materialize,
            transitions,
            opening,
            opening_architecture,
            product_fused_materialize_incremental_bytes: (INSTRUCTION_CLAIM_UNIQUE_OPERAND_BYTES
                + INSTRUCTION_CLAIM_FIELD_BYTES)
                * rows,
            shared_row_standalone_materialize_bytes: (INSTRUCTION_CLAIM_PRODUCT_ROW_BYTES
                + INSTRUCTION_CLAIM_UNIQUE_OPERAND_BYTES
                + INSTRUCTION_CLAIM_FIELD_BYTES)
                * rows,
            final_state_readback_bytes: 2 * INSTRUCTION_CLAIM_FIELD_BYTES,
        })
    }

    pub const fn useful_field_products(self) -> u128 {
        self.materialize.useful_field_products
            + self.transitions.useful_field_products
            + self.opening.useful_field_products
    }

    pub const fn issued_field_product_lanes(self) -> u128 {
        self.materialize.issued_field_product_lanes
            + self.transitions.issued_field_product_lanes
            + self.opening.issued_field_product_lanes
    }

    pub const fn cache_optimistic_bytes(self) -> u128 {
        self.materialize.cache_optimistic_bytes()
            + self.transitions.cache_optimistic_bytes()
            + self.opening.cache_optimistic_bytes()
            + self.final_state_readback_bytes
    }

    pub const fn shader_logical_bytes(self) -> u128 {
        self.materialize.shader_logical_bytes()
            + self.transitions.shader_logical_bytes()
            + self.opening.shader_logical_bytes()
            + self.final_state_readback_bytes
    }
}

fn round_up_to_simd(value: u128) -> u128 {
    value.div_ceil(INSTRUCTION_CLAIM_SIMD_WIDTH as u128) * INSTRUCTION_CLAIM_SIMD_WIDTH as u128
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimCpuWorkPlan {
    pub combined_useful_products: u128,
    pub combined_scalar_fmadds: u128,
    pub message_useful_products: u128,
    pub bind_useful_products: u128,
    pub opening_useful_products: u128,
    pub collection_payload_write_bytes: u128,
    pub combined_pass_bytes: u128,
    pub message_state_read_bytes: u128,
    pub bind_state_bytes: u128,
    pub opening_payload_read_bytes: u128,
}

impl InstructionClaimCpuWorkPlan {
    pub fn new(geometry: InstructionClaimGeometry) -> Result<Self, InstructionClaimShapeError> {
        let rows = geometry.rows() as u128;
        let mut message_pairs = 0u128;
        let mut message_outer = 0u128;
        for round in 0..geometry.log_t() {
            let weights = geometry.message(round)?.weights();
            message_pairs += weights.e_in_length() as u128 * weights.e_out_length() as u128;
            message_outer += weights.e_out_length() as u128;
        }
        let opening_out = geometry.opening().e_out_length() as u128;
        Ok(Self {
            combined_useful_products: 4 * rows,
            combined_scalar_fmadds: 7 * rows,
            message_useful_products: 3 * message_pairs
                + 3 * message_outer
                + 3 * geometry.log_t() as u128,
            bind_useful_products: rows - 1 + 2 * geometry.log_t() as u128,
            opening_useful_products: 5 * rows + 5 * opening_out,
            collection_payload_write_bytes: INSTRUCTION_CLAIM_NATIVE_ROW_BYTES * rows,
            combined_pass_bytes: (INSTRUCTION_CLAIM_NATIVE_ROW_BYTES
                + INSTRUCTION_CLAIM_FIELD_BYTES)
                * rows,
            message_state_read_bytes: 2 * INSTRUCTION_CLAIM_FIELD_BYTES * (rows - 1),
            bind_state_bytes: 3 * INSTRUCTION_CLAIM_FIELD_BYTES * (rows - 1),
            opening_payload_read_bytes: INSTRUCTION_CLAIM_NATIVE_ROW_BYTES * rows,
        })
    }

    pub const fn useful_field_products(self) -> u128 {
        self.combined_useful_products
            + self.message_useful_products
            + self.bind_useful_products
            + self.opening_useful_products
    }

    /// Does not include the opaque source reads needed to extract the native
    /// rows or the small equality tables.
    pub const fn visible_payload_bytes(self) -> u128 {
        self.collection_payload_write_bytes
            + self.combined_pass_bytes
            + self.message_state_read_bytes
            + self.bind_state_bytes
            + self.opening_payload_read_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimPromotionGates {
    pub cpu_median_ns: u64,
    pub five_x_wall_ns: u64,
    pub eight_x_wall_ns: u64,
    pub materialize_active_ns: u64,
    pub transitions_active_ns: u64,
    pub opening_active_ns: u64,
    pub total_active_ns: u64,
}

impl InstructionClaimPromotionGates {
    pub fn target(
        opening_architecture: InstructionClaimOpeningArchitecture,
    ) -> Result<Self, InstructionClaimShapeError> {
        let geometry = InstructionClaimGeometry::new(INSTRUCTION_CLAIM_TARGET_ROWS)?;
        let plan = InstructionClaimWorkPlan::new(geometry, opening_architecture)?;
        let materialize_floor = phase_floor_ns(
            plan.materialize.compulsory_bytes,
            plan.materialize.useful_field_products,
            INSTRUCTION_CLAIM_MULTI_ACCUM_PRODUCTS_PER_SECOND,
        );
        let transitions_floor = phase_floor_ns(
            plan.transitions.compulsory_bytes,
            plan.transitions.useful_field_products,
            INSTRUCTION_CLAIM_TRANSITION_PRODUCTS_PER_SECOND,
        );
        let opening_floor = phase_floor_ns(
            plan.opening.compulsory_bytes,
            plan.opening.useful_field_products,
            INSTRUCTION_CLAIM_MULTI_ACCUM_PRODUCTS_PER_SECOND,
        );
        let materialize_active_ns = eighty_percent_cap(materialize_floor);
        let transitions_active_ns = eighty_percent_cap(transitions_floor);
        let opening_active_ns = eighty_percent_cap(opening_floor);
        Ok(Self {
            cpu_median_ns: INSTRUCTION_CLAIM_CPU_MEDIAN_NS,
            five_x_wall_ns: INSTRUCTION_CLAIM_CPU_MEDIAN_NS / INSTRUCTION_CLAIM_REQUIRED_SPEEDUP,
            eight_x_wall_ns: INSTRUCTION_CLAIM_CPU_MEDIAN_NS / INSTRUCTION_CLAIM_STRETCH_SPEEDUP,
            materialize_active_ns,
            transitions_active_ns,
            opening_active_ns,
            total_active_ns: materialize_active_ns + transitions_active_ns + opening_active_ns,
        })
    }
}

fn phase_floor_ns(bytes: u128, products: u128, products_per_second: u128) -> u64 {
    let traffic = div_ceil_u128(
        bytes * 1_000_000_000,
        INSTRUCTION_CLAIM_COPY_BYTES_PER_SECOND,
    );
    let compute = div_ceil_u128(products * 1_000_000_000, products_per_second);
    u64::try_from(traffic.max(compute)).unwrap_or(u64::MAX)
}

fn eighty_percent_cap(floor_ns: u64) -> u64 {
    floor_ns.saturating_mul(5).div_ceil(4)
}

const fn div_ceil_u128(lhs: u128, rhs: u128) -> u128 {
    lhs.div_ceil(rhs)
}
