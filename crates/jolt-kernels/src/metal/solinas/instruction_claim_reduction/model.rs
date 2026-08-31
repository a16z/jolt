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
    copy_field_getters! { pub, {
        e_in_length: usize,
        e_out_length: usize,
    }}
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
    copy_field_getters! { pub, {
        round: usize,
        source_elements: usize,
        state_elements: usize,
        weights: InstructionClaimWeightGeometry,
    }}
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

    copy_field_getters! { pub, {
        rows: usize,
        log_t: usize,
    }}

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
    copy_field_getters! { pub, {
        input_count: usize,
        output_count: usize,
        dispatched_threads: usize,
    }}
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

    copy_field_getters! { pub, { columns: usize }}

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

    copy_field_getters! { pub, {
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
    }}

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
