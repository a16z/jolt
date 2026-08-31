//! Native operand ABI and host-side protocol boundary.

use core::mem::{align_of, size_of};

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use thiserror::Error;

use crate::optimized::instruction_claim_reduction::InstructionOperandRow;

use super::model::{validate_rows, InstructionClaimShapeError};

pub const INSTRUCTION_CLAIM_AKITA_OFFSET: u32 = super::super::AKITA_OFFSET_FFFFA7F7;
pub const INSTRUCTION_CLAIM_CORE_WORDS: usize = 5;
pub const INSTRUCTION_CLAIM_RIGHT_INPUT_WORDS: usize = 2;
pub const INSTRUCTION_CLAIM_NONTRIVIAL_GAMMA_POWERS: usize = 4;
pub const INSTRUCTION_CLAIM_MESSAGE_COLUMNS: usize = 2;
pub const INSTRUCTION_CLAIM_ALIASED_OPENINGS: usize = 2;
pub const INSTRUCTION_CLAIM_CORE_OPENINGS: usize = 4;
pub const INSTRUCTION_CLAIM_ALL_OPENINGS: usize = 5;
pub const INSTRUCTION_CLAIM_SIMD_WIDTH: usize = 32;

pub const MATERIALIZE_PIPELINE: &str = "solinas_instruction_claim_materialize_message";
pub const TRANSITION_PIPELINE: &str = "solinas_instruction_claim_bind_message";
pub const CORE_OPENING_PIPELINE: &str = "solinas_instruction_claim_open_core";
pub const ALIASED_OPENING_PIPELINE: &str = "solinas_instruction_claim_open_lookup_operands";
pub const ALL_OPENING_PIPELINE: &str = "solinas_instruction_claim_open_all";
pub const REDUCTION_PIPELINE: &str = "solinas_instruction_claim_reduce";

const CORE_LOOKUP_OUTPUT: usize = 0;
const CORE_LEFT_LOOKUP_OPERAND: usize = 1;
const CORE_RIGHT_LOOKUP_LOW: usize = 2;
const CORE_RIGHT_LOOKUP_HIGH: usize = 3;
const CORE_LEFT_INSTRUCTION_INPUT: usize = 4;

/// One row's four unsigned operands for host-side staging and reference checks.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionClaimCoreRow {
    words: [u64; INSTRUCTION_CLAIM_CORE_WORDS],
}

/// The signed 128-bit instruction input in two's-complement word order.
///
/// This is a separate 16-byte plane; all bit patterns are canonical `i128`
/// encodings.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionClaimRightInput {
    words: [u64; INSTRUCTION_CLAIM_RIGHT_INPUT_WORDS],
}

/// The unsigned 128-bit lookup operand in little-endian word order.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionClaimRightLookup {
    words: [u64; 2],
}

const _: [(); 40] = [(); size_of::<InstructionClaimCoreRow>()];
const _: [(); 8] = [(); align_of::<InstructionClaimCoreRow>()];
const _: [(); 16] = [(); size_of::<InstructionClaimRightInput>()];
const _: [(); 8] = [(); align_of::<InstructionClaimRightInput>()];
const _: [(); 16] = [(); size_of::<InstructionClaimRightLookup>()];
const _: [(); 8] = [(); align_of::<InstructionClaimRightLookup>()];

impl InstructionClaimCoreRow {
    pub const fn new(
        lookup_output: u64,
        left_lookup_operand: u64,
        right_lookup_operand: u128,
        left_instruction_input: u64,
    ) -> Self {
        Self {
            words: [
                lookup_output,
                left_lookup_operand,
                right_lookup_operand as u64,
                (right_lookup_operand >> 64) as u64,
                left_instruction_input,
            ],
        }
    }

    pub const fn from_words(words: [u64; INSTRUCTION_CLAIM_CORE_WORDS]) -> Self {
        Self { words }
    }

    copy_field_getters! { pub, { words: [u64; INSTRUCTION_CLAIM_CORE_WORDS] }}

    pub const fn lookup_output(self) -> u64 {
        self.words[CORE_LOOKUP_OUTPUT]
    }

    pub const fn left_lookup_operand(self) -> u64 {
        self.words[CORE_LEFT_LOOKUP_OPERAND]
    }

    pub const fn right_lookup_operand(self) -> u128 {
        self.words[CORE_RIGHT_LOOKUP_LOW] as u128
            | ((self.words[CORE_RIGHT_LOOKUP_HIGH] as u128) << 64)
    }

    pub const fn left_instruction_input(self) -> u64 {
        self.words[CORE_LEFT_INSTRUCTION_INPUT]
    }

    /// Evaluates the gamma-combined operand at this row.
    pub fn combined<F: Field>(self, right_input: InstructionClaimRightInput, gamma: F) -> F {
        let powers = nontrivial_gamma_powers(gamma);
        F::from_u64(self.lookup_output())
            + powers[0] * F::from_u64(self.left_lookup_operand())
            + powers[1] * F::from_u128(self.right_lookup_operand())
            + powers[2] * F::from_u64(self.left_instruction_input())
            + powers[3] * F::from_i128(right_input.value())
    }

    /// Returns the four core opening columns in protocol order.
    pub fn fields<F: Field>(self) -> [F; 4] {
        [
            F::from_u64(self.lookup_output()),
            F::from_u64(self.left_lookup_operand()),
            F::from_u128(self.right_lookup_operand()),
            F::from_u64(self.left_instruction_input()),
        ]
    }
}

impl InstructionClaimRightInput {
    pub const fn new(value: i128) -> Self {
        let bits = value as u128;
        Self {
            words: [bits as u64, (bits >> 64) as u64],
        }
    }

    pub const fn from_words(words: [u64; INSTRUCTION_CLAIM_RIGHT_INPUT_WORDS]) -> Self {
        Self { words }
    }

    copy_field_getters! { pub, { words: [u64; INSTRUCTION_CLAIM_RIGHT_INPUT_WORDS] }}

    pub const fn value(self) -> i128 {
        (self.words[0] as u128 | ((self.words[1] as u128) << 64)) as i128
    }

    pub fn field<F: Field>(self) -> F {
        F::from_i128(self.value())
    }
}

impl InstructionClaimRightLookup {
    pub const fn new(value: u128) -> Self {
        Self {
            words: [value as u64, (value >> 64) as u64],
        }
    }

    pub const fn from_words(words: [u64; 2]) -> Self {
        Self { words }
    }

    copy_field_getters! { pub, { words: [u64; 2] }}

    pub const fn value(self) -> u128 {
        self.words[0] as u128 | ((self.words[1] as u128) << 64)
    }
}

/// Five Metal-visible operand planes. No log-26 buffer exceeds 1 GiB.
#[derive(Debug, Eq, PartialEq)]
pub struct InstructionClaimOperandPlanes {
    lookup_output: Vec<u64>,
    left_lookup_operand: Vec<u64>,
    right_lookup_operand: Vec<InstructionClaimRightLookup>,
    left_instruction_input: Vec<u64>,
    right_instruction_input: Vec<InstructionClaimRightInput>,
}

impl InstructionClaimOperandPlanes {
    pub fn new(
        lookup_output: Vec<u64>,
        left_lookup_operand: Vec<u64>,
        right_lookup_operand: Vec<InstructionClaimRightLookup>,
        left_instruction_input: Vec<u64>,
        right_instruction_input: Vec<InstructionClaimRightInput>,
    ) -> Result<Self, InstructionClaimShapeError> {
        let rows = lookup_output.len();
        validate_rows(rows)?;
        for (name, len) in [
            ("left lookup operand", left_lookup_operand.len()),
            ("right lookup operand", right_lookup_operand.len()),
            ("left instruction input", left_instruction_input.len()),
            ("right instruction input", right_instruction_input.len()),
        ] {
            if len != rows {
                return Err(InstructionClaimShapeError::OperandPlaneLength {
                    name,
                    expected: rows,
                    got: len,
                });
            }
        }
        Ok(Self {
            lookup_output,
            left_lookup_operand,
            right_lookup_operand,
            left_instruction_input,
            right_instruction_input,
        })
    }

    pub const fn len(&self) -> usize {
        self.lookup_output.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.lookup_output.is_empty()
    }

    pub fn lookup_output(&self) -> &[u64] {
        &self.lookup_output
    }

    pub fn left_lookup_operand(&self) -> &[u64] {
        &self.left_lookup_operand
    }

    pub fn right_lookup_operand(&self) -> &[InstructionClaimRightLookup] {
        &self.right_lookup_operand
    }

    pub fn left_instruction_input(&self) -> &[u64] {
        &self.left_instruction_input
    }

    pub fn right_instruction_input(&self) -> &[InstructionClaimRightInput] {
        &self.right_instruction_input
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub(super) fn row(
        &self,
        index: usize,
    ) -> (InstructionClaimCoreRow, InstructionClaimRightInput) {
        (
            InstructionClaimCoreRow::new(
                self.lookup_output[index],
                self.left_lookup_operand[index],
                self.right_lookup_operand[index].value(),
                self.left_instruction_input[index],
            ),
            self.right_instruction_input[index],
        )
    }
}

/// Converts the optimized CPU row store into the checked Metal-visible planes.
pub fn split_operand_rows(
    rows: &[InstructionOperandRow],
) -> Result<InstructionClaimOperandPlanes, InstructionClaimShapeError> {
    let mut lookup_output = Vec::with_capacity(rows.len());
    let mut left_lookup_operand = Vec::with_capacity(rows.len());
    let mut right_lookup_operand = Vec::with_capacity(rows.len());
    let mut left_instruction_input = Vec::with_capacity(rows.len());
    let mut right_instruction_input = Vec::with_capacity(rows.len());
    for row in rows {
        lookup_output.push(row.lookup_output.0);
        left_lookup_operand.push(row.left_lookup_operand.0);
        right_lookup_operand.push(InstructionClaimRightLookup::new(row.right_lookup_operand.0));
        left_instruction_input.push(row.left_instruction_input.0);
        right_instruction_input.push(InstructionClaimRightInput::new(
            row.right_instruction_input.0,
        ));
    }
    InstructionClaimOperandPlanes::new(
        lookup_output,
        left_lookup_operand,
        right_lookup_operand,
        left_instruction_input,
        right_instruction_input,
    )
}

pub fn nontrivial_gamma_powers<F: Field>(gamma: F) -> [F; 4] {
    let gamma_squared = gamma * gamma;
    [
        gamma,
        gamma_squared,
        gamma_squared * gamma,
        gamma_squared * gamma_squared,
    ]
}

/// Selects the self-contained opening scan required after all sumcheck rounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InstructionClaimOpeningMode {
    CoreAndRecover,
    AllColumns,
}

impl InstructionClaimOpeningMode {
    pub fn for_gamma<F: Field>(gamma: F) -> Self {
        if gamma == F::zero() {
            Self::AllColumns
        } else {
            Self::CoreAndRecover
        }
    }

    pub const fn columns(self) -> usize {
        match self {
            Self::CoreAndRecover => INSTRUCTION_CLAIM_CORE_OPENINGS,
            Self::AllColumns => INSTRUCTION_CLAIM_ALL_OPENINGS,
        }
    }

    pub const fn pipeline(self) -> &'static str {
        match self {
            Self::CoreAndRecover => CORE_OPENING_PIPELINE,
            Self::AllColumns => ALL_OPENING_PIPELINE,
        }
    }

    pub const fn aliased_pipeline() -> &'static str {
        ALIASED_OPENING_PIPELINE
    }
}

/// The five reduced openings in protocol declaration order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimOpenings<F> {
    pub lookup_output: F,
    pub left_lookup_operand: F,
    pub right_lookup_operand: F,
    pub left_instruction_input: F,
    pub right_instruction_input: F,
}

/// Openings supplied by ProductRemainder at the same reduced point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionClaimAliasedOpenings<F> {
    pub lookup_output: F,
    pub left_instruction_input: F,
    pub right_instruction_input: F,
}

impl<F: Field> InstructionClaimOpenings<F> {
    pub const fn into_array(self) -> [F; INSTRUCTION_CLAIM_ALL_OPENINGS] {
        [
            self.lookup_output,
            self.left_lookup_operand,
            self.right_lookup_operand,
            self.left_instruction_input,
            self.right_instruction_input,
        ]
    }

    pub fn combined(self, gamma: F) -> F {
        let powers = nontrivial_gamma_powers(gamma);
        self.lookup_output
            + powers[0] * self.left_lookup_operand
            + powers[1] * self.right_lookup_operand
            + powers[2] * self.left_instruction_input
            + powers[3] * self.right_instruction_input
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum InstructionClaimOpeningError {
    #[error("gamma-zero instruction openings require the scanned right-input column")]
    ZeroGammaRecovery,
    #[error("all-column instruction openings are missing the scanned right-input column")]
    MissingRightInputOpening,
    #[error("nonzero gamma unexpectedly had no fourth-power inverse")]
    GammaInverseFailed,
    #[error("scanned instruction openings do not recombine to the final resident claim")]
    CombinedClaimMismatch,
}

/// Applies the final low-to-high bind after reading the last two resident values.
pub fn finish_bind<F: Field>(state: [F; 2], challenge: F) -> F {
    state[0] + challenge * (state[1] - state[0])
}

/// Applies the host Gruen linear factor to device-returned `q(0), q(2)`.
pub fn scale_q_endpoints<F: Field>(q_endpoints: [F; 2], linear_evals: [F; 2]) -> [F; 2] {
    let l_at_2 = linear_evals[1] + linear_evals[1] - linear_evals[0];
    [linear_evals[0] * q_endpoints[0], l_at_2 * q_endpoints[1]]
}

/// Constructs the member polynomial returned to the generic batched driver.
///
/// The driver, not this helper, combines members, absorbs the batched
/// polynomial, and draws the next challenge.
pub fn round_polynomial_from_q_endpoints<F: jolt_field::JoltField>(
    previous_claim: F,
    q_endpoints: [F; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
    linear_evals: [F; 2],
) -> UnivariatePoly<F> {
    let scaled = scale_q_endpoints(q_endpoints, linear_evals);
    UnivariatePoly::from_evals_and_hint(previous_claim, &scaled)
}

pub fn recover_right_input<F: Field>(
    gamma: F,
    combined_claim: F,
    core_openings: [F; INSTRUCTION_CLAIM_CORE_OPENINGS],
) -> Result<F, InstructionClaimOpeningError> {
    if gamma == F::zero() {
        return Err(InstructionClaimOpeningError::ZeroGammaRecovery);
    }
    let powers = nontrivial_gamma_powers(gamma);
    let gamma_fourth_inverse = powers[3]
        .inverse()
        .ok_or(InstructionClaimOpeningError::GammaInverseFailed)?;
    let remainder = combined_claim
        - core_openings[0]
        - powers[0] * core_openings[1]
        - powers[1] * core_openings[2]
        - powers[2] * core_openings[3];
    Ok(gamma_fourth_inverse * remainder)
}

pub fn finalize_openings<F: Field>(
    mode: InstructionClaimOpeningMode,
    gamma: F,
    combined_claim: F,
    core_openings: [F; INSTRUCTION_CLAIM_CORE_OPENINGS],
    scanned_right_input: Option<F>,
) -> Result<InstructionClaimOpenings<F>, InstructionClaimOpeningError> {
    let right_instruction_input = match mode {
        InstructionClaimOpeningMode::CoreAndRecover => {
            recover_right_input(gamma, combined_claim, core_openings)?
        }
        InstructionClaimOpeningMode::AllColumns => {
            scanned_right_input.ok_or(InstructionClaimOpeningError::MissingRightInputOpening)?
        }
    };
    let openings = InstructionClaimOpenings {
        lookup_output: core_openings[0],
        left_lookup_operand: core_openings[1],
        right_lookup_operand: core_openings[2],
        left_instruction_input: core_openings[3],
        right_instruction_input,
    };
    if openings.combined(gamma) != combined_claim {
        return Err(InstructionClaimOpeningError::CombinedClaimMismatch);
    }
    Ok(openings)
}

/// Reconstructs the five outputs when ProductRemainder supplies its aliases.
pub fn finalize_aliased_openings<F: Field>(
    gamma: F,
    combined_claim: F,
    lookup_operands: [F; INSTRUCTION_CLAIM_ALIASED_OPENINGS],
    aliases: InstructionClaimAliasedOpenings<F>,
) -> Result<InstructionClaimOpenings<F>, InstructionClaimOpeningError> {
    let openings = InstructionClaimOpenings {
        lookup_output: aliases.lookup_output,
        left_lookup_operand: lookup_operands[0],
        right_lookup_operand: lookup_operands[1],
        left_instruction_input: aliases.left_instruction_input,
        right_instruction_input: aliases.right_instruction_input,
    };
    if openings.combined(gamma) != combined_claim {
        return Err(InstructionClaimOpeningError::CombinedClaimMismatch);
    }
    Ok(openings)
}

/// The verifier's output expression for this member.
pub fn verifier_output_term<F: Field>(
    eq_spartan: F,
    openings: InstructionClaimOpenings<F>,
    gamma: F,
) -> F {
    eq_spartan * openings.combined(gamma)
}
