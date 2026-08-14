//! Checked host/shader ABI for the InstructionInput successor.

use core::mem::{align_of, size_of};

use thiserror::Error;

pub const INSTRUCTION_INPUT_SUCCESSOR_TABLES: usize = 8;
pub const INSTRUCTION_INPUT_SUCCESSOR_COEFFICIENTS: usize = 3;
pub const INSTRUCTION_INPUT_SUCCESSOR_ROW_WORDS: usize = 6;
pub const INSTRUCTION_INPUT_SUCCESSOR_ROW_BYTES: usize = 48;
pub const INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH: usize = 32;

pub const MATERIALIZE_PIPELINE: &str = "solinas_instruction_input_successor_materialize";
pub const DENSE_MESSAGE_PIPELINE: &str = "solinas_instruction_input_successor_dense_message";

pub const ROW_RS1: usize = 0;
pub const ROW_UNEXPANDED_PC: usize = 1;
pub const ROW_EFFECTIVE_RS2: usize = 2;
pub const ROW_IMM_LOW: usize = 3;
pub const ROW_IMM_HIGH: usize = 4;
pub const ROW_FLAGS: usize = 5;

pub const FLAG_LOAD: u32 = 0;
pub const FLAG_IMM_POSITIVE: u32 = 18;
pub const FLAG_LEFT_OPERAND_IS_RS1: u32 = 20;
pub const FLAG_LEFT_OPERAND_IS_PC: u32 = 21;
pub const FLAG_RIGHT_OPERAND_IS_RS2: u32 = 22;
pub const FLAG_RIGHT_OPERAND_IS_IMM: u32 = 23;

/// Canonical table and output-claim order.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InstructionInputSuccessorTable {
    LeftOperandIsRs1 = 0,
    Rs1Value = 1,
    LeftOperandIsPc = 2,
    UnexpandedPc = 3,
    RightOperandIsRs2 = 4,
    Rs2Value = 5,
    RightOperandIsImm = 6,
    Imm = 7,
}

impl InstructionInputSuccessorTable {
    pub const ALL: [Self; INSTRUCTION_INPUT_SUCCESSOR_TABLES] = [
        Self::LeftOperandIsRs1,
        Self::Rs1Value,
        Self::LeftOperandIsPc,
        Self::UnexpandedPc,
        Self::RightOperandIsRs2,
        Self::Rs2Value,
        Self::RightOperandIsImm,
        Self::Imm,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Byte-for-byte view of the resident production `InstructionInputRow`.
///
/// Integration must borrow the existing allocation. Constructing a second
/// array of this type would violate the producer/consumer lifetime contract.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionInputSuccessorRow {
    words: [u64; INSTRUCTION_INPUT_SUCCESSOR_ROW_WORDS],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InstructionInputSuccessorSelectors {
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
}

impl InstructionInputSuccessorSelectors {
    pub const fn from_array(values: [bool; 4]) -> Self {
        Self {
            left_is_rs1: values[0],
            left_is_pc: values[1],
            right_is_rs2: values[2],
            right_is_imm: values[3],
        }
    }
}

impl InstructionInputSuccessorRow {
    pub const fn from_words(words: [u64; INSTRUCTION_INPUT_SUCCESSOR_ROW_WORDS]) -> Self {
        Self { words }
    }

    pub fn from_components(
        rs1: u64,
        unexpanded_pc: u64,
        effective_rs2: u64,
        imm: i128,
        selectors: InstructionInputSuccessorSelectors,
    ) -> Self {
        let magnitude = imm.unsigned_abs();
        let mut flags = 0u64;
        for (bit, value) in [
            (FLAG_IMM_POSITIVE, imm >= 0),
            (FLAG_LEFT_OPERAND_IS_RS1, selectors.left_is_rs1),
            (FLAG_LEFT_OPERAND_IS_PC, selectors.left_is_pc),
            (FLAG_RIGHT_OPERAND_IS_RS2, selectors.right_is_rs2),
            (FLAG_RIGHT_OPERAND_IS_IMM, selectors.right_is_imm),
        ] {
            flags |= u64::from(value) << bit;
        }
        Self {
            words: [
                rs1,
                unexpanded_pc,
                effective_rs2,
                magnitude as u64,
                (magnitude >> 64) as u64,
                flags,
            ],
        }
    }

    pub const fn words(self) -> [u64; INSTRUCTION_INPUT_SUCCESSOR_ROW_WORDS] {
        self.words
    }

    pub const fn word(self, index: usize) -> u64 {
        self.words[index]
    }

    pub const fn flag(self, bit: u32) -> bool {
        ((self.words[ROW_FLAGS] >> bit) & 1) != 0
    }

    pub const fn imm_magnitude(self) -> u128 {
        self.words[ROW_IMM_LOW] as u128 | ((self.words[ROW_IMM_HIGH] as u128) << 64)
    }

    pub fn validate(self) -> Result<(), InstructionInputSuccessorError> {
        if self.flag(FLAG_LOAD) && self.words[ROW_EFFECTIVE_RS2] != 0 {
            return Err(InstructionInputSuccessorError::UnmaskedLoadRs2);
        }
        let magnitude = self.imm_magnitude();
        let positive = self.flag(FLAG_IMM_POSITIVE);
        if magnitude == 0 && !positive {
            return Err(InstructionInputSuccessorError::NegativeZeroImmediate);
        }
        if magnitude > (1u128 << 127) || (magnitude == (1u128 << 127) && positive) {
            return Err(InstructionInputSuccessorError::InvalidImmediateEncoding);
        }
        Ok(())
    }
}

const _: [(); INSTRUCTION_INPUT_SUCCESSOR_ROW_BYTES] =
    [(); size_of::<InstructionInputSuccessorRow>()];
const _: [(); 16] = [(); align_of::<InstructionInputSuccessorRow>()];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InstructionInputSuccessorKernel {
    Materialize,
    DenseMessage,
}

impl InstructionInputSuccessorKernel {
    pub const ALL: [Self; 2] = [Self::Materialize, Self::DenseMessage];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Materialize => MATERIALIZE_PIPELINE,
            Self::DenseMessage => DENSE_MESSAGE_PIPELINE,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct InstructionInputSuccessorMaterializeParams {
    pub(crate) source_elements: u32,
    pub(crate) bound_elements: u32,
    pub(crate) reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct InstructionInputSuccessorDenseMessageParams {
    pub(crate) table_elements: u32,
    pub(crate) e_in_length: u32,
    pub(crate) e_out_length: u32,
    pub(crate) reserved: u32,
}

const _: [(); 16] = [(); size_of::<InstructionInputSuccessorMaterializeParams>()];
const _: [(); 4] = [(); align_of::<InstructionInputSuccessorMaterializeParams>()];
const _: [(); 16] = [(); size_of::<InstructionInputSuccessorDenseMessageParams>()];
const _: [(); 4] = [(); align_of::<InstructionInputSuccessorDenseMessageParams>()];

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum InstructionInputSuccessorError {
    #[error("InstructionInput successor requires a power-of-two row count of at least four")]
    InvalidRows,
    #[error("InstructionInput successor geometry exceeds the shader's 32-bit index space")]
    ShaderIndexOverflow,
    #[error("InstructionInput successor geometry arithmetic overflowed")]
    GeometryOverflow,
    #[error(
        "InstructionInput successor buffer requires {requested} bytes but Metal allows {maximum}"
    )]
    BufferTooLong { requested: u64, maximum: u64 },
    #[error("InstructionInput successor equality split is inconsistent: tables={table_elements}, e_in={e_in}, e_out={e_out}")]
    InvalidEqualitySplit {
        table_elements: usize,
        e_in: usize,
        e_out: usize,
    },
    #[error("InstructionInput successor table storage has {got} values; expected {expected}")]
    InvalidTableStorage { expected: usize, got: usize },
    #[error("InstructionInput successor threadgroup width must be a nonzero SIMD-width multiple")]
    InvalidThreadgroupWidth,
    #[error("InstructionInput successor row encodes a negative zero immediate")]
    NegativeZeroImmediate,
    #[error("InstructionInput successor row is not a signed i128 magnitude/sign encoding")]
    InvalidImmediateEncoding,
    #[error("InstructionInput successor load row contains a nonzero effective rs2")]
    UnmaskedLoadRs2,
}
