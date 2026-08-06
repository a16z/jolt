//! Stable host and shader ABI for the isolated primitive probes.

use core::mem::{align_of, size_of};

use thiserror::Error;

pub const MUL_U64_PIPELINE: &str = "solinas_half_width_mul_u64_probe";
pub const MUL_SIGNED_U64_PIPELINE: &str = "solinas_half_width_mul_signed_u64_probe";
pub const MUL_U64_DELTA_PIPELINE: &str = "solinas_half_width_mul_u64_delta_probe";

/// Interpretation of the two-word operand ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfWidthDomain {
    /// `coefficient * primary` for `primary in 0..=u64::MAX`.
    Unsigned,
    /// `coefficient * (-1)^secondary * primary`; `secondary` is zero or one.
    SignedMagnitude,
    /// `coefficient * (primary - secondary)` over two unsigned endpoints.
    UnsignedDelta,
}

impl HalfWidthDomain {
    /// Minimum semantic input bytes per element, excluding coefficient/output.
    pub const fn semantic_operand_bytes(self) -> usize {
        match self {
            Self::Unsigned => size_of::<u64>(),
            Self::SignedMagnitude => size_of::<u64>() + size_of::<u8>(),
            Self::UnsignedDelta => 2 * size_of::<u64>(),
        }
    }
}

/// Two-word ABI shared by all domains.
///
/// Unsigned probes require `secondary == 0`. Signed-magnitude probes interpret
/// `secondary` as a one-bit negative flag. Delta probes interpret the words as
/// the ordered endpoints `primary - secondary`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HalfWidthOperand {
    pub primary: u64,
    pub secondary: u64,
}

impl HalfWidthOperand {
    pub const fn unsigned(scalar: u64) -> Self {
        Self {
            primary: scalar,
            secondary: 0,
        }
    }

    /// Encodes `magnitude` with a canonical sign. Zero is never negative.
    pub const fn signed_magnitude(magnitude: u64, negative: bool) -> Self {
        Self {
            primary: magnitude,
            secondary: (negative && magnitude != 0) as u64,
        }
    }

    /// Encodes the mathematical integer `minuend - subtrahend` without an
    /// `i64` narrowing. Its magnitude can be `u64::MAX`.
    pub const fn delta(minuend: u64, subtrahend: u64) -> Self {
        Self {
            primary: minuend,
            secondary: subtrahend,
        }
    }

    /// Raw constructor for ABI decoding and negative validation tests.
    pub const fn from_words(words: [u64; 2]) -> Self {
        Self {
            primary: words[0],
            secondary: words[1],
        }
    }

    pub const fn words(self) -> [u64; 2] {
        [self.primary, self.secondary]
    }

    pub fn validate(self, domain: HalfWidthDomain) -> Result<(), HalfWidthOperandError> {
        match domain {
            HalfWidthDomain::Unsigned if self.secondary != 0 => Err(
                HalfWidthOperandError::NonzeroUnsignedPadding(self.secondary),
            ),
            HalfWidthDomain::SignedMagnitude if self.secondary > 1 => {
                Err(HalfWidthOperandError::InvalidSignWord(self.secondary))
            }
            HalfWidthDomain::SignedMagnitude if self.primary == 0 && self.secondary == 1 => {
                Err(HalfWidthOperandError::NegativeZero)
            }
            _ => Ok(()),
        }
    }
}

const _: [(); 16] = [(); size_of::<HalfWidthOperand>()];
const _: [(); 16] = [(); align_of::<HalfWidthOperand>()];

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum HalfWidthOperandError {
    #[error("unsigned half-width padding must be zero, got {0}")]
    NonzeroUnsignedPadding(u64),
    #[error("signed half-width sign word must be zero or one, got {0}")]
    InvalidSignWord(u64),
    #[error("signed half-width zero must be nonnegative")]
    NegativeZero,
}

/// Probe entry points. Chain variants assign `ILP` independent field values
/// to each Metal thread and reuse their operands for every iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfWidthProbe {
    MulU64,
    MulSignedU64,
    MulU64Delta,
    ChainU64Ilp1,
    ChainU64Ilp2,
    ChainU64Ilp4,
    ChainU64Ilp8,
    ChainSignedU64Ilp1,
    ChainSignedU64Ilp2,
    ChainSignedU64Ilp4,
    ChainSignedU64Ilp8,
    ChainU64DeltaIlp1,
    ChainU64DeltaIlp2,
    ChainU64DeltaIlp4,
    ChainU64DeltaIlp8,
}

impl HalfWidthProbe {
    pub const ALL: [Self; 15] = [
        Self::MulU64,
        Self::MulSignedU64,
        Self::MulU64Delta,
        Self::ChainU64Ilp1,
        Self::ChainU64Ilp2,
        Self::ChainU64Ilp4,
        Self::ChainU64Ilp8,
        Self::ChainSignedU64Ilp1,
        Self::ChainSignedU64Ilp2,
        Self::ChainSignedU64Ilp4,
        Self::ChainSignedU64Ilp8,
        Self::ChainU64DeltaIlp1,
        Self::ChainU64DeltaIlp2,
        Self::ChainU64DeltaIlp4,
        Self::ChainU64DeltaIlp8,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::MulU64 => MUL_U64_PIPELINE,
            Self::MulSignedU64 => MUL_SIGNED_U64_PIPELINE,
            Self::MulU64Delta => MUL_U64_DELTA_PIPELINE,
            Self::ChainU64Ilp1 => "solinas_half_width_chain_u64_ilp1",
            Self::ChainU64Ilp2 => "solinas_half_width_chain_u64_ilp2",
            Self::ChainU64Ilp4 => "solinas_half_width_chain_u64_ilp4",
            Self::ChainU64Ilp8 => "solinas_half_width_chain_u64_ilp8",
            Self::ChainSignedU64Ilp1 => "solinas_half_width_chain_signed_u64_ilp1",
            Self::ChainSignedU64Ilp2 => "solinas_half_width_chain_signed_u64_ilp2",
            Self::ChainSignedU64Ilp4 => "solinas_half_width_chain_signed_u64_ilp4",
            Self::ChainSignedU64Ilp8 => "solinas_half_width_chain_signed_u64_ilp8",
            Self::ChainU64DeltaIlp1 => "solinas_half_width_chain_u64_delta_ilp1",
            Self::ChainU64DeltaIlp2 => "solinas_half_width_chain_u64_delta_ilp2",
            Self::ChainU64DeltaIlp4 => "solinas_half_width_chain_u64_delta_ilp4",
            Self::ChainU64DeltaIlp8 => "solinas_half_width_chain_u64_delta_ilp8",
        }
    }

    pub const fn domain(self) -> HalfWidthDomain {
        match self {
            Self::MulU64
            | Self::ChainU64Ilp1
            | Self::ChainU64Ilp2
            | Self::ChainU64Ilp4
            | Self::ChainU64Ilp8 => HalfWidthDomain::Unsigned,
            Self::MulSignedU64
            | Self::ChainSignedU64Ilp1
            | Self::ChainSignedU64Ilp2
            | Self::ChainSignedU64Ilp4
            | Self::ChainSignedU64Ilp8 => HalfWidthDomain::SignedMagnitude,
            Self::MulU64Delta
            | Self::ChainU64DeltaIlp1
            | Self::ChainU64DeltaIlp2
            | Self::ChainU64DeltaIlp4
            | Self::ChainU64DeltaIlp8 => HalfWidthDomain::UnsignedDelta,
        }
    }

    pub const fn independent_chains(self) -> usize {
        match self {
            Self::ChainU64Ilp2 | Self::ChainSignedU64Ilp2 | Self::ChainU64DeltaIlp2 => 2,
            Self::ChainU64Ilp4 | Self::ChainSignedU64Ilp4 | Self::ChainU64DeltaIlp4 => 4,
            Self::ChainU64Ilp8 | Self::ChainSignedU64Ilp8 | Self::ChainU64DeltaIlp8 => 8,
            _ => 1,
        }
    }

    pub const fn is_chain(self) -> bool {
        !matches!(self, Self::MulU64 | Self::MulSignedU64 | Self::MulU64Delta)
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum HalfWidthProbeError {
    #[error("half-width probes require at least one coefficient")]
    EmptyInput,
    #[error("half-width coefficient and operand lengths differ: coefficients={coefficients}, operands={operands}")]
    LengthMismatch {
        coefficients: usize,
        operands: usize,
    },
    #[error("half-width probe iteration count must be nonzero")]
    ZeroIterations,
    #[error("pointwise half-width probes require one iteration, got {0}")]
    PointwiseIterations(u32),
    #[error("Solinas offset must be nonzero")]
    InvalidOffset,
    #[error("half-width input length {0} exceeds the shader's 32-bit element count")]
    InputTooLong(usize),
    #[error("{probe} requires an element count divisible by its ILP ({ilp})")]
    MisalignedElementCount { probe: &'static str, ilp: usize },
    #[error("half-width buffer length overflow for {0} elements")]
    BufferLengthOverflow(usize),
    #[error("half-width buffer requires {requested} bytes but Metal allows {maximum}")]
    BufferTooLong { requested: u64, maximum: u64 },
    #[error("half-width coefficient[{index}] is not canonical for 2^128 - {offset}")]
    NonCanonicalCoefficient { index: usize, offset: u32 },
    #[error("unsigned half-width operand[{index}] has nonzero padding {value}")]
    NonzeroUnsignedOperand { index: usize, value: u64 },
    #[error("signed half-width operand[{index}] has invalid sign word {value}")]
    InvalidSignedOperand { index: usize, value: u64 },
    #[error("signed half-width operand[{index}] encodes negative zero")]
    NegativeZeroOperand { index: usize },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HalfWidthProbeParams {
    pub(super) elements: u32,
    pub(super) iterations: u32,
}

const _: [(); 8] = [(); size_of::<HalfWidthProbeParams>()];
const _: [(); 4] = [(); align_of::<HalfWidthProbeParams>()];
