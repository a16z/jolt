//! Checked geometry, instruction-shape budget, and fail-closed promotion gates.

use core::mem::size_of;

use super::super::Fp128;
use super::abi::{
    HalfWidthDomain, HalfWidthOperand, HalfWidthOperandError, HalfWidthProbe, HalfWidthProbeError,
    HalfWidthProbeParams,
};

pub const HALF_WIDTH_AKITA_OFFSET: u32 = 0xffff_a7f7;
pub const HALF_WIDTH_AKITA_MODULUS: u128 = u128::MAX - HALF_WIDTH_AKITA_OFFSET as u128 + 1;
pub const HALF_WIDTH_SIMD_WIDTH: usize = 32;

pub const FULL_WIDTH_CONTROL_PRODUCTS_PER_SECOND: u64 = 16_420_000_000;
pub const MINIMUM_RELATIVE_SPEEDUP_BPS: u64 = 16_000;
pub const MINIMUM_HALF_WIDTH_PRODUCTS_PER_SECOND: u64 = 26_272_000_000;
pub const MAXIMUM_RELATIVE_MAD_BPS: u64 = 300;
pub const MINIMUM_RESIDENT_THREADGROUPS: u32 = 2;
pub const TARGET_RESIDENT_THREADGROUPS: u32 = 4;
pub const TARGET_CHAIN_ELEMENTS: usize = 1 << 20;
pub const TARGET_CHAIN_ITERATIONS: u32 = 512;
pub const TARGET_CHAIN_OPERATIONS: u64 =
    TARGET_CHAIN_ELEMENTS as u64 * TARGET_CHAIN_ITERATIONS as u64;
pub const TARGET_MAX_GPU_ACTIVE_NS: u64 = 20_435_099;

const FP128_BYTES: u64 = size_of::<Fp128>() as u64;
const OPERAND_BYTES: u64 = size_of::<HalfWidthOperand>() as u64;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HalfWidthProbeShape {
    params: HalfWidthProbeParams,
    grid_threads: usize,
    field_buffer_bytes: u64,
    operand_buffer_bytes: u64,
    allocated_bytes: u64,
    semantic_bytes: u64,
    operation_count: u64,
}

impl HalfWidthProbeShape {
    pub(super) const fn params(self) -> HalfWidthProbeParams {
        self.params
    }

    pub const fn grid_threads(self) -> usize {
        self.grid_threads
    }

    /// Bytes in each coefficient and output buffer.
    pub const fn field_buffer_bytes(self) -> u64 {
        self.field_buffer_bytes
    }

    /// Bytes in the common two-word operand buffer.
    pub const fn operand_buffer_bytes(self) -> u64 {
        self.operand_buffer_bytes
    }

    /// Coefficient, operand, output, and parameter allocation bytes.
    pub const fn allocated_bytes(self) -> u64 {
        self.allocated_bytes
    }

    /// Minimum meaningful bytes, before ABI padding or cache-line effects.
    pub const fn semantic_bytes(self) -> u64 {
        self.semantic_bytes
    }

    pub const fn operation_count(self) -> u64 {
        self.operation_count
    }
}

pub fn checked_probe_shape(
    probe: HalfWidthProbe,
    coefficients: &[Fp128],
    operands: &[HalfWidthOperand],
    iterations: u32,
    offset: u32,
    max_buffer_length: u64,
) -> Result<HalfWidthProbeShape, HalfWidthProbeError> {
    if coefficients.is_empty() {
        return Err(HalfWidthProbeError::EmptyInput);
    }
    if coefficients.len() != operands.len() {
        return Err(HalfWidthProbeError::LengthMismatch {
            coefficients: coefficients.len(),
            operands: operands.len(),
        });
    }
    if iterations == 0 {
        return Err(HalfWidthProbeError::ZeroIterations);
    }
    if !probe.is_chain() && iterations != 1 {
        return Err(HalfWidthProbeError::PointwiseIterations(iterations));
    }
    if offset == 0 {
        return Err(HalfWidthProbeError::InvalidOffset);
    }

    let ilp = probe.independent_chains();
    if !coefficients.len().is_multiple_of(ilp) {
        return Err(HalfWidthProbeError::MisalignedElementCount {
            probe: probe.name(),
            ilp,
        });
    }
    let elements = u32::try_from(coefficients.len())
        .map_err(|_| HalfWidthProbeError::InputTooLong(coefficients.len()))?;
    let elements_u64 = u64::from(elements);
    let field_buffer_bytes =
        elements_u64
            .checked_mul(FP128_BYTES)
            .ok_or(HalfWidthProbeError::BufferLengthOverflow(
                coefficients.len(),
            ))?;
    let operand_buffer_bytes = elements_u64.checked_mul(OPERAND_BYTES).ok_or(
        HalfWidthProbeError::BufferLengthOverflow(coefficients.len()),
    )?;
    for requested in [field_buffer_bytes, operand_buffer_bytes] {
        if requested > max_buffer_length {
            return Err(HalfWidthProbeError::BufferTooLong {
                requested,
                maximum: max_buffer_length,
            });
        }
    }
    if let Some((index, _)) = coefficients
        .iter()
        .enumerate()
        .find(|(_, coefficient)| !coefficient.is_canonical(offset))
    {
        return Err(HalfWidthProbeError::NonCanonicalCoefficient { index, offset });
    }
    for (index, operand) in operands.iter().copied().enumerate() {
        if let Err(error) = operand.validate(probe.domain()) {
            return Err(match error {
                HalfWidthOperandError::NonzeroUnsignedPadding(value) => {
                    HalfWidthProbeError::NonzeroUnsignedOperand { index, value }
                }
                HalfWidthOperandError::InvalidSignWord(value) => {
                    HalfWidthProbeError::InvalidSignedOperand { index, value }
                }
                HalfWidthOperandError::NegativeZero => {
                    HalfWidthProbeError::NegativeZeroOperand { index }
                }
            });
        }
    }

    let allocated_bytes = field_buffer_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(operand_buffer_bytes))
        .and_then(|bytes| bytes.checked_add(size_of::<HalfWidthProbeParams>() as u64))
        .ok_or(HalfWidthProbeError::BufferLengthOverflow(
            coefficients.len(),
        ))?;
    let semantic_per_element =
        (2 * size_of::<Fp128>() + probe.domain().semantic_operand_bytes()) as u64;
    let semantic_bytes = elements_u64.checked_mul(semantic_per_element).ok_or(
        HalfWidthProbeError::BufferLengthOverflow(coefficients.len()),
    )?;
    let rounds = if probe.is_chain() { iterations } else { 1 };
    let operation_count = elements_u64 * u64::from(rounds);

    Ok(HalfWidthProbeShape {
        params: HalfWidthProbeParams {
            elements,
            iterations,
        },
        grid_threads: coefficients.len() / ilp,
        field_buffer_bytes,
        operand_buffer_bytes,
        allocated_bytes,
        semantic_bytes,
        operation_count,
    })
}

/// Source-level limb-product budget before Metal code generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HalfWidthInstructionBudget {
    pub coefficient_products: u32,
    pub high_fold_products: u32,
    pub carry_fold_products: u32,
    pub total_products: u32,
}

impl HalfWidthInstructionBudget {
    pub const fn half_width() -> Self {
        Self {
            coefficient_products: 4 * 2,
            high_fold_products: 2,
            carry_fold_products: 1,
            total_products: 11,
        }
    }

    pub const fn full_width() -> Self {
        Self {
            coefficient_products: 4 * 4,
            high_fold_products: 4,
            carry_fold_products: 1,
            total_products: 21,
        }
    }
}

/// Structural live values before compiler temporaries and allocation.
///
/// These are 32-bit-word equivalents, not a prediction of physical registers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HalfWidthRegisterFloor {
    pub independent_chains: usize,
    pub persistent_words_per_chain: usize,
    pub helper_scratch_words: usize,
    pub minimum_live_words: usize,
}

impl HalfWidthRegisterFloor {
    pub const fn for_probe(probe: HalfWidthProbe) -> Self {
        let persistent_words_per_chain = match probe.domain() {
            HalfWidthDomain::Unsigned => 6,
            HalfWidthDomain::SignedMagnitude | HalfWidthDomain::UnsignedDelta => 7,
        };
        let independent_chains = probe.independent_chains();
        let helper_scratch_words = 12;
        Self {
            independent_chains,
            persistent_words_per_chain,
            helper_scratch_words,
            minimum_live_words: independent_chains * persistent_words_per_chain
                + helper_scratch_words,
        }
    }
}

/// The exact range proof used by the Akita-specialized reducer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HalfWidthReductionBounds {
    pub coefficient_bits: u32,
    pub scalar_bits: u32,
    pub product_bits: u32,
    pub high_bits: u32,
    pub offset_bits: u32,
    pub high_times_offset_bits: u32,
    pub first_fold_carry_max: u32,
    pub second_fold_carry_max: u32,
    pub canonical_subtractions: u32,
}

pub const AKITA_REDUCTION_BOUNDS: HalfWidthReductionBounds = HalfWidthReductionBounds {
    coefficient_bits: 128,
    scalar_bits: 64,
    product_bits: 192,
    high_bits: 64,
    offset_bits: 32,
    high_times_offset_bits: 96,
    first_fold_carry_max: 1,
    second_fold_carry_max: 0,
    canonical_subtractions: 1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfWidthGateStatus {
    ParityMissing,
    CandidateCompilerShapeUnmeasured,
    CandidateCompilerShapeFailed,
    ControlCompilerShapeUnmeasured,
    ControlCompilerShapeFailed,
    CandidateOccupancyUnmeasured,
    CandidateSpillDetected,
    CandidateInsufficientResidency,
    ControlOccupancyUnmeasured,
    ControlSpillDetected,
    ControlInsufficientResidency,
    ThroughputUnmeasured,
    AbsoluteThroughputFailed,
    RelativeThroughputFailed,
    Noisy,
    Pass,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HalfWidthGateEvidence {
    pub parity_passed: bool,
    pub candidate_compiler_shape_passed: Option<bool>,
    pub control_compiler_shape_passed: Option<bool>,
    pub candidate_spills_detected: Option<bool>,
    pub control_spills_detected: Option<bool>,
    pub candidate_resident_threadgroups: Option<u32>,
    pub control_resident_threadgroups: Option<u32>,
    pub full_width_products_per_second: Option<u64>,
    pub half_width_products_per_second: Option<u64>,
    pub relative_mad_bps: Option<u64>,
}

pub fn gate_status(evidence: HalfWidthGateEvidence) -> HalfWidthGateStatus {
    if !evidence.parity_passed {
        return HalfWidthGateStatus::ParityMissing;
    }
    match evidence.candidate_compiler_shape_passed {
        None => return HalfWidthGateStatus::CandidateCompilerShapeUnmeasured,
        Some(false) => return HalfWidthGateStatus::CandidateCompilerShapeFailed,
        Some(true) => {}
    }
    match evidence.control_compiler_shape_passed {
        None => return HalfWidthGateStatus::ControlCompilerShapeUnmeasured,
        Some(false) => return HalfWidthGateStatus::ControlCompilerShapeFailed,
        Some(true) => {}
    }
    match evidence.candidate_spills_detected {
        None => return HalfWidthGateStatus::CandidateOccupancyUnmeasured,
        Some(true) => return HalfWidthGateStatus::CandidateSpillDetected,
        Some(false) => {}
    }
    let Some(candidate_resident) = evidence.candidate_resident_threadgroups else {
        return HalfWidthGateStatus::CandidateOccupancyUnmeasured;
    };
    if candidate_resident < MINIMUM_RESIDENT_THREADGROUPS {
        return HalfWidthGateStatus::CandidateInsufficientResidency;
    }
    match evidence.control_spills_detected {
        None => return HalfWidthGateStatus::ControlOccupancyUnmeasured,
        Some(true) => return HalfWidthGateStatus::ControlSpillDetected,
        Some(false) => {}
    }
    let Some(control_resident) = evidence.control_resident_threadgroups else {
        return HalfWidthGateStatus::ControlOccupancyUnmeasured;
    };
    if control_resident < MINIMUM_RESIDENT_THREADGROUPS {
        return HalfWidthGateStatus::ControlInsufficientResidency;
    }
    let (Some(full), Some(half), Some(mad)) = (
        evidence.full_width_products_per_second,
        evidence.half_width_products_per_second,
        evidence.relative_mad_bps,
    ) else {
        return HalfWidthGateStatus::ThroughputUnmeasured;
    };
    if half < MINIMUM_HALF_WIDTH_PRODUCTS_PER_SECOND {
        return HalfWidthGateStatus::AbsoluteThroughputFailed;
    }
    if u128::from(half) * 10_000 < u128::from(full) * u128::from(MINIMUM_RELATIVE_SPEEDUP_BPS) {
        return HalfWidthGateStatus::RelativeThroughputFailed;
    }
    if mad > MAXIMUM_RELATIVE_MAD_BPS {
        return HalfWidthGateStatus::Noisy;
    }
    HalfWidthGateStatus::Pass
}

pub const fn maximum_active_ns(operations: u64, products_per_second: u64) -> u64 {
    ((operations as u128 * NANOS_PER_SECOND).div_ceil(products_per_second as u128)) as u64
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfWidthCandidate {
    SpartanShiftNativeU64,
    IncrementClaimSignedMagnitude,
    RamOutputCheckU64,
    AddressSuffixU64,
    InstructionClaimMixedWidths,
    AddressRafMixedWidths,
    ProductRemainderMixedWidths,
    RegistersReadWriteFirstRound,
    BytecodeRawIncrementFirstMessage,
    RegistersClaimUnreducedAccumulator,
    BoundMultilinearState,
    SignedOrUnsigned128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfWidthCandidatePolicy {
    /// Every selected multiply has an exact 64-bit operand and may use the
    /// shared primitive after it passes the promotion gate.
    MayUseAfterPromotion,
    /// Only a statically identified subset is 64-bit. The kernel must retain
    /// its full-width path for other columns, rounds, or suffix widths.
    HybridOnly,
    /// Reducing each product would destroy an existing deferred-accumulation
    /// advantage; retain the specialized accumulator.
    RetainDeferredAccumulator,
    /// The operand is an arbitrary field value or genuinely wider integer.
    FullWidthRequired,
}

pub const fn candidate_policy(candidate: HalfWidthCandidate) -> HalfWidthCandidatePolicy {
    match candidate {
        HalfWidthCandidate::SpartanShiftNativeU64
        | HalfWidthCandidate::IncrementClaimSignedMagnitude
        | HalfWidthCandidate::RamOutputCheckU64
        | HalfWidthCandidate::AddressSuffixU64 => HalfWidthCandidatePolicy::MayUseAfterPromotion,
        HalfWidthCandidate::InstructionClaimMixedWidths
        | HalfWidthCandidate::AddressRafMixedWidths
        | HalfWidthCandidate::ProductRemainderMixedWidths
        | HalfWidthCandidate::RegistersReadWriteFirstRound
        | HalfWidthCandidate::BytecodeRawIncrementFirstMessage => {
            HalfWidthCandidatePolicy::HybridOnly
        }
        HalfWidthCandidate::RegistersClaimUnreducedAccumulator => {
            HalfWidthCandidatePolicy::RetainDeferredAccumulator
        }
        HalfWidthCandidate::BoundMultilinearState | HalfWidthCandidate::SignedOrUnsigned128 => {
            HalfWidthCandidatePolicy::FullWidthRequired
        }
    }
}
