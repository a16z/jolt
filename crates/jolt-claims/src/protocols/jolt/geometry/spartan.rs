use jolt_field::Ring;
use jolt_riscv::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS};

use crate::{derived, opening};

use super::super::{
    JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    SpartanProductVirtualizationPublic,
};

pub(crate) const OUTER_REMAINDER_DEGREE: usize = 3;
pub(crate) const PRODUCT_REMAINDER_DEGREE: usize = 3;
pub(crate) const SHIFT_DEGREE: usize = 2;

pub const SPARTAN_OUTER_R1CS_INPUTS: [JoltVirtualPolynomial; 20 + NUM_CIRCUIT_FLAGS] = [
    JoltVirtualPolynomial::LeftInstructionInput,
    JoltVirtualPolynomial::RightInstructionInput,
    JoltVirtualPolynomial::Product,
    JoltVirtualPolynomial::ShouldBranch,
    JoltVirtualPolynomial::PC,
    JoltVirtualPolynomial::UnexpandedPC,
    JoltVirtualPolynomial::Imm,
    JoltVirtualPolynomial::RamAddress,
    JoltVirtualPolynomial::Rs1Value,
    JoltVirtualPolynomial::Rs2Value,
    JoltVirtualPolynomial::RdWriteValue,
    JoltVirtualPolynomial::RamReadValue,
    JoltVirtualPolynomial::RamWriteValue,
    JoltVirtualPolynomial::LeftLookupOperand,
    JoltVirtualPolynomial::RightLookupOperand,
    JoltVirtualPolynomial::NextUnexpandedPC,
    JoltVirtualPolynomial::NextPC,
    JoltVirtualPolynomial::NextIsVirtual,
    JoltVirtualPolynomial::NextIsFirstInSequence,
    JoltVirtualPolynomial::LookupOutput,
    JoltVirtualPolynomial::OpFlags(CircuitFlags::AddOperands),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::SubtractOperands),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::MultiplyOperands),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::Load),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::Store),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::Jump),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::Assert),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::Advice),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::IsCompressed),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::IsLastInSequence),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::Halt),
    JoltVirtualPolynomial::OpFlags(CircuitFlags::HaltPanic),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldAdd),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldSub),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldMul),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldInv),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldAssertEq),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldLoadAccumulateFromRegister),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldAssertZero),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldLoadImm),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldLoadAccumulateFromMemory),
    #[cfg(feature = "field-inline")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::FieldAdviceLimb),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanOuterDimensions {
    log_t: usize,
    variables: Vec<JoltVirtualPolynomial>,
    include_affine_terms: bool,
}

impl SpartanOuterDimensions {
    pub fn new(
        log_t: usize,
        variables: Vec<JoltVirtualPolynomial>,
        include_affine_terms: bool,
    ) -> Option<Self> {
        if variables.is_empty() {
            return None;
        }
        Some(Self {
            log_t,
            variables,
            include_affine_terms,
        })
    }

    pub fn variables(&self) -> &[JoltVirtualPolynomial] {
        &self.variables
    }

    pub fn log_t(&self) -> usize {
        self.log_t
    }

    /// Whether the `Az`/`Bz` linear forms carry their public-column constants
    /// (the affine parts — the source of the expanded form's linear and
    /// constant terms).
    pub fn include_affine_terms(&self) -> bool {
        self.include_affine_terms
    }

    pub const fn remainder_rounds(&self) -> usize {
        1 + self.log_t
    }

    pub fn rv64(log_t: usize) -> Self {
        Self {
            log_t,
            variables: SPARTAN_OUTER_R1CS_INPUTS.to_vec(),
            include_affine_terms: true,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpartanProductDimensions {
    log_t: usize,
}

impl SpartanProductDimensions {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }
}

pub fn outer_opening(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltRelationId::SpartanOuter)
}

pub fn outer_uniskip_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::UnivariateSkip)
}

pub(crate) fn product_weight<F>(index: usize) -> JoltExpr<F>
where
    F: Ring,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::LagrangeWeight(index),
    ))
}

pub(crate) fn product_uniskip_weight<F>(index: usize) -> JoltExpr<F>
where
    F: Ring,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index),
    ))
}

pub(crate) fn product_tau_kernel<F>() -> JoltExpr<F>
where
    F: Ring,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::TauKernel,
    ))
}

fn product_claimed_panic<F>() -> JoltExpr<F>
where
    F: Ring,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::ClaimedPanic,
    ))
}

/// One term of product lane 3's output: `(constant + per_panic·Π)·OpFlags(flag)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HaltLaneTerm {
    pub(crate) flag: CircuitFlags,
    pub(crate) constant: i8,
    pub(crate) per_panic: i8,
}

/// Product lane 3's constant left factor; its right factor is
/// `NextIsNoop − IsNoop`.
pub const PRODUCT_HALT_LANE_LEFT_FACTOR: u8 = 2;

/// The output of product lane 3: every cycle satisfies
/// `2·(NextIsNoop − IsNoop) = (2 − Π)·Halt + HaltPanic`, with `Π` the claimed
/// panic bit. Over boolean flags the left side is −2, 0 or 2, and the output
/// is 0 on a row with neither flag, 2 on a halt whose `HaltPanic` equals `Π`,
/// and 1 or 3 on a halt of the other kind or on `HaltPanic` without `Halt`.
/// A field of characteristic other than 2, 3 and 5 keeps −2, 0, 1, 2 and 3
/// distinct, so the lane holds exactly when `IsNoop` is unchanged across a
/// cycle that carries neither flag, or rises from 0 to 1 across a halt of the
/// claimed kind. The lane does not read `Jump`: that a halt is a jump is a
/// fact about the halt kinds, pinned by jolt-riscv's
/// `halt_flags_mark_exactly_the_halt_kinds`.
pub(crate) const PRODUCT_HALT_LANE_OUTPUT: [HaltLaneTerm; 2] = [
    HaltLaneTerm {
        flag: CircuitFlags::Halt,
        constant: 2,
        per_panic: -1,
    },
    HaltLaneTerm {
        flag: CircuitFlags::HaltPanic,
        constant: 1,
        per_panic: 0,
    },
];

/// [`PRODUCT_HALT_LANE_OUTPUT`] over the Spartan-outer flag openings, with `Π`
/// the `ClaimedPanic` derived public.
pub(crate) fn product_halt_lane_output<F>() -> JoltExpr<F>
where
    F: Ring,
{
    PRODUCT_HALT_LANE_OUTPUT
        .into_iter()
        .fold(JoltExpr::zero(), |output, term| {
            let flag = opening(outer_opening(JoltVirtualPolynomial::OpFlags(term.flag)));
            output
                + i128::from(term.constant) * flag.clone()
                + i128::from(term.per_panic) * (product_claimed_panic() * flag)
        })
}

pub fn product_uniskip_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnivariateSkip,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn product_outer_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::Product)
}

pub fn product_should_branch_outer_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::ShouldBranch)
}

pub fn left_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn right_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn lookup_output_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn is_noop_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn write_lookup_output_to_rd_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn branch_flag_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn next_is_noop_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::NextIsNoop,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn virtual_instruction_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn unexpanded_pc_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::SpartanShift,
    )
}

pub fn pc_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltRelationId::SpartanShift)
}

pub fn is_virtual_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
        JoltRelationId::SpartanShift,
    )
}

pub fn is_first_in_sequence_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
        JoltRelationId::SpartanShift,
    )
}

pub fn is_noop_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
        JoltRelationId::SpartanShift,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outer_dimensions_rejects_empty_variables() {
        assert_eq!(SpartanOuterDimensions::new(8, Vec::new(), false), None);
    }
}
