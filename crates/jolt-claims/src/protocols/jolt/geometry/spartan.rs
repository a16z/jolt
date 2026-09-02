use jolt_field::Ring;
use jolt_riscv::{CircuitFlags, InstructionFlags};

use crate::derived;

use super::super::{
    JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    SpartanProductVirtualizationPublic,
};

pub(crate) const OUTER_REMAINDER_DEGREE: usize = 3;
pub(crate) const PRODUCT_REMAINDER_DEGREE: usize = 3;
pub(crate) const SHIFT_DEGREE: usize = 2;

pub const SPARTAN_OUTER_R1CS_INPUTS: [JoltVirtualPolynomial;
    35 + 4 * cfg!(feature = "implicit-carry") as usize] = [
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
    JoltVirtualPolynomial::ShouldJump,
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
    #[cfg(feature = "implicit-carry")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::UsesCarry),
    #[cfg(feature = "implicit-carry")]
    JoltVirtualPolynomial::OpFlags(CircuitFlags::ProducesCarry),
    #[cfg(feature = "implicit-carry")]
    JoltVirtualPolynomial::CarryUsed,
    #[cfg(feature = "implicit-carry")]
    JoltVirtualPolynomial::NextCarry,
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

pub fn product_should_jump_outer_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::ShouldJump)
}

#[cfg(feature = "implicit-carry")]
pub fn uses_carry_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::UsesCarry),
        JoltRelationId::SpartanProductVirtualization,
    )
}

#[cfg(feature = "implicit-carry")]
pub fn carry_product() -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::Carry,
        JoltRelationId::SpartanProductVirtualization,
    )
}

#[cfg(feature = "implicit-carry")]
pub fn carry_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::Carry,
        JoltRelationId::CarryClaimReduction,
    )
}

#[cfg(feature = "implicit-carry")]
pub fn carry_shift() -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::Carry,
        JoltRelationId::SpartanShift,
    )
}

#[cfg(feature = "implicit-carry")]
pub fn next_carry_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::NextCarry,
        JoltRelationId::SpartanOuter,
    )
}

#[cfg(feature = "implicit-carry")]
pub fn product_carry_used_outer_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::CarryUsed,
        JoltRelationId::SpartanOuter,
    )
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

pub fn jump_flag_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::Jump),
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

pub(crate) fn next_unexpanded_pc_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextUnexpandedPC)
}

pub(crate) fn next_pc_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextPC)
}

pub(crate) fn next_is_virtual_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextIsVirtual)
}

pub(crate) fn next_is_first_in_sequence_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextIsFirstInSequence)
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

    #[test]
    fn default_outer_dimensions_match_r1cs_input_catalog() {
        let dimensions = SpartanOuterDimensions::rv64(8);

        assert_eq!(dimensions.log_t(), 8);
        assert_eq!(dimensions.variables(), &SPARTAN_OUTER_R1CS_INPUTS);
        assert!(dimensions.include_affine_terms());
    }
}
