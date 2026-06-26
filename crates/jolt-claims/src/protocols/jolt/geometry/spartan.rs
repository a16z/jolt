use std::fmt;

use jolt_field::{Field, RingCore};
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel, CenteredIntegerDomainError},
    EqPolynomial,
};
use jolt_riscv::{CircuitFlags, InstructionFlags};

use crate::derived;

use super::super::{
    JoltExpr, JoltOpeningId, JoltDerivedId, JoltRelationId, JoltVirtualPolynomial,
    SpartanOuterPublic, SpartanProductVirtualizationPublic,
};
use super::dimensions::{
    JoltSumcheckSpec, OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    PRODUCT_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};

const OUTER_REMAINDER_DEGREE: usize = 3;
const PRODUCT_REMAINDER_DEGREE: usize = 3;
pub(crate) const SHIFT_DEGREE: usize = 2;
const SPARTAN_OUTER_RV64_ROW_COUNT: usize = 19;
const SPARTAN_OUTER_FIRST_GROUP_ROWS: [usize; OUTER_UNISKIP_DOMAIN_SIZE] =
    [1, 2, 3, 4, 5, 6, 11, 14, 17, 18];
const SPARTAN_OUTER_SECOND_GROUP_ROWS: [usize; 9] = [0, 7, 8, 9, 10, 12, 13, 15, 16];

pub const SPARTAN_OUTER_R1CS_INPUTS: [JoltVirtualPolynomial; 35] = [
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
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpartanOuterClaimError {
    InvalidUniskipDomain(CenteredIntegerDomainError),
    ChallengeLengthMismatch { expected: usize, got: usize },
    LinearFormLengthMismatch { expected: usize, got: usize },
    UnsupportedR1csInput { variable: JoltVirtualPolynomial },
}

impl fmt::Display for SpartanOuterClaimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUniskipDomain(error) => write!(f, "{error}"),
            Self::ChallengeLengthMismatch { expected, got } => {
                write!(
                    f,
                    "challenge length mismatch: expected {expected}, got {got}"
                )
            }
            Self::LinearFormLengthMismatch { expected, got } => {
                write!(
                    f,
                    "linear form length mismatch: expected {expected}, got {got}"
                )
            }
            Self::UnsupportedR1csInput { variable } => {
                write!(f, "unsupported Spartan outer R1CS input {variable:?}")
            }
        }
    }
}

impl std::error::Error for SpartanOuterClaimError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanOuterDimensions {
    log_t: usize,
    variables: Vec<JoltVirtualPolynomial>,
    include_linear_terms: bool,
    include_constant_term: bool,
}

impl SpartanOuterDimensions {
    pub fn new(
        log_t: usize,
        variables: Vec<JoltVirtualPolynomial>,
        include_linear_terms: bool,
        include_constant_term: bool,
    ) -> Option<Self> {
        if variables.is_empty() {
            return None;
        }
        Some(Self {
            log_t,
            variables,
            include_linear_terms,
            include_constant_term,
        })
    }

    pub fn variables(&self) -> &[JoltVirtualPolynomial] {
        &self.variables
    }

    pub fn log_t(&self) -> usize {
        self.log_t
    }

    pub fn include_linear_terms(&self) -> bool {
        self.include_linear_terms
    }

    pub fn include_constant_term(&self) -> bool {
        self.include_constant_term
    }

    pub const fn uniskip_sumcheck(&self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::centered_integer(
            OUTER_UNISKIP_DOMAIN_SIZE,
            1,
            OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        )
    }

    pub const fn remainder_sumcheck(&self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(1 + self.log_t, OUTER_REMAINDER_DEGREE)
    }

    pub fn rv64(log_t: usize) -> Self {
        Self {
            log_t,
            variables: SPARTAN_OUTER_R1CS_INPUTS.to_vec(),
            include_linear_terms: true,
            include_constant_term: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanOuterLinearForms<F> {
    pub az_coefficients: Vec<F>,
    pub bz_coefficients: Vec<F>,
    pub az_constant: F,
    pub bz_constant: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanOuterRemainderPlan {
    variables: Vec<JoltVirtualPolynomial>,
    include_linear_terms: bool,
    include_constant_term: bool,
}

impl SpartanOuterRemainderPlan {
    pub fn from_dimensions(dimensions: &SpartanOuterDimensions) -> Self {
        Self {
            variables: dimensions.variables().to_vec(),
            include_linear_terms: dimensions.include_linear_terms(),
            include_constant_term: dimensions.include_constant_term(),
        }
    }

    pub fn variables(&self) -> &[JoltVirtualPolynomial] {
        &self.variables
    }

    pub fn r1cs_input_indices(&self) -> Result<Vec<usize>, SpartanOuterClaimError> {
        self.variables
            .iter()
            .copied()
            .map(spartan_outer_r1cs_input_index)
            .collect()
    }

    pub fn row_weights<F: Field>(
        &self,
        r0: F,
        r_stream: F,
    ) -> Result<Vec<F>, SpartanOuterClaimError> {
        let lagrange_weights = centered_lagrange_evals(OUTER_UNISKIP_DOMAIN_SIZE, r0)
            .map_err(SpartanOuterClaimError::InvalidUniskipDomain)?;
        let mut weights = vec![F::zero(); SPARTAN_OUTER_RV64_ROW_COUNT];

        for (index, &row) in SPARTAN_OUTER_FIRST_GROUP_ROWS.iter().enumerate() {
            weights[row] += (F::one() - r_stream) * lagrange_weights[index];
        }
        for (index, &row) in SPARTAN_OUTER_SECOND_GROUP_ROWS
            .iter()
            .take(OUTER_UNISKIP_DOMAIN_SIZE)
            .enumerate()
        {
            weights[row] += r_stream * lagrange_weights[index];
        }

        Ok(weights)
    }

    pub fn tau_kernel<F: Field>(
        &self,
        tau: &[F],
        r0: F,
        remainder_challenges: &[F],
    ) -> Result<F, SpartanOuterClaimError> {
        let expected = remainder_challenges.len() + 1;
        if tau.len() != expected {
            return Err(SpartanOuterClaimError::ChallengeLengthMismatch {
                expected,
                got: tau.len(),
            });
        }

        let tau_high = tau[tau.len() - 1];
        let tau_high_bound_r0 = centered_lagrange_kernel(OUTER_UNISKIP_DOMAIN_SIZE, tau_high, r0)
            .map_err(SpartanOuterClaimError::InvalidUniskipDomain)?;
        let mut reversed_challenges = remainder_challenges.to_vec();
        reversed_challenges.reverse();
        Ok(tau_high_bound_r0 * EqPolynomial::<F>::mle(&tau[..tau.len() - 1], &reversed_challenges))
    }

    pub fn public_claims<F: Field>(
        &self,
        tau_kernel: F,
        linear_forms: &SpartanOuterLinearForms<F>,
    ) -> Result<Vec<(SpartanOuterPublic, F)>, SpartanOuterClaimError> {
        let expected = self.variables.len();
        check_linear_form_len(expected, linear_forms.az_coefficients.len())?;
        check_linear_form_len(expected, linear_forms.bz_coefficients.len())?;

        let mut claims = Vec::with_capacity(expected * expected + 2 * expected + 1);
        for left in 0..expected {
            for right in 0..expected {
                claims.push((
                    SpartanOuterPublic::QuadraticCoefficient { left, right },
                    tau_kernel
                        * linear_forms.az_coefficients[left]
                        * linear_forms.bz_coefficients[right],
                ));
            }
        }

        if self.include_linear_terms {
            for index in 0..expected {
                let claim = linear_forms.az_coefficients[index] * linear_forms.bz_constant
                    + linear_forms.az_constant * linear_forms.bz_coefficients[index];
                claims.push((
                    SpartanOuterPublic::LinearCoefficient(index),
                    tau_kernel * claim,
                ));
            }
        }

        if self.include_constant_term {
            claims.push((
                SpartanOuterPublic::ConstantCoefficient,
                tau_kernel * linear_forms.az_constant * linear_forms.bz_constant,
            ));
        }

        Ok(claims)
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

    pub const fn uniskip_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::centered_integer(
            PRODUCT_UNISKIP_DOMAIN_SIZE,
            1,
            PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
        )
    }

    pub const fn remainder_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t, PRODUCT_REMAINDER_DEGREE)
    }
}

pub fn outer_opening(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltRelationId::SpartanOuter)
}

pub fn outer_uniskip_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::UnivariateSkip)
}

pub fn product_remainder_output_openings() -> [JoltOpeningId; 8] {
    [
        left_instruction_input_product(),
        right_instruction_input_product(),
        jump_flag_product(),
        write_lookup_output_to_rd_product(),
        lookup_output_product(),
        branch_flag_product(),
        next_is_noop_product(),
        virtual_instruction_product(),
    ]
}

pub fn shift_output_openings() -> [JoltOpeningId; 5] {
    [
        unexpanded_pc_shift(),
        pc_shift(),
        is_virtual_shift(),
        is_first_in_sequence_shift(),
        is_noop_shift(),
    ]
}

fn check_linear_form_len(expected: usize, got: usize) -> Result<(), SpartanOuterClaimError> {
    if got == expected {
        Ok(())
    } else {
        Err(SpartanOuterClaimError::LinearFormLengthMismatch { expected, got })
    }
}

fn spartan_outer_r1cs_input_index(
    variable: JoltVirtualPolynomial,
) -> Result<usize, SpartanOuterClaimError> {
    SPARTAN_OUTER_R1CS_INPUTS
        .iter()
        .position(|candidate| *candidate == variable)
        .ok_or(SpartanOuterClaimError::UnsupportedR1csInput { variable })
}

pub(crate) fn product_weight<F>(index: usize) -> JoltExpr<F>
where
    F: RingCore,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::LagrangeWeight(index),
    ))
}

pub(crate) fn product_uniskip_weight<F>(index: usize) -> JoltExpr<F>
where
    F: RingCore,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index),
    ))
}

pub(crate) fn product_tau_kernel<F>() -> JoltExpr<F>
where
    F: RingCore,
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

pub(crate) fn left_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn right_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn lookup_output_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn jump_flag_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::Jump),
        JoltRelationId::SpartanProductVirtualization,
    )
}

fn write_lookup_output_to_rd_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn branch_flag_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn next_is_noop_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::NextIsNoop,
        JoltRelationId::SpartanProductVirtualization,
    )
}

fn virtual_instruction_product() -> JoltOpeningId {
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

pub(crate) fn unexpanded_pc_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::SpartanShift,
    )
}

pub(crate) fn pc_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltRelationId::SpartanShift)
}

pub(crate) fn is_virtual_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
        JoltRelationId::SpartanShift,
    )
}

pub(crate) fn is_first_in_sequence_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
        JoltRelationId::SpartanShift,
    )
}

pub(crate) fn is_noop_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
        JoltRelationId::SpartanShift,
    )
}

#[cfg(test)]
#[expect(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn outer_dimensions() -> SpartanOuterDimensions {
        match SpartanOuterDimensions::new(
            8,
            vec![
                JoltVirtualPolynomial::PC,
                JoltVirtualPolynomial::LookupOutput,
            ],
            true,
            true,
        ) {
            Some(dimensions) => dimensions,
            None => panic!("test Spartan outer dimensions should be valid"),
        }
    }

    #[test]
    fn outer_dimensions_rejects_empty_variables() {
        assert_eq!(
            SpartanOuterDimensions::new(8, Vec::new(), false, false),
            None
        );
    }

    #[test]
    fn default_outer_dimensions_match_r1cs_input_catalog() {
        let dimensions = SpartanOuterDimensions::rv64(8);

        assert_eq!(dimensions.log_t(), 8);
        assert_eq!(dimensions.variables(), &SPARTAN_OUTER_R1CS_INPUTS);
        assert!(dimensions.include_linear_terms());
        assert!(dimensions.include_constant_term());
    }

    #[test]
    fn outer_remainder_plan_maps_variables_to_r1cs_inputs() {
        let dimensions = outer_dimensions();
        let plan = SpartanOuterRemainderPlan::from_dimensions(&dimensions);

        assert_eq!(plan.r1cs_input_indices(), Ok(vec![4, 19]));

        let unsupported =
            SpartanOuterDimensions::new(8, vec![JoltVirtualPolynomial::NextIsNoop], true, true)
                .unwrap();
        assert!(matches!(
            SpartanOuterRemainderPlan::from_dimensions(&unsupported).r1cs_input_indices(),
            Err(SpartanOuterClaimError::UnsupportedR1csInput {
                variable: JoltVirtualPolynomial::NextIsNoop
            })
        ));
    }

    #[test]
    fn outer_remainder_plan_computes_group_row_weights() {
        let plan = SpartanOuterRemainderPlan::from_dimensions(&outer_dimensions());
        let r0 = Fr::from_i64(-4);

        let first_group = plan.row_weights(r0, Fr::from_u64(0)).unwrap();
        assert_eq!(first_group[1], Fr::from_u64(1));
        assert_eq!(first_group[0], Fr::from_u64(0));

        let second_group = plan.row_weights(r0, Fr::from_u64(1)).unwrap();
        assert_eq!(second_group[0], Fr::from_u64(1));
        assert_eq!(second_group[1], Fr::from_u64(0));
    }

    #[test]
    fn outer_remainder_plan_computes_tau_kernel() {
        let plan = SpartanOuterRemainderPlan::from_dimensions(&outer_dimensions());
        let tau = [Fr::from_u64(0), Fr::from_u64(0), Fr::from_i64(-4)];
        let challenges = [Fr::from_u64(0), Fr::from_u64(0)];

        assert_eq!(
            plan.tau_kernel(&tau, Fr::from_i64(-4), &challenges),
            Ok(Fr::from_u64(1))
        );
    }

    #[test]
    fn outer_remainder_plan_expands_public_coefficients() {
        let plan = SpartanOuterRemainderPlan::from_dimensions(&outer_dimensions());
        let linear_forms = SpartanOuterLinearForms {
            az_coefficients: vec![Fr::from_u64(2), Fr::from_u64(3)],
            bz_coefficients: vec![Fr::from_u64(5), Fr::from_u64(7)],
            az_constant: Fr::from_u64(11),
            bz_constant: Fr::from_u64(13),
        };

        let claims = plan.public_claims(Fr::from_u64(17), &linear_forms).unwrap();

        assert_eq!(
            claims,
            vec![
                (
                    SpartanOuterPublic::QuadraticCoefficient { left: 0, right: 0 },
                    Fr::from_u64(170),
                ),
                (
                    SpartanOuterPublic::QuadraticCoefficient { left: 0, right: 1 },
                    Fr::from_u64(238),
                ),
                (
                    SpartanOuterPublic::QuadraticCoefficient { left: 1, right: 0 },
                    Fr::from_u64(255),
                ),
                (
                    SpartanOuterPublic::QuadraticCoefficient { left: 1, right: 1 },
                    Fr::from_u64(357),
                ),
                (SpartanOuterPublic::LinearCoefficient(0), Fr::from_u64(1377),),
                (SpartanOuterPublic::LinearCoefficient(1), Fr::from_u64(1972),),
                (SpartanOuterPublic::ConstantCoefficient, Fr::from_u64(2431),),
            ]
        );
    }
}
