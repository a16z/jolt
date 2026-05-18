use std::fmt;

use jolt_field::{Field, RingCore};
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel, CenteredIntegerDomainError},
    EqPolynomial,
};
use jolt_riscv::{CircuitFlags, InstructionFlags};

use crate::{challenge, opening, public};

use super::super::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId, JoltStageClaims, JoltStageId,
    JoltVirtualPolynomial, SpartanOuterPublic, SpartanProductVirtualizationPublic,
    SpartanShiftChallenge, SpartanShiftPublic,
};
use super::dimensions::{
    JoltSumcheckSpec, TraceDimensions, OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    PRODUCT_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};

const OUTER_REMAINDER_DEGREE: usize = 3;
const PRODUCT_REMAINDER_DEGREE: usize = 3;
const SHIFT_DEGREE: usize = 2;
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

impl From<usize> for SpartanOuterDimensions {
    fn from(log_t: usize) -> Self {
        Self::rv64(log_t)
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

impl From<usize> for SpartanProductDimensions {
    fn from(log_t: usize) -> Self {
        Self::new(log_t)
    }
}

pub fn outer_uniskip<F>(dimensions: &SpartanOuterDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    JoltStageClaims::new(
        JoltStageId::SpartanOuter,
        dimensions.uniskip_sumcheck(),
        JoltExpr::zero(),
        opening(outer_uniskip_opening()),
    )
}

pub fn outer_remainder<F>(dimensions: &SpartanOuterDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let mut output = JoltExpr::zero();

    for (left_index, left_variable) in dimensions.variables().iter().copied().enumerate() {
        for (right_index, right_variable) in dimensions.variables().iter().copied().enumerate() {
            output = output
                + public(JoltPublicId::from(
                    SpartanOuterPublic::QuadraticCoefficient {
                        left: left_index,
                        right: right_index,
                    },
                )) * opening(outer_opening(left_variable))
                    * opening(outer_opening(right_variable));
        }
    }

    if dimensions.include_linear_terms() {
        for (index, variable) in dimensions.variables().iter().copied().enumerate() {
            output = output
                + public(JoltPublicId::from(SpartanOuterPublic::LinearCoefficient(
                    index,
                ))) * opening(outer_opening(variable));
        }
    }

    if dimensions.include_constant_term() {
        output = output + public(JoltPublicId::from(SpartanOuterPublic::ConstantCoefficient));
    }

    JoltStageClaims::new(
        JoltStageId::SpartanOuter,
        dimensions.remainder_sumcheck(),
        opening(outer_uniskip_opening()),
        output,
    )
}

pub fn outer_opening(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltStageId::SpartanOuter)
}

pub fn outer_uniskip_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::UnivariateSkip)
}

pub fn product_uniskip<F>(dimensions: SpartanProductDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let input = product_weight(0) * opening(product_outer())
        + product_weight(1) * opening(should_branch_outer())
        + product_weight(2) * opening(should_jump_outer());

    JoltStageClaims::new(
        JoltStageId::SpartanProductVirtualization,
        dimensions.uniskip_sumcheck(),
        input,
        opening(product_uniskip_claim()),
    )
}

pub fn product_remainder<F>(dimensions: SpartanProductDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let left = product_weight(0) * opening(left_instruction_input_product())
        + product_weight(1) * opening(lookup_output_product())
        + product_weight(2) * opening(jump_flag_product());
    let right = product_weight(0) * opening(right_instruction_input_product())
        + product_weight(1) * opening(branch_flag_product())
        + product_weight(2) * (JoltExpr::one() - opening(next_is_noop_product()));

    JoltStageClaims::new(
        JoltStageId::SpartanProductVirtualization,
        dimensions.remainder_sumcheck(),
        opening(product_uniskip_claim()),
        product_tau_kernel() * left * right,
    )
}

pub fn shift<F>(dimensions: TraceDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let gamma = shift_challenge(SpartanShiftChallenge::Gamma);
    let input = opening(next_unexpanded_pc_outer())
        + gamma.clone() * opening(next_pc_outer())
        + gamma.clone().pow(2) * opening(next_is_virtual_outer())
        + gamma.clone().pow(3) * opening(next_is_first_in_sequence_outer())
        + gamma.clone().pow(4) * (JoltExpr::one() - opening(next_is_noop_product()));

    let output = shift_public(SpartanShiftPublic::EqPlusOneOuter)
        * (opening(unexpanded_pc_shift())
            + gamma.clone() * opening(pc_shift())
            + gamma.clone().pow(2) * opening(is_virtual_shift())
            + gamma.clone().pow(3) * opening(is_first_in_sequence_shift()))
        + shift_public(SpartanShiftPublic::EqPlusOneProduct)
            * gamma.pow(4)
            * (JoltExpr::one() - opening(is_noop_shift()));

    JoltStageClaims::new(
        JoltStageId::SpartanShift,
        dimensions.sumcheck(SHIFT_DEGREE),
        input,
        output,
    )
}

fn shift_challenge<F>(id: SpartanShiftChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn shift_public<F>(id: SpartanShiftPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
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

fn product_weight<F>(index: usize) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(
        SpartanProductVirtualizationPublic::LagrangeWeight(index),
    ))
}

fn product_tau_kernel<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(
        SpartanProductVirtualizationPublic::TauKernel,
    ))
}

fn product_uniskip_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnivariateSkip,
        JoltStageId::SpartanProductVirtualization,
    )
}

fn product_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::Product)
}

fn should_branch_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::ShouldBranch)
}

fn should_jump_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::ShouldJump)
}

fn left_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltStageId::SpartanProductVirtualization,
    )
}

fn right_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltStageId::SpartanProductVirtualization,
    )
}

fn lookup_output_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltStageId::SpartanProductVirtualization,
    )
}

fn jump_flag_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::Jump),
        JoltStageId::SpartanProductVirtualization,
    )
}

fn branch_flag_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
        JoltStageId::SpartanProductVirtualization,
    )
}

fn next_is_noop_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::NextIsNoop,
        JoltStageId::SpartanProductVirtualization,
    )
}

fn next_unexpanded_pc_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextUnexpandedPC)
}

fn next_pc_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextPC)
}

fn next_is_virtual_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextIsVirtual)
}

fn next_is_first_in_sequence_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextIsFirstInSequence)
}

fn unexpanded_pc_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltStageId::SpartanShift,
    )
}

fn pc_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltStageId::SpartanShift)
}

fn is_virtual_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
        JoltStageId::SpartanShift,
    )
}

fn is_first_in_sequence_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
        JoltStageId::SpartanShift,
    )
}

fn is_noop_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
        JoltStageId::SpartanShift,
    )
}

#[cfg(test)]
#[expect(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

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
    fn outer_remainder_exposes_expected_dependencies() {
        let dimensions = outer_dimensions();
        let claims = outer_remainder::<Fr>(&dimensions);

        assert_eq!(claims.id, JoltStageId::SpartanOuter);
        assert_eq!(claims.sumcheck, dimensions.remainder_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![outer_uniskip_opening()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![
                outer_opening(JoltVirtualPolynomial::PC),
                outer_opening(JoltVirtualPolynomial::LookupOutput),
            ]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(SpartanOuterPublic::QuadraticCoefficient { left: 0, right: 0 }),
                JoltPublicId::from(SpartanOuterPublic::QuadraticCoefficient { left: 0, right: 1 }),
                JoltPublicId::from(SpartanOuterPublic::QuadraticCoefficient { left: 1, right: 0 }),
                JoltPublicId::from(SpartanOuterPublic::QuadraticCoefficient { left: 1, right: 1 }),
                JoltPublicId::from(SpartanOuterPublic::LinearCoefficient(0)),
                JoltPublicId::from(SpartanOuterPublic::LinearCoefficient(1)),
                JoltPublicId::from(SpartanOuterPublic::ConstantCoefficient),
            ]
        );
        assert!(claims.required_challenges().is_empty());
    }

    #[test]
    fn outer_split_claims_are_connected() {
        let dimensions = outer_dimensions();
        let first = outer_uniskip::<Fr>(&dimensions);
        let remainder = outer_remainder::<Fr>(&dimensions);

        assert_eq!(first.sumcheck, dimensions.uniskip_sumcheck());
        assert_eq!(first.output, remainder.input);
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

    #[test]
    fn product_uniskip_exposes_expected_dependencies() {
        let dimensions = SpartanProductDimensions::from(7);
        let claims = product_uniskip::<Fr>(dimensions);

        assert_eq!(claims.id, JoltStageId::SpartanProductVirtualization);
        assert_eq!(claims.sumcheck, dimensions.uniskip_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![product_outer(), should_branch_outer(), should_jump_outer()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![product_uniskip_claim()]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(SpartanProductVirtualizationPublic::LagrangeWeight(0)),
                JoltPublicId::from(SpartanProductVirtualizationPublic::LagrangeWeight(1)),
                JoltPublicId::from(SpartanProductVirtualizationPublic::LagrangeWeight(2)),
            ]
        );
    }

    #[test]
    fn product_split_claims_are_connected() {
        let dimensions = SpartanProductDimensions::from(7);
        let first = product_uniskip::<Fr>(dimensions);
        let remainder = product_remainder::<Fr>(dimensions);

        assert_eq!(remainder.sumcheck, dimensions.remainder_sumcheck());
        assert_eq!(first.output, remainder.input);
    }

    #[test]
    fn product_remainder_evaluates_like_core_formula() {
        let claims = product_remainder::<Fr>(7.into());

        let left_input = Fr::from_u64(2);
        let lookup_output = Fr::from_u64(3);
        let jump = Fr::from_u64(5);
        let right_input = Fr::from_u64(7);
        let branch = Fr::from_u64(11);
        let next_is_noop = Fr::from_u64(13);
        let weights = [Fr::from_u64(17), Fr::from_u64(19), Fr::from_u64(23)];
        let tau_kernel = Fr::from_u64(29);
        let zero = Fr::from_u64(0);

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == left_instruction_input_product() => left_input,
                id if id == lookup_output_product() => lookup_output,
                id if id == jump_flag_product() => jump,
                id if id == right_instruction_input_product() => right_input,
                id if id == branch_flag_product() => branch,
                id if id == next_is_noop_product() => next_is_noop,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::SpartanProductVirtualization(
                    SpartanProductVirtualizationPublic::LagrangeWeight(index),
                ) => weights[index],
                JoltPublicId::SpartanProductVirtualization(
                    SpartanProductVirtualizationPublic::TauKernel,
                ) => tau_kernel,
                _ => zero,
            },
        );

        assert_eq!(
            output,
            tau_kernel
                * (weights[0] * left_input + weights[1] * lookup_output + weights[2] * jump)
                * (weights[0] * right_input
                    + weights[1] * branch
                    + weights[2] * (Fr::from_u64(1) - next_is_noop))
        );
    }

    #[test]
    fn shift_exposes_expected_dependencies() {
        let dimensions = TraceDimensions::from(5);
        let claims = shift::<Fr>(dimensions);

        assert_eq!(claims.id, JoltStageId::SpartanShift);
        assert_eq!(claims.sumcheck, dimensions.sumcheck(SHIFT_DEGREE));
        assert_eq!(
            claims.input.required_openings,
            vec![
                next_unexpanded_pc_outer(),
                next_pc_outer(),
                next_is_virtual_outer(),
                next_is_first_in_sequence_outer(),
                next_is_noop_product(),
            ]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![
                unexpanded_pc_shift(),
                pc_shift(),
                is_virtual_shift(),
                is_first_in_sequence_shift(),
                is_noop_shift(),
            ]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(SpartanShiftChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(SpartanShiftPublic::EqPlusOneOuter),
                JoltPublicId::from(SpartanShiftPublic::EqPlusOneProduct),
            ]
        );
    }

    #[test]
    fn shift_evaluates_like_core_formula() {
        let claims = shift::<Fr>(5.into());

        let next_unexpanded_pc = Fr::from_u64(3);
        let next_pc = Fr::from_u64(5);
        let next_virtual = Fr::from_u64(7);
        let next_first = Fr::from_u64(11);
        let next_noop = Fr::from_u64(13);
        let unexpanded_pc = Fr::from_u64(17);
        let pc = Fr::from_u64(19);
        let is_virtual = Fr::from_u64(23);
        let is_first = Fr::from_u64(29);
        let is_noop = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let eq_outer = Fr::from_u64(41);
        let eq_product = Fr::from_u64(43);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == next_unexpanded_pc_outer() => next_unexpanded_pc,
                id if id == next_pc_outer() => next_pc,
                id if id == next_is_virtual_outer() => next_virtual,
                id if id == next_is_first_in_sequence_outer() => next_first,
                id if id == next_is_noop_product() => next_noop,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );
        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == unexpanded_pc_shift() => unexpanded_pc,
                id if id == pc_shift() => pc,
                id if id == is_virtual_shift() => is_virtual,
                id if id == is_first_in_sequence_shift() => is_first,
                id if id == is_noop_shift() => is_noop,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::SpartanShift(SpartanShiftPublic::EqPlusOneOuter) => eq_outer,
                JoltPublicId::SpartanShift(SpartanShiftPublic::EqPlusOneProduct) => eq_product,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            next_unexpanded_pc
                + gamma * next_pc
                + gamma_power(gamma, 2) * next_virtual
                + gamma_power(gamma, 3) * next_first
                + gamma_power(gamma, 4) * (Fr::from_u64(1) - next_noop)
        );
        assert_eq!(
            output,
            eq_outer
                * (unexpanded_pc
                    + gamma * pc
                    + gamma_power(gamma, 2) * is_virtual
                    + gamma_power(gamma, 3) * is_first)
                + eq_product * gamma_power(gamma, 4) * (Fr::from_u64(1) - is_noop)
        );
    }
}
