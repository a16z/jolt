use jolt_field::RingCore;
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
        opening(outer_uniskip_claim()),
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
                )) * opening(outer_variable(left_variable))
                    * opening(outer_variable(right_variable));
        }
    }

    if dimensions.include_linear_terms() {
        for (index, variable) in dimensions.variables().iter().copied().enumerate() {
            output = output
                + public(JoltPublicId::from(SpartanOuterPublic::LinearCoefficient(
                    index,
                ))) * opening(outer_variable(variable));
        }
    }

    if dimensions.include_constant_term() {
        output = output + public(JoltPublicId::from(SpartanOuterPublic::ConstantCoefficient));
    }

    JoltStageClaims::new(
        JoltStageId::SpartanOuter,
        dimensions.remainder_sumcheck(),
        opening(outer_uniskip_claim()),
        output,
    )
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

fn outer_variable(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltStageId::SpartanOuter)
}

fn outer_uniskip_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnivariateSkip,
        JoltStageId::SpartanOuter,
    )
}

fn product_uniskip_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnivariateSkip,
        JoltStageId::SpartanProductVirtualization,
    )
}

fn product_outer() -> JoltOpeningId {
    outer_variable(JoltVirtualPolynomial::Product)
}

fn should_branch_outer() -> JoltOpeningId {
    outer_variable(JoltVirtualPolynomial::ShouldBranch)
}

fn should_jump_outer() -> JoltOpeningId {
    outer_variable(JoltVirtualPolynomial::ShouldJump)
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
    outer_variable(JoltVirtualPolynomial::NextUnexpandedPC)
}

fn next_pc_outer() -> JoltOpeningId {
    outer_variable(JoltVirtualPolynomial::NextPC)
}

fn next_is_virtual_outer() -> JoltOpeningId {
    outer_variable(JoltVirtualPolynomial::NextIsVirtual)
}

fn next_is_first_in_sequence_outer() -> JoltOpeningId {
    outer_variable(JoltVirtualPolynomial::NextIsFirstInSequence)
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
#[expect(clippy::panic)]
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
        assert_eq!(claims.input.required_openings, vec![outer_uniskip_claim()]);
        assert_eq!(
            claims.output.required_openings,
            vec![
                outer_variable(JoltVirtualPolynomial::PC),
                outer_variable(JoltVirtualPolynomial::LookupOutput),
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
