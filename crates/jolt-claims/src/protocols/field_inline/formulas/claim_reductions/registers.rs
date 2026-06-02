use jolt_field::RingCore;

use crate::{challenge, opening};

use super::super::super::{
    FieldInlineChallengeId, FieldInlineExpr, FieldInlineOpeningId, FieldInlineRelationClaims,
    FieldInlineRelationId, FieldInlineVirtualPolynomial, FieldRegistersClaimReductionChallenge,
};
use super::super::dimensions::FieldRegistersTraceDimensions;

pub fn claim_reduction<F>(dimensions: FieldRegistersTraceDimensions) -> FieldInlineRelationClaims<F>
where
    F: RingCore,
{
    let gamma = reduction_challenge(FieldRegistersClaimReductionChallenge::Gamma);
    let eq_spartan = reduction_challenge(FieldRegistersClaimReductionChallenge::EqSpartan);

    let input = opening(field_rd_value_spartan())
        + gamma.clone() * opening(field_rs1_value_spartan())
        + gamma.clone().pow(2) * opening(field_rs2_value_spartan());

    let output = eq_spartan.clone() * opening(field_rd_value_reduced())
        + eq_spartan.clone() * gamma.clone() * opening(field_rs1_value_reduced())
        + eq_spartan * gamma.pow(2) * opening(field_rs2_value_reduced());

    FieldInlineRelationClaims::new(
        FieldInlineRelationId::FieldRegistersClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

pub fn claim_reduction_input_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rd_value_spartan(),
        field_rs1_value_spartan(),
        field_rs2_value_spartan(),
    ]
}

pub fn claim_reduction_output_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rd_value_reduced(),
        field_rs1_value_reduced(),
        field_rs2_value_reduced(),
    ]
}

fn reduction_challenge<F>(id: FieldRegistersClaimReductionChallenge) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    challenge(FieldInlineChallengeId::from(id))
}

fn field_rd_value_spartan() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

fn field_rs1_value_spartan() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

fn field_rs2_value_spartan() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

fn field_rd_value_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

fn field_rs1_value_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

fn field_rs2_value_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let claims = claim_reduction::<Fr>(dimensions());

        assert_eq!(
            claims.id,
            FieldInlineRelationId::FieldRegistersClaimReduction
        );
        assert_eq!(claims.sumcheck, dimensions().sumcheck(2));
        assert_eq!(
            claims.input.required_openings,
            claim_reduction_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            claim_reduction_output_openings().to_vec()
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![FieldInlineChallengeId::from(
                FieldRegistersClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![
                FieldInlineChallengeId::from(FieldRegistersClaimReductionChallenge::EqSpartan),
                FieldInlineChallengeId::from(FieldRegistersClaimReductionChallenge::Gamma),
            ]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                FieldInlineChallengeId::from(FieldRegistersClaimReductionChallenge::Gamma),
                FieldInlineChallengeId::from(FieldRegistersClaimReductionChallenge::EqSpartan),
            ]
        );
        assert_eq!(
            claims.challenge_index(FieldInlineChallengeId::from(
                FieldRegistersClaimReductionChallenge::EqSpartan
            )),
            Some(1)
        );
        assert!(claims.required_publics().is_empty());
        assert_eq!(claims.num_challenges(), 2);
    }

    #[test]
    fn claim_reduction_evaluates_like_field_register_twist_formula() {
        let claims = claim_reduction::<Fr>(dimensions());

        let rd_spartan = Fr::from_u64(3);
        let rs1_spartan = Fr::from_u64(5);
        let rs2_spartan = Fr::from_u64(7);
        let rd_reduced = Fr::from_u64(11);
        let rs1_reduced = Fr::from_u64(13);
        let rs2_reduced = Fr::from_u64(17);
        let gamma = Fr::from_u64(19);
        let eq_spartan = Fr::from_u64(23);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == field_rd_value_spartan() => rd_spartan,
                id if id == field_rs1_value_spartan() => rs1_spartan,
                id if id == field_rs2_value_spartan() => rs2_spartan,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersClaimReduction(
                    FieldRegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == field_rd_value_reduced() => rd_reduced,
                id if id == field_rs1_value_reduced() => rs1_reduced,
                id if id == field_rs2_value_reduced() => rs2_reduced,
                _ => zero,
            },
            |id| match *id {
                FieldInlineChallengeId::FieldRegistersClaimReduction(
                    FieldRegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                FieldInlineChallengeId::FieldRegistersClaimReduction(
                    FieldRegistersClaimReductionChallenge::EqSpartan,
                ) => eq_spartan,
                _ => zero,
            },
            |_| zero,
        );

        assert_eq!(
            input,
            rd_spartan + gamma * rs1_spartan + gamma * gamma * rs2_spartan
        );
        assert_eq!(
            output,
            eq_spartan * (rd_reduced + gamma * rs1_reduced + gamma * gamma * rs2_reduced)
        );
    }
}
