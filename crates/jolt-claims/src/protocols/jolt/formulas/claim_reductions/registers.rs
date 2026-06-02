use jolt_field::RingCore;

use crate::{challenge, opening};

use super::super::super::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltRelationClaims, JoltRelationId,
    JoltVirtualPolynomial, RegistersClaimReductionChallenge,
};
use super::super::dimensions::TraceDimensions;

pub fn claim_reduction<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = reduction_challenge(RegistersClaimReductionChallenge::Gamma);
    let eq_spartan = reduction_challenge(RegistersClaimReductionChallenge::EqSpartan);

    let input = opening(rd_write_value_spartan())
        + gamma.clone() * opening(rs1_value_spartan())
        + gamma.clone().pow(2) * opening(rs2_value_spartan());

    let output = eq_spartan.clone() * opening(rd_write_value_reduced())
        + eq_spartan.clone() * gamma.clone() * opening(rs1_value_reduced())
        + eq_spartan * gamma.pow(2) * opening(rs2_value_reduced());

    JoltRelationClaims::new(
        JoltRelationId::RegistersClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

pub fn claim_reduction_input_openings() -> [JoltOpeningId; 3] {
    [
        rd_write_value_spartan(),
        rs1_value_spartan(),
        rs2_value_spartan(),
    ]
}

pub fn claim_reduction_output_openings() -> [JoltOpeningId; 3] {
    [
        rd_write_value_reduced(),
        rs1_value_reduced(),
        rs2_value_reduced(),
    ]
}

fn reduction_challenge<F>(id: RegistersClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn rd_write_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::SpartanOuter,
    )
}

fn rs1_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::SpartanOuter,
    )
}

fn rs2_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::SpartanOuter,
    )
}

fn rd_write_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::RegistersClaimReduction,
    )
}

fn rs1_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

fn rs2_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let claims = claim_reduction::<Fr>(dimensions());

        assert_eq!(claims.id, JoltRelationId::RegistersClaimReduction);
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
            vec![JoltChallengeId::from(
                RegistersClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![
                JoltChallengeId::from(RegistersClaimReductionChallenge::EqSpartan),
                JoltChallengeId::from(RegistersClaimReductionChallenge::Gamma),
            ]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                JoltChallengeId::from(RegistersClaimReductionChallenge::Gamma),
                JoltChallengeId::from(RegistersClaimReductionChallenge::EqSpartan),
            ]
        );
        assert_eq!(
            claims.challenge_index(JoltChallengeId::from(
                RegistersClaimReductionChallenge::EqSpartan
            )),
            Some(1)
        );
        assert!(claims.required_publics().is_empty());
        assert_eq!(claims.num_challenges(), 2);
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() {
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
                id if id == rd_write_value_spartan() => rd_spartan,
                id if id == rs1_value_spartan() => rs1_spartan,
                id if id == rs2_value_spartan() => rs2_spartan,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersClaimReduction(
                    RegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RegistersClaimReduction(
                    RegistersClaimReductionChallenge::EqSpartan,
                )
                | JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == rd_write_value_reduced() => rd_reduced,
                id if id == rs1_value_reduced() => rs1_reduced,
                id if id == rs2_value_reduced() => rs2_reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersClaimReduction(
                    RegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RegistersClaimReduction(
                    RegistersClaimReductionChallenge::EqSpartan,
                ) => eq_spartan,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::SpartanShift(_) => zero,
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
