use jolt_field::RingCore;

use crate::{challenge, public};

use super::super::super::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationClaims, JoltRelationId,
    JoltVirtualPolynomial, RegistersClaimReductionChallenge, RegistersClaimReductionPublic,
};
use super::super::dimensions::TraceDimensions;

pub fn claim_reduction<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    use crate::protocols::jolt::relations::claim_reductions::registers::ClaimReduction;
    use crate::SymbolicSumcheck;
    let r = ClaimReduction::new(dimensions);
    JoltRelationClaims::new(
        ClaimReduction::id(),
        r.spec(),
        r.input_expression::<F>(),
        r.output_expression::<F>(),
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

pub(crate) fn reduction_challenge<F>(id: RegistersClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn reduction_public<F>(id: RegistersClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn rd_write_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn rs1_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn rs2_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn rd_write_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs1_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs2_value_reduced() -> JoltOpeningId {
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
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
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
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::RegistersClaimReduction(RegistersClaimReductionPublic::EqSpartan) => {
                    eq_spartan
                }
                _ => zero,
            },
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
