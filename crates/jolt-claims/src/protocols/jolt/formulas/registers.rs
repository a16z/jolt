use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::{
    JoltChallengeId, JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltPublicId,
    JoltRelationClaims, JoltRelationId, JoltVirtualPolynomial, RegistersReadWriteChallenge,
    RegistersReadWritePublic, RegistersValEvaluationPublic,
};
use super::dimensions::{JoltSumcheckSpec, ReadWriteDimensions, TraceDimensions};

pub const fn read_write_checking_sumcheck(dimensions: ReadWriteDimensions) -> JoltSumcheckSpec {
    dimensions.read_write_sumcheck()
}

pub fn read_write_checking<F>(dimensions: ReadWriteDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = read_write_challenge(RegistersReadWriteChallenge::Gamma);
    let eq_cycle = read_write_public(RegistersReadWritePublic::EqCycle);

    let input = opening(rd_write_value_claim())
        + gamma.clone() * opening(rs1_value_claim())
        + gamma.clone().pow(2) * opening(rs2_value_claim());

    let output = eq_cycle.clone() * opening(rd_wa_read_write()) * opening(rd_inc_read_write())
        + eq_cycle.clone() * opening(rd_wa_read_write()) * opening(registers_val_read_write())
        + eq_cycle.clone()
            * gamma.clone()
            * opening(rs1_ra_read_write())
            * opening(registers_val_read_write())
        + eq_cycle
            * gamma.pow(2)
            * opening(rs2_ra_read_write())
            * opening(registers_val_read_write());

    JoltRelationClaims::new(
        JoltRelationId::RegistersReadWriteChecking,
        read_write_checking_sumcheck(dimensions),
        input,
        output,
    )
}

pub fn val_evaluation<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let input = opening(registers_val_read_write());
    let output = val_evaluation_public(RegistersValEvaluationPublic::LtCycle)
        * opening(rd_inc_val_evaluation())
        * opening(rd_wa_val_evaluation());

    JoltRelationClaims::new(
        JoltRelationId::RegistersValEvaluation,
        dimensions.sumcheck(3),
        input,
        output,
    )
}

pub fn read_write_checking_input_openings() -> [JoltOpeningId; 3] {
    [rd_write_value_claim(), rs1_value_claim(), rs2_value_claim()]
}

pub fn read_write_checking_output_openings() -> [JoltOpeningId; 5] {
    [
        registers_val_read_write(),
        rs1_ra_read_write(),
        rs2_ra_read_write(),
        rd_wa_read_write(),
        rd_inc_read_write(),
    ]
}

pub fn val_evaluation_input_openings() -> [JoltOpeningId; 1] {
    [registers_val_read_write()]
}

pub fn val_evaluation_output_openings() -> [JoltOpeningId; 2] {
    [rd_inc_val_evaluation(), rd_wa_val_evaluation()]
}

fn read_write_challenge<F>(id: RegistersReadWriteChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn read_write_public<F>(id: RegistersReadWritePublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn val_evaluation_public<F>(id: RegistersValEvaluationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn rd_write_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::RegistersClaimReduction,
    )
}

fn rs1_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

fn rs2_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub fn registers_val_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RegistersVal,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn rs1_ra_read_write_opening() -> JoltOpeningId {
    rs1_ra_read_write()
}

pub fn rs2_ra_read_write_opening() -> JoltOpeningId {
    rs2_ra_read_write()
}

pub fn rd_wa_read_write_opening() -> JoltOpeningId {
    rd_wa_read_write()
}

pub fn rd_inc_read_write_opening() -> JoltOpeningId {
    rd_inc_read_write()
}

fn rs1_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

fn rs2_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

fn rd_wa_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

fn rd_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn rd_inc_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersValEvaluation,
    )
}

pub fn rd_wa_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersValEvaluation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 7, 2, 1)
    }

    #[test]
    fn read_write_claims_expose_expected_dependencies() {
        let claims = read_write_checking::<Fr>(read_write_dimensions());

        assert_eq!(claims.id, JoltRelationId::RegistersReadWriteChecking);
        assert_eq!(
            claims.sumcheck,
            read_write_checking_sumcheck(read_write_dimensions())
        );
        assert_eq!(
            claims.input.required_openings,
            read_write_checking_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            vec![
                rd_wa_read_write(),
                rd_inc_read_write(),
                registers_val_read_write(),
                rs1_ra_read_write(),
                rs2_ra_read_write(),
            ]
        );
        assert_eq!(
            read_write_checking_output_openings(),
            [
                registers_val_read_write(),
                rs1_ra_read_write(),
                rs2_ra_read_write(),
                rd_wa_read_write(),
                rd_inc_read_write(),
            ]
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(RegistersReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(RegistersReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(RegistersReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(RegistersReadWritePublic::EqCycle)]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn read_write_claims_evaluate_like_core_formula() {
        let claims = read_write_checking::<Fr>(read_write_dimensions());

        let rd_write_value = Fr::from_u64(3);
        let rs1_value = Fr::from_u64(5);
        let rs2_value = Fr::from_u64(7);
        let val = Fr::from_u64(11);
        let rs1_ra = Fr::from_u64(13);
        let rs2_ra = Fr::from_u64(17);
        let rd_wa = Fr::from_u64(19);
        let inc = Fr::from_u64(23);
        let gamma = Fr::from_u64(29);
        let eq_cycle = Fr::from_u64(31);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == rd_write_value_claim() => rd_write_value,
                id if id == rs1_value_claim() => rs1_value,
                id if id == rs2_value_claim() => rs2_value,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersClaimReduction(_)
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
                id if id == registers_val_read_write() => val,
                id if id == rs1_ra_read_write() => rs1_ra,
                id if id == rs2_ra_read_write() => rs2_ra,
                id if id == rd_wa_read_write() => rd_wa,
                id if id == rd_inc_read_write() => inc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersClaimReduction(_)
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
                JoltPublicId::RegistersReadWrite(RegistersReadWritePublic::EqCycle) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            rd_write_value + gamma * rs1_value + gamma * gamma * rs2_value
        );
        assert_eq!(
            output,
            eq_cycle * (rd_wa * (inc + val) + gamma * rs1_ra * val + gamma * gamma * rs2_ra * val)
        );
    }

    #[test]
    fn val_evaluation_claims_expose_expected_dependencies() {
        let claims = val_evaluation::<Fr>(trace_dimensions());

        assert_eq!(claims.id, JoltRelationId::RegistersValEvaluation);
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(3));
        assert_eq!(
            claims.input.required_openings,
            val_evaluation_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            val_evaluation_output_openings().to_vec()
        );
        assert!(claims.output.required_challenges.is_empty());
        assert!(claims.required_challenges().is_empty());
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(RegistersValEvaluationPublic::LtCycle)]
        );
        assert_eq!(claims.num_challenges(), 0);
    }

    #[test]
    fn val_evaluation_claims_evaluate_like_core_formula() {
        let claims = val_evaluation::<Fr>(trace_dimensions());

        let val = Fr::from_u64(3);
        let inc = Fr::from_u64(5);
        let wa = Fr::from_u64(7);
        let lt_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == registers_val_read_write() => val,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == rd_inc_val_evaluation() => inc,
                id if id == rd_wa_val_evaluation() => wa,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
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
                JoltPublicId::RegistersValEvaluation(RegistersValEvaluationPublic::LtCycle) => {
                    lt_cycle
                }
                _ => zero,
            },
        );

        assert_eq!(input, val);
        assert_eq!(output, lt_cycle * inc * wa);
    }
}
