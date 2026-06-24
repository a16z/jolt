//! registers symbolic sumcheck relations.

use jolt_field::RingCore;

use crate::opening;
use crate::protocols::jolt::formulas::registers::{
    rd_inc_read_write, rd_inc_val_evaluation, rd_wa_read_write, rd_wa_val_evaluation,
    rd_write_value_claim, read_write_challenge, read_write_checking_sumcheck, read_write_public,
    registers_val_read_write, rs1_ra_read_write, rs1_value_claim, rs2_ra_read_write,
    rs2_value_claim, val_evaluation_public,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, JoltSumcheckSpec, ReadWriteDimensions, RegistersReadWriteChallenge,
    RegistersReadWritePublic, RegistersValEvaluationPublic, TraceDimensions,
};
use crate::SymbolicSumcheck;

/// The registers read/write checking sumcheck: relates the read-value claims
/// (`RdWriteValue`, `Rs1Value`, `Rs2Value`) folded by `gamma` to the register
/// `val`/`ra`/`inc` openings weighted by the `EqCycle` public.
pub struct ReadWriteChecking {
    shape: ReadWriteDimensions,
}

impl SymbolicSumcheck for ReadWriteChecking {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RegistersReadWriteChecking
    }

    fn spec(&self) -> JoltSumcheckSpec {
        read_write_checking_sumcheck(self.shape)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = read_write_challenge(RegistersReadWriteChallenge::Gamma);
        opening(rd_write_value_claim())
            + gamma.clone() * opening(rs1_value_claim())
            + gamma.clone().pow(2) * opening(rs2_value_claim())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = read_write_challenge(RegistersReadWriteChallenge::Gamma);
        let eq_cycle = read_write_public(RegistersReadWritePublic::EqCycle);
        eq_cycle.clone() * opening(rd_wa_read_write()) * opening(rd_inc_read_write())
            + eq_cycle.clone() * opening(rd_wa_read_write()) * opening(registers_val_read_write())
            + eq_cycle.clone()
                * gamma.clone()
                * opening(rs1_ra_read_write())
                * opening(registers_val_read_write())
            + eq_cycle
                * gamma.pow(2)
                * opening(rs2_ra_read_write())
                * opening(registers_val_read_write())
    }
}

/// The registers val-evaluation sumcheck: relates the register `val` opening to
/// `rd_inc * rd_wa` weighted by the `LtCycle` public.
pub struct ValEvaluation {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ValEvaluation {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RegistersValEvaluation
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(3)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(registers_val_read_write())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        val_evaluation_public(RegistersValEvaluationPublic::LtCycle)
            * opening(rd_inc_val_evaluation())
            * opening(rd_wa_val_evaluation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltPublicId};
    use jolt_field::Fr;

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 7, 2, 1)
    }

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn read_write_checking_symbolic_matches_dependencies() {
        let relation = ReadWriteChecking::new(read_write_dimensions());
        assert_eq!(
            ReadWriteChecking::id(),
            JoltRelationId::RegistersReadWriteChecking
        );
        assert_eq!(
            relation.spec(),
            read_write_checking_sumcheck(read_write_dimensions())
        );
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                rd_write_value_claim(),
                rs1_value_claim(),
                rs2_value_claim(),
                rd_wa_read_write(),
                rd_inc_read_write(),
                registers_val_read_write(),
                rs1_ra_read_write(),
                rs2_ra_read_write(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(RegistersReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(RegistersReadWritePublic::EqCycle)]
        );
    }

    #[test]
    fn val_evaluation_symbolic_matches_dependencies() {
        let relation = ValEvaluation::new(trace_dimensions());
        assert_eq!(ValEvaluation::id(), JoltRelationId::RegistersValEvaluation);
        assert_eq!(relation.spec(), trace_dimensions().sumcheck(3));
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                registers_val_read_write(),
                rd_inc_val_evaluation(),
                rd_wa_val_evaluation(),
            ]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(RegistersValEvaluationPublic::LtCycle)]
        );
    }
}
