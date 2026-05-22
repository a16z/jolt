use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::super::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltCommittedPolynomial,
    JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationClaims, JoltRelationId,
};
use super::super::dimensions::TraceDimensions;

pub fn claim_reduction<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = inc_challenge(IncClaimReductionChallenge::Gamma);

    let input = opening(ram_inc_read_write())
        + gamma.clone() * opening(ram_inc_val_check())
        + gamma.clone().pow(2) * opening(rd_inc_read_write())
        + gamma.clone().pow(3) * opening(rd_inc_val_evaluation());

    let ram_output_coeff = inc_public(IncClaimReductionPublic::EqRamReadWrite)
        + gamma.clone() * inc_public(IncClaimReductionPublic::EqRamValCheck);
    let rd_output_coeff = inc_public(IncClaimReductionPublic::EqRegistersReadWrite)
        + gamma.clone() * inc_public(IncClaimReductionPublic::EqRegistersValEvaluation);
    let output = ram_output_coeff * opening(ram_inc_reduced())
        + gamma.pow(2) * rd_output_coeff * opening(rd_inc_reduced());

    JoltRelationClaims::new(
        JoltRelationId::IncClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

fn inc_challenge<F>(id: IncClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn inc_public<F>(id: IncClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub fn claim_reduction_input_openings() -> [JoltOpeningId; 4] {
    [
        ram_inc_read_write(),
        ram_inc_val_check(),
        rd_inc_read_write(),
        rd_inc_val_evaluation(),
    ]
}

pub fn claim_reduction_output_openings() -> [JoltOpeningId; 2] {
    [ram_inc_reduced(), rd_inc_reduced()]
}

pub fn ram_inc_read_write_opening() -> JoltOpeningId {
    ram_inc_read_write()
}

pub fn ram_inc_val_check_opening() -> JoltOpeningId {
    ram_inc_val_check()
}

pub fn rd_inc_read_write_opening() -> JoltOpeningId {
    rd_inc_read_write()
}

pub fn rd_inc_val_evaluation_opening() -> JoltOpeningId {
    rd_inc_val_evaluation()
}

pub fn ram_inc_reduced_opening() -> JoltOpeningId {
    ram_inc_reduced()
}

pub fn rd_inc_reduced_opening() -> JoltOpeningId {
    rd_inc_reduced()
}

fn ram_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

fn ram_inc_val_check() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

fn rd_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

fn rd_inc_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersValEvaluation,
    )
}

fn ram_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::IncClaimReduction,
    )
}

fn rd_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::IncClaimReduction,
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

        assert_eq!(claims.id, JoltRelationId::IncClaimReduction);
        assert_eq!(claims.sumcheck, dimensions().sumcheck(2));
        assert_eq!(
            claims.input.required_openings,
            claim_reduction_input_openings()
        );
        assert_eq!(
            claims.output.required_openings,
            claim_reduction_output_openings()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(IncClaimReductionChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(IncClaimReductionPublic::EqRamReadWrite),
                JoltPublicId::from(IncClaimReductionPublic::EqRamValCheck),
                JoltPublicId::from(IncClaimReductionPublic::EqRegistersReadWrite),
                JoltPublicId::from(IncClaimReductionPublic::EqRegistersValEvaluation),
            ]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() {
        let claims = claim_reduction::<Fr>(dimensions());

        let ram_rw = Fr::from_u64(3);
        let ram_val = Fr::from_u64(5);
        let rd_rw = Fr::from_u64(7);
        let rd_val = Fr::from_u64(11);
        let ram_reduced = Fr::from_u64(13);
        let rd_reduced = Fr::from_u64(17);
        let eq_ram_rw = Fr::from_u64(19);
        let eq_ram_val = Fr::from_u64(23);
        let eq_rd_rw = Fr::from_u64(29);
        let eq_rd_val = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_inc_read_write() => ram_rw,
                id if id == ram_inc_val_check() => ram_val,
                id if id == rd_inc_read_write() => rd_rw,
                id if id == rd_inc_val_evaluation() => rd_val,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_inc_reduced() => ram_reduced,
                id if id == rd_inc_reduced() => rd_reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamReadWrite) => {
                    eq_ram_rw
                }
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamValCheck) => {
                    eq_ram_val
                }
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRegistersReadWrite) => {
                    eq_rd_rw
                }
                JoltPublicId::IncClaimReduction(
                    IncClaimReductionPublic::EqRegistersValEvaluation,
                ) => eq_rd_val,
                _ => zero,
            },
        );

        let gamma_2 = gamma * gamma;
        assert_eq!(
            input,
            ram_rw + gamma * ram_val + gamma_2 * rd_rw + gamma_2 * gamma * rd_val
        );
        assert_eq!(
            output,
            ram_reduced * (eq_ram_rw + gamma * eq_ram_val)
                + gamma_2 * rd_reduced * (eq_rd_rw + gamma * eq_rd_val)
        );
    }
}
