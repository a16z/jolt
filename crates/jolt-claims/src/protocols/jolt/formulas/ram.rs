use jolt_field::RingCore;

use crate::{challenge, opening};

use super::super::{
    JoltChallengeId, JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltStageClaims,
    JoltStageId, JoltVirtualPolynomial, RamReadWriteChallenge,
};

pub fn read_write_checking<F>() -> JoltStageClaims<F>
where
    F: RingCore,
{
    let input = opening(ram_read_value())
        + read_write_challenge(RamReadWriteChallenge::Gamma) * opening(ram_write_value());

    let output = read_write_challenge(RamReadWriteChallenge::EqOnePlusGamma)
        * opening(ram_ra())
        * opening(ram_val())
        + read_write_challenge(RamReadWriteChallenge::EqGamma)
            * opening(ram_ra())
            * opening(ram_inc());

    JoltStageClaims::new(JoltStageId::RamReadWriteChecking, input, output)
}

fn read_write_challenge<F>(id: RamReadWriteChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn ram_read_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamReadValue,
        JoltStageId::SpartanOuter,
    )
}

fn ram_write_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamWriteValue,
        JoltStageId::SpartanOuter,
    )
}

fn ram_ra() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltStageId::RamReadWriteChecking,
    )
}

fn ram_val() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamVal,
        JoltStageId::RamReadWriteChecking,
    )
}

fn ram_inc() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltStageId::RamReadWriteChecking,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn read_write_claims_expose_expected_dependencies() {
        let claims = read_write_checking::<Fr>();

        assert_eq!(claims.id, JoltStageId::RamReadWriteChecking);
        assert_eq!(
            claims.input.required_openings,
            vec![ram_read_value(), ram_write_value()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![ram_ra(), ram_val(), ram_inc()]
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(RamReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![
                JoltChallengeId::from(RamReadWriteChallenge::EqOnePlusGamma),
                JoltChallengeId::from(RamReadWriteChallenge::EqGamma),
            ]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                JoltChallengeId::from(RamReadWriteChallenge::Gamma),
                JoltChallengeId::from(RamReadWriteChallenge::EqOnePlusGamma),
                JoltChallengeId::from(RamReadWriteChallenge::EqGamma),
            ]
        );
        assert_eq!(
            claims.challenge_index(JoltChallengeId::from(RamReadWriteChallenge::EqGamma)),
            Some(2)
        );
        assert!(claims.required_publics().is_empty());
        assert_eq!(claims.num_challenges(), 3);
    }

    #[test]
    fn read_write_claims_evaluate_like_core_formula() {
        let claims = read_write_checking::<Fr>();

        let read = Fr::from_u64(3);
        let write = Fr::from_u64(5);
        let ra = Fr::from_u64(7);
        let val = Fr::from_u64(11);
        let inc = Fr::from_u64(13);
        let gamma = Fr::from_u64(17);
        let eq = Fr::from_u64(19);
        let zero = Fr::from_u64(0);
        let one = Fr::from_u64(1);

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == ram_read_value() => read,
                id if id == ram_write_value() => write,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(
                    RamReadWriteChallenge::EqOnePlusGamma | RamReadWriteChallenge::EqGamma,
                ) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == ram_ra() => ra,
                id if id == ram_val() => val,
                id if id == ram_inc() => inc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::EqOnePlusGamma) => {
                    eq * (one + gamma)
                }
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::EqGamma) => eq * gamma,
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => zero,
            },
            |_| zero,
        );

        assert_eq!(input, read + gamma * write);
        assert_eq!(output, eq * ra * (val + gamma * (val + inc)));
    }
}
