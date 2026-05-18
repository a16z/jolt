use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::{
    BooleanityChallenge, BooleanityPublic, JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId,
    JoltStageClaims, JoltStageId,
};
use super::dimensions::JoltSumcheckSpec;
use super::ra::JoltRaPolynomialLayout;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BooleanityDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_t: usize,
    pub log_k_chunk: usize,
}

impl BooleanityDimensions {
    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k_chunk, 3)
    }
}

impl From<(JoltRaPolynomialLayout, usize, usize)> for BooleanityDimensions {
    fn from((layout, log_t, log_k_chunk): (JoltRaPolynomialLayout, usize, usize)) -> Self {
        Self {
            layout,
            log_t,
            log_k_chunk,
        }
    }
}

pub fn booleanity<F>(dimensions: BooleanityDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let gamma = booleanity_challenge(BooleanityChallenge::Gamma);
    let eq_address_cycle = booleanity_public(BooleanityPublic::EqAddressCycle);
    let mut output = JoltExpr::zero();

    for (i, opening_id) in booleanity_openings(dimensions.layout)
        .into_iter()
        .enumerate()
    {
        let ra = opening(opening_id);
        output = output + gamma.clone().pow(2 * i) * (ra.clone() * ra.clone() - ra);
    }

    JoltStageClaims::new(
        JoltStageId::Booleanity,
        dimensions.sumcheck(),
        JoltExpr::zero(),
        eq_address_cycle * output,
    )
    .with_input_challenges([JoltChallengeId::from(BooleanityChallenge::Gamma)])
}

fn booleanity_challenge<F>(id: BooleanityChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn booleanity_public<F>(id: BooleanityPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn booleanity_openings(layout: JoltRaPolynomialLayout) -> Vec<JoltOpeningId> {
    layout.openings(JoltStageId::Booleanity).collect()
}

#[cfg(test)]
mod tests {
    use super::super::super::JoltCommittedPolynomial;
    use super::super::dimensions::JoltFormulaDimensionsError;
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn layout(
        instruction: usize,
        bytecode: usize,
        ram: usize,
    ) -> Result<JoltRaPolynomialLayout, JoltFormulaDimensionsError> {
        JoltRaPolynomialLayout::new(instruction, bytecode, ram)
    }

    fn dimensions(layout: JoltRaPolynomialLayout) -> BooleanityDimensions {
        (layout, 5, 8).into()
    }

    #[test]
    fn booleanity_exposes_expected_dependencies() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(1, 1, 1)?;
        let claims = booleanity::<Fr>(dimensions(layout));

        assert_eq!(claims.id, JoltStageId::Booleanity);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(13, 3));
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(claims.output.required_openings, booleanity_openings(layout));
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(BooleanityChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(BooleanityChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(BooleanityChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(BooleanityPublic::EqAddressCycle)]
        );
        assert_eq!(claims.num_challenges(), 1);
        Ok(())
    }

    #[test]
    fn booleanity_preserves_gamma_dependency_for_single_polynomial(
    ) -> Result<(), JoltFormulaDimensionsError> {
        let claims = booleanity::<Fr>(dimensions(layout(1, 0, 0)?));

        assert!(claims.output.required_challenges.is_empty());
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(BooleanityChallenge::Gamma)]
        );
        Ok(())
    }

    #[test]
    fn booleanity_evaluates_like_core_formula() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(1, 1, 1)?;
        let claims = booleanity::<Fr>(dimensions(layout));

        let instruction_ra = Fr::from_u64(3);
        let bytecode_ra = Fr::from_u64(5);
        let ram_ra = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let eq_address_cycle = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::InstructionRa(0),
                        JoltStageId::Booleanity,
                    ) =>
                {
                    instruction_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::BytecodeRa(0),
                        JoltStageId::Booleanity,
                    ) =>
                {
                    bytecode_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::RamRa(0),
                        JoltStageId::Booleanity,
                    ) =>
                {
                    ram_ra
                }
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::Booleanity(BooleanityPublic::EqAddressCycle) => eq_address_cycle,
                _ => zero,
            },
        );

        let gamma_2 = gamma * gamma;
        let gamma_4 = gamma_2 * gamma_2;
        assert_eq!(
            output,
            eq_address_cycle
                * ((instruction_ra * instruction_ra - instruction_ra)
                    + gamma_2 * (bytecode_ra * bytecode_ra - bytecode_ra)
                    + gamma_4 * (ram_ra * ram_ra - ram_ra))
        );
        Ok(())
    }
}
