use jolt_field::{Field, RingCore};
use jolt_poly::{EqPolynomial, Polynomial};

use crate::{challenge, opening, public};

use super::super::{
    BooleanityChallenge, BooleanityPublic, JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId,
    JoltRelationClaims, JoltRelationId, JoltVirtualPolynomial,
};
use super::dimensions::{JoltFormulaPointError, JoltSumcheckSpec};
use super::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BooleanityDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_t: usize,
    pub log_k_chunk: usize,
}

impl BooleanityDimensions {
    pub const fn new(layout: JoltRaPolynomialLayout, log_t: usize, log_k_chunk: usize) -> Self {
        Self {
            layout,
            log_t,
            log_k_chunk,
        }
    }

    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k_chunk, 3)
    }

    pub const fn address_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_k_chunk, 3)
    }

    pub const fn cycle_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t, 3)
    }

    pub fn opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<BooleanityOpeningPoint<F>, JoltFormulaPointError> {
        let expected = self.log_t + self.log_k_chunk;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let (r_address, r_cycle) = challenges.split_at(self.log_k_chunk);
        let r_address = r_address.iter().rev().copied().collect::<Vec<_>>();
        let r_cycle = r_cycle.iter().rev().copied().collect::<Vec<_>>();
        let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();

        Ok(BooleanityOpeningPoint {
            r_address,
            r_cycle,
            opening_point,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityOpeningPoint<F: Field> {
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
}

pub fn booleanity<F>(dimensions: BooleanityDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    JoltRelationClaims::new(
        JoltRelationId::Booleanity,
        dimensions.sumcheck(),
        JoltExpr::zero(),
        booleanity_cycle_output(dimensions),
    )
    .with_input_challenges([JoltChallengeId::from(BooleanityChallenge::Gamma)])
}

pub fn booleanity_address_phase<F>(dimensions: BooleanityDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    JoltRelationClaims::new(
        JoltRelationId::Booleanity,
        dimensions.address_sumcheck(),
        JoltExpr::zero(),
        opening(booleanity_address_phase_opening()),
    )
}

pub fn booleanity_cycle_phase<F>(dimensions: BooleanityDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    JoltRelationClaims::new(
        JoltRelationId::Booleanity,
        dimensions.cycle_sumcheck(),
        opening(booleanity_address_phase_opening()),
        booleanity_cycle_output(dimensions),
    )
    .with_input_challenges([JoltChallengeId::from(BooleanityChallenge::Gamma)])
}

fn booleanity_cycle_output<F>(dimensions: BooleanityDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = booleanity_challenge(BooleanityChallenge::Gamma);
    let eq_address_cycle = booleanity_public(BooleanityPublic::EqAddressCycle);
    let mut output = JoltExpr::zero();

    for (i, opening_id) in booleanity_output_openings(dimensions.layout)
        .into_iter()
        .enumerate()
    {
        let ra = opening(opening_id);
        output = output + gamma.clone().pow(2 * i) * (ra.clone() * ra.clone() - ra);
    }

    eq_address_cycle * output
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

pub fn booleanity_output_openings(layout: JoltRaPolynomialLayout) -> Vec<JoltOpeningId> {
    layout.openings(JoltRelationId::Booleanity).collect()
}

pub fn booleanity_address_phase_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::BooleanityAddrClaim,
        JoltRelationId::Booleanity,
    )
}

pub fn eq_address_cycle_polynomial<F>(
    reference_address: &[F],
    reference_cycle: &[F],
) -> Polynomial<F>
where
    F: Field,
{
    let eq_point = reference_address
        .iter()
        .rev()
        .chain(reference_cycle.iter().rev())
        .copied()
        .collect::<Vec<_>>();
    Polynomial::new(EqPolynomial::<F>::evals(&eq_point, None))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanityOutputOpeningGroups {
    pub instruction_ra: Vec<JoltOpeningId>,
    pub bytecode_ra: Vec<JoltOpeningId>,
    pub ram_ra: Vec<JoltOpeningId>,
}

impl BooleanityOutputOpeningGroups {
    pub fn total_len(&self) -> usize {
        self.instruction_ra.len() + self.bytecode_ra.len() + self.ram_ra.len()
    }
}

pub fn booleanity_output_opening_groups(
    layout: JoltRaPolynomialLayout,
) -> BooleanityOutputOpeningGroups {
    BooleanityOutputOpeningGroups {
        instruction_ra: (0..layout.instruction())
            .map(|index| JoltRaPolynomial::Instruction(index).opening(JoltRelationId::Booleanity))
            .collect(),
        bytecode_ra: (0..layout.bytecode())
            .map(|index| JoltRaPolynomial::Bytecode(index).opening(JoltRelationId::Booleanity))
            .collect(),
        ram_ra: (0..layout.ram())
            .map(|index| JoltRaPolynomial::Ram(index).opening(JoltRelationId::Booleanity))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::JoltCommittedPolynomial;
    use super::super::dimensions::JoltFormulaDimensionsError;
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;

    fn layout(
        instruction: usize,
        bytecode: usize,
        ram: usize,
    ) -> Result<JoltRaPolynomialLayout, JoltFormulaDimensionsError> {
        JoltRaPolynomialLayout::new(instruction, bytecode, ram)
    }

    fn dimensions(layout: JoltRaPolynomialLayout) -> BooleanityDimensions {
        BooleanityDimensions::new(layout, 5, 8)
    }

    #[test]
    fn booleanity_exposes_expected_dependencies() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(1, 1, 1)?;
        let claims = booleanity::<Fr>(dimensions(layout));

        assert_eq!(claims.id, JoltRelationId::Booleanity);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(13, 3));
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(
            claims.output.required_openings,
            booleanity_output_openings(layout)
        );
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
    fn booleanity_groups_output_openings_by_ra_family() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(2, 1, 2)?;
        let groups = booleanity_output_opening_groups(layout);

        assert_eq!(groups.instruction_ra.len(), 2);
        assert_eq!(groups.bytecode_ra.len(), 1);
        assert_eq!(groups.ram_ra.len(), 2);
        assert_eq!(groups.total_len(), 5);
        assert_eq!(
            groups
                .instruction_ra
                .iter()
                .chain(&groups.bytecode_ra)
                .chain(&groups.ram_ra)
                .copied()
                .collect::<Vec<_>>(),
            booleanity_output_openings(layout)
        );
        Ok(())
    }

    #[test]
    fn eq_address_cycle_polynomial_reverses_address_then_cycle() {
        let address = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let cycle = vec![Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(11)];
        let eq_point = vec![
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(11),
            Fr::from_u64(7),
            Fr::from_u64(5),
        ];

        assert_eq!(
            eq_address_cycle_polynomial(&address, &cycle).evals(),
            EqPolynomial::<Fr>::evals(&eq_point, None)
        );
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

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::InstructionRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    instruction_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::BytecodeRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    bytecode_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::RamRa(0),
                        JoltRelationId::Booleanity,
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
