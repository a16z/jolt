use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::super::{
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic, JoltChallengeId,
    JoltExpr, JoltOpeningId, JoltPublicId, JoltStageClaims, JoltStageId, JoltVirtualPolynomial,
};
use super::super::dimensions::JoltSumcheckSpec;
use super::super::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HammingWeightClaimReductionDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_k_chunk: usize,
}

impl HammingWeightClaimReductionDimensions {
    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_k_chunk, 2)
    }
}

impl From<(JoltRaPolynomialLayout, usize)> for HammingWeightClaimReductionDimensions {
    fn from((layout, log_k_chunk): (JoltRaPolynomialLayout, usize)) -> Self {
        Self {
            layout,
            log_k_chunk,
        }
    }
}

pub fn claim_reduction<F>(dimensions: HammingWeightClaimReductionDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let gamma = hamming_weight_challenge(HammingWeightClaimReductionChallenge::Gamma);
    let mut input = JoltExpr::zero();
    let mut output = JoltExpr::zero();

    for (i, polynomial) in dimensions.layout.polynomials().enumerate() {
        input = input
            + gamma.clone().pow(3 * i) * hamming_weight_claim(polynomial)
            + gamma.clone().pow(3 * i + 1) * opening(booleanity_claim(polynomial))
            + gamma.clone().pow(3 * i + 2) * opening(virtualization_claim(polynomial));

        let output_coeff = gamma.clone().pow(3 * i)
            + gamma.clone().pow(3 * i + 1)
                * hamming_weight_public(HammingWeightClaimReductionPublic::EqBooleanity)
            + gamma.clone().pow(3 * i + 2)
                * hamming_weight_public(HammingWeightClaimReductionPublic::EqVirtualization(i));
        output = output + output_coeff * opening(reduced_claim(polynomial));
    }

    JoltStageClaims::new(
        JoltStageId::HammingWeightClaimReduction,
        dimensions.sumcheck(),
        input,
        output,
    )
}

fn hamming_weight_challenge<F>(id: HammingWeightClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn hamming_weight_public<F>(id: HammingWeightClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn hamming_weight_claim<F>(polynomial: JoltRaPolynomial) -> JoltExpr<F>
where
    F: RingCore,
{
    match polynomial {
        JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => JoltExpr::one(),
        JoltRaPolynomial::Ram(_) => opening(ram_hamming_weight()),
    }
}

fn booleanity_claim(polynomial: JoltRaPolynomial) -> JoltOpeningId {
    polynomial.opening(JoltStageId::Booleanity)
}

fn virtualization_claim(polynomial: JoltRaPolynomial) -> JoltOpeningId {
    match polynomial {
        JoltRaPolynomial::Instruction(_) => JoltOpeningId::committed(
            polynomial.committed(),
            JoltStageId::InstructionRaVirtualization,
        ),
        JoltRaPolynomial::Bytecode(_) => {
            JoltOpeningId::committed(polynomial.committed(), JoltStageId::BytecodeReadRaf)
        }
        JoltRaPolynomial::Ram(_) => {
            JoltOpeningId::committed(polynomial.committed(), JoltStageId::RamRaVirtualization)
        }
    }
}

fn reduced_claim(polynomial: JoltRaPolynomial) -> JoltOpeningId {
    polynomial.opening(JoltStageId::HammingWeightClaimReduction)
}

fn ram_hamming_weight() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamHammingWeight,
        JoltStageId::RamHammingBooleanity,
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::dimensions::JoltFormulaDimensionsError;
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn layout(
        instruction: usize,
        bytecode: usize,
        ram: usize,
    ) -> Result<JoltRaPolynomialLayout, JoltFormulaDimensionsError> {
        JoltRaPolynomialLayout::new(instruction, bytecode, ram)
    }

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

    fn dimensions(layout: JoltRaPolynomialLayout) -> HammingWeightClaimReductionDimensions {
        (layout, 8).into()
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(1, 1, 1)?;
        let claims = claim_reduction::<Fr>(dimensions(layout));

        let instruction = JoltRaPolynomial::Instruction(0);
        let bytecode = JoltRaPolynomial::Bytecode(0);
        let ram = JoltRaPolynomial::Ram(0);

        assert_eq!(claims.id, JoltStageId::HammingWeightClaimReduction);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(8, 2));
        assert_eq!(
            claims.input.required_openings,
            vec![
                booleanity_claim(instruction),
                virtualization_claim(instruction),
                booleanity_claim(bytecode),
                virtualization_claim(bytecode),
                ram_hamming_weight(),
                booleanity_claim(ram),
                virtualization_claim(ram),
            ]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![
                reduced_claim(instruction),
                reduced_claim(bytecode),
                reduced_claim(ram),
            ]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                HammingWeightClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqBooleanity),
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(0)),
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(1)),
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(2)),
            ]
        );
        assert_eq!(claims.num_challenges(), 1);
        Ok(())
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(1, 1, 1)?;
        let claims = claim_reduction::<Fr>(dimensions(layout));

        let instruction = JoltRaPolynomial::Instruction(0);
        let bytecode = JoltRaPolynomial::Bytecode(0);
        let ram = JoltRaPolynomial::Ram(0);

        let ram_hw = Fr::from_u64(3);
        let bool_instruction = Fr::from_u64(5);
        let virt_instruction = Fr::from_u64(7);
        let bool_bytecode = Fr::from_u64(11);
        let virt_bytecode = Fr::from_u64(13);
        let bool_ram = Fr::from_u64(17);
        let virt_ram = Fr::from_u64(19);
        let reduced_instruction = Fr::from_u64(23);
        let reduced_bytecode = Fr::from_u64(29);
        let reduced_ram = Fr::from_u64(31);
        let eq_bool = Fr::from_u64(37);
        let eq_virt_instruction = Fr::from_u64(41);
        let eq_virt_bytecode = Fr::from_u64(43);
        let eq_virt_ram = Fr::from_u64(47);
        let gamma = Fr::from_u64(53);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == ram_hamming_weight() => ram_hw,
                id if id == booleanity_claim(instruction) => bool_instruction,
                id if id == virtualization_claim(instruction) => virt_instruction,
                id if id == booleanity_claim(bytecode) => bool_bytecode,
                id if id == virtualization_claim(bytecode) => virt_bytecode,
                id if id == booleanity_claim(ram) => bool_ram,
                id if id == virtualization_claim(ram) => virt_ram,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == reduced_claim(instruction) => reduced_instruction,
                id if id == reduced_claim(bytecode) => reduced_bytecode,
                id if id == reduced_claim(ram) => reduced_ram,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionPublic::EqBooleanity,
                ) => eq_bool,
                JoltPublicId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionPublic::EqVirtualization(0),
                ) => eq_virt_instruction,
                JoltPublicId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionPublic::EqVirtualization(1),
                ) => eq_virt_bytecode,
                JoltPublicId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionPublic::EqVirtualization(2),
                ) => eq_virt_ram,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            gamma_power(gamma, 0)
                + gamma_power(gamma, 1) * bool_instruction
                + gamma_power(gamma, 2) * virt_instruction
                + gamma_power(gamma, 3)
                + gamma_power(gamma, 4) * bool_bytecode
                + gamma_power(gamma, 5) * virt_bytecode
                + gamma_power(gamma, 6) * ram_hw
                + gamma_power(gamma, 7) * bool_ram
                + gamma_power(gamma, 8) * virt_ram
        );
        assert_eq!(
            output,
            reduced_instruction
                * (gamma_power(gamma, 0)
                    + gamma_power(gamma, 1) * eq_bool
                    + gamma_power(gamma, 2) * eq_virt_instruction)
                + reduced_bytecode
                    * (gamma_power(gamma, 3)
                        + gamma_power(gamma, 4) * eq_bool
                        + gamma_power(gamma, 5) * eq_virt_bytecode)
                + reduced_ram
                    * (gamma_power(gamma, 6)
                        + gamma_power(gamma, 7) * eq_bool
                        + gamma_power(gamma, 8) * eq_virt_ram)
        );
        Ok(())
    }
}
