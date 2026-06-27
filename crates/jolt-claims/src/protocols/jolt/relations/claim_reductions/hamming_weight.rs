//! Hamming-weight claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, hamming_weight_claim, reduced_claim, virtualization_claim,
    HammingWeightClaimReductionDimensions,
};
use crate::protocols::jolt::{
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic, JoltChallengeId,
    JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

/// Produced one-hot `Ra` opening claims, grouped by family (instruction,
/// bytecode, RAM) in canonical layout order. Every produced opening shares the
/// single hamming-weight opening point. Generic over the cell (`F` on the wire,
/// `Vec<F>` for ZK points, `OpeningClaim<F>` on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(HammingWeightClaimReduction)]
pub struct HammingWeightClaimReductionOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// Consumed claims reduced by the hamming-weight sumcheck: the RAM hamming-weight
/// claim (from RAM hamming booleanity) plus the per-family booleanity and
/// virtualization claims (each wired from its producing stage-6 relation).
/// Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct HammingWeightClaimReductionInputClaims<C> {
    #[opening(RamHammingWeight, from = RamHammingBooleanity)]
    pub ram_hamming_weight: C,
    #[opening(committed = InstructionRa, from = Booleanity)]
    pub instruction_booleanity: Vec<C>,
    #[opening(committed = BytecodeRa, from = Booleanity)]
    pub bytecode_booleanity: Vec<C>,
    #[opening(committed = RamRa, from = Booleanity)]
    pub ram_booleanity: Vec<C>,
    #[opening(committed = InstructionRa, from = InstructionRaVirtualization)]
    pub instruction_virtualization: Vec<C>,
    #[opening(committed = BytecodeRa, from = BytecodeReadRaf)]
    pub bytecode_virtualization: Vec<C>,
    #[opening(committed = RamRa, from = RamRaVirtualization)]
    pub ram_virtualization: Vec<C>,
}

/// Fiat-Shamir challenge drawn by the hamming-weight claim-reduction sumcheck.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct HammingWeightClaimReductionChallenges<F> {
    #[challenge(HammingWeightClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches each RA polynomial's hamming-weight, booleanity, and virtualization
/// claims by powers of `gamma` and reduces them to the per-polynomial
/// hamming-weight-claim-reduction openings weighted by the eq publics.
pub struct ClaimReduction {
    shape: HammingWeightClaimReductionDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = HammingWeightClaimReductionDimensions;
    type Challenges<F> = HammingWeightClaimReductionChallenges<F>;
    type Inputs<C> = HammingWeightClaimReductionInputClaims<C>;
    type Outputs<C> = HammingWeightClaimReductionOutputClaims<C>;

    fn new(shape: HammingWeightClaimReductionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::HammingWeightClaimReduction
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let mut input = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            input = input
                + gamma.clone().pow(3 * i) * hamming_weight_claim(polynomial)
                + gamma.clone().pow(3 * i + 1) * opening(booleanity_claim(polynomial))
                + gamma.clone().pow(3 * i + 2) * opening(virtualization_claim(polynomial));
        }

        input
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let mut output = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let output_coeff = gamma.clone().pow(3 * i)
                + gamma.clone().pow(3 * i + 1)
                    * derived(HammingWeightClaimReductionPublic::EqBooleanity)
                + gamma.clone().pow(3 * i + 2)
                    * derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            output = output + output_coeff * opening(reduced_claim(polynomial));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::ram_hamming_weight;
    use crate::protocols::jolt::geometry::dimensions::JoltFormulaDimensionsError;
    use crate::protocols::jolt::geometry::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn layout(
        instruction: usize,
        bytecode: usize,
        ram: usize,
    ) -> Result<JoltRaPolynomialLayout, JoltFormulaDimensionsError> {
        JoltRaPolynomialLayout::new(instruction, bytecode, ram)
    }

    fn dimensions(layout: JoltRaPolynomialLayout) -> HammingWeightClaimReductionDimensions {
        HammingWeightClaimReductionDimensions::new(layout, 8)
    }

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(1, 1, 1)?;
        let relation = ClaimReduction::new(dimensions(layout));

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

        let input = relation.input_expression::<Fr>().evaluate(
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

        let output = relation.output_expression::<Fr>().evaluate(
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
                JoltDerivedId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionPublic::EqBooleanity,
                ) => eq_bool,
                JoltDerivedId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionPublic::EqVirtualization(0),
                ) => eq_virt_instruction,
                JoltDerivedId::HammingWeightClaimReduction(
                    HammingWeightClaimReductionPublic::EqVirtualization(1),
                ) => eq_virt_bytecode,
                JoltDerivedId::HammingWeightClaimReduction(
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

    #[test]
    fn claim_reduction_exposes_expected_dependencies() -> Result<(), JoltFormulaDimensionsError> {
        let layout = layout(1, 1, 1)?;
        let relation = ClaimReduction::new(dimensions(layout));

        let instruction = JoltRaPolynomial::Instruction(0);
        let bytecode = JoltRaPolynomial::Bytecode(0);
        let ram = JoltRaPolynomial::Ram(0);

        assert_eq!(
            ClaimReduction::id(),
            JoltRelationId::HammingWeightClaimReduction
        );
        assert_eq!(relation.spec(), JoltSumcheckSpec::boolean(8, 2));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
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
            relation.output_expression::<Fr>().required_openings(),
            vec![
                reduced_claim(instruction),
                reduced_claim(bytecode),
                reduced_claim(ram),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(
                HammingWeightClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqBooleanity),
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqVirtualization(0)),
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqVirtualization(1)),
                JoltDerivedId::from(HammingWeightClaimReductionPublic::EqVirtualization(2)),
            ]
        );
        Ok(())
    }
}
