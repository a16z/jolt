use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, gamma_power_expr, has_hamming_weight_leg, reduced_claim,
    virtualization_claim, HammingWeightClaimReductionDimensions,
};
use crate::protocols::jolt::{
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic, JoltChallengeId,
    JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// Produced one-hot `Ra` opening claims, grouped by family (instruction,
/// bytecode, RAM) in canonical layout order. Every produced opening shares the
/// single hamming-weight opening point. Generic over the opening cell (`F` for
/// the serialized wire value, `Vec<F>` for the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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

/// Consumed claims reduced by the hamming-weight sumcheck: the per-family
/// booleanity and virtualization claims (each wired from its producing stage-6
/// relation). The instruction and bytecode Hamming-weight claims are the public
/// constant one; RAM has none. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct HammingWeightClaimReductionInputClaims<C> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct HammingWeightClaimReductionChallenges<F> {
    #[challenge(HammingWeightClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches each RA polynomial's booleanity and virtualization claims, and the
/// instruction and bytecode Hamming-weight claims, by powers of `gamma` and
/// reduces them to the per-polynomial hamming-weight-claim-reduction openings
/// weighted by the eq publics.
#[derive(Clone)]
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

    fn rounds(&self) -> usize {
        self.shape.log_k_chunk
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let mut input = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            if has_hamming_weight_leg(polynomial) {
                input = input + gamma_power_expr(3 * i);
            }
            input = input
                + gamma_power_expr(3 * i + 1) * opening(booleanity_claim(polynomial))
                + gamma_power_expr(3 * i + 2) * opening(virtualization_claim(polynomial));
        }

        input
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let mut output = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let mut output_coeff = gamma_power_expr(3 * i + 1)
                * derived(HammingWeightClaimReductionPublic::EqBooleanity)
                + gamma_power_expr(3 * i + 2)
                    * derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            if has_hamming_weight_leg(polynomial) {
                output_coeff = gamma_power_expr(3 * i) + output_coeff;
            }
            output = output + output_coeff * opening(reduced_claim(polynomial));
        }

        output
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
    use jolt_field::Fr;

    #[test]
    fn ram_reduces_booleanity_and_virtualization_legs_only() {
        let layout = JoltRaPolynomialLayout::new(1, 1, 1).unwrap();
        let relation = ClaimReduction::new(HammingWeightClaimReductionDimensions::new(layout, 4));

        let gamma = Fr::from_u64(3);
        let eq_bool = Fr::from_u64(13);
        let eq_virt = [Fr::from_u64(19), Fr::from_u64(23), Fr::from_u64(29)];
        let booleanity = [Fr::from_u64(41), Fr::from_u64(47), Fr::from_u64(61)];
        let virtualization = [Fr::from_u64(43), Fr::from_u64(53), Fr::from_u64(67)];
        let out = [Fr::from_u64(71), Fr::from_u64(73), Fr::from_u64(79)];
        let zero = Fr::from_u64(0);
        let power = |exponent: usize| {
            (0..exponent).fold(Fr::from_u64(1), |accumulator, _| accumulator * gamma)
        };

        let opening_value = |id: &JoltOpeningId| match *id {
            id if id == booleanity_claim(JoltRaPolynomial::Instruction(0)) => booleanity[0],
            id if id == virtualization_claim(JoltRaPolynomial::Instruction(0)) => virtualization[0],
            id if id == booleanity_claim(JoltRaPolynomial::Bytecode(0)) => booleanity[1],
            id if id == virtualization_claim(JoltRaPolynomial::Bytecode(0)) => virtualization[1],
            id if id == booleanity_claim(JoltRaPolynomial::Ram(0)) => booleanity[2],
            id if id == virtualization_claim(JoltRaPolynomial::Ram(0)) => virtualization[2],
            id if id == reduced_claim(JoltRaPolynomial::Instruction(0)) => out[0],
            id if id == reduced_claim(JoltRaPolynomial::Bytecode(0)) => out[1],
            id if id == reduced_claim(JoltRaPolynomial::Ram(0)) => out[2],
            _ => zero,
        };
        let challenge_value = |id: &JoltChallengeId| match *id {
            JoltChallengeId::HammingWeightClaimReduction(
                HammingWeightClaimReductionChallenge::Gamma,
            ) => gamma,
            _ => zero,
        };
        let derived_value = |id: &JoltDerivedId| match *id {
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::GammaPow(exponent),
            ) => power(exponent),
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqBooleanity,
            ) => eq_bool,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(index),
            ) => eq_virt[index],
            _ => zero,
        };

        let input = relation.input_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_input = power(0)
            + power(1) * booleanity[0]
            + power(2) * virtualization[0]
            + power(3)
            + power(4) * booleanity[1]
            + power(5) * virtualization[1]
            + power(7) * booleanity[2]
            + power(8) * virtualization[2];
        assert_eq!(input, expected_input);

        let output = relation.output_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_output = out[0] * (power(0) + power(1) * eq_bool + power(2) * eq_virt[0])
            + out[1] * (power(3) + power(4) * eq_bool + power(5) * eq_virt[1])
            + out[2] * (power(7) * eq_bool + power(8) * eq_virt[2]);
        assert_eq!(output, expected_output);
    }
}
