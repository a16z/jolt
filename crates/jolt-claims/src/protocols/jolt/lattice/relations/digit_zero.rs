//! Stage-7 digit-zero reduction for instruction and bytecode RA columns.
//! RAM retains its Hamming-weight, Booleanity, and virtualization legs.

use std::collections::BTreeSet;

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, gamma_power_expr, reduced_claim, virtualization_claim,
    HammingWeightClaimReductionDimensions,
};
use crate::protocols::jolt::geometry::ra::JoltRaPolynomial;
use crate::protocols::jolt::geometry::ram::ram_hamming_weight;
use crate::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionChallenges;
use crate::protocols::jolt::{
    HammingWeightClaimReductionPublic, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct LatticeDigitZeroClaimReductionInputClaims<C> {
    /// The RAM access indicator produced by `RamHammingBooleanity`. RAM is not
    /// virtualized, so this remains the base Hamming-weight leg.
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

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(HammingWeightClaimReduction)]
pub struct LatticeDigitZeroClaimReductionOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// The lattice instantiation of the stage-7 reduction slot, selected by the verifier's
/// `akita` feature (`jolt-verifier/src/stages/stage7/hamming_weight_claim_reduction.rs`,
/// the `mode` module — the single place base and lattice are swapped).
///
/// [`Self::id`] deliberately returns the base `JoltRelationId::HammingWeightClaimReduction`:
/// the relation id is protocol data (opening ids, transcript labels, tamper-manifest
/// paths), RAM still carries a genuine Hamming-weight leg here, and base mode must keep
/// producing bit-identical proofs. So the *slot* is named for the base algebra while this
/// type is named for the lattice one — the verifier aliases between the two names once,
/// at the `mode` seam, and nowhere else.
#[derive(Clone)]
pub struct LatticeDigitZeroClaimReduction {
    shape: HammingWeightClaimReductionDimensions,
}

impl SymbolicSumcheck for LatticeDigitZeroClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = HammingWeightClaimReductionDimensions;
    type Challenges<F> = HammingWeightClaimReductionChallenges<F>;
    type Inputs<C> = LatticeDigitZeroClaimReductionInputClaims<C>;
    type Outputs<C> = LatticeDigitZeroClaimReductionOutputClaims<C>;

    fn new(shape: Self::Shape) -> Self {
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
        let eq_booleanity_digit_zero =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero);
        let mut input = JoltExpr::zero();
        let mut power = 0usize;
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            match polynomial {
                // RAM: base three legs, no reconstruction. The committed column
                // includes the digit-zero row.
                JoltRaPolynomial::Ram(_) => {
                    input = input
                        + gamma_power_expr(power) * opening(ram_hamming_weight())
                        + gamma_power_expr(power + 1) * opening(booleanity_claim(polynomial))
                        + gamma_power_expr(power + 2) * opening(virtualization_claim(polynomial));
                    power += 3;
                }
                // Public M_mu = 1: fold eq(r_address, 0) into each input claim.
                JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => {
                    let eq_virtualization_digit_zero =
                        derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
                    input = input
                        + gamma_power_expr(power)
                            * (opening(booleanity_claim(polynomial))
                                - eq_booleanity_digit_zero.clone())
                        + gamma_power_expr(power + 1)
                            * (opening(virtualization_claim(polynomial))
                                - eq_virtualization_digit_zero);
                    power += 2;
                }
            }
        }
        input
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let eq_booleanity = derived(HammingWeightClaimReductionPublic::EqBooleanity);
        let eq_booleanity_digit_zero =
            derived(HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero);
        let mut output = JoltExpr::zero();
        let mut power = 0usize;

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let eq_virtualization = derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            let coefficient = match polynomial {
                JoltRaPolynomial::Ram(_) => {
                    let c = gamma_power_expr(power)
                        + gamma_power_expr(power + 1) * eq_booleanity.clone()
                        + gamma_power_expr(power + 2) * eq_virtualization;
                    power += 3;
                    c
                }
                // The committed rows use eq(r_address, k_i) - eq(r_address, 0).
                JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => {
                    let eq_virtualization_digit_zero =
                        derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
                    let c = gamma_power_expr(power)
                        * (eq_booleanity.clone() - eq_booleanity_digit_zero.clone())
                        + gamma_power_expr(power + 1)
                            * (eq_virtualization - eq_virtualization_digit_zero);
                    power += 2;
                    c
                }
            };
            output = output + coefficient * opening(reduced_claim(polynomial));
        }
        output
    }

    /// Every output term is one reduced opening times a coefficient built from
    /// public values and challenges alone, so the openings are listed
    /// without expanding the output.
    fn expected_output_openings<F: Ring>(&self) -> BTreeSet<JoltOpeningId> {
        self.shape
            .layout
            .polynomials()
            .map(reduced_claim)
            .collect()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use crate::protocols::jolt::{
        HammingWeightClaimReductionChallenge, JoltChallengeId, JoltDerivedId,
    };
    use crate::referenced_openings;
    use jolt_field::{Fr, Ring};

    #[test]
    fn expected_output_openings_match_expanded_output() {
        let layout = JoltRaPolynomialLayout::new(2, 3, 2).unwrap();
        let relation = LatticeDigitZeroClaimReduction::new(
            HammingWeightClaimReductionDimensions::new(layout, 32),
        );
        assert_eq!(
            relation.expected_output_openings::<Fr>(),
            referenced_openings([relation.output_expression::<Fr>()])
        );
    }

    #[test]
    fn public_unit_activation_ra_is_reconstructed_but_ram_keeps_base_legs() {
        let layout = JoltRaPolynomialLayout::new(1, 1, 1).unwrap();
        let relation = LatticeDigitZeroClaimReduction::new(
            HammingWeightClaimReductionDimensions::new(layout, 4),
        );

        let gamma = Fr::from_u64(3);
        let eq_bool = Fr::from_u64(13);
        let eq_bool_zero = Fr::from_u64(17);
        let eq_virt = [Fr::from_u64(19), Fr::from_u64(23), Fr::from_u64(29)];
        let eq_virt_zero = [Fr::from_u64(31), Fr::from_u64(37)];
        let inst_bool = Fr::from_u64(41);
        let inst_virt = Fr::from_u64(43);
        let bytecode_bool = Fr::from_u64(47);
        let bytecode_virt = Fr::from_u64(53);
        let ram_hamming = Fr::from_u64(59);
        let ram_bool = Fr::from_u64(61);
        let ram_virt = Fr::from_u64(67);
        let out = [Fr::from_u64(71), Fr::from_u64(73), Fr::from_u64(79)];
        let zero = Fr::from_u64(0);
        let power = |exponent: usize| {
            (0..exponent).fold(Fr::from_u64(1), |accumulator, _| accumulator * gamma)
        };

        let opening_value = |id: &JoltOpeningId| match *id {
            id if id == booleanity_claim(JoltRaPolynomial::Instruction(0)) => inst_bool,
            id if id == virtualization_claim(JoltRaPolynomial::Instruction(0)) => inst_virt,
            id if id == booleanity_claim(JoltRaPolynomial::Bytecode(0)) => bytecode_bool,
            id if id == virtualization_claim(JoltRaPolynomial::Bytecode(0)) => bytecode_virt,
            id if id == ram_hamming_weight() => ram_hamming,
            id if id == booleanity_claim(JoltRaPolynomial::Ram(0)) => ram_bool,
            id if id == virtualization_claim(JoltRaPolynomial::Ram(0)) => ram_virt,
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
                HammingWeightClaimReductionPublic::EqBooleanityAtDigitZero,
            ) => eq_bool_zero,
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(index),
            ) => eq_virt[index],
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(index),
            ) => eq_virt_zero[index],
            JoltDerivedId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::ReservedBalancedIncValueAtAddress,
            ) => zero,
            _ => zero,
        };

        let input = relation.input_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_input = power(0) * (inst_bool - eq_bool_zero)
            + power(1) * (inst_virt - eq_virt_zero[0])
            + power(2) * (bytecode_bool - eq_bool_zero)
            + power(3) * (bytecode_virt - eq_virt_zero[1])
            + power(4) * ram_hamming
            + power(5) * ram_bool
            + power(6) * ram_virt;
        assert_eq!(input, expected_input);

        let output = relation.output_expression::<Fr>().evaluate(
            opening_value,
            challenge_value,
            derived_value,
        );
        let expected_output = out[0]
            * (power(0) * (eq_bool - eq_bool_zero) + power(1) * (eq_virt[0] - eq_virt_zero[0]))
            + out[1]
                * (power(2) * (eq_bool - eq_bool_zero) + power(3) * (eq_virt[1] - eq_virt_zero[1]))
            + out[2] * (power(4) + power(5) * eq_bool + power(6) * eq_virt[2]);
        assert_eq!(output, expected_output);
    }
}
