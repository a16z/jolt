//! Stage-7 digit-zero reduction. Instruction and bytecode RA columns are
//! digit-zero virtualized; RAM commits its digit-zero row and reduces unrecentered
//! booleanity and virtualization legs.

use std::collections::BTreeSet;

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, gamma_power_expr, reduced_claim, virtualization_claim,
    HammingWeightClaimReductionDimensions,
};
use crate::protocols::jolt::geometry::ra::JoltRaPolynomial;
use crate::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionChallenges;
use crate::protocols::jolt::{
    HammingWeightClaimReductionPublic, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct LatticeDigitZeroClaimReductionInputClaims<C> {
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
/// paths) shared with the base reduction that fills the same slot. So the *slot* is named
/// for the base algebra while this type is named for the lattice one — the verifier
/// aliases between the two names once, at the `mode` seam, and nowhere else.
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
        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            match polynomial {
                // RAM: no reconstruction. The committed column includes the
                // digit-zero row.
                JoltRaPolynomial::Ram(_) => {
                    input = input
                        + gamma_power_expr(2 * i) * opening(booleanity_claim(polynomial))
                        + gamma_power_expr(2 * i + 1) * opening(virtualization_claim(polynomial));
                }
                // Public M_mu = 1: fold eq(r_address, 0) into each input claim.
                JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => {
                    let eq_virtualization_digit_zero =
                        derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
                    input = input
                        + gamma_power_expr(2 * i)
                            * (opening(booleanity_claim(polynomial))
                                - eq_booleanity_digit_zero.clone())
                        + gamma_power_expr(2 * i + 1)
                            * (opening(virtualization_claim(polynomial))
                                - eq_virtualization_digit_zero);
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

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let eq_virtualization = derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            let coefficient = match polynomial {
                JoltRaPolynomial::Ram(_) => {
                    gamma_power_expr(2 * i) * eq_booleanity.clone()
                        + gamma_power_expr(2 * i + 1) * eq_virtualization
                }
                // The committed rows use eq(r_address, k_i) - eq(r_address, 0).
                JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => {
                    let eq_virtualization_digit_zero =
                        derived(HammingWeightClaimReductionPublic::EqVirtualizationAtDigitZero(i));
                    gamma_power_expr(2 * i)
                        * (eq_booleanity.clone() - eq_booleanity_digit_zero.clone())
                        + gamma_power_expr(2 * i + 1)
                            * (eq_virtualization - eq_virtualization_digit_zero)
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
        self.shape.layout.polynomials().map(reduced_claim).collect()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use crate::referenced_openings;
    use jolt_field::Fr;

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
}
