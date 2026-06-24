//! Hamming-weight claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::claim_reductions::hamming_weight::{
    booleanity_claim, hamming_weight_challenge, hamming_weight_claim, hamming_weight_public,
    reduced_claim, virtualization_claim, HammingWeightClaimReductionDimensions,
};
use crate::protocols::jolt::{
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic, JoltChallengeId,
    JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, SymbolicSumcheck};

/// Batches each RA polynomial's hamming-weight, booleanity, and virtualization
/// claims by powers of `gamma` and reduces them to the per-polynomial
/// hamming-weight-claim-reduction openings weighted by the eq publics.
pub struct ClaimReduction {
    shape: HammingWeightClaimReductionDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = HammingWeightClaimReductionDimensions;

    fn new(shape: HammingWeightClaimReductionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::HammingWeightClaimReduction
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = hamming_weight_challenge(HammingWeightClaimReductionChallenge::Gamma);
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
        let gamma = hamming_weight_challenge(HammingWeightClaimReductionChallenge::Gamma);
        let mut output = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let output_coeff = gamma.clone().pow(3 * i)
                + gamma.clone().pow(3 * i + 1)
                    * hamming_weight_public(HammingWeightClaimReductionPublic::EqBooleanity)
                + gamma.clone().pow(3 * i + 2)
                    * hamming_weight_public(HammingWeightClaimReductionPublic::EqVirtualization(i));
            output = output + output_coeff * opening(reduced_claim(polynomial));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::formulas::claim_reductions::hamming_weight::ram_hamming_weight;
    use crate::protocols::jolt::formulas::dimensions::JoltFormulaDimensionsError;
    use crate::protocols::jolt::formulas::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
    use jolt_field::Fr;

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
        assert_eq!(relation.sumcheck(), JoltSumcheckSpec::boolean(8, 2));
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
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqBooleanity),
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(0)),
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(1)),
                JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(2)),
            ]
        );
        Ok(())
    }
}
