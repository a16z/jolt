//! Jolt committed-polynomial opening order used by the final PCS check.

use jolt_field::Field;

use super::super::{JoltCommittedPolynomial, JoltOpeningId, JoltStageId};
use super::ra::JoltRaPolynomialLayout;

pub fn proof_commitment_order(layout: JoltRaPolynomialLayout) -> Vec<JoltCommittedPolynomial> {
    let mut polynomials = Vec::with_capacity(2 + layout.total());
    polynomials.push(JoltCommittedPolynomial::RdInc);
    polynomials.push(JoltCommittedPolynomial::RamInc);
    polynomials.extend((0..layout.instruction()).map(JoltCommittedPolynomial::InstructionRa));
    polynomials.extend((0..layout.ram()).map(JoltCommittedPolynomial::RamRa));
    polynomials.extend((0..layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa));
    polynomials
}

pub fn final_opening_polynomial_order(
    layout: JoltRaPolynomialLayout,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
) -> Vec<JoltCommittedPolynomial> {
    let mut polynomials = Vec::with_capacity(
        2 + layout.total()
            + usize::from(include_trusted_advice)
            + usize::from(include_untrusted_advice),
    );
    polynomials.push(JoltCommittedPolynomial::RamInc);
    polynomials.push(JoltCommittedPolynomial::RdInc);
    polynomials.extend((0..layout.instruction()).map(JoltCommittedPolynomial::InstructionRa));
    polynomials.extend((0..layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa));
    polynomials.extend((0..layout.ram()).map(JoltCommittedPolynomial::RamRa));
    if include_trusted_advice {
        polynomials.push(JoltCommittedPolynomial::TrustedAdvice);
    }
    if include_untrusted_advice {
        polynomials.push(JoltCommittedPolynomial::UntrustedAdvice);
    }
    polynomials
}

pub fn final_opening_ids(
    layout: JoltRaPolynomialLayout,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
) -> Vec<JoltOpeningId> {
    let mut ids = Vec::with_capacity(
        2 + layout.total()
            + usize::from(include_trusted_advice)
            + usize::from(include_untrusted_advice),
    );
    ids.push(JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltStageId::IncClaimReduction,
    ));
    ids.push(JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltStageId::IncClaimReduction,
    ));
    ids.extend((0..layout.instruction()).map(|index| {
        JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(index),
            JoltStageId::HammingWeightClaimReduction,
        )
    }));
    ids.extend((0..layout.bytecode()).map(|index| {
        JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltStageId::HammingWeightClaimReduction,
        )
    }));
    ids.extend((0..layout.ram()).map(|index| {
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RamRa(index),
            JoltStageId::HammingWeightClaimReduction,
        )
    }));
    if include_trusted_advice {
        ids.push(JoltOpeningId::trusted_advice(
            JoltStageId::AdviceClaimReduction,
        ));
    }
    if include_untrusted_advice {
        ids.push(JoltOpeningId::untrusted_advice(
            JoltStageId::AdviceClaimReduction,
        ));
    }
    ids
}

pub fn advice_commitment_embedding_scale<F: Field>(
    opening_point: &[F],
    advice_opening_point: &[F],
) -> F {
    opening_point
        .iter()
        .map(|challenge| {
            if advice_opening_point.contains(challenge) {
                F::one()
            } else {
                F::one() - challenge
            }
        })
        .product()
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn layout() -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(2, 1, 2).unwrap_or_else(|error| {
            panic!("test layout should be valid: {error}");
        })
    }

    #[test]
    fn proof_commitment_order_matches_jolt_proof_payload_order() {
        assert_eq!(
            proof_commitment_order(layout()),
            vec![
                JoltCommittedPolynomial::RdInc,
                JoltCommittedPolynomial::RamInc,
                JoltCommittedPolynomial::InstructionRa(0),
                JoltCommittedPolynomial::InstructionRa(1),
                JoltCommittedPolynomial::RamRa(0),
                JoltCommittedPolynomial::RamRa(1),
                JoltCommittedPolynomial::BytecodeRa(0),
            ]
        );
    }

    #[test]
    fn final_opening_order_matches_stage8_rlc_order() {
        assert_eq!(
            final_opening_polynomial_order(layout(), true, true),
            vec![
                JoltCommittedPolynomial::RamInc,
                JoltCommittedPolynomial::RdInc,
                JoltCommittedPolynomial::InstructionRa(0),
                JoltCommittedPolynomial::InstructionRa(1),
                JoltCommittedPolynomial::BytecodeRa(0),
                JoltCommittedPolynomial::RamRa(0),
                JoltCommittedPolynomial::RamRa(1),
                JoltCommittedPolynomial::TrustedAdvice,
                JoltCommittedPolynomial::UntrustedAdvice,
            ]
        );
    }

    #[test]
    fn final_opening_ids_use_sumcheck_sources() {
        assert_eq!(
            final_opening_ids(layout(), true, false),
            vec![
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamInc,
                    JoltStageId::IncClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RdInc,
                    JoltStageId::IncClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::InstructionRa(0),
                    JoltStageId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::InstructionRa(1),
                    JoltStageId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::BytecodeRa(0),
                    JoltStageId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamRa(0),
                    JoltStageId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamRa(1),
                    JoltStageId::HammingWeightClaimReduction,
                ),
                JoltOpeningId::trusted_advice(JoltStageId::AdviceClaimReduction),
            ]
        );
    }

    #[test]
    fn advice_embedding_scale_selects_variables_outside_advice_point() {
        let opening_point = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let advice_point = [Fr::from_u64(3)];

        assert_eq!(
            advice_commitment_embedding_scale(&opening_point, &advice_point),
            (Fr::from_u64(1) - Fr::from_u64(2)) * (Fr::from_u64(1) - Fr::from_u64(5))
        );
    }
}
