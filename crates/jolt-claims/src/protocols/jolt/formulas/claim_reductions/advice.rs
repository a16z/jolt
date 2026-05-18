use jolt_field::RingCore;

use crate::{opening, public};

use super::super::super::{
    AdviceClaimReductionPublic, JoltAdviceKind, JoltOpeningId, JoltPublicId, JoltStageClaims,
    JoltStageId,
};
use super::super::dimensions::AdviceClaimReductionDimensions;

pub fn cycle_phase<F>(
    kind: JoltAdviceKind,
    dimensions: AdviceClaimReductionDimensions,
) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let input = opening(ram_val_check_advice(kind));
    let output = if dimensions.has_address_phase() {
        opening(cycle_phase_advice(kind))
    } else {
        public(JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
            kind,
        ))) * opening(final_advice(kind))
    };

    JoltStageClaims::new(
        JoltStageId::AdviceClaimReductionCyclePhase,
        dimensions.cycle_sumcheck(),
        input,
        output,
    )
}

pub fn address_phase<F>(
    kind: JoltAdviceKind,
    dimensions: AdviceClaimReductionDimensions,
) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let input = opening(cycle_phase_advice(kind));
    let output = public(JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
        kind,
    ))) * opening(final_advice(kind));

    JoltStageClaims::new(
        JoltStageId::AdviceClaimReduction,
        dimensions.address_sumcheck(),
        input,
        output,
    )
}

fn ram_val_check_advice(kind: JoltAdviceKind) -> JoltOpeningId {
    advice_opening(kind, JoltStageId::RamValCheck)
}

fn cycle_phase_advice(kind: JoltAdviceKind) -> JoltOpeningId {
    advice_opening(kind, JoltStageId::AdviceClaimReductionCyclePhase)
}

fn final_advice(kind: JoltAdviceKind) -> JoltOpeningId {
    advice_opening(kind, JoltStageId::AdviceClaimReduction)
}

fn advice_opening(kind: JoltAdviceKind, stage: JoltStageId) -> JoltOpeningId {
    match kind {
        JoltAdviceKind::Trusted => JoltOpeningId::trusted_advice(stage),
        JoltAdviceKind::Untrusted => JoltOpeningId::untrusted_advice(stage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn with_address_phase() -> AdviceClaimReductionDimensions {
        (4, 3).into()
    }

    fn without_address_phase() -> AdviceClaimReductionDimensions {
        (4, 0).into()
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let claims = cycle_phase::<Fr>(JoltAdviceKind::Trusted, with_address_phase());

        assert_eq!(claims.id, JoltStageId::AdviceClaimReductionCyclePhase);
        assert_eq!(claims.sumcheck, with_address_phase().cycle_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![ram_val_check_advice(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![cycle_phase_advice(JoltAdviceKind::Trusted)]
        );
        assert!(claims.required_challenges().is_empty());
        assert!(claims.required_publics().is_empty());
    }

    #[test]
    fn cycle_phase_without_address_phase_exposes_final_scale() {
        let claims = cycle_phase::<Fr>(JoltAdviceKind::Untrusted, without_address_phase());

        assert_eq!(claims.id, JoltStageId::AdviceClaimReductionCyclePhase);
        assert_eq!(
            claims.required_openings(),
            vec![
                ram_val_check_advice(JoltAdviceKind::Untrusted),
                final_advice(JoltAdviceKind::Untrusted),
            ]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Untrusted
            ))]
        );
    }

    #[test]
    fn address_phase_exposes_expected_dependencies() {
        let claims = address_phase::<Fr>(JoltAdviceKind::Trusted, with_address_phase());

        assert_eq!(claims.id, JoltStageId::AdviceClaimReduction);
        assert_eq!(claims.sumcheck, with_address_phase().address_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![cycle_phase_advice(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![final_advice(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Trusted
            ))]
        );
    }

    #[test]
    fn cycle_phase_without_address_phase_evaluates_like_core_formula() {
        let claims = cycle_phase::<Fr>(JoltAdviceKind::Trusted, without_address_phase());

        let input_advice = Fr::from_u64(3);
        let final_advice_claim = Fr::from_u64(5);
        let final_scale = Fr::from_u64(7);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == ram_val_check_advice(JoltAdviceKind::Trusted) => input_advice,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == final_advice(JoltAdviceKind::Trusted) => final_advice_claim,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                    JoltAdviceKind::Trusted,
                )) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, input_advice);
        assert_eq!(output, final_scale * final_advice_claim);
    }

    #[test]
    fn address_phase_evaluates_like_core_formula() {
        let claims = address_phase::<Fr>(JoltAdviceKind::Untrusted, with_address_phase());

        let cycle_claim = Fr::from_u64(11);
        let final_advice_claim = Fr::from_u64(13);
        let final_scale = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == cycle_phase_advice(JoltAdviceKind::Untrusted) => cycle_claim,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == final_advice(JoltAdviceKind::Untrusted) => final_advice_claim,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                    JoltAdviceKind::Untrusted,
                )) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, cycle_claim);
        assert_eq!(output, final_scale * final_advice_claim);
    }
}
