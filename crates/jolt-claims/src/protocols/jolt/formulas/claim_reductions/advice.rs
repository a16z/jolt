//! Two-phase advice claim reduction (Stage 6 cycle -> Stage 7 address).

use jolt_field::{Field, RingCore};
use jolt_poly::EqPolynomial;

use crate::{opening, public};

use super::super::super::{
    AdviceClaimReductionPublic, JoltAdviceKind, JoltOpeningId, JoltPublicId, JoltRelationClaims,
    JoltRelationId,
};
use super::super::dimensions::{CommitmentMatrixShape, JoltSumcheckSpec, TracePolynomialOrder};
use super::super::error::JoltFormulaPointError;
use super::precommitted::{
    precommitted_skip_round_scale, PrecommittedClaimReduction, PrecommittedSchedulingReference,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaimReductionLayout {
    advice_shape: CommitmentMatrixShape,
    precommitted: PrecommittedClaimReduction,
}

/// Total-var counts of the present advice polynomials, in the order core feeds
/// them to the shared precommitted scheduling reference (trusted first).
pub fn precommitted_candidates(
    trusted_max_advice_size_bytes: Option<usize>,
    untrusted_max_advice_size_bytes: Option<usize>,
) -> Vec<usize> {
    trusted_max_advice_size_bytes
        .into_iter()
        .chain(untrusted_max_advice_size_bytes)
        .map(|max_bytes| CommitmentMatrixShape::advice_from_max_bytes(max_bytes).total_vars())
        .collect()
}

impl AdviceClaimReductionLayout {
    pub fn balanced(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        max_advice_size_bytes: usize,
    ) -> Self {
        let advice_shape = CommitmentMatrixShape::advice_from_max_bytes(max_advice_size_bytes);
        let precommitted = PrecommittedClaimReduction::new(
            advice_shape.row_vars(),
            advice_shape.column_vars(),
            scheduling_reference,
            trace_order,
            log_t,
        );
        Self {
            advice_shape,
            precommitted,
        }
    }

    pub const fn advice_shape(&self) -> CommitmentMatrixShape {
        self.advice_shape
    }

    pub const fn precommitted(&self) -> &PrecommittedClaimReduction {
        &self.precommitted
    }

    pub fn dimensions(&self) -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(
            self.precommitted.cycle_phase_total_rounds(),
            self.precommitted.address_phase_total_rounds(),
            self.precommitted.num_address_phase_rounds() > 0,
        )
    }

    pub fn cycle_phase_opening_point<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.precommitted.cycle_phase_opening_point(challenges)
    }

    pub fn cycle_phase_variable_challenges<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.precommitted
            .cycle_phase_variable_challenges(challenges)
    }

    pub fn address_phase_opening_point<F: Field>(
        &self,
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        self.precommitted
            .address_phase_opening_point(cycle_var_challenges, challenges)
    }

    /// `FinalScale` value when the reduction completes in the cycle phase
    /// (i.e. no active address-phase rounds remain).
    pub fn cycle_phase_final_output_scale<F: Field>(
        &self,
        reference_opening_point: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self
            .precommitted
            .cycle_phase_permuted_opening_point(challenges)?;
        let eq_eval = final_advice_eq_eval(reference_opening_point, &opening_point)?;
        Ok(eq_eval * self.precommitted.cycle_phase_skip_scale::<F>())
    }

    /// `FinalScale` value when the reduction completes in the address phase.
    pub fn address_phase_final_output_scale<F: Field>(
        &self,
        reference_opening_point: &[F],
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self
            .precommitted
            .address_phase_opening_point(cycle_var_challenges, challenges)?;
        let eq_eval = final_advice_eq_eval(reference_opening_point, &opening_point)?;
        Ok(eq_eval * precommitted_skip_round_scale::<F>(&self.precommitted))
    }
}

fn final_advice_eq_eval<F: Field>(
    reference_opening_point: &[F],
    opening_point: &[F],
) -> Result<F, JoltFormulaPointError> {
    if reference_opening_point.len() != opening_point.len() {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected: reference_opening_point.len(),
            got: opening_point.len(),
        });
    }
    Ok(EqPolynomial::<F>::mle(
        opening_point,
        reference_opening_point,
    ))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaimReductionDimensions {
    cycle_phase_rounds: usize,
    address_phase_rounds: usize,
    has_address_phase: bool,
}

impl AdviceClaimReductionDimensions {
    pub const fn new(
        cycle_phase_rounds: usize,
        address_phase_rounds: usize,
        has_address_phase: bool,
    ) -> Self {
        Self {
            cycle_phase_rounds,
            address_phase_rounds,
            has_address_phase,
        }
    }

    pub const fn cycle_phase_rounds(self) -> usize {
        self.cycle_phase_rounds
    }

    pub const fn address_phase_rounds(self) -> usize {
        self.address_phase_rounds
    }

    /// The address phase only runs for this advice polynomial when it has
    /// active address-phase rounds; otherwise the reduction finalizes at the
    /// cycle-phase handoff.
    pub const fn has_address_phase(self) -> bool {
        self.has_address_phase
    }

    pub const fn cycle_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.cycle_phase_rounds, 2)
    }

    pub const fn address_sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.address_phase_rounds, 2)
    }
}

pub fn cycle_phase<F>(
    kind: JoltAdviceKind,
    dimensions: AdviceClaimReductionDimensions,
) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let input = opening(ram_val_check_advice_opening(kind));
    let output = if dimensions.has_address_phase() {
        opening(cycle_phase_advice_opening(kind))
    } else {
        public(JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
            kind,
        ))) * opening(final_advice_opening(kind))
    };

    JoltRelationClaims::new(
        JoltRelationId::AdviceClaimReductionCyclePhase,
        dimensions.cycle_sumcheck(),
        input,
        output,
    )
}

pub fn address_phase<F>(
    kind: JoltAdviceKind,
    dimensions: AdviceClaimReductionDimensions,
) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let input = opening(cycle_phase_advice_opening(kind));
    let output = public(JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
        kind,
    ))) * opening(final_advice_opening(kind));

    JoltRelationClaims::new(
        JoltRelationId::AdviceClaimReduction,
        dimensions.address_sumcheck(),
        input,
        output,
    )
}

pub fn cycle_phase_output_openings(
    kind: JoltAdviceKind,
    dimensions: AdviceClaimReductionDimensions,
) -> Vec<JoltOpeningId> {
    if dimensions.has_address_phase() {
        vec![cycle_phase_advice_opening(kind)]
    } else {
        vec![final_advice_opening(kind)]
    }
}

pub fn ram_val_check_advice_opening(kind: JoltAdviceKind) -> JoltOpeningId {
    advice_opening(kind, JoltRelationId::RamValCheck)
}

pub fn cycle_phase_advice_opening(kind: JoltAdviceKind) -> JoltOpeningId {
    advice_opening(kind, JoltRelationId::AdviceClaimReductionCyclePhase)
}

pub fn final_advice_opening(kind: JoltAdviceKind) -> JoltOpeningId {
    advice_opening(kind, JoltRelationId::AdviceClaimReduction)
}

fn advice_opening(kind: JoltAdviceKind, relation: JoltRelationId) -> JoltOpeningId {
    match kind {
        JoltAdviceKind::Trusted => JoltOpeningId::trusted_advice(relation),
        JoltAdviceKind::Untrusted => JoltOpeningId::untrusted_advice(relation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn with_address_phase() -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(4, 3, true)
    }

    fn without_address_phase() -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(4, 3, false)
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let claims = cycle_phase::<Fr>(JoltAdviceKind::Trusted, with_address_phase());

        assert_eq!(claims.id, JoltRelationId::AdviceClaimReductionCyclePhase);
        assert_eq!(claims.sumcheck, with_address_phase().cycle_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![ram_val_check_advice_opening(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            claims.output.required_openings,
            cycle_phase_output_openings(JoltAdviceKind::Trusted, with_address_phase())
        );
        assert!(claims.required_challenges().is_empty());
        assert!(claims.required_publics().is_empty());
    }

    #[test]
    fn cycle_phase_without_address_phase_exposes_final_scale() {
        let claims = cycle_phase::<Fr>(JoltAdviceKind::Untrusted, without_address_phase());
        let mut expected_openings = vec![ram_val_check_advice_opening(JoltAdviceKind::Untrusted)];
        expected_openings.extend(cycle_phase_output_openings(
            JoltAdviceKind::Untrusted,
            without_address_phase(),
        ));

        assert_eq!(claims.id, JoltRelationId::AdviceClaimReductionCyclePhase);
        assert_eq!(claims.required_openings(), expected_openings);
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

        assert_eq!(claims.id, JoltRelationId::AdviceClaimReduction);
        assert_eq!(claims.sumcheck, with_address_phase().address_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![cycle_phase_advice_opening(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![final_advice_opening(JoltAdviceKind::Trusted)]
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

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_val_check_advice_opening(JoltAdviceKind::Trusted) => input_advice,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == final_advice_opening(JoltAdviceKind::Trusted) => final_advice_claim,
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

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == cycle_phase_advice_opening(JoltAdviceKind::Untrusted) => cycle_claim,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == final_advice_opening(JoltAdviceKind::Untrusted) => final_advice_claim,
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
