use std::{cmp::min, ops::Range};

use jolt_field::{Field, RingCore};
use jolt_poly::EqPolynomial;

use crate::{opening, public};

use super::super::super::{
    AdviceClaimReductionPublic, JoltAdviceKind, JoltOpeningId, JoltPublicId, JoltRelationClaims,
    JoltRelationId,
};
use super::super::dimensions::{CommitmentMatrixShape, JoltSumcheckSpec, TracePolynomialOrder};
use super::super::error::JoltFormulaPointError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaimReductionLayout {
    trace_order: TracePolynomialOrder,
    log_t: usize,
    log_k_chunk: usize,
    main_shape: CommitmentMatrixShape,
    advice_shape: CommitmentMatrixShape,
    cycle_phase_col_rounds: Range<usize>,
    cycle_phase_row_rounds: Range<usize>,
}

impl AdviceClaimReductionLayout {
    pub fn balanced(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        max_advice_size_bytes: usize,
    ) -> Self {
        Self::new(
            trace_order,
            log_t,
            log_k_chunk,
            CommitmentMatrixShape::balanced(log_k_chunk + log_t),
            CommitmentMatrixShape::advice_from_max_bytes(max_advice_size_bytes),
        )
    }

    pub fn new(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        main_shape: CommitmentMatrixShape,
        advice_shape: CommitmentMatrixShape,
    ) -> Self {
        let (cycle_phase_col_rounds, cycle_phase_row_rounds) =
            cycle_phase_round_schedule(trace_order, log_t, log_k_chunk, main_shape, advice_shape);
        Self {
            trace_order,
            log_t,
            log_k_chunk,
            main_shape,
            advice_shape,
            cycle_phase_col_rounds,
            cycle_phase_row_rounds,
        }
    }

    pub const fn trace_order(&self) -> TracePolynomialOrder {
        self.trace_order
    }

    pub const fn log_t(&self) -> usize {
        self.log_t
    }

    pub const fn log_k_chunk(&self) -> usize {
        self.log_k_chunk
    }

    pub const fn main_shape(&self) -> CommitmentMatrixShape {
        self.main_shape
    }

    pub const fn advice_shape(&self) -> CommitmentMatrixShape {
        self.advice_shape
    }

    pub fn cycle_phase_col_rounds(&self) -> Range<usize> {
        self.cycle_phase_col_rounds.clone()
    }

    pub fn cycle_phase_row_rounds(&self) -> Range<usize> {
        self.cycle_phase_row_rounds.clone()
    }

    pub fn active_cycle_phase_rounds(&self) -> usize {
        self.cycle_phase_col_rounds.len() + self.cycle_phase_row_rounds.len()
    }

    pub fn cycle_phase_rounds(&self) -> usize {
        if !self.cycle_phase_row_rounds.is_empty() {
            self.cycle_phase_row_rounds.end - self.cycle_phase_col_rounds.start
        } else {
            self.cycle_phase_col_rounds.len()
        }
    }

    pub fn address_phase_rounds(&self) -> usize {
        self.advice_shape
            .total_vars()
            .saturating_sub(self.active_cycle_phase_rounds())
    }

    pub fn dimensions(&self) -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(self.cycle_phase_rounds(), self.address_phase_rounds())
    }

    pub fn cycle_phase_opening_point<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let mut advice_var_challenges = self.cycle_phase_variable_challenges(challenges)?;
        advice_var_challenges.reverse();
        Ok(advice_var_challenges)
    }

    pub fn cycle_phase_variable_challenges<F: Field>(
        &self,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let expected = self.cycle_phase_rounds();
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let mut advice_var_challenges = Vec::with_capacity(self.active_cycle_phase_rounds());
        advice_var_challenges.extend_from_slice(&challenges[self.cycle_phase_col_rounds.clone()]);
        advice_var_challenges.extend_from_slice(&challenges[self.cycle_phase_row_rounds.clone()]);
        Ok(advice_var_challenges)
    }

    pub fn address_phase_opening_point<F: Field>(
        &self,
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let expected_cycle = self.active_cycle_phase_rounds();
        if cycle_var_challenges.len() != expected_cycle {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: expected_cycle,
                got: cycle_var_challenges.len(),
            });
        }
        let expected_address = self.address_phase_rounds();
        if challenges.len() != expected_address {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: expected_address,
                got: challenges.len(),
            });
        }

        let mut point = match self.trace_order {
            TracePolynomialOrder::CycleMajor => [cycle_var_challenges, challenges].concat(),
            TracePolynomialOrder::AddressMajor => [challenges, cycle_var_challenges].concat(),
        };
        point.reverse();
        Ok(point)
    }

    pub fn cycle_phase_final_output_scale<F: Field>(
        &self,
        reference_opening_point: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self.cycle_phase_opening_point(challenges)?;
        self.final_output_scale_at(reference_opening_point, &opening_point)
    }

    pub fn address_phase_final_output_scale<F: Field>(
        &self,
        reference_opening_point: &[F],
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let opening_point = self.address_phase_opening_point(cycle_var_challenges, challenges)?;
        self.final_output_scale_at(reference_opening_point, &opening_point)
    }

    fn final_output_scale_at<F: Field>(
        &self,
        reference_opening_point: &[F],
        opening_point: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        if reference_opening_point.len() != opening_point.len() {
            return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: reference_opening_point.len(),
                got: opening_point.len(),
            });
        }

        Ok(
            EqPolynomial::<F>::mle(opening_point, reference_opening_point)
                * self.dummy_cycle_phase_scale::<F>(),
        )
    }

    fn dummy_cycle_phase_scale<F: Field>(&self) -> F {
        let two_inv = F::from_u64(2).inv_or_zero();
        (0..self.dummy_cycle_phase_rounds()).fold(F::one(), |scale, _| scale * two_inv)
    }

    pub fn dummy_cycle_phase_rounds(&self) -> usize {
        self.cycle_phase_rounds()
            .saturating_sub(self.active_cycle_phase_rounds())
    }
}

fn cycle_phase_round_schedule(
    trace_order: TracePolynomialOrder,
    log_t: usize,
    log_k_chunk: usize,
    main_shape: CommitmentMatrixShape,
    advice_shape: CommitmentMatrixShape,
) -> (Range<usize>, Range<usize>) {
    match trace_order {
        TracePolynomialOrder::CycleMajor => {
            let col_binding_rounds = 0..min(log_t, advice_shape.column_vars());
            let row_binding_rounds = min(log_t, main_shape.column_vars())
                ..min(log_t, main_shape.column_vars() + advice_shape.row_vars());
            (col_binding_rounds, row_binding_rounds)
        }
        TracePolynomialOrder::AddressMajor => {
            let col_binding_rounds = 0..advice_shape.column_vars().saturating_sub(log_k_chunk);
            let row_binding_rounds = main_shape.column_vars().saturating_sub(log_k_chunk)
                ..min(
                    log_t,
                    main_shape.column_vars().saturating_sub(log_k_chunk) + advice_shape.row_vars(),
                );
            (col_binding_rounds, row_binding_rounds)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaimReductionDimensions {
    cycle_phase_rounds: usize,
    address_phase_rounds: usize,
}

impl AdviceClaimReductionDimensions {
    pub const fn new(cycle_phase_rounds: usize, address_phase_rounds: usize) -> Self {
        Self {
            cycle_phase_rounds,
            address_phase_rounds,
        }
    }

    pub const fn cycle_phase_rounds(self) -> usize {
        self.cycle_phase_rounds
    }

    pub const fn address_phase_rounds(self) -> usize {
        self.address_phase_rounds
    }

    pub const fn has_address_phase(self) -> bool {
        self.address_phase_rounds > 0
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

pub fn cycle_phase_input_openings(kind: JoltAdviceKind) -> [JoltOpeningId; 1] {
    [ram_val_check_advice_opening(kind)]
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

pub fn address_phase_input_openings(kind: JoltAdviceKind) -> [JoltOpeningId; 1] {
    [cycle_phase_advice_opening(kind)]
}

pub fn address_phase_output_openings(kind: JoltAdviceKind) -> [JoltOpeningId; 1] {
    [final_advice_opening(kind)]
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
        AdviceClaimReductionDimensions::new(4, 3)
    }

    fn without_address_phase() -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(4, 0)
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let claims = cycle_phase::<Fr>(JoltAdviceKind::Trusted, with_address_phase());

        assert_eq!(claims.id, JoltRelationId::AdviceClaimReductionCyclePhase);
        assert_eq!(claims.sumcheck, with_address_phase().cycle_sumcheck());
        assert_eq!(
            claims.input.required_openings,
            cycle_phase_input_openings(JoltAdviceKind::Trusted).to_vec()
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
        let mut expected_openings = cycle_phase_input_openings(JoltAdviceKind::Untrusted).to_vec();
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
            address_phase_input_openings(JoltAdviceKind::Trusted).to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            address_phase_output_openings(JoltAdviceKind::Trusted).to_vec()
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
