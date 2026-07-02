//! Two-phase advice claim reduction (Stage 6 cycle -> Stage 7 address).

use std::{cmp::min, ops::Range};

use jolt_field::Field;
use jolt_poly::EqPolynomial;

use super::super::super::{JoltAdviceKind, JoltOpeningId, JoltRelationId};
use super::super::dimensions::{CommitmentMatrixShape, TracePolynomialOrder};
use super::super::error::JoltFormulaPointError;
use super::precommitted::{
    precommitted_skip_round_scale, PrecommittedClaimReduction, PrecommittedReductionDimensions,
    PrecommittedReductionLayout, PrecommittedSchedulingReference,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceClaimReductionLayout {
    trace_order: TracePolynomialOrder,
    log_t: usize,
    log_k_chunk: usize,
    advice_shape: CommitmentMatrixShape,
    precommitted: PrecommittedClaimReduction,
    cycle_phase_col_rounds: Range<usize>,
    cycle_phase_row_rounds: Range<usize>,
}

/// Total-var counts of the present advice polynomials, in the order core feeds
/// them to the shared precommitted scheduling reference (trusted first).
pub fn candidate_total_vars(
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
    ) -> Result<Self, JoltFormulaPointError> {
        let advice_shape = CommitmentMatrixShape::advice_from_max_bytes(max_advice_size_bytes);
        let log_k_chunk = scheduling_reference.address_rounds;
        let precommitted = PrecommittedClaimReduction::new(
            advice_shape.row_vars(),
            advice_shape.column_vars(),
            scheduling_reference,
            trace_order,
            log_t,
        )?;
        let main_shape = CommitmentMatrixShape::balanced(log_k_chunk + log_t);
        let (cycle_phase_col_rounds, cycle_phase_row_rounds) =
            cycle_phase_round_schedule(trace_order, log_t, log_k_chunk, main_shape, advice_shape);
        Ok(Self {
            trace_order,
            log_t,
            log_k_chunk,
            advice_shape,
            precommitted,
            cycle_phase_col_rounds,
            cycle_phase_row_rounds,
        })
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

    pub fn main_shape(&self) -> CommitmentMatrixShape {
        CommitmentMatrixShape::balanced(self.log_k_chunk + self.log_t)
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

    /// `FinalScale` value from the reduction's already-derived cycle-phase
    /// opening point, rather than re-deriving it from the sumcheck challenges.
    /// Lets the cycle-phase relation object's `resolve_public` recover the scale
    /// from the opening point it produced in `derive_opening_points`.
    pub fn cycle_phase_scale_at_opening_point<F: Field>(
        &self,
        reference_opening_point: &[F],
        opening_point: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let permuted = self
            .precommitted
            .cycle_phase_permuted_from_opening_point(opening_point)?;
        let eq_eval = final_advice_eq_eval(reference_opening_point, &permuted)?;
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
        self.address_phase_scale_at_opening_point(reference_opening_point, &opening_point)
    }

    /// `FinalScale` value from the reduction's already-derived address-phase
    /// opening point, rather than re-deriving it from the cycle/sumcheck
    /// challenges. Lets the stage 7 relation object's `resolve_public` recover the
    /// scale from the opening point it produced in `derive_opening_points`.
    pub fn address_phase_scale_at_opening_point<F: Field>(
        &self,
        reference_opening_point: &[F],
        opening_point: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let eq_eval = final_advice_eq_eval(reference_opening_point, opening_point)?;
        Ok(eq_eval * precommitted_skip_round_scale::<F>(&self.precommitted))
    }
}

impl PrecommittedReductionLayout for AdviceClaimReductionLayout {
    fn precommitted(&self) -> &PrecommittedClaimReduction {
        &self.precommitted
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

pub fn cycle_phase_output_openings(
    kind: JoltAdviceKind,
    dimensions: PrecommittedReductionDimensions,
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

/// Compute the column-side and row-side active round ranges for the advice
/// cycle-phase sumcheck, given the main/advice matrix shapes and trace order.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn with_address_phase() -> PrecommittedReductionDimensions {
        PrecommittedReductionDimensions::new(4, 3, true)
    }

    fn without_address_phase() -> PrecommittedReductionDimensions {
        PrecommittedReductionDimensions::new(4, 3, false)
    }

    #[test]
    fn cycle_phase_output_openings_track_address_phase_presence() {
        assert_eq!(
            cycle_phase_output_openings(JoltAdviceKind::Trusted, with_address_phase()),
            vec![cycle_phase_advice_opening(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            cycle_phase_output_openings(JoltAdviceKind::Untrusted, without_address_phase()),
            vec![final_advice_opening(JoltAdviceKind::Untrusted)]
        );
    }
}
