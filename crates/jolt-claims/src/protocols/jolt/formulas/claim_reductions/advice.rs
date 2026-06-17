use std::{cmp::min, ops::Range};

use jolt_field::{Field, RingCore};
use jolt_poly::{eq_index_msb, BindingOrder, EqPolynomial, Polynomial};

use crate::{opening, public};

use super::super::super::{
    AdviceClaimReductionPublic, JoltAdviceKind, JoltOpeningId, JoltPublicId, JoltRelationClaims,
    JoltRelationId,
};
use super::super::dimensions::{CommitmentMatrixShape, JoltSumcheckSpec, TracePolynomialOrder};
use super::super::error::{JoltFormulaDimensionsError, JoltFormulaPointError};

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

    pub fn cycle_phase_binding_challenges<F: Field>(
        &self,
        opening_point: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let expected = self.active_cycle_phase_rounds();
        if opening_point.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: opening_point.len(),
            });
        }

        let mut binding_challenges = opening_point.to_vec();
        binding_challenges.reverse();
        Ok(binding_challenges)
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
                * self.cycle_phase_dummy_scale::<F>(),
        )
    }

    pub fn cycle_phase_dummy_scale<F: Field>(&self) -> F {
        advice_cycle_phase_dummy_scale(self.dummy_cycle_phase_rounds())
    }

    pub fn dummy_cycle_phase_rounds(&self) -> usize {
        self.cycle_phase_rounds()
            .saturating_sub(self.active_cycle_phase_rounds())
    }

    pub fn cycle_phase_batch_offset(
        &self,
        max_num_vars: usize,
    ) -> Result<usize, JoltFormulaDimensionsError> {
        let trace_rounds = self.log_k_chunk.checked_add(self.log_t).ok_or(
            JoltFormulaDimensionsError::Overflow {
                name: "advice cycle-phase trace rounds",
            },
        )?;
        let trace_offset = max_num_vars.checked_sub(trace_rounds).ok_or(
            JoltFormulaDimensionsError::Underflow {
                name: "advice cycle-phase batch offset",
            },
        )?;
        trace_offset
            .checked_add(self.log_k_chunk)
            .ok_or(JoltFormulaDimensionsError::Overflow {
                name: "advice cycle-phase batch offset",
            })
    }

    pub fn advice_index_to_address_cycle(&self, index: usize) -> (usize, usize) {
        advice_index_to_address_cycle(
            self.trace_order,
            self.log_t,
            self.log_k_chunk,
            self.main_shape.column_vars(),
            self.advice_shape.column_vars(),
            index,
        )
    }

    pub fn cycle_phase_coefficients<F, T>(
        &self,
        reference_opening_point: &[F],
        values: Vec<T>,
    ) -> Result<(Vec<T>, Vec<F>), JoltFormulaPointError>
    where
        F: Field,
    {
        let advice_vars = self.advice_shape.total_vars();
        if reference_opening_point.len() != advice_vars {
            return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: advice_vars,
                got: reference_opening_point.len(),
            });
        }
        let expected_domain = 1usize << advice_vars;
        if values.len() != expected_domain {
            return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
                expected: expected_domain,
                got: values.len(),
            });
        }

        let mut permuted = values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                (
                    self.advice_index_to_address_cycle(index),
                    value,
                    eq_index_msb(reference_opening_point, index),
                )
            })
            .collect::<Vec<_>>();
        permuted.sort_by_key(|(key, _, _)| *key);
        Ok(permuted
            .into_iter()
            .map(|(_, value, eq_value)| (value, eq_value))
            .unzip())
    }

    pub fn cycle_phase_polynomials<F>(
        &self,
        reference_opening_point: &[F],
        values: Vec<F>,
    ) -> Result<(Polynomial<F>, Polynomial<F>), JoltFormulaPointError>
    where
        F: Field,
    {
        let (advice_coeffs, eq_coeffs) =
            self.cycle_phase_coefficients(reference_opening_point, values)?;
        Ok((Polynomial::new(advice_coeffs), Polynomial::new(eq_coeffs)))
    }

    pub fn cycle_phase_opening_claim<F>(
        &self,
        reference_opening_point: &[F],
        values: Vec<F>,
        opening_point: &[F],
    ) -> Result<F, JoltFormulaPointError>
    where
        F: Field,
    {
        let (mut advice_poly, mut eq_poly) =
            self.cycle_phase_polynomials(reference_opening_point, values)?;
        for challenge in self.cycle_phase_binding_challenges(opening_point)? {
            advice_poly.bind_with_order(challenge, BindingOrder::LowToHigh);
            eq_poly.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        Ok(advice_poly
            .evals()
            .iter()
            .zip(eq_poly.evals())
            .map(|(&advice, &eq)| advice * eq)
            .sum::<F>()
            * self.cycle_phase_dummy_scale::<F>())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdviceCyclePhaseRelationState<F: Field> {
    advice: Polynomial<F>,
    eq: Polynomial<F>,
    col_rounds: Range<usize>,
    row_rounds: Range<usize>,
    scale: F,
}

impl<F: Field> AdviceCyclePhaseRelationState<F> {
    pub fn new(
        advice: Polynomial<F>,
        eq: Polynomial<F>,
        col_rounds: Range<usize>,
        row_rounds: Range<usize>,
    ) -> Self {
        Self {
            advice,
            eq,
            col_rounds,
            row_rounds,
            scale: F::one(),
        }
    }

    pub fn bind(&mut self, local_round: usize, challenge: F) {
        if self.is_active_round(local_round) {
            self.advice
                .bind_with_order(challenge, BindingOrder::LowToHigh);
            self.eq.bind_with_order(challenge, BindingOrder::LowToHigh);
        } else {
            self.scale *= F::from_u64(2).inv_or_zero();
        }
    }

    pub fn round_rows(&self, local_round: usize) -> usize {
        if self.is_active_round(local_round) {
            self.advice.len() / 2
        } else {
            self.advice.len()
        }
    }

    pub fn round_eval(&self, local_round: usize, index: usize, point: F) -> F {
        if self.is_active_round(local_round) {
            self.scale
                * self
                    .advice
                    .sumcheck_round_eval_with_order(index, point, BindingOrder::LowToHigh)
                * self
                    .eq
                    .sumcheck_round_eval_with_order(index, point, BindingOrder::LowToHigh)
        } else {
            self.scale
                * self.advice.evals()[index]
                * self.eq.evals()[index]
                * F::from_u64(2).inv_or_zero()
        }
    }

    fn is_active_round(&self, local_round: usize) -> bool {
        self.col_rounds.contains(&local_round) || self.row_rounds.contains(&local_round)
    }
}

pub fn advice_index_to_address_cycle(
    trace_order: TracePolynomialOrder,
    log_t: usize,
    log_k_chunk: usize,
    main_column_vars: usize,
    advice_column_vars: usize,
    index: usize,
) -> (usize, usize) {
    let advice_cols = 1usize << advice_column_vars;
    let row = index / advice_cols;
    let col = index % advice_cols;
    let main_cols = 1usize << main_column_vars;
    let global_index = row * main_cols + col;
    trace_order.index_to_address_cycle(global_index, 1usize << log_k_chunk, 1usize << log_t)
}

pub fn advice_cycle_phase_dummy_scale<F: Field>(dummy_rounds: usize) -> F {
    let two_inv = F::from_u64(2).inv_or_zero();
    (0..dummy_rounds).fold(F::one(), |scale, _| scale * two_inv)
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
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};

    fn with_address_phase() -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(4, 3)
    }

    fn without_address_phase() -> AdviceClaimReductionDimensions {
        AdviceClaimReductionDimensions::new(4, 0)
    }

    #[test]
    fn advice_index_to_address_cycle_embeds_advice_matrix_in_main_trace_order() {
        let main_shape = CommitmentMatrixShape::new(3, 2);
        let advice_shape = CommitmentMatrixShape::new(1, 2);
        let cycle_major = AdviceClaimReductionLayout::new(
            TracePolynomialOrder::CycleMajor,
            2,
            3,
            main_shape,
            advice_shape,
        );
        let address_major = AdviceClaimReductionLayout::new(
            TracePolynomialOrder::AddressMajor,
            2,
            3,
            main_shape,
            advice_shape,
        );

        assert_eq!(cycle_major.advice_index_to_address_cycle(5), (4, 1));
        assert_eq!(address_major.advice_index_to_address_cycle(5), (1, 2));
        assert_eq!(
            advice_index_to_address_cycle(TracePolynomialOrder::CycleMajor, 2, 3, 3, 1, 5),
            (4, 1)
        );
        assert_eq!(
            advice_index_to_address_cycle(TracePolynomialOrder::AddressMajor, 2, 3, 3, 1, 5),
            (1, 2)
        );
    }

    #[test]
    fn cycle_phase_binding_helpers_follow_layout_schedule() {
        let layout =
            AdviceClaimReductionLayout::balanced(TracePolynomialOrder::CycleMajor, 8, 4, 64);
        let opening_point = vec![Fr::from_u64(7), Fr::from_u64(2), Fr::from_u64(1)];
        let two_inv = Fr::from_u64(2).inv_or_zero();

        assert_eq!(
            layout
                .cycle_phase_binding_challenges(&opening_point)
                .map_err(|error| error.to_string()),
            Ok(vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(7)])
        );
        assert_eq!(
            layout.cycle_phase_dummy_scale::<Fr>(),
            two_inv * two_inv * two_inv * two_inv
        );
        assert_eq!(
            advice_cycle_phase_dummy_scale::<Fr>(layout.dummy_cycle_phase_rounds()),
            layout.cycle_phase_dummy_scale::<Fr>()
        );
    }

    #[test]
    fn cycle_phase_batch_offset_aligns_cycle_rounds_inside_padded_batch() {
        let layout = AdviceClaimReductionLayout::new(
            TracePolynomialOrder::AddressMajor,
            2,
            3,
            CommitmentMatrixShape::new(3, 2),
            CommitmentMatrixShape::new(1, 2),
        );

        assert_eq!(layout.cycle_phase_batch_offset(8), Ok(6));
        assert_eq!(
            layout.cycle_phase_batch_offset(4),
            Err(JoltFormulaDimensionsError::Underflow {
                name: "advice cycle-phase batch offset",
            })
        );
    }

    #[test]
    fn cycle_phase_coefficients_sort_by_address_cycle_order() {
        let layout = AdviceClaimReductionLayout::new(
            TracePolynomialOrder::AddressMajor,
            2,
            3,
            CommitmentMatrixShape::new(3, 2),
            CommitmentMatrixShape::new(1, 2),
        );
        let values = (0..8).collect::<Vec<_>>();
        let zero = Fr::from_u64(0);
        let point = vec![zero, zero, zero];

        let coefficients = layout
            .cycle_phase_coefficients(&point, values)
            .map_err(|error| error.to_string());

        assert_eq!(
            coefficients,
            Ok((
                vec![0, 2, 4, 6, 1, 3, 5, 7],
                vec![Fr::from_u64(1), zero, zero, zero, zero, zero, zero, zero,],
            ))
        );
    }

    #[test]
    fn cycle_phase_polynomials_wrap_sorted_coefficients() {
        let layout = AdviceClaimReductionLayout::new(
            TracePolynomialOrder::AddressMajor,
            2,
            3,
            CommitmentMatrixShape::new(3, 2),
            CommitmentMatrixShape::new(1, 2),
        );
        let values = (0..8).map(Fr::from_u64).collect::<Vec<_>>();
        let zero = Fr::from_u64(0);
        let point = vec![zero, zero, zero];

        let polynomials = layout
            .cycle_phase_polynomials(&point, values)
            .map(|(advice, eq)| (advice.evals().to_vec(), eq.evals().to_vec()))
            .map_err(|error| error.to_string());

        assert_eq!(
            polynomials,
            Ok((
                vec![
                    Fr::from_u64(0),
                    Fr::from_u64(2),
                    Fr::from_u64(4),
                    Fr::from_u64(6),
                    Fr::from_u64(1),
                    Fr::from_u64(3),
                    Fr::from_u64(5),
                    Fr::from_u64(7),
                ],
                vec![Fr::from_u64(1), zero, zero, zero, zero, zero, zero, zero],
            )),
        );
    }

    #[test]
    fn cycle_phase_opening_claim_binds_sorted_coefficients() {
        let layout = AdviceClaimReductionLayout::new(
            TracePolynomialOrder::AddressMajor,
            2,
            3,
            CommitmentMatrixShape::new(3, 2),
            CommitmentMatrixShape::new(1, 2),
        );
        let values = (10..18).map(Fr::from_u64).collect::<Vec<_>>();
        let point = vec![Fr::from_u64(0), Fr::from_u64(0), Fr::from_u64(0)];
        let opening_point = vec![Fr::from_u64(0), Fr::from_u64(0)];

        assert_eq!(
            layout
                .cycle_phase_opening_claim(&point, values, &opening_point)
                .map_err(|error| error.to_string()),
            Ok(Fr::from_u64(10))
        );
    }

    #[test]
    fn cycle_phase_relation_state_evaluates_active_and_dummy_rounds() {
        let advice = Polynomial::new(vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ]);
        let eq = Polynomial::new(vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ]);
        let point = Fr::from_u64(23);
        let state = AdviceCyclePhaseRelationState::new(advice.clone(), eq.clone(), 0..1, 2..3);
        let two_inv = Fr::from_u64(2).inv_or_zero();

        assert_eq!(state.round_rows(0), 2);
        assert_eq!(
            state.round_eval(0, 0, point),
            advice.sumcheck_round_eval_with_order(0, point, BindingOrder::LowToHigh)
                * eq.sumcheck_round_eval_with_order(0, point, BindingOrder::LowToHigh)
        );
        assert_eq!(state.round_rows(1), 4);
        assert_eq!(
            state.round_eval(1, 2, point),
            advice.evals()[2] * eq.evals()[2] * two_inv
        );
    }

    #[test]
    fn cycle_phase_relation_state_binds_active_rounds_and_scales_dummy_rounds() {
        let challenge = Fr::from_u64(29);
        let dummy_challenge = Fr::from_u64(31);
        let mut advice =
            Polynomial::new((0..4).map(|value| Fr::from_u64(value as u64 + 2)).collect());
        let mut eq = Polynomial::new((0..4).map(|value| Fr::from_u64(value as u64 + 7)).collect());
        let two_inv = Fr::from_u64(2).inv_or_zero();
        let mut state = AdviceCyclePhaseRelationState::new(advice.clone(), eq.clone(), 0..1, 2..3);

        state.bind(0, challenge);
        advice.bind_with_order(challenge, BindingOrder::LowToHigh);
        eq.bind_with_order(challenge, BindingOrder::LowToHigh);
        assert_eq!(state.advice, advice);
        assert_eq!(state.eq, eq);
        assert_eq!(state.scale, Fr::from_u64(1));

        state.bind(1, dummy_challenge);
        assert_eq!(state.advice, advice);
        assert_eq!(state.eq, eq);
        assert_eq!(state.scale, two_inv);
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
