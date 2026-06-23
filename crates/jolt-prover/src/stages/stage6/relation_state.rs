use std::ops::Range;

use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial};

use crate::ProverError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AdviceCyclePhaseRelationState<F: Field> {
    advice: Polynomial<F>,
    eq: Polynomial<F>,
    col_rounds: Range<usize>,
    row_rounds: Range<usize>,
    scale: F,
}

impl<F: Field> AdviceCyclePhaseRelationState<F> {
    pub(super) fn new(
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

    fn bind(&mut self, local_round: usize, challenge: F) {
        if self.is_active_round(local_round) {
            self.advice
                .bind_with_order(challenge, BindingOrder::LowToHigh);
            self.eq.bind_with_order(challenge, BindingOrder::LowToHigh);
        } else {
            self.scale *= F::from_u64(2).inv_or_zero();
        }
    }

    fn round_rows(&self, local_round: usize) -> usize {
        if self.is_active_round(local_round) {
            self.advice.len() / 2
        } else {
            self.advice.len()
        }
    }

    fn round_eval(&self, local_round: usize, index: usize, point: F) -> F {
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

pub(super) struct Stage6RelationState<F: Field> {
    advice: AdviceCyclePhaseRelationState<F>,
}

impl<F: Field> Stage6RelationState<F> {
    pub(super) fn advice(advice: AdviceCyclePhaseRelationState<F>) -> Self {
        Self { advice }
    }

    pub(super) fn round_sum(&self, local_round: usize, point: F) -> Result<F, ProverError> {
        let mut sum = F::zero();
        for index in 0..self.advice.round_rows(local_round) {
            sum += self.advice.round_eval(local_round, index, point);
        }
        Ok(sum)
    }

    pub(super) fn bind(&mut self, local_round: usize, challenge: F) {
        self.advice.bind(local_round, challenge);
    }
}
