use std::cmp::Ordering;

use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{
    BackendError, SumcheckAdviceTraceOrder, SumcheckStage7AdviceAddressState,
    SumcheckStage7AdviceAddressStateRequest,
};

#[derive(Clone, Copy)]
pub(in crate::cpu::sumcheck) struct Stage7AdviceKernelContext {
    backend: &'static str,
    task: &'static str,
}

impl Stage7AdviceKernelContext {
    pub(in crate::cpu::sumcheck) const fn new(backend: &'static str, task: &'static str) -> Self {
        Self { backend, task }
    }

    fn invalid<T>(self, reason: impl Into<String>) -> Result<T, BackendError> {
        Err(BackendError::InvalidRequest {
            backend: self.backend,
            task: self.task,
            reason: reason.into(),
        })
    }
}

pub(in crate::cpu::sumcheck) fn build_state<F, N>(
    context: Stage7AdviceKernelContext,
    request: &SumcheckStage7AdviceAddressStateRequest<F, N>,
    advice_words: Vec<u64>,
) -> Result<SumcheckStage7AdviceAddressState<F>, BackendError>
where
    F: Field,
    N: jolt_witness::WitnessNamespace,
{
    let total_vars = request.total_vars();
    if request.reference_opening_point.len() != total_vars {
        return context.invalid(format!(
            "Stage 7 advice reference opening point has {} coordinates, expected {total_vars}",
            request.reference_opening_point.len()
        ));
    }
    if request.cycle_phase_variables.len() > total_vars {
        return context.invalid(format!(
            "Stage 7 advice cycle phase has {} variables for {total_vars} advice variables",
            request.cycle_phase_variables.len()
        ));
    }
    let expected_len =
        1usize
            .checked_shl(total_vars as u32)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: context.backend,
                task: context.task,
                reason: format!("Stage 7 advice variable count {total_vars} overflows usize"),
            })?;
    if advice_words.len() != expected_len {
        return context.invalid(format!(
            "Stage 7 advice stream has {} rows, expected {expected_len}",
            advice_words.len()
        ));
    }

    let eq_evals = EqPolynomial::<F>::evals(&request.reference_opening_point, None);
    let mut permuted = advice_words
        .into_par_iter()
        .zip(eq_evals.into_par_iter())
        .enumerate()
        .collect::<Vec<_>>();

    let trace_order = request.trace_order;
    let log_t = request.log_t;
    let log_k_chunk = request.log_k_chunk;
    let main_column_vars = request.main_column_vars;
    let advice_column_vars = request.advice_column_vars;
    permuted.par_sort_by(|(left_index, _), (right_index, _)| {
        compare_address_cycle(
            trace_order,
            log_t,
            log_k_chunk,
            main_column_vars,
            advice_column_vars,
            *left_index,
            *right_index,
        )
    });

    let (advice, eq): (Vec<_>, Vec<_>) = permuted
        .into_par_iter()
        .map(|(_, (advice, eq))| (advice, eq))
        .unzip();
    let mut state = SumcheckStage7AdviceAddressState::new(
        request.label,
        request.address_phase_rounds(),
        advice,
        Polynomial::from(eq),
        dummy_cycle_phase_scale::<F>(request.dummy_cycle_phase_rounds),
    );

    for &challenge in &request.cycle_phase_variables {
        bind_state(context, &mut state, challenge)?;
    }
    Ok(state)
}

pub(in crate::cpu::sumcheck) fn evaluate_round<F>(
    context: Stage7AdviceKernelContext,
    state: &SumcheckStage7AdviceAddressState<F>,
    previous_claim: F,
    max_num_rounds: usize,
) -> Result<UnivariatePoly<F>, BackendError>
where
    F: Field,
{
    validate_state(context, state)?;
    let num_rounds = state.num_rounds();
    let trailing_rounds =
        max_num_rounds
            .checked_sub(num_rounds)
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: context.backend,
                task: context.task,
                reason: format!(
                "Stage 7 advice state {} has {num_rounds} rounds, exceeding max {max_num_rounds}",
                state.label
            ),
            })?;
    let scaling_factor = state.scale * F::one().mul_pow_2(trailing_rounds);
    let previous_unscaled = previous_claim
        * scaling_factor
            .inverse()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: context.backend,
                task: context.task,
                reason: format!(
                    "Stage 7 advice state {} has non-invertible scaling factor",
                    state.label
                ),
            })?;

    let half = advice_len(state) / 2;
    if half == 0 {
        return context.invalid(format!(
            "Stage 7 advice state {} has no unbound variables",
            state.label
        ));
    }
    let evals = (0..half)
        .into_par_iter()
        .fold(
            || [F::zero(); 2],
            |mut running, index| {
                let advice = advice_sumcheck_evals_array::<F, 2>(state, index);
                let eq = sumcheck_evals_array::<F, 2>(&state.eq, index, BindingOrder::LowToHigh);
                running[0] += advice[0] * eq[0];
                running[1] += advice[1] * eq[1];
                running
            },
        )
        .reduce(
            || [F::zero(); 2],
            |mut left, right| {
                left[0] += right[0];
                left[1] += right[1];
                left
            },
        );
    Ok(UnivariatePoly::from_evals_and_hint(previous_unscaled, &evals) * scaling_factor)
}

pub(in crate::cpu::sumcheck) fn bind_state<F>(
    context: Stage7AdviceKernelContext,
    state: &mut SumcheckStage7AdviceAddressState<F>,
    challenge: F,
) -> Result<(), BackendError>
where
    F: Field,
{
    validate_state(context, state)?;
    bind_advice(state, challenge);
    state.eq.bind_with_order(challenge, BindingOrder::LowToHigh);
    Ok(())
}

fn compare_address_cycle(
    trace_order: SumcheckAdviceTraceOrder,
    log_t: usize,
    log_k_chunk: usize,
    main_column_vars: usize,
    advice_column_vars: usize,
    left: usize,
    right: usize,
) -> Ordering {
    let left_key = advice_index_to_address_cycle(
        trace_order,
        log_t,
        log_k_chunk,
        main_column_vars,
        advice_column_vars,
        left,
    );
    let right_key = advice_index_to_address_cycle(
        trace_order,
        log_t,
        log_k_chunk,
        main_column_vars,
        advice_column_vars,
        right,
    );
    left_key.cmp(&right_key)
}

fn advice_index_to_address_cycle(
    trace_order: SumcheckAdviceTraceOrder,
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
    match trace_order {
        SumcheckAdviceTraceOrder::CycleMajor => {
            let global_index = row as u128 * main_cols as u128 + col as u128;
            let address = global_index / (1u128 << log_t);
            let cycle = global_index % (1u128 << log_t);
            (address as usize, cycle as usize)
        }
        SumcheckAdviceTraceOrder::AddressMajor => {
            let global_index = row as u128 * main_cols as u128 + col as u128;
            let address = global_index % (1u128 << log_k_chunk);
            let cycle = global_index / (1u128 << log_k_chunk);
            (address as usize, cycle as usize)
        }
    }
}

fn dummy_cycle_phase_scale<F: Field>(dummy_rounds: usize) -> F {
    let two_inv = F::from_u64(2).inv_or_zero();
    (0..dummy_rounds).fold(F::one(), |scale, _| scale * two_inv)
}

fn validate_state<F>(
    context: Stage7AdviceKernelContext,
    state: &SumcheckStage7AdviceAddressState<F>,
) -> Result<(), BackendError>
where
    F: Field,
{
    let len = advice_len(state);
    if len == 0 || !len.is_power_of_two() {
        return context.invalid(format!(
            "Stage 7 advice state {} length {len} is not a nonzero power of two",
            state.label
        ));
    }
    if state.eq.len() != len {
        return context.invalid(format!(
            "Stage 7 advice state {} EQ table has {} rows, expected {len}",
            state.label,
            state.eq.len()
        ));
    }
    Ok(())
}

fn advice_len<F: Field>(state: &SumcheckStage7AdviceAddressState<F>) -> usize {
    state
        .bound_advice
        .as_ref()
        .map_or(state.advice_words.len(), |poly| poly.len())
}

fn advice_sumcheck_evals_array<F: Field, const DEGREE: usize>(
    state: &SumcheckStage7AdviceAddressState<F>,
    index: usize,
) -> [F; DEGREE] {
    if let Some(advice) = &state.bound_advice {
        return sumcheck_evals_array(advice, index, BindingOrder::LowToHigh);
    }
    debug_assert!(DEGREE > 0);
    let lo = F::from_u64(state.advice_words[2 * index]);
    let hi = F::from_u64(state.advice_words[2 * index + 1]);
    let mut evals = [F::zero(); DEGREE];
    evals[0] = lo;
    if DEGREE == 1 {
        return evals;
    }
    let step = hi - lo;
    let mut value = hi;
    for eval in evals.iter_mut().skip(1) {
        value += step;
        *eval = value;
    }
    evals
}

fn bind_advice<F: Field>(state: &mut SumcheckStage7AdviceAddressState<F>, challenge: F) {
    if let Some(advice) = &mut state.bound_advice {
        advice.bind_with_order(challenge, BindingOrder::LowToHigh);
        return;
    }
    let half = state.advice_words.len() / 2;
    let bound = (0..half)
        .into_par_iter()
        .map(|index| {
            let lo = F::from_u64(state.advice_words[2 * index]);
            let hi = F::from_u64(state.advice_words[2 * index + 1]);
            lo + challenge * (hi - lo)
        })
        .collect::<Vec<_>>();
    state.advice_words = Vec::new();
    state.bound_advice = Some(Polynomial::from(bound));
}

fn sumcheck_evals_array<F: Field, const DEGREE: usize>(
    polynomial: &Polynomial<F>,
    index: usize,
    order: BindingOrder,
) -> [F; DEGREE] {
    debug_assert!(DEGREE > 0);
    let (lo, hi) = polynomial.sumcheck_eval_pair(index, order);
    let mut evals = [F::zero(); DEGREE];
    evals[0] = lo;
    if DEGREE == 1 {
        return evals;
    }
    let step = hi - lo;
    let mut value = hi;
    for eval in evals.iter_mut().skip(1) {
        value += step;
        *eval = value;
    }
    evals
}
