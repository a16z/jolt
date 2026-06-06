use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{BackendError, SumcheckStage7HammingState};

#[derive(Clone, Copy)]
pub(in crate::cpu::sumcheck) struct Stage7HammingKernelContext {
    backend: &'static str,
    task: &'static str,
}

impl Stage7HammingKernelContext {
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

pub(in crate::cpu::sumcheck) fn build_state<F>(
    context: Stage7HammingKernelContext,
    label: &'static str,
    g_tables: Vec<Vec<F>>,
    eq_bool_table: Vec<F>,
    eq_virt_tables: Vec<Vec<F>>,
    gamma_powers: Vec<F>,
) -> Result<SumcheckStage7HammingState<F>, BackendError>
where
    F: Field,
{
    if g_tables.is_empty() {
        return context.invalid("Stage 7 hamming state has no RA pushforward tables");
    }
    if g_tables.len() != eq_virt_tables.len() {
        return context.invalid(format!(
            "Stage 7 hamming state has {} G tables but {} virtualization eq tables",
            g_tables.len(),
            eq_virt_tables.len()
        ));
    }
    if gamma_powers.len() != 3 * g_tables.len() {
        return context.invalid(format!(
            "Stage 7 hamming state has {} gamma powers for {} RA polynomials",
            gamma_powers.len(),
            g_tables.len()
        ));
    }
    let Some(first_len) = g_tables.first().map(Vec::len) else {
        return context.invalid("Stage 7 hamming state has no G tables");
    };
    if first_len == 0 || !first_len.is_power_of_two() {
        return context.invalid(format!(
            "Stage 7 hamming state length {first_len} is not a nonzero power of two"
        ));
    }
    if eq_bool_table.len() != first_len {
        return context.invalid(format!(
            "Stage 7 hamming booleanity eq table has {} rows, expected {first_len}",
            eq_bool_table.len()
        ));
    }
    for (index, table) in g_tables.iter().enumerate() {
        if table.len() != first_len {
            return context.invalid(format!(
                "Stage 7 hamming G table {index} has {} rows, expected {first_len}",
                table.len()
            ));
        }
    }
    for (index, table) in eq_virt_tables.iter().enumerate() {
        if table.len() != first_len {
            return context.invalid(format!(
                "Stage 7 hamming virtualization eq table {index} has {} rows, expected {first_len}",
                table.len()
            ));
        }
    }

    Ok(SumcheckStage7HammingState::new(
        label,
        g_tables.into_iter().map(Polynomial::from).collect(),
        Polynomial::from(eq_bool_table),
        eq_virt_tables.into_iter().map(Polynomial::from).collect(),
        gamma_powers,
    ))
}

pub(in crate::cpu::sumcheck) fn evaluate_round<F>(
    context: Stage7HammingKernelContext,
    state: &SumcheckStage7HammingState<F>,
    previous_claim: F,
) -> Result<UnivariatePoly<F>, BackendError>
where
    F: Field,
{
    validate_state(context, state)?;
    let half = state.g[0].len() / 2;
    if half == 0 {
        return context.invalid(format!(
            "Stage 7 hamming state {} has no unbound variables",
            state.label
        ));
    }

    let evals = (0..half)
        .into_par_iter()
        .fold(
            || [F::zero(); 2],
            |mut running, index| {
                let eq_bool =
                    sumcheck_evals_array::<F, 2>(&state.eq_bool, index, BindingOrder::LowToHigh);
                for poly_idx in 0..state.num_polys() {
                    let g = sumcheck_evals_array::<F, 2>(
                        &state.g[poly_idx],
                        index,
                        BindingOrder::LowToHigh,
                    );
                    let eq_virt = sumcheck_evals_array::<F, 2>(
                        &state.eq_virt[poly_idx],
                        index,
                        BindingOrder::LowToHigh,
                    );
                    let gamma_hw = state.gamma_powers[3 * poly_idx];
                    let gamma_bool = state.gamma_powers[3 * poly_idx + 1];
                    let gamma_virt = state.gamma_powers[3 * poly_idx + 2];
                    for eval_index in 0..2 {
                        running[eval_index] += g[eval_index]
                            * (gamma_hw
                                + gamma_bool * eq_bool[eval_index]
                                + gamma_virt * eq_virt[eval_index]);
                    }
                }
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

    Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
}

pub(in crate::cpu::sumcheck) fn bind_state<F>(
    context: Stage7HammingKernelContext,
    state: &mut SumcheckStage7HammingState<F>,
    challenge: F,
) -> Result<(), BackendError>
where
    F: Field,
{
    validate_state(context, state)?;
    if state.scratch_g.len() < state.g.len() {
        state.scratch_g.resize_with(state.g.len(), Vec::new);
    }
    if state.scratch_eq_virt.len() < state.eq_virt.len() {
        state
            .scratch_eq_virt
            .resize_with(state.eq_virt.len(), Vec::new);
    }

    let SumcheckStage7HammingState {
        g,
        eq_bool,
        eq_virt,
        scratch_g,
        scratch_eq_bool,
        scratch_eq_virt,
        ..
    } = state;

    rayon::scope(|scope| {
        scope.spawn(move |_| {
            g.iter_mut()
                .zip(scratch_g.iter_mut())
                .for_each(|(poly, scratch)| {
                    poly.bind_low_to_high_reusing_scratch(challenge, scratch);
                });
        });
        scope.spawn(move |_| {
            eq_bool.bind_low_to_high_reusing_scratch(challenge, scratch_eq_bool);
        });
        scope.spawn(move |_| {
            eq_virt
                .iter_mut()
                .zip(scratch_eq_virt.iter_mut())
                .for_each(|(poly, scratch)| {
                    poly.bind_low_to_high_reusing_scratch(challenge, scratch);
                });
        });
    });
    Ok(())
}

fn validate_state<F>(
    context: Stage7HammingKernelContext,
    state: &SumcheckStage7HammingState<F>,
) -> Result<(), BackendError>
where
    F: Field,
{
    if state.g.is_empty() {
        return context.invalid(format!(
            "Stage 7 hamming state {} has no G tables",
            state.label
        ));
    }
    if state.g.len() != state.eq_virt.len() {
        return context.invalid(format!(
            "Stage 7 hamming state {} has {} G tables but {} virtualization eq tables",
            state.label,
            state.g.len(),
            state.eq_virt.len()
        ));
    }
    if state.gamma_powers.len() != 3 * state.g.len() {
        return context.invalid(format!(
            "Stage 7 hamming state {} has {} gamma powers for {} RA polynomials",
            state.label,
            state.gamma_powers.len(),
            state.g.len()
        ));
    }
    let len = state.g[0].len();
    if len == 0 || !len.is_power_of_two() {
        return context.invalid(format!(
            "Stage 7 hamming state {} length {len} is not a nonzero power of two",
            state.label
        ));
    }
    if state.eq_bool.len() != len {
        return context.invalid(format!(
            "Stage 7 hamming state {} booleanity eq table has {} rows, expected {len}",
            state.label,
            state.eq_bool.len()
        ));
    }
    for (index, table) in state.g.iter().enumerate().skip(1) {
        if table.len() != len {
            return context.invalid(format!(
                "Stage 7 hamming state {} G table {index} has {} rows, expected {len}",
                state.label,
                table.len()
            ));
        }
    }
    for (index, table) in state.eq_virt.iter().enumerate() {
        if table.len() != len {
            return context.invalid(format!(
                "Stage 7 hamming state {} virtualization eq table {index} has {} rows, expected {len}",
                state.label,
                table.len()
            ));
        }
    }
    Ok(())
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
