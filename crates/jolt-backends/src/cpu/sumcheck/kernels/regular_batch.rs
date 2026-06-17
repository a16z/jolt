use jolt_field::{AdditiveAccumulator, Field, RingAccumulator, WithAccumulator};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use rayon::prelude::*;

use crate::{
    BackendError, SumcheckRegularBatchInstance, SumcheckRegularBatchLinearFactor,
    SumcheckRegularBatchRound, SumcheckRegularBatchState,
};

const MAX_CACHED_POLYNOMIALS: usize = 16;

#[derive(Clone, Copy)]
pub(in crate::cpu::sumcheck) struct RegularBatchKernelContext {
    backend: &'static str,
    task: &'static str,
}

impl RegularBatchKernelContext {
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

pub(in crate::cpu::sumcheck) fn evaluate_round<F>(
    context: RegularBatchKernelContext,
    state: &mut SumcheckRegularBatchState<F>,
    round: usize,
    max_rounds: usize,
    previous_claims: &[F],
) -> Result<Vec<SumcheckRegularBatchRound<F>>, BackendError>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    validate_state_once(context, state)?;
    if previous_claims.len() != state.instances.len() {
        return context.invalid(format!(
            "regular batch has {} previous claims for {} instances",
            previous_claims.len(),
            state.instances.len()
        ));
    }
    if round >= max_rounds {
        return context.invalid(format!(
            "regular batch round {round} is outside max round count {max_rounds}"
        ));
    }

    let two_inv = F::from_u64(2)
        .inverse()
        .ok_or_else(|| BackendError::InvalidRequest {
            backend: context.backend,
            task: context.task,
            reason: "field element 2 is not invertible".to_owned(),
        })?;

    if let Some(rounds) =
        evaluate_stage2_product_instruction_tail(state, round, max_rounds, previous_claims)
    {
        return Ok(rounds);
    }
    if let Some(rounds) =
        evaluate_stage3_instruction_register_tail(state, round, max_rounds, previous_claims)
    {
        return Ok(rounds);
    }

    state
        .instances
        .iter()
        .zip(previous_claims)
        .enumerate()
        .map(|(instance_index, (instance, &previous_claim))| {
            let offset = max_rounds - instance.num_rounds();
            let polynomial = if round < offset {
                UnivariatePoly::new(vec![previous_claim * two_inv])
            } else {
                compute_instance_message(context, instance, previous_claim)?
            };
            Ok(SumcheckRegularBatchRound::new(instance_index, polynomial))
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Stage2ProductTailShape<F: Field> {
    scale: F,
}

#[derive(Clone, Copy)]
struct Stage2LinearReductionShape;

#[derive(Clone, Copy)]
struct Stage3InstructionShape<F: Field> {
    gamma: F,
}

#[derive(Clone, Copy)]
struct Stage3RegistersShape<F: Field> {
    gamma: F,
    gamma2: F,
}

fn evaluate_stage2_product_instruction_tail<F>(
    state: &SumcheckRegularBatchState<F>,
    round: usize,
    max_rounds: usize,
    previous_claims: &[F],
) -> Option<Vec<SumcheckRegularBatchRound<F>>>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let [product, instruction, rest @ ..] = state.instances.as_slice() else {
        return None;
    };
    if rest.len() > 1 {
        return None;
    }
    let product_offset = max_rounds.checked_sub(product.num_rounds())?;
    if round < product_offset {
        return None;
    }
    let product_shape = stage2_product_tail_shape(product)?;
    let _instruction_shape =
        stage2_linear_reduction_shape(instruction, "instruction claim-reduction")?;
    let field_registers_shape = match rest.first() {
        Some(instance) => Some(stage2_linear_reduction_shape(
            instance,
            "field-registers claim-reduction",
        )?),
        None => None,
    };
    let half = product.polynomials.first()?.len() / 2;
    if half == 0
        || product.polynomials.get(1)?.len() / 2 != half
        || product.polynomials.get(2)?.len() / 2 != half
        || instruction.polynomials.get(1)?.len() / 2 != half
        || rest.first().is_some_and(|instance| {
            instance.polynomials.get(1).map_or(0, Polynomial::len) / 2 != half
        })
    {
        return None;
    }

    let (product_accumulators, instruction_accumulators, field_registers_accumulators) = (0..half)
        .into_par_iter()
        .fold(
            || {
                (
                    [<F as WithAccumulator>::Accumulator::default(); 3],
                    [<F as WithAccumulator>::Accumulator::default(); 2],
                    [<F as WithAccumulator>::Accumulator::default(); 2],
                )
            },
            |(mut product_running, mut instruction_running, mut field_registers_running), index| {
                let product_pairs = [
                    product.polynomials[0].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    product.polynomials[1].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    product.polynomials[2].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                ];
                for (point_index, value) in product_running.iter_mut().enumerate() {
                    let point = sumcheck_hint_point::<F>(point_index);
                    let eq = eval_pair_at(product_pairs[0], point);
                    let left = eval_pair_at(product_pairs[1], point);
                    let right = eval_pair_at(product_pairs[2], point);
                    value.fmadd(product_shape.scale * eq, left * right);
                }

                let instruction_pairs = [
                    product_pairs[0],
                    instruction.polynomials[1].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                ];
                for (point_index, value) in instruction_running.iter_mut().enumerate() {
                    let point = sumcheck_hint_point::<F>(point_index);
                    let eq = eval_pair_at(instruction_pairs[0], point);
                    let reduced = eval_pair_at(instruction_pairs[1], point);
                    value.fmadd(eq, reduced);
                }
                if field_registers_shape.is_some() {
                    let field_registers = &rest[0];
                    let field_registers_pairs = [
                        product_pairs[0],
                        field_registers.polynomials[1]
                            .sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    ];
                    for (point_index, value) in field_registers_running.iter_mut().enumerate() {
                        let point = sumcheck_hint_point::<F>(point_index);
                        let eq = eval_pair_at(field_registers_pairs[0], point);
                        let reduced = eval_pair_at(field_registers_pairs[1], point);
                        value.fmadd(eq, reduced);
                    }
                }
                (
                    product_running,
                    instruction_running,
                    field_registers_running,
                )
            },
        )
        .reduce(
            || {
                (
                    [<F as WithAccumulator>::Accumulator::default(); 3],
                    [<F as WithAccumulator>::Accumulator::default(); 2],
                    [<F as WithAccumulator>::Accumulator::default(); 2],
                )
            },
            |(mut left_product, mut left_instruction, mut left_field_registers),
             (right_product, right_instruction, right_field_registers)| {
                for (left, right) in left_product.iter_mut().zip(right_product) {
                    left.merge(right);
                }
                for (left, right) in left_instruction.iter_mut().zip(right_instruction) {
                    left.merge(right);
                }
                for (left, right) in left_field_registers.iter_mut().zip(right_field_registers) {
                    left.merge(right);
                }
                (left_product, left_instruction, left_field_registers)
            },
        );
    let product_evals = product_accumulators.map(AdditiveAccumulator::reduce);
    let instruction_evals = instruction_accumulators.map(AdditiveAccumulator::reduce);
    let field_registers_evals = field_registers_accumulators.map(AdditiveAccumulator::reduce);

    let mut rounds = vec![
        SumcheckRegularBatchRound::new(
            0,
            UnivariatePoly::from_evals_and_hint(previous_claims[0], &product_evals),
        ),
        SumcheckRegularBatchRound::new(
            1,
            UnivariatePoly::from_evals_and_hint(previous_claims[1], &instruction_evals),
        ),
    ];
    if field_registers_shape.is_some() {
        rounds.push(SumcheckRegularBatchRound::new(
            2,
            UnivariatePoly::from_evals_and_hint(previous_claims[2], &field_registers_evals),
        ));
    }
    Some(rounds)
}

fn stage2_product_tail_shape<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
) -> Option<Stage2ProductTailShape<F>> {
    if instance.label != "product remainder" || instance.polynomials.len() != 3 {
        return None;
    }
    let [product] = instance.products.as_slice() else {
        return None;
    };
    let [eq_factor, left_factor, right_factor] = product.factors.as_slice() else {
        return None;
    };
    if !single_term_factor(eq_factor, 0, F::one())
        || !single_term_factor(left_factor, 1, F::one())
        || !single_term_factor(right_factor, 2, F::one())
    {
        return None;
    }
    Some(Stage2ProductTailShape {
        scale: product.scale,
    })
}

fn stage2_linear_reduction_shape<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
    label: &'static str,
) -> Option<Stage2LinearReductionShape> {
    if instance.label != label || instance.polynomials.len() != 2 {
        return None;
    }
    let [product] = instance.products.as_slice() else {
        return None;
    };
    if product.scale != F::one() {
        return None;
    }
    let [eq_factor, reduced_factor] = product.factors.as_slice() else {
        return None;
    };
    if !single_term_factor(eq_factor, 0, F::one()) {
        return None;
    }
    if !single_term_factor(reduced_factor, 1, F::one()) {
        return None;
    }
    Some(Stage2LinearReductionShape)
}

fn evaluate_stage3_instruction_register_tail<F>(
    state: &SumcheckRegularBatchState<F>,
    round: usize,
    max_rounds: usize,
    previous_claims: &[F],
) -> Option<Vec<SumcheckRegularBatchRound<F>>>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let [instruction, registers] = state.instances.as_slice() else {
        return None;
    };
    let instruction_offset = max_rounds.checked_sub(instruction.num_rounds())?;
    if round < instruction_offset || registers.num_rounds() != instruction.num_rounds() {
        return None;
    }
    let instruction_shape = stage3_instruction_shape(instruction)?;
    let registers_shape = stage3_registers_shape(registers)?;
    let half = instruction.polynomials.first()?.len() / 2;
    if half == 0 || registers.polynomials.first()?.len() / 2 != half {
        return None;
    }

    let (instruction_accumulators, registers_accumulators) = (0..half)
        .into_par_iter()
        .fold(
            || {
                (
                    [<F as WithAccumulator>::Accumulator::default(); 3],
                    [<F as WithAccumulator>::Accumulator::default(); 2],
                )
            },
            |(mut instruction_running, mut registers_running), index| {
                let instruction_pairs = [
                    instruction.polynomials[0].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[1].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[2].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[3].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[4].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[5].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[6].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[7].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    instruction.polynomials[8].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                ];
                for (point_index, value) in instruction_running.iter_mut().enumerate() {
                    let point = sumcheck_hint_point::<F>(point_index);
                    let eq = eval_pair_at(instruction_pairs[0], point);
                    let right = eval_pair_at(instruction_pairs[1], point)
                        * eval_pair_at(instruction_pairs[2], point)
                        + eval_pair_at(instruction_pairs[3], point)
                            * eval_pair_at(instruction_pairs[4], point);
                    let left = eval_pair_at(instruction_pairs[5], point)
                        * eval_pair_at(instruction_pairs[6], point)
                        + eval_pair_at(instruction_pairs[7], point)
                            * eval_pair_at(instruction_pairs[8], point);
                    value.fmadd(eq, right + instruction_shape.gamma * left);
                }

                let registers_pairs = [
                    registers.polynomials[0].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    registers.polynomials[1].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    registers.polynomials[2].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                    registers.polynomials[3].sumcheck_eval_pair(index, BindingOrder::HighToLow),
                ];
                for (point_index, value) in registers_running.iter_mut().enumerate() {
                    let point = sumcheck_hint_point::<F>(point_index);
                    let eq = eval_pair_at(registers_pairs[0], point);
                    let reduced = eval_pair_at(registers_pairs[1], point)
                        + registers_shape.gamma * eval_pair_at(registers_pairs[2], point)
                        + registers_shape.gamma2 * eval_pair_at(registers_pairs[3], point);
                    value.fmadd(eq, reduced);
                }
                (instruction_running, registers_running)
            },
        )
        .reduce(
            || {
                (
                    [<F as WithAccumulator>::Accumulator::default(); 3],
                    [<F as WithAccumulator>::Accumulator::default(); 2],
                )
            },
            |(mut left_instruction, mut left_registers), (right_instruction, right_registers)| {
                for (left, right) in left_instruction.iter_mut().zip(right_instruction) {
                    left.merge(right);
                }
                for (left, right) in left_registers.iter_mut().zip(right_registers) {
                    left.merge(right);
                }
                (left_instruction, left_registers)
            },
        );
    let instruction_evals = instruction_accumulators.map(AdditiveAccumulator::reduce);
    let registers_evals = registers_accumulators.map(AdditiveAccumulator::reduce);

    Some(vec![
        SumcheckRegularBatchRound::new(
            0,
            UnivariatePoly::from_evals_and_hint(previous_claims[0], &instruction_evals),
        ),
        SumcheckRegularBatchRound::new(
            1,
            UnivariatePoly::from_evals_and_hint(previous_claims[1], &registers_evals),
        ),
    ])
}

fn stage3_instruction_shape<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
) -> Option<Stage3InstructionShape<F>> {
    if instance.label != "stage3.instruction_input" || instance.polynomials.len() != 9 {
        return None;
    }
    let [right_rs2, right_imm, left_rs1, left_pc] = instance.products.as_slice() else {
        return None;
    };
    if !stage3_eq_product_shape(right_rs2, F::one(), 1, 2)
        || !stage3_eq_product_shape(right_imm, F::one(), 3, 4)
    {
        return None;
    }
    let gamma = left_rs1.scale;
    if !stage3_eq_product_shape(left_rs1, gamma, 5, 6)
        || !stage3_eq_product_shape(left_pc, gamma, 7, 8)
    {
        return None;
    }
    Some(Stage3InstructionShape { gamma })
}

fn stage3_registers_shape<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
) -> Option<Stage3RegistersShape<F>> {
    if instance.label != "stage3.registers_claim_reduction" || instance.polynomials.len() != 4 {
        return None;
    }
    let [product] = instance.products.as_slice() else {
        return None;
    };
    if product.scale != F::one() {
        return None;
    }
    let [eq_factor, reduced_factor] = product.factors.as_slice() else {
        return None;
    };
    if !single_term_factor(eq_factor, 0, F::one()) {
        return None;
    }
    let [one, gamma, gamma2] =
        coefficient_set(reduced_factor, &[(1, Some(F::one())), (2, None), (3, None)])?;
    if one != F::one() {
        return None;
    }
    Some(Stage3RegistersShape { gamma, gamma2 })
}

fn stage3_eq_product_shape<F: Field>(
    product: &crate::SumcheckRegularBatchProduct<F>,
    scale: F,
    left: usize,
    right: usize,
) -> bool {
    let [eq_factor, left_factor, right_factor] = product.factors.as_slice() else {
        return false;
    };
    product.scale == scale
        && single_term_factor(eq_factor, 0, F::one())
        && single_term_factor(left_factor, left, F::one())
        && single_term_factor(right_factor, right, F::one())
}

fn single_term_factor<F: Field>(
    factor: &SumcheckRegularBatchLinearFactor<F>,
    polynomial: usize,
    coefficient: F,
) -> bool {
    factor.constant == F::zero()
        && factor.terms.len() == 1
        && factor.terms[0].polynomial == polynomial
        && factor.terms[0].coefficient == coefficient
}

fn coefficient_set<F: Field, const N: usize>(
    factor: &SumcheckRegularBatchLinearFactor<F>,
    expected: &[(usize, Option<F>); N],
) -> Option<[F; N]> {
    if factor.constant != F::zero() || factor.terms.len() != N {
        return None;
    }
    let mut coefficients = [F::zero(); N];
    for (index, (polynomial, expected_coefficient)) in expected.iter().copied().enumerate() {
        let coefficient = factor
            .terms
            .iter()
            .find(|term| term.polynomial == polynomial)
            .map(|term| term.coefficient)?;
        if expected_coefficient.is_some_and(|expected| expected != coefficient) {
            return None;
        }
        coefficients[index] = coefficient;
    }
    Some(coefficients)
}

pub(in crate::cpu::sumcheck) fn bind_state<F>(
    context: RegularBatchKernelContext,
    state: &mut SumcheckRegularBatchState<F>,
    round: usize,
    max_rounds: usize,
    challenge: F,
) -> Result<(), BackendError>
where
    F: Field,
{
    validate_state_once(context, state)?;
    if round >= max_rounds {
        return context.invalid(format!(
            "regular batch bind round {round} is outside max round count {max_rounds}"
        ));
    }
    if bind_stage2_product_instruction_tail(state, round, max_rounds, challenge).is_some() {
        return Ok(());
    }
    state.instances.par_iter_mut().for_each(|instance| {
        let offset = max_rounds - instance.num_rounds();
        if round >= offset {
            instance.polynomials.par_iter_mut().for_each(|polynomial| {
                polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
            });
        }
    });
    Ok(())
}

fn bind_stage2_product_instruction_tail<F>(
    state: &mut SumcheckRegularBatchState<F>,
    round: usize,
    max_rounds: usize,
    challenge: F,
) -> Option<()>
where
    F: Field,
{
    let [product, instruction, rest @ ..] = state.instances.as_mut_slice() else {
        return None;
    };
    if rest.len() > 1 {
        return None;
    }
    let product_offset = max_rounds.checked_sub(product.num_rounds())?;
    if round < product_offset {
        return Some(());
    }
    let _product_shape = stage2_product_tail_shape(product)?;
    let _instruction_shape =
        stage2_linear_reduction_shape(instruction, "instruction claim-reduction")?;
    let has_field_registers = match rest.first() {
        Some(instance) => {
            let _field_registers_shape =
                stage2_linear_reduction_shape(instance, "field-registers claim-reduction")?;
            true
        }
        None => false,
    };

    if product.polynomials.len() != 3
        || instruction.polynomials.len() != 2
        || rest
            .first()
            .is_some_and(|instance| instance.polynomials.len() != 2)
    {
        return None;
    }

    rayon::scope(|scope| {
        let product_polynomials = &mut product.polynomials;
        scope.spawn(move |_| {
            product_polynomials.par_iter_mut().for_each(|polynomial| {
                polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
            });
        });

        let instruction_reduced = &mut instruction.polynomials[1];
        scope.spawn(move |_| {
            instruction_reduced.bind_with_order(challenge, BindingOrder::HighToLow);
        });

        if has_field_registers {
            let field_registers_reduced = &mut rest[0].polynomials[1];
            scope.spawn(move |_| {
                field_registers_reduced.bind_with_order(challenge, BindingOrder::HighToLow);
            });
        }
    });
    Some(())
}

fn validate_state_once<F: Field>(
    context: RegularBatchKernelContext,
    state: &mut SumcheckRegularBatchState<F>,
) -> Result<(), BackendError> {
    if state.is_validated() {
        return Ok(());
    }
    validate_state(context, state)?;
    state.mark_validated();
    Ok(())
}

fn validate_state<F: Field>(
    context: RegularBatchKernelContext,
    state: &SumcheckRegularBatchState<F>,
) -> Result<(), BackendError> {
    if state.instances.is_empty() {
        return context.invalid("regular batch has no instances");
    }
    for (instance_index, instance) in state.instances.iter().enumerate() {
        validate_instance(context, instance_index, instance)?;
    }
    Ok(())
}

fn validate_instance<F: Field>(
    context: RegularBatchKernelContext,
    instance_index: usize,
    instance: &SumcheckRegularBatchInstance<F>,
) -> Result<(), BackendError> {
    let Some(first) = instance.polynomials.first() else {
        return context.invalid(format!(
            "regular batch instance {instance_index} ({}) has no polynomials",
            instance.label
        ));
    };
    if instance.products.is_empty() {
        return context.invalid(format!(
            "regular batch instance {instance_index} ({}) has no product terms",
            instance.label
        ));
    }
    let len = first.len();
    if len == 0 || !len.is_power_of_two() {
        return context.invalid(format!(
            "regular batch instance {instance_index} ({}) length {len} is not a nonzero power of two",
            instance.label
        ));
    }
    for (polynomial_index, polynomial) in instance.polynomials.iter().enumerate().skip(1) {
        if polynomial.len() != len {
            return context.invalid(format!(
                "regular batch instance {instance_index} ({}) polynomial {polynomial_index} has length {}, expected {len}",
                instance.label,
                polynomial.len()
            ));
        }
    }
    for (product_index, product) in instance.products.iter().enumerate() {
        if product.factors.is_empty() {
            return context.invalid(format!(
                "regular batch instance {instance_index} ({}) product term {product_index} has no factors",
                instance.label
            ));
        }
        for (factor_index, factor) in product.factors.iter().enumerate() {
            for term in &factor.terms {
                if term.polynomial >= instance.polynomials.len() {
                    return context.invalid(format!(
                        "regular batch instance {instance_index} ({}) product {product_index} factor {factor_index} references polynomial {}, but only {} polynomials exist",
                        instance.label,
                        term.polynomial,
                        instance.polynomials.len()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn compute_instance_message<F>(
    context: RegularBatchKernelContext,
    instance: &SumcheckRegularBatchInstance<F>,
    previous_claim: F,
) -> Result<UnivariatePoly<F>, BackendError>
where
    F: Field,
{
    let half = instance.polynomials[0].len() / 2;
    if half == 0 {
        return context.invalid(format!(
            "regular batch instance {} has no unbound variables",
            instance.label
        ));
    }
    let eval_count = instance.degree();
    match eval_count {
        1 => {
            return Ok(compute_instance_message_array::<F, 1>(
                instance,
                previous_claim,
            ))
        }
        2 => {
            return Ok(compute_instance_message_array::<F, 2>(
                instance,
                previous_claim,
            ))
        }
        3 => {
            return Ok(compute_instance_message_array::<F, 3>(
                instance,
                previous_claim,
            ))
        }
        4 => {
            return Ok(compute_instance_message_array::<F, 4>(
                instance,
                previous_claim,
            ))
        }
        5 => {
            return Ok(compute_instance_message_array::<F, 5>(
                instance,
                previous_claim,
            ))
        }
        6 => {
            return Ok(compute_instance_message_array::<F, 6>(
                instance,
                previous_claim,
            ))
        }
        _ => {}
    }
    let evals = (0..half)
        .into_par_iter()
        .fold(
            || vec![F::zero(); eval_count],
            |mut running, index| {
                if instance.polynomials.len() <= MAX_CACHED_POLYNOMIALS {
                    let mut pairs = [(F::zero(), F::zero()); MAX_CACHED_POLYNOMIALS];
                    for (pair, polynomial) in pairs.iter_mut().zip(&instance.polynomials) {
                        *pair = polynomial.sumcheck_eval_pair(index, BindingOrder::HighToLow);
                    }
                    for (point_index, value) in running.iter_mut().enumerate() {
                        let point = sumcheck_hint_point::<F>(point_index);
                        let values =
                            evaluated_pairs_for_point(&pairs, instance.polynomials.len(), point);
                        *value += evaluate_cached_point_values(instance, &values);
                    }
                } else {
                    for (point_index, value) in running.iter_mut().enumerate() {
                        let point = sumcheck_hint_point::<F>(point_index);
                        *value += evaluate_pair(instance, index, point);
                    }
                }
                running
            },
        )
        .reduce(
            || vec![F::zero(); eval_count],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        );
    Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
}

fn compute_instance_message_array<F, const EVAL_COUNT: usize>(
    instance: &SumcheckRegularBatchInstance<F>,
    previous_claim: F,
) -> UnivariatePoly<F>
where
    F: Field,
{
    let half = instance.polynomials[0].len() / 2;
    let evals = (0..half)
        .into_par_iter()
        .fold(
            || [F::zero(); EVAL_COUNT],
            |mut running, index| {
                if instance.polynomials.len() <= MAX_CACHED_POLYNOMIALS {
                    let mut pairs = [(F::zero(), F::zero()); MAX_CACHED_POLYNOMIALS];
                    for (pair, polynomial) in pairs.iter_mut().zip(&instance.polynomials) {
                        *pair = polynomial.sumcheck_eval_pair(index, BindingOrder::HighToLow);
                    }
                    let values = evaluated_pairs_for_points::<F, EVAL_COUNT>(
                        &pairs,
                        instance.polynomials.len(),
                    );
                    for (point_index, value) in running.iter_mut().enumerate() {
                        *value += evaluate_cached_point_values(instance, &values[point_index]);
                    }
                } else {
                    for (point_index, value) in running.iter_mut().enumerate() {
                        let point = sumcheck_hint_point::<F>(point_index);
                        *value += evaluate_pair(instance, index, point);
                    }
                }
                running
            },
        )
        .reduce(
            || [F::zero(); EVAL_COUNT],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        );
    UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
}

fn sumcheck_hint_point<F: Field>(point_index: usize) -> F {
    if point_index == 0 {
        F::zero()
    } else {
        F::from_u64(point_index as u64 + 1)
    }
}

fn evaluated_pairs_for_points<F: Field, const EVAL_COUNT: usize>(
    pairs: &[(F, F); MAX_CACHED_POLYNOMIALS],
    polynomial_count: usize,
) -> [[F; MAX_CACHED_POLYNOMIALS]; EVAL_COUNT] {
    debug_assert!(polynomial_count <= MAX_CACHED_POLYNOMIALS);
    let mut values = [[F::zero(); MAX_CACHED_POLYNOMIALS]; EVAL_COUNT];
    for (point_index, point_values) in values.iter_mut().enumerate() {
        let point = sumcheck_hint_point::<F>(point_index);
        for (value, &(lo, hi)) in point_values.iter_mut().zip(&pairs[..polynomial_count]) {
            *value = lo + point * (hi - lo);
        }
    }
    values
}

fn evaluated_pairs_for_point<F: Field>(
    pairs: &[(F, F); MAX_CACHED_POLYNOMIALS],
    polynomial_count: usize,
    point: F,
) -> [F; MAX_CACHED_POLYNOMIALS] {
    debug_assert!(polynomial_count <= MAX_CACHED_POLYNOMIALS);
    let mut values = [F::zero(); MAX_CACHED_POLYNOMIALS];
    for (value, &(lo, hi)) in values.iter_mut().zip(&pairs[..polynomial_count]) {
        *value = lo + point * (hi - lo);
    }
    values
}

fn evaluate_cached_point_values<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
    values: &[F; MAX_CACHED_POLYNOMIALS],
) -> F {
    instance
        .products
        .iter()
        .map(|product| {
            product
                .factors
                .iter()
                .map(|factor| evaluate_factor_from_values(factor, values))
                .fold(product.scale, |acc, value| acc * value)
        })
        .sum()
}

fn evaluate_factor_from_values<F: Field>(
    factor: &SumcheckRegularBatchLinearFactor<F>,
    values: &[F; MAX_CACHED_POLYNOMIALS],
) -> F {
    factor.terms.iter().fold(factor.constant, |value, term| {
        value + term.coefficient * values[term.polynomial]
    })
}

fn evaluate_pair<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
    index: usize,
    point: F,
) -> F {
    instance
        .products
        .iter()
        .map(|product| {
            product
                .factors
                .iter()
                .map(|factor| {
                    factor.terms.iter().fold(factor.constant, |value, term| {
                        value
                            + term.coefficient
                                * factor_at(&instance.polynomials[term.polynomial], index, point)
                    })
                })
                .fold(product.scale, |acc, value| acc * value)
        })
        .sum()
}

fn eval_pair_at<F: Field>((lo, hi): (F, F), point: F) -> F {
    lo + point * (hi - lo)
}

fn factor_at<F: Field>(polynomial: &Polynomial<F>, index: usize, point: F) -> F {
    let (lo, hi) = polynomial.sumcheck_eval_pair(index, BindingOrder::HighToLow);
    eval_pair_at((lo, hi), point)
}
