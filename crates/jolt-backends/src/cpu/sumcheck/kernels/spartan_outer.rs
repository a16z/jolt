use std::collections::{HashMap, HashSet};

use jolt_field::{
    signed::{S128, S160, S192, S64},
    AdditiveAccumulator, Field, RingAccumulator, WithAccumulator,
};
use jolt_poly::{lagrange::centered_lagrange_evals, TensorEqTable};
use rayon::prelude::*;

use super::sparse_product::{CpuKernelContext, SparseProductKernel};
use crate::{
    BackendError, BackendValueSlot, SumcheckLinearProductOutput, SumcheckLinearProductRequest,
    SumcheckPrefixProductSumQuery, SumcheckPrefixProductSumRequest,
    SumcheckSpartanOuterRemainderQuery, SumcheckSpartanOuterRemainderRequest,
    SumcheckSpartanOuterRemainderRound, SumcheckSpartanOuterRemainderRowStateRequest,
    SumcheckSpartanOuterRemainderState, SumcheckSpartanOuterRemainderStateRequest,
    SumcheckSpartanOuterRow, SumcheckSpartanOuterUniskipQuery, SumcheckSpartanOuterUniskipRequest,
};

const JOLT_VM_NAMESPACE: &str = "jolt_vm";
const UNISKIP_RELATION: &str = "spartan_outer.uniskip_first_round";
const REMAINDER_RELATION: &str = "spartan_outer.remainder";
const BASE_FIRST_GROUP_TERMS: usize = 10;
const BASE_SECOND_GROUP_TERMS: usize = 9;

// This module is intentionally reserved for the coarse Spartan-outer CPU port.
// The fallback path below keeps correctness tests moving, but core parity
// should replace it with the trace/window/split-eq routine from `jolt-core`.

pub(in crate::cpu::sumcheck) fn matches_linear_product<F: Field>(
    request: &SumcheckLinearProductRequest<F>,
) -> bool {
    request.kernel.relation.is_some_and(|relation| {
        relation.namespace == JOLT_VM_NAMESPACE
            && matches!(relation.name, UNISKIP_RELATION | REMAINDER_RELATION)
    })
}

pub(in crate::cpu::sumcheck) fn matches_prefix_product_sum<F: Field>(
    request: &SumcheckPrefixProductSumRequest<F>,
) -> bool {
    request.kernel.relation.is_some_and(|relation| {
        relation.namespace == JOLT_VM_NAMESPACE
            && matches!(relation.name, UNISKIP_RELATION | REMAINDER_RELATION)
    })
}

pub(in crate::cpu::sumcheck) fn evaluate_linear_products<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckLinearProductRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
{
    super::evaluate_linear_product_queries(backend, task, request)
}

pub(in crate::cpu::sumcheck) fn evaluate_prefix_product_sums<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckPrefixProductSumRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
{
    validate_unique_query_slots(
        backend,
        task,
        request.queries.iter().map(|query| query.slot),
    )?;
    let context = CpuKernelContext::new(backend, task);
    let kernel = SparseProductKernel::new(
        context,
        request.witness_polynomials,
        request.input_columns,
        request.constant_column,
        request.left_rows,
        request.right_rows,
    )?;

    let mut groups = HashMap::<Vec<F>, Vec<(usize, &SumcheckPrefixProductSumQuery<F>)>>::new();
    for (query_index, query) in request.queries.iter().enumerate() {
        groups
            .entry(query.eq_point.clone())
            .or_default()
            .push((query_index, query));
    }

    let mut outputs = Vec::with_capacity(request.queries.len());
    outputs.resize_with(request.queries.len(), || None);
    for (eq_point, queries) in groups {
        if queries
            .iter()
            .all(|(_, query)| query.fixed_prefix.is_empty())
        {
            for (query_index, output) in
                evaluate_full_hypercube_group(context, &kernel, &eq_point, &queries)?
            {
                outputs[query_index] = Some(output);
            }
        } else if queries
            .iter()
            .all(|(_, query)| query.fixed_prefix.len() == 1)
        {
            for (query_index, output) in
                evaluate_fixed_stream_group(context, &kernel, &eq_point, &queries)?
            {
                outputs[query_index] = Some(output);
            }
        } else if can_evaluate_bound_prefix_group(&queries) {
            for (query_index, output) in
                evaluate_bound_prefix_group(context, &kernel, &eq_point, &queries)?
            {
                outputs[query_index] = Some(output);
            }
        } else {
            for (query_index, query) in queries {
                let value = evaluate_prefix_product_sum(context, &kernel, query)
                    .map_err(|error| with_query_context(error, query_index))?;
                outputs[query_index] = Some(SumcheckLinearProductOutput::new(query.slot, value));
            }
        }
    }

    outputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| BackendError::InvalidRequest {
            backend,
            task,
            reason: "prefix product sum evaluation did not produce every requested slot".to_owned(),
        })
}

pub(in crate::cpu::sumcheck) fn evaluate_spartan_outer_uniskip_rows<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckSpartanOuterUniskipRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    RawSpartanOuterUniskipKernel::new(CpuKernelContext::new(backend, task), request)?
        .evaluate(CpuKernelContext::new(backend, task), request)
}

pub(in crate::cpu::sumcheck) fn evaluate_spartan_outer_remainder_rows<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckSpartanOuterRemainderRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    RawSpartanOuterRemainderKernel::new(CpuKernelContext::new(backend, task), request)?
        .evaluate(CpuKernelContext::new(backend, task), request)
}

pub(in crate::cpu::sumcheck) fn materialize_spartan_outer_remainder_state<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckSpartanOuterRemainderStateRequest<F>,
) -> Result<SumcheckSpartanOuterRemainderState<F>, BackendError>
where
    F: Field,
{
    let context = CpuKernelContext::new(backend, task);
    let kernel = SparseProductKernel::new(
        context,
        request.witness_polynomials,
        request.input_columns,
        request.constant_column,
        request.left_rows,
        request.right_rows,
    )?;
    if request.eq_point.len() != kernel.log_rows() + 1 {
        return context.invalid(format!(
            "Spartan outer remainder state equality point has {} variables, expected {}",
            request.eq_point.len(),
            kernel.log_rows() + 1
        ));
    }
    if request.row_weights_at_zero.len() != request.row_weights_at_one.len() {
        return context.invalid(format!(
            "row weight selector lengths differ: zero has {}, one has {}",
            request.row_weights_at_zero.len(),
            request.row_weights_at_one.len()
        ));
    }
    if request.row_weights_at_zero.len() != kernel.sparse_row_count() {
        return context.invalid(format!(
            "request has {} row weights, expected {}",
            request.row_weights_at_zero.len(),
            kernel.sparse_row_count()
        ));
    }

    let stream_eq = eq_factor(
        request.eq_point[kernel.log_rows()],
        request.stream_challenge,
    );
    let row_weights = blend_row_weights(
        request.stream_challenge,
        &request.row_weights_at_zero,
        &request.row_weights_at_one,
    );
    let (left_form, right_form) = kernel.combine_weighted_forms(context, &row_weights)?;
    let mut left = vec![F::zero(); kernel.rows()];
    let mut right = vec![F::zero(); kernel.rows()];
    left.par_iter_mut()
        .zip(right.par_iter_mut())
        .enumerate()
        .for_each(|(row_index, (left, right))| {
            let (left_value, right_value) =
                kernel.evaluate_combined_forms_at_boolean_row(&left_form, &right_form, row_index);
            *left = left_value;
            *right = right_value;
        });

    Ok(SumcheckSpartanOuterRemainderState::new(
        request.label,
        request.eq_point[..kernel.log_rows()].to_vec(),
        left,
        right,
        kernel.rows(),
        request.scale * stream_eq,
    ))
}

pub(in crate::cpu::sumcheck) fn materialize_spartan_outer_remainder_row_state<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckSpartanOuterRemainderRowStateRequest<F>,
) -> Result<SumcheckSpartanOuterRemainderState<F>, BackendError>
where
    F: Field,
{
    let context = CpuKernelContext::new(backend, task);
    let row_count = request.rows.len();
    if row_count == 0 || !row_count.is_power_of_two() {
        return context.invalid(format!(
            "Spartan outer raw-row remainder state row count {row_count} is not a nonzero power of two"
        ));
    }
    let log_rows = row_count.trailing_zeros() as usize;
    if request.eq_point.len() != log_rows + 1 {
        return context.invalid(format!(
            "Spartan outer raw-row remainder state equality point has {} variables, expected {}",
            request.eq_point.len(),
            log_rows + 1
        ));
    }

    let lagrange = spartan_outer_lagrange_weights_for_domain(
        context,
        request.uniskip_challenge,
        request.uniskip_domain_size,
    )?;
    let stream_eq = eq_factor(request.eq_point[log_rows], request.stream_challenge);
    let mut left = vec![F::zero(); row_count];
    let mut right = vec![F::zero(); row_count];
    left.par_iter_mut()
        .zip(right.par_iter_mut())
        .zip(request.rows.par_iter())
        .for_each(|((left, right), row)| {
            let (left_value, right_value) =
                spartan_outer_remainder_forms(row, &lagrange, request.stream_challenge);
            *left = left_value;
            *right = right_value;
        });

    Ok(SumcheckSpartanOuterRemainderState::new(
        request.label,
        request.eq_point[..log_rows].to_vec(),
        left,
        right,
        row_count,
        request.scale * stream_eq,
    ))
}

pub(in crate::cpu::sumcheck) fn evaluate_spartan_outer_remainder_round<F>(
    backend: &'static str,
    task: &'static str,
    state: &SumcheckSpartanOuterRemainderState<F>,
) -> Result<SumcheckSpartanOuterRemainderRound<F>, BackendError>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let context = CpuKernelContext::new(backend, task);
    validate_spartan_outer_remainder_state(context, state)?;
    if state.active_len == 1 {
        return context.invalid("Spartan outer remainder state has no unbound variables");
    }

    let active_log = state.active_len.trailing_zeros() as usize;
    let remaining_vars = active_log - 1;
    let eq_tensor = &state.eq_tables[remaining_vars];

    let (q_at_zero, q_at_infinity) = sum_bound_pair_endpoints(
        eq_tensor,
        &state.left[..state.active_len],
        &state.right[..state.active_len],
    );
    Ok(SumcheckSpartanOuterRemainderRound::new(
        q_at_zero,
        q_at_infinity,
    ))
}

pub(in crate::cpu::sumcheck) fn bind_spartan_outer_remainder_state<F>(
    backend: &'static str,
    task: &'static str,
    state: &mut SumcheckSpartanOuterRemainderState<F>,
    challenge: F,
) -> Result<(), BackendError>
where
    F: Field,
{
    let context = CpuKernelContext::new(backend, task);
    validate_spartan_outer_remainder_state(context, state)?;
    if state.active_len == 1 {
        return context.invalid("Spartan outer remainder state has no unbound variables to bind");
    }
    let active_log = state.active_len.trailing_zeros() as usize;
    let challenge_index = active_log - 1;
    let Some(&eq_challenge) = state.eq_point.get(challenge_index) else {
        return context.invalid(format!(
            "Spartan outer remainder state missing equality challenge {challenge_index}"
        ));
    };
    state.scale *= eq_factor(eq_challenge, challenge);
    let next_len = bind_low_variable_in_place(&mut state.left, state.active_len, challenge);
    let right_next_len = bind_low_variable_in_place(&mut state.right, state.active_len, challenge);
    debug_assert_eq!(next_len, right_next_len);
    state.active_len = next_len;
    Ok(())
}

fn validate_spartan_outer_remainder_state<F: Field>(
    context: CpuKernelContext,
    state: &SumcheckSpartanOuterRemainderState<F>,
) -> Result<(), BackendError> {
    if state.active_len == 0 || !state.active_len.is_power_of_two() {
        return context.invalid(format!(
            "Spartan outer remainder state active length {} is not a nonzero power of two",
            state.active_len
        ));
    }
    if state.left.len() != state.right.len() {
        return context.invalid(format!(
            "Spartan outer remainder state left length {} differs from right length {}",
            state.left.len(),
            state.right.len()
        ));
    }
    if state.active_len > state.left.len() {
        return context.invalid(format!(
            "Spartan outer remainder state active length {} exceeds stored row length {}",
            state.active_len,
            state.left.len()
        ));
    }
    let active_log = state.active_len.trailing_zeros() as usize;
    if state.eq_point.len() < active_log {
        return context.invalid(format!(
            "Spartan outer remainder state has {} equality variables, expected at least {}",
            state.eq_point.len(),
            active_log
        ));
    }
    if state.eq_tables.len() < active_log {
        return context.invalid(format!(
            "Spartan outer remainder state has {} cached equality tables, expected {}",
            state.eq_tables.len(),
            active_log
        ));
    }
    if state.active_len > 1 {
        let eq_tensor = &state.eq_tables[active_log - 1];
        if eq_tensor.len() * 2 != state.active_len {
            return context.invalid(format!(
                "Spartan outer remainder state equality tensor has {} row pairs, expected {}",
                eq_tensor.len(),
                state.active_len / 2
            ));
        }
    }
    Ok(())
}

struct RawSpartanOuterUniskipKernel<'a> {
    rows: &'a [SumcheckSpartanOuterRow],
    log_rows: usize,
}

impl<'a> RawSpartanOuterUniskipKernel<'a> {
    fn new<F: Field>(
        context: CpuKernelContext,
        request: &'a SumcheckSpartanOuterUniskipRequest<F>,
    ) -> Result<Self, BackendError> {
        let row_count = request.rows.len();
        if row_count == 0 || !row_count.is_power_of_two() {
            return context.invalid(format!(
                "Spartan outer row count {row_count} is not a nonzero power of two"
            ));
        }
        Ok(Self {
            rows: request.rows,
            log_rows: row_count.trailing_zeros() as usize,
        })
    }

    fn evaluate<F>(
        &self,
        context: CpuKernelContext,
        request: &SumcheckSpartanOuterUniskipRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
    where
        F: Field,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        validate_unique_query_slots(
            context.backend,
            context.task,
            request.queries.iter().map(|query| query.slot),
        )?;

        let mut groups =
            HashMap::<Vec<F>, Vec<(usize, &SumcheckSpartanOuterUniskipQuery<F>)>>::new();
        for (query_index, query) in request.queries.iter().enumerate() {
            groups
                .entry(query.eq_point.clone())
                .or_default()
                .push((query_index, query));
        }

        let mut outputs = Vec::with_capacity(request.queries.len());
        outputs.resize_with(request.queries.len(), || None);
        for (eq_point, queries) in groups {
            for (query_index, output) in self.evaluate_group(context, &eq_point, &queries)? {
                outputs[query_index] = Some(output);
            }
        }

        outputs
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: context.backend,
                task: context.task,
                reason: "Spartan outer uniskip row evaluation did not produce every requested slot"
                    .to_owned(),
            })
    }

    fn evaluate_group<F>(
        &self,
        context: CpuKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckSpartanOuterUniskipQuery<F>)],
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
    where
        F: Field,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let total_vars = self.log_rows + 1;
        if eq_point.len() != total_vars {
            return context.invalid(format!(
                "Spartan outer uniskip equality point has {} variables, expected {total_vars}",
                eq_point.len()
            ));
        }
        let compiled = queries
            .iter()
            .map(|&(query_index, query)| {
                RawSpartanOuterUniskipQuery::new(context, query_index, query)
                    .map_err(|error| with_query_context(error, query_index))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let domain_size = compiled[0].coeffs.len();
        if compiled
            .iter()
            .any(|query| query.coeffs.len() != domain_size)
        {
            return context
                .invalid("Spartan outer uniskip row queries do not share the domain size");
        }
        validate_supported_uniskip_domain(context, domain_size)?;
        let extended_evals = domain_size - 1;
        if compiled.len() != extended_evals {
            return context.invalid(format!(
                "Spartan outer uniskip row kernel expects {extended_evals} queries, got {}",
                compiled.len()
            ));
        }

        let eq_tensor = TensorEqTable::<F>::new(eq_point);
        if eq_tensor.len() != self.rows.len() * 2 {
            return context.invalid(format!(
                "Spartan outer uniskip equality tensor has {} rows, expected {}",
                eq_tensor.len(),
                self.rows.len() * 2
            ));
        }

        let values = eq_tensor.par_fold_out_in(
            || SpartanOuterUniskipAccumulator::<F>::new(compiled.len()),
            |inner, full_row_index, x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                let row = &self.rows[full_row_index >> 1];
                let second_group = (x_in & 1) == 1;
                if second_group {
                    let terms = SpartanOuterSecondGroupTerms::new(row);
                    for (query_index, query) in compiled.iter().enumerate() {
                        inner.totals[query_index].fmadd(e_in, terms.product::<F>(&query.coeffs));
                    }
                } else {
                    let terms = SpartanOuterFirstGroupTerms::new(row);
                    for (query_index, query) in compiled.iter().enumerate() {
                        inner.totals[query_index].fmadd(e_in, terms.product::<F>(&query.coeffs));
                    }
                }
            },
            |_x_out, e_out, inner| inner.scale(e_out),
            SpartanOuterUniskipAccumulator::merge,
        );

        Ok(values
            .into_values()
            .zip(compiled)
            .map(|(value, query)| {
                (
                    query.query_index,
                    SumcheckLinearProductOutput::new(query.slot, query.scale * value),
                )
            })
            .collect())
    }
}

struct RawSpartanOuterUniskipQuery<F: Field> {
    query_index: usize,
    slot: BackendValueSlot,
    coeffs: Vec<i32>,
    scale: F,
}

impl<F: Field> RawSpartanOuterUniskipQuery<F> {
    fn new(
        context: CpuKernelContext,
        query_index: usize,
        query: &SumcheckSpartanOuterUniskipQuery<F>,
    ) -> Result<Self, BackendError> {
        validate_supported_uniskip_domain(context, query.coeffs.len())?;
        Ok(Self {
            query_index,
            slot: query.slot,
            coeffs: query.coeffs.clone(),
            scale: query.scale,
        })
    }
}

struct SpartanOuterUniskipAccumulator<F: Field>
where
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    totals: Vec<<F as WithAccumulator>::Accumulator>,
}

impl<F> SpartanOuterUniskipAccumulator<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn new(output_count: usize) -> Self {
        Self {
            totals: vec![<F as WithAccumulator>::Accumulator::default(); output_count],
        }
    }

    fn scale(self, scale: F) -> Self {
        let mut scaled = Self::new(self.totals.len());
        if !scale.is_zero() {
            for (out, total) in scaled.totals.iter_mut().zip(self.totals) {
                out.fmadd(scale, total.reduce());
            }
        }
        scaled
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.totals.iter_mut().zip(other.totals) {
            left.merge(right);
        }
        self
    }

    fn into_values(self) -> impl Iterator<Item = F> {
        self.totals.into_iter().map(|total| total.reduce())
    }
}

struct RawSpartanOuterRemainderKernel<'a> {
    rows: &'a [SumcheckSpartanOuterRow],
    log_rows: usize,
}

impl<'a> RawSpartanOuterRemainderKernel<'a> {
    fn new<F: Field>(
        context: CpuKernelContext,
        request: &'a SumcheckSpartanOuterRemainderRequest<F>,
    ) -> Result<Self, BackendError> {
        let row_count = request.rows.len();
        if row_count == 0 || !row_count.is_power_of_two() {
            return context.invalid(format!(
                "Spartan outer remainder row count {row_count} is not a nonzero power of two"
            ));
        }
        Ok(Self {
            rows: request.rows,
            log_rows: row_count.trailing_zeros() as usize,
        })
    }

    fn evaluate<F>(
        &self,
        context: CpuKernelContext,
        request: &SumcheckSpartanOuterRemainderRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
    where
        F: Field,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        validate_unique_query_slots(
            context.backend,
            context.task,
            request.queries.iter().map(|query| query.slot),
        )?;

        let mut groups =
            HashMap::<Vec<F>, Vec<(usize, &SumcheckSpartanOuterRemainderQuery<F>)>>::new();
        for (query_index, query) in request.queries.iter().enumerate() {
            groups
                .entry(query.eq_point.clone())
                .or_default()
                .push((query_index, query));
        }

        let mut outputs = Vec::with_capacity(request.queries.len());
        outputs.resize_with(request.queries.len(), || None);
        for (eq_point, queries) in groups {
            for (query_index, output) in self.evaluate_group(context, &eq_point, &queries)? {
                outputs[query_index] = Some(output);
            }
        }

        outputs
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| BackendError::InvalidRequest {
                backend: context.backend,
                task: context.task,
                reason:
                    "Spartan outer remainder row evaluation did not produce every requested slot"
                        .to_owned(),
            })
    }

    fn evaluate_group<F>(
        &self,
        context: CpuKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckSpartanOuterRemainderQuery<F>)],
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
    where
        F: Field,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        let Some(&(_, first)) = queries.first() else {
            return Ok(Vec::new());
        };
        let total_vars = self.log_rows + 1;
        if eq_point.len() != total_vars {
            return context.invalid(format!(
                "Spartan outer remainder equality point has {} variables, expected {total_vars}",
                eq_point.len()
            ));
        }
        let fixed_len = first.fixed_prefix.len();
        if fixed_len == 0 {
            return context.invalid("Spartan outer remainder queries must fix the stream variable");
        }

        for &(query_index, query) in queries {
            validate_spartan_outer_remainder_query(context, self.log_rows, query)
                .map_err(|error| with_query_context(error, query_index))?;
            if query.uniskip_challenge != first.uniskip_challenge
                || query.uniskip_domain_size != first.uniskip_domain_size
                || query.fixed_prefix.len() != fixed_len
                || query.suffix_vars != first.suffix_vars
            {
                return Err(with_query_context(
                    BackendError::InvalidRequest {
                        backend: context.backend,
                        task: context.task,
                        reason:
                            "Spartan outer remainder row queries do not share the reusable shape"
                                .to_owned(),
                    },
                    query_index,
                ));
            }
            if fixed_len > 1
                && query.fixed_prefix[..fixed_len - 1] != first.fixed_prefix[..fixed_len - 1]
            {
                return Err(with_query_context(
                    BackendError::InvalidRequest {
                        backend: context.backend,
                        task: context.task,
                        reason:
                            "Spartan outer remainder row queries do not share the reusable prefix"
                                .to_owned(),
                    },
                    query_index,
                ));
            }
        }

        let lagrange = spartan_outer_lagrange_weights_for_domain(
            context,
            first.uniskip_challenge,
            first.uniskip_domain_size,
        )?;
        if fixed_len == 1 {
            return self.evaluate_stream_round_group(context, eq_point, queries, &lagrange);
        }
        self.evaluate_bound_cycle_group(context, eq_point, queries, &lagrange)
    }

    fn evaluate_stream_round_group<F>(
        &self,
        context: CpuKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckSpartanOuterRemainderQuery<F>)],
        lagrange: &[F],
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
    where
        F: Field,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        let stream_challenge = eq_point[self.log_rows];
        let compiled = queries
            .iter()
            .map(|&(query_index, query)| {
                let stream = query.fixed_prefix[0];
                Ok(SpartanOuterRemainderPointQuery {
                    query_index,
                    slot: query.slot,
                    point: stream,
                    scale: query.scale * eq_factor(stream_challenge, stream),
                })
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let eq_tensor = TensorEqTable::<F>::new(&eq_point[..self.log_rows]);
        if eq_tensor.len() != self.rows.len() {
            return context.invalid(format!(
                "Spartan outer remainder equality tensor has {} rows, expected {}",
                eq_tensor.len(),
                self.rows.len()
            ));
        }

        let values = eq_tensor.par_fold_out_in(
            || SpartanOuterRemainderAccumulator::new(compiled.len()),
            |inner: &mut SpartanOuterRemainderAccumulator<F>, row_index, _x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                let row = &self.rows[row_index];
                let (first_left, first_right) = spartan_outer_first_group_forms(row, lagrange);
                let (second_left, second_right) = spartan_outer_second_group_forms(row, lagrange);
                for (query_index, query) in compiled.iter().enumerate() {
                    if query.scale.is_zero() {
                        continue;
                    }
                    let one_minus_point = F::one() - query.point;
                    let left = one_minus_point * first_left + query.point * second_left;
                    let right = one_minus_point * first_right + query.point * second_right;
                    inner.totals[query_index].fmadd(e_in * query.scale, left * right);
                }
            },
            |_x_out, e_out, inner: SpartanOuterRemainderAccumulator<F>| inner.scale(e_out),
            SpartanOuterRemainderAccumulator::merge,
        );

        Ok(values
            .into_values()
            .zip(compiled)
            .map(|(value, query)| {
                (
                    query.query_index,
                    SumcheckLinearProductOutput::new(query.slot, value),
                )
            })
            .collect())
    }

    fn evaluate_bound_cycle_group<F>(
        &self,
        context: CpuKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckSpartanOuterRemainderQuery<F>)],
        lagrange: &[F],
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
    where
        F: Field,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        let first = queries[0].1;
        let fixed_len = first.fixed_prefix.len();
        let stream = first.fixed_prefix[0];
        let stream_eq = eq_factor(eq_point[self.log_rows], stream);
        if fixed_len == 2 {
            return self.evaluate_single_cycle_group(
                context, eq_point, queries, lagrange, stream, stream_eq,
            );
        }

        let mut left = vec![F::zero(); self.rows.len()];
        let mut right = vec![F::zero(); self.rows.len()];
        left.par_iter_mut()
            .zip(right.par_iter_mut())
            .enumerate()
            .for_each(|(row_index, (left, right))| {
                let (left_value, right_value) =
                    spartan_outer_remainder_forms(&self.rows[row_index], lagrange, stream);
                *left = left_value;
                *right = right_value;
            });

        let mut active_len = self.rows.len();
        let mut shared_fixed_eq = F::one();
        for (fixed_index, &value) in first.fixed_prefix[1..fixed_len - 1].iter().enumerate() {
            let challenge = eq_point[self.log_rows - fixed_index - 1];
            shared_fixed_eq *= eq_factor(challenge, value);
            active_len = bind_low_variable_in_place(&mut left, active_len, value);
            let right_active_len = bind_low_variable_in_place(&mut right, active_len * 2, value);
            debug_assert_eq!(right_active_len, active_len);
        }

        let remaining_vars = self.log_rows + 1 - fixed_len;
        let eq_tensor = TensorEqTable::<F>::new(&eq_point[..remaining_vars]);
        if eq_tensor.len() != active_len / 2 {
            return context.invalid(format!(
                "Spartan outer remainder bound equality tensor has {} rows, expected {}",
                eq_tensor.len(),
                active_len / 2
            ));
        }

        let last_fixed_index = fixed_len - 2;
        let last_challenge = eq_point[self.log_rows - last_fixed_index - 1];
        queries
            .iter()
            .map(|&(query_index, query)| {
                let last_value = query.fixed_prefix[fixed_len - 1];
                let query_scale = query.scale
                    * stream_eq
                    * shared_fixed_eq
                    * eq_factor(last_challenge, last_value);
                if query_scale.is_zero() {
                    return Ok((
                        query_index,
                        SumcheckLinearProductOutput::new(query.slot, F::zero()),
                    ));
                }
                let value = sum_bound_pairs(
                    &eq_tensor,
                    &left[..active_len],
                    &right[..active_len],
                    last_value,
                );
                Ok((
                    query_index,
                    SumcheckLinearProductOutput::new(query.slot, query_scale * value),
                ))
            })
            .collect()
    }

    fn evaluate_single_cycle_group<F>(
        &self,
        context: CpuKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckSpartanOuterRemainderQuery<F>)],
        lagrange: &[F],
        stream: F,
        stream_eq: F,
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
    where
        F: Field,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        let remaining_vars = self.log_rows - 1;
        let eq_tensor = TensorEqTable::<F>::new(&eq_point[..remaining_vars]);
        if eq_tensor.len() * 2 != self.rows.len() {
            return context.invalid(format!(
                "Spartan outer remainder single-cycle equality tensor has {} row pairs, expected {}",
                eq_tensor.len(),
                self.rows.len() / 2,
            ));
        }
        let last_challenge = eq_point[self.log_rows - 1];
        let compiled = queries
            .iter()
            .map(|&(query_index, query)| SpartanOuterRemainderPointQuery {
                query_index,
                slot: query.slot,
                point: query.fixed_prefix[1],
                scale: query.scale * stream_eq * eq_factor(last_challenge, query.fixed_prefix[1]),
            })
            .collect::<Vec<_>>();

        let values = eq_tensor.par_fold_out_in(
            || SpartanOuterRemainderAccumulator::new(compiled.len()),
            |inner: &mut SpartanOuterRemainderAccumulator<F>, row_index, _x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                let low_index = 2 * row_index;
                let high_index = low_index + 1;
                let (low_left, low_right) =
                    spartan_outer_remainder_forms(&self.rows[low_index], lagrange, stream);
                let (high_left, high_right) =
                    spartan_outer_remainder_forms(&self.rows[high_index], lagrange, stream);
                for (query_index, query) in compiled.iter().enumerate() {
                    if query.scale.is_zero() {
                        continue;
                    }
                    let one_minus_point = F::one() - query.point;
                    let left = one_minus_point * low_left + query.point * high_left;
                    let right = one_minus_point * low_right + query.point * high_right;
                    inner.totals[query_index].fmadd(e_in * query.scale, left * right);
                }
            },
            |_x_out, e_out, inner: SpartanOuterRemainderAccumulator<F>| inner.scale(e_out),
            SpartanOuterRemainderAccumulator::merge,
        );

        Ok(values
            .into_values()
            .zip(compiled)
            .map(|(value, query)| {
                (
                    query.query_index,
                    SumcheckLinearProductOutput::new(query.slot, value),
                )
            })
            .collect())
    }
}

struct SpartanOuterRemainderPointQuery<F: Field> {
    query_index: usize,
    slot: BackendValueSlot,
    point: F,
    scale: F,
}

struct SpartanOuterRemainderAccumulator<F: Field>
where
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    totals: Vec<<F as WithAccumulator>::Accumulator>,
}

impl<F> SpartanOuterRemainderAccumulator<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn new(query_count: usize) -> Self {
        Self {
            totals: vec![<F as WithAccumulator>::Accumulator::default(); query_count],
        }
    }

    fn scale(self, scale: F) -> Self {
        let mut scaled = Self::new(self.totals.len());
        if !scale.is_zero() {
            for (out, total) in scaled.totals.iter_mut().zip(self.totals) {
                out.fmadd(scale, total.reduce());
            }
        }
        scaled
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.totals.iter_mut().zip(other.totals) {
            left.merge(right);
        }
        self
    }

    fn into_values(self) -> impl Iterator<Item = F> {
        self.totals.into_iter().map(|total| total.reduce())
    }
}

fn validate_spartan_outer_remainder_query<F: Field>(
    context: CpuKernelContext,
    log_rows: usize,
    query: &SumcheckSpartanOuterRemainderQuery<F>,
) -> Result<(), BackendError> {
    let total_vars = log_rows + 1;
    if query.eq_point.len() != total_vars {
        return context.invalid(format!(
            "Spartan outer remainder equality point has {} variables, expected {total_vars}",
            query.eq_point.len()
        ));
    }
    if query.fixed_prefix.len() + query.suffix_vars != total_vars {
        return context.invalid(format!(
            "Spartan outer remainder query fixes {} variables and leaves {} suffix variables, expected {total_vars} total variables",
            query.fixed_prefix.len(),
            query.suffix_vars,
        ));
    }
    validate_supported_uniskip_domain(context, query.uniskip_domain_size)?;
    Ok(())
}

fn validate_supported_uniskip_domain(
    context: CpuKernelContext,
    domain_size: usize,
) -> Result<(), BackendError> {
    if domain_size < BASE_FIRST_GROUP_TERMS {
        return context.invalid(format!(
            "Spartan outer uniskip domain size {domain_size} cannot cover {BASE_FIRST_GROUP_TERMS} base first-group rows",
        ));
    }
    Ok(())
}

fn spartan_outer_lagrange_weights_for_domain<F: Field>(
    context: CpuKernelContext,
    uniskip: F,
    domain_size: usize,
) -> Result<Vec<F>, BackendError> {
    validate_supported_uniskip_domain(context, domain_size)?;
    centered_lagrange_evals(domain_size, uniskip).map_err(|error| BackendError::InvalidRequest {
        backend: context.backend,
        task: context.task,
        reason: error.to_string(),
    })
}

fn spartan_outer_remainder_forms<F: Field>(
    row: &SumcheckSpartanOuterRow,
    lagrange: &[F],
    stream: F,
) -> (F, F) {
    let (first_left, first_right) = spartan_outer_first_group_forms(row, lagrange);
    let (second_left, second_right) = spartan_outer_second_group_forms(row, lagrange);
    let one_minus_stream = F::one() - stream;
    (
        one_minus_stream * first_left + stream * second_left,
        one_minus_stream * first_right + stream * second_right,
    )
}

fn spartan_outer_first_group_forms<F: Field>(
    row: &SumcheckSpartanOuterRow,
    lagrange: &[F],
) -> (F, F) {
    let mut left = F::zero();
    let mut right = F::zero();
    add_form_term(
        &mut left,
        &mut right,
        lagrange[0],
        F::one() - F::from_bool(row.flag_load) - F::from_bool(row.flag_store),
        F::from_u64(row.ram_address),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[1],
        F::from_bool(row.flag_load),
        F::from_u64(row.ram_read_value) - F::from_u64(row.ram_write_value),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[2],
        F::from_bool(row.flag_load),
        F::from_u64(row.ram_read_value) - F::from_u64(row.rd_write_value),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[3],
        F::from_bool(row.flag_store),
        F::from_u64(row.rs2_value) - F::from_u64(row.ram_write_value),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[4],
        F::from_bool(row.flag_add_operands)
            + F::from_bool(row.flag_subtract_operands)
            + F::from_bool(row.flag_multiply_operands),
        F::from_u64(row.left_lookup_operand),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[5],
        F::one()
            - F::from_bool(row.flag_add_operands)
            - F::from_bool(row.flag_subtract_operands)
            - F::from_bool(row.flag_multiply_operands),
        F::from_u64(row.left_lookup_operand) - F::from_u64(row.left_instruction_input),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[6],
        F::from_bool(row.flag_assert),
        F::from_u64(row.lookup_output) - F::one(),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[7],
        F::from_bool(row.should_jump),
        F::from_u64(row.next_unexpanded_pc) - F::from_u64(row.lookup_output),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[8],
        F::from_bool(row.flag_virtual_instruction) - F::from_bool(row.flag_is_last_in_sequence),
        F::from_u64(row.next_pc) - F::from_u64(row.pc) - F::one(),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[9],
        F::from_bool(row.next_is_virtual) - F::from_bool(row.next_is_first_in_sequence),
        F::one() - F::from_bool(row.flag_do_not_update_unexpanded_pc),
    );
    if let Some(&finv_weight) = lagrange.get(13) {
        right -= finv_weight;
    }
    (left, right)
}

fn spartan_outer_second_group_forms<F: Field>(
    row: &SumcheckSpartanOuterRow,
    lagrange: &[F],
) -> (F, F) {
    let mut left = F::zero();
    let mut right = F::zero();
    add_form_term(
        &mut left,
        &mut right,
        lagrange[0],
        F::from_bool(row.flag_load) + F::from_bool(row.flag_store),
        F::from_u64(row.ram_address) - F::from_u64(row.rs1_value) - F::from_i128(row.imm),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[1],
        F::from_bool(row.flag_add_operands),
        F::from_u128(row.right_lookup_operand)
            - F::from_u64(row.left_instruction_input)
            - F::from_i128(row.right_instruction_input),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[2],
        F::from_bool(row.flag_subtract_operands),
        F::from_u128(row.right_lookup_operand) - F::from_u64(row.left_instruction_input)
            + F::from_i128(row.right_instruction_input)
            - F::from_u128(1u128 << 64),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[3],
        F::from_bool(row.flag_multiply_operands),
        F::from_u128(row.right_lookup_operand) - signed_product_field::<F>(row),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[4],
        F::one()
            - F::from_bool(row.flag_add_operands)
            - F::from_bool(row.flag_subtract_operands)
            - F::from_bool(row.flag_multiply_operands)
            - F::from_bool(row.flag_advice),
        F::from_u128(row.right_lookup_operand) - F::from_i128(row.right_instruction_input),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[5],
        F::from_bool(row.flag_write_lookup_output_to_rd),
        F::from_u64(row.rd_write_value) - F::from_u64(row.lookup_output),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[6],
        F::from_bool(row.flag_jump),
        F::from_u64(row.rd_write_value) - F::from_u64(row.unexpanded_pc) - F::from_u64(4)
            + F::from_u64(2) * F::from_bool(row.flag_is_compressed),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[7],
        F::from_bool(row.should_branch),
        F::from_u64(row.next_unexpanded_pc)
            - F::from_u64(row.unexpanded_pc)
            - F::from_i128(row.imm),
    );
    add_form_term(
        &mut left,
        &mut right,
        lagrange[8],
        F::one() - F::from_bool(row.should_branch) - F::from_bool(row.flag_jump),
        F::from_u64(row.next_unexpanded_pc) - F::from_u64(row.unexpanded_pc) - F::from_u64(4)
            + F::from_u64(4) * F::from_bool(row.flag_do_not_update_unexpanded_pc)
            + F::from_u64(2) * F::from_bool(row.flag_is_compressed),
    );
    if let Some(&load_from_x_weight) = lagrange.get(10) {
        right -= load_from_x_weight * F::from_u64(row.rs1_value);
    }
    if let Some(&store_to_x_weight) = lagrange.get(11) {
        right += store_to_x_weight * F::from_u64(row.rd_write_value);
    }
    if let Some(&load_imm_weight) = lagrange.get(12) {
        right -= load_imm_weight * F::from_i128(row.imm);
    }
    (left, right)
}

fn add_form_term<F: Field>(left: &mut F, right: &mut F, weight: F, a: F, b: F) {
    if weight.is_zero() {
        return;
    }
    *left += weight * a;
    *right += weight * b;
}

fn signed_product_field<F: Field>(row: &SumcheckSpartanOuterRow) -> F {
    let magnitude = F::from_u128(row.product_magnitude);
    if row.product_is_positive {
        magnitude
    } else {
        -magnitude
    }
}

struct SpartanOuterFirstGroupTerms {
    guards: [bool; BASE_FIRST_GROUP_TERMS],
    values: [i128; BASE_FIRST_GROUP_TERMS],
}

impl SpartanOuterFirstGroupTerms {
    #[inline(always)]
    fn new(row: &SumcheckSpartanOuterRow) -> Self {
        let not_load_store = !(row.flag_load || row.flag_store);
        let add_sub_mul =
            row.flag_add_operands || row.flag_subtract_operands || row.flag_multiply_operands;
        let not_add_sub_mul = !add_sub_mul;
        Self {
            guards: [
                not_load_store,
                row.flag_load,
                row.flag_load,
                row.flag_store,
                add_sub_mul,
                not_add_sub_mul,
                row.flag_assert,
                row.should_jump,
                row.flag_virtual_instruction && !row.flag_is_last_in_sequence,
                row.next_is_virtual && !row.next_is_first_in_sequence,
            ],
            values: [
                i128::from(row.ram_address),
                i128::from(row.ram_read_value) - i128::from(row.ram_write_value),
                i128::from(row.ram_read_value) - i128::from(row.rd_write_value),
                i128::from(row.rs2_value) - i128::from(row.ram_write_value),
                i128::from(row.left_lookup_operand),
                i128::from(row.left_lookup_operand) - i128::from(row.left_instruction_input),
                i128::from(row.lookup_output) - 1,
                i128::from(row.next_unexpanded_pc) - i128::from(row.lookup_output),
                i128::from(row.next_pc) - i128::from(row.pc) - 1,
                i128::from(!row.flag_do_not_update_unexpanded_pc),
            ],
        }
    }

    #[inline(always)]
    fn product<F: Field>(&self, coeffs: &[i32]) -> F {
        let mut az = 0i64;
        let mut bz = 0i128;
        for ((&guard, &value), &coeff) in self.guards.iter().zip(&self.values).zip(coeffs) {
            if guard {
                az += i64::from(coeff);
            } else {
                bz += i128::from(coeff) * value;
            }
        }
        if let Some(&finv_coeff) = coeffs.get(13) {
            bz -= i128::from(finv_coeff);
        }
        signed_192_to_field(S64::from_i64(az).mul_trunc::<2, 3>(&S128::from_i128(bz)))
    }
}

struct SpartanOuterSecondGroupTerms {
    guards: [bool; BASE_SECOND_GROUP_TERMS],
    values: [S160; BASE_SECOND_GROUP_TERMS],
    inactive_field_values: [S160; 4],
}

impl SpartanOuterSecondGroupTerms {
    #[inline(always)]
    fn new(row: &SumcheckSpartanOuterRow) -> Self {
        let not_add_sub_mul_advice = !(row.flag_add_operands
            || row.flag_subtract_operands
            || row.flag_multiply_operands
            || row.flag_advice);
        Self {
            guards: [
                row.flag_load || row.flag_store,
                row.flag_add_operands,
                row.flag_subtract_operands,
                row.flag_multiply_operands,
                not_add_sub_mul_advice,
                row.flag_write_lookup_output_to_rd,
                row.flag_jump,
                row.should_branch,
                !row.flag_jump && !row.should_branch,
            ],
            values: [
                S160::from(i128::from(row.ram_address) - i128::from(row.rs1_value) - row.imm),
                S160::from_u128_minus_i128(
                    row.right_lookup_operand,
                    i128::from(row.left_instruction_input) + row.right_instruction_input,
                ),
                S160::from_u128_minus_i128(
                    row.right_lookup_operand,
                    i128::from(row.left_instruction_input) - row.right_instruction_input
                        + (1i128 << 64),
                ),
                if row.product_is_positive {
                    S160::from_diff_u128(row.right_lookup_operand, row.product_magnitude)
                } else {
                    S160::from_sum_u128(row.right_lookup_operand, row.product_magnitude)
                },
                S160::from_u128_minus_i128(row.right_lookup_operand, row.right_instruction_input),
                S160::from_diff_u64(row.rd_write_value, row.lookup_output),
                S160::from(
                    i128::from(row.rd_write_value) - i128::from(row.unexpanded_pc) - 4
                        + if row.flag_is_compressed { 2 } else { 0 },
                ),
                S160::from(
                    i128::from(row.next_unexpanded_pc) - i128::from(row.unexpanded_pc) - row.imm,
                ),
                S160::from(
                    i128::from(row.next_unexpanded_pc) - i128::from(row.unexpanded_pc) - 4
                        + if row.flag_do_not_update_unexpanded_pc {
                            4
                        } else {
                            0
                        }
                        + if row.flag_is_compressed { 2 } else { 0 },
                ),
            ],
            inactive_field_values: [
                S160::zero(),
                S160::from(-i128::from(row.rs1_value)),
                S160::from(i128::from(row.rd_write_value)),
                S160::from(-row.imm),
            ],
        }
    }

    #[inline(always)]
    fn product<F: Field>(&self, coeffs: &[i32]) -> F {
        let mut az = 0i64;
        let mut bz = S160::zero();
        for ((&guard, value), &coeff) in self.guards.iter().zip(&self.values).zip(coeffs) {
            if guard {
                az += i64::from(coeff);
            } else {
                bz += *value * S160::from(i64::from(coeff));
            }
        }
        for (value, &coeff) in self
            .inactive_field_values
            .iter()
            .zip(coeffs.iter().skip(BASE_SECOND_GROUP_TERMS))
        {
            bz += *value * S160::from(i64::from(coeff));
        }
        signed_192_to_field(S64::from_i64(az).mul_trunc::<3, 3>(&bz.to_signed_bigint_nplus1::<3>()))
    }
}

#[inline(always)]
fn signed_192_to_field<F: Field>(value: S192) -> F {
    let limbs = value.magnitude_limbs();
    if limbs[2] == 0 {
        let magnitude = ((limbs[1] as u128) << 64) | limbs[0] as u128;
        let field_value = F::from_u128(magnitude);
        return if value.is_positive {
            field_value
        } else {
            -field_value
        };
    }

    let mut bytes = [0u8; 24];
    for (index, limb) in limbs.iter().copied().enumerate() {
        bytes[index * 8..(index + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let magnitude = F::from_le_bytes_mod_order(&bytes);
    if value.is_positive {
        magnitude
    } else {
        -magnitude
    }
}

fn evaluate_full_hypercube_group<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    eq_point: &[F],
    queries: &[(usize, &SumcheckPrefixProductSumQuery<F>)],
) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
where
    F: Field,
{
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let total_vars = kernel.log_rows() + 1;
    if eq_point.len() != total_vars {
        return context.invalid(format!(
            "prefix product equality point has {} variables, expected {total_vars}",
            eq_point.len()
        ));
    }
    for &(query_index, query) in queries {
        if query.suffix_vars != total_vars {
            return Err(with_query_context(
                BackendError::InvalidRequest {
                    backend: context.backend,
                    task: context.task,
                    reason: format!(
                        "full-hypercube query leaves {} suffix variables, expected {total_vars}",
                        query.suffix_vars
                    ),
                },
                query_index,
            ));
        }
    }

    let compiled = queries
        .iter()
        .map(|&(query_index, query)| {
            FullHypercubeQuery::new(context, kernel, query_index, query)
                .map_err(|error| with_query_context(error, query_index))
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    let stream_eq_zero = F::one() - eq_point[kernel.log_rows()];
    let stream_eq_one = eq_point[kernel.log_rows()];
    let eq_tensor = TensorEqTable::<F>::new(&eq_point[..kernel.log_rows()]);
    if eq_tensor.len() != kernel.rows() {
        return context.invalid(format!(
            "cycle equality tensor has {} rows, expected {}",
            eq_tensor.len(),
            kernel.rows()
        ));
    }

    let values = eq_tensor.par_fold_out_in(
        || PrefixProductGroupAccumulator::new(compiled.len(), kernel.sparse_row_count()),
        |inner, row_index, _x_in, e_in| {
            if e_in.is_zero() {
                return;
            }
            kernel.evaluate_sparse_rows_at_boolean_row(
                row_index,
                &mut inner.left_rows,
                &mut inner.right_rows,
            );
            for (total, query) in inner.totals.iter_mut().zip(&compiled) {
                let zero = weighted_product(
                    &query.row_weights_at_zero,
                    &inner.left_rows,
                    &inner.right_rows,
                );
                let one = weighted_product(
                    &query.row_weights_at_one,
                    &inner.left_rows,
                    &inner.right_rows,
                );
                *total += e_in * query.scale * (stream_eq_zero * zero + stream_eq_one * one);
            }
        },
        |_x_out, e_out, inner| inner.scale(e_out),
        PrefixProductGroupAccumulator::merge,
    );

    Ok(values
        .totals
        .into_iter()
        .zip(compiled)
        .map(|(value, query)| {
            (
                query.query_index,
                SumcheckLinearProductOutput::new(query.slot, value),
            )
        })
        .collect())
}

fn evaluate_fixed_stream_group<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    eq_point: &[F],
    queries: &[(usize, &SumcheckPrefixProductSumQuery<F>)],
) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
where
    F: Field,
{
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let total_vars = kernel.log_rows() + 1;
    if eq_point.len() != total_vars {
        return context.invalid(format!(
            "prefix product equality point has {} variables, expected {total_vars}",
            eq_point.len()
        ));
    }
    let stream_challenge = eq_point[kernel.log_rows()];
    let compiled = queries
        .iter()
        .map(|&(query_index, query)| {
            FixedStreamQuery::new(
                context,
                kernel,
                query_index,
                query,
                stream_challenge,
                total_vars,
            )
            .map_err(|error| with_query_context(error, query_index))
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    let eq_tensor = TensorEqTable::<F>::new(&eq_point[..kernel.log_rows()]);
    if eq_tensor.len() != kernel.rows() {
        return context.invalid(format!(
            "cycle equality tensor has {} rows, expected {}",
            eq_tensor.len(),
            kernel.rows()
        ));
    }

    let values = eq_tensor.par_fold_out_in(
        || PrefixProductGroupAccumulator::new(compiled.len(), kernel.sparse_row_count()),
        |inner, row_index, _x_in, e_in| {
            if e_in.is_zero() {
                return;
            }
            kernel.evaluate_sparse_rows_at_boolean_row(
                row_index,
                &mut inner.left_rows,
                &mut inner.right_rows,
            );
            for (total, query) in inner.totals.iter_mut().zip(&compiled) {
                let product =
                    weighted_product(&query.row_weights, &inner.left_rows, &inner.right_rows);
                *total += e_in * query.scale * product;
            }
        },
        |_x_out, e_out, inner| inner.scale(e_out),
        PrefixProductGroupAccumulator::merge,
    );

    Ok(values
        .totals
        .into_iter()
        .zip(compiled)
        .map(|(value, query)| {
            (
                query.query_index,
                SumcheckLinearProductOutput::new(query.slot, value),
            )
        })
        .collect())
}

struct FullHypercubeQuery<F: Field> {
    query_index: usize,
    slot: BackendValueSlot,
    scale: F,
    row_weights_at_zero: Vec<(usize, F)>,
    row_weights_at_one: Vec<(usize, F)>,
}

impl<F: Field> FullHypercubeQuery<F> {
    fn new(
        context: CpuKernelContext,
        kernel: &SparseProductKernel<'_, F>,
        query_index: usize,
        query: &SumcheckPrefixProductSumQuery<F>,
    ) -> Result<Self, BackendError> {
        if query.row_weights_at_zero.len() != query.row_weights_at_one.len() {
            return context.invalid(format!(
                "row weight selector lengths differ: zero has {}, one has {}",
                query.row_weights_at_zero.len(),
                query.row_weights_at_one.len()
            ));
        }
        if query.row_weights_at_zero.len() != kernel.sparse_row_count() {
            return context.invalid(format!(
                "query has {} row weights, expected {}",
                query.row_weights_at_zero.len(),
                kernel.sparse_row_count()
            ));
        }
        Ok(Self {
            query_index,
            slot: query.slot,
            scale: query.scale,
            row_weights_at_zero: nonzero_row_weights(&query.row_weights_at_zero),
            row_weights_at_one: nonzero_row_weights(&query.row_weights_at_one),
        })
    }
}

struct FixedStreamQuery<F: Field> {
    query_index: usize,
    slot: BackendValueSlot,
    scale: F,
    row_weights: Vec<(usize, F)>,
}

impl<F: Field> FixedStreamQuery<F> {
    fn new(
        context: CpuKernelContext,
        kernel: &SparseProductKernel<'_, F>,
        query_index: usize,
        query: &SumcheckPrefixProductSumQuery<F>,
        stream_challenge: F,
        total_vars: usize,
    ) -> Result<Self, BackendError> {
        if query.suffix_vars != kernel.log_rows() {
            return context.invalid(format!(
                "fixed-stream query leaves {} suffix variables, expected {}",
                query.suffix_vars,
                kernel.log_rows()
            ));
        }
        if query.fixed_prefix.len() != 1 {
            return context.invalid(format!(
                "fixed-stream query fixes {} variables, expected 1",
                query.fixed_prefix.len()
            ));
        }
        if query.eq_point.len() != total_vars {
            return context.invalid(format!(
                "prefix product equality point has {} variables, expected {total_vars}",
                query.eq_point.len()
            ));
        }
        if query.row_weights_at_zero.len() != query.row_weights_at_one.len() {
            return context.invalid(format!(
                "row weight selector lengths differ: zero has {}, one has {}",
                query.row_weights_at_zero.len(),
                query.row_weights_at_one.len()
            ));
        }
        if query.row_weights_at_zero.len() != kernel.sparse_row_count() {
            return context.invalid(format!(
                "query has {} row weights, expected {}",
                query.row_weights_at_zero.len(),
                kernel.sparse_row_count()
            ));
        }
        let stream = query.fixed_prefix[0];
        let stream_eq =
            stream_challenge * stream + (F::one() - stream_challenge) * (F::one() - stream);
        Ok(Self {
            query_index,
            slot: query.slot,
            scale: query.scale * stream_eq,
            row_weights: nonzero_row_weights(&blend_row_weights(
                stream,
                &query.row_weights_at_zero,
                &query.row_weights_at_one,
            )),
        })
    }
}

fn nonzero_row_weights<F: Field>(weights: &[F]) -> Vec<(usize, F)> {
    weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| !weight.is_zero())
        .collect()
}

fn weighted_product<F: Field>(weights: &[(usize, F)], left_rows: &[F], right_rows: &[F]) -> F {
    let (left, right) = weighted_forms(weights, left_rows, right_rows);
    left * right
}

fn weighted_forms<F: Field>(weights: &[(usize, F)], left_rows: &[F], right_rows: &[F]) -> (F, F) {
    let mut left = F::zero();
    let mut right = F::zero();
    for &(index, weight) in weights {
        left += weight * left_rows[index];
        right += weight * right_rows[index];
    }
    (left, right)
}

struct PrefixProductGroupAccumulator<F: Field> {
    totals: Vec<F>,
    left_rows: Vec<F>,
    right_rows: Vec<F>,
}

impl<F: Field> PrefixProductGroupAccumulator<F> {
    fn new(len: usize, row_count: usize) -> Self {
        Self {
            totals: vec![F::zero(); len],
            left_rows: vec![F::zero(); row_count],
            right_rows: vec![F::zero(); row_count],
        }
    }

    fn scale(mut self, scale: F) -> Self {
        if scale.is_zero() {
            self.totals.fill(F::zero());
        } else {
            for value in &mut self.totals {
                *value *= scale;
            }
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.totals.iter_mut().zip(other.totals) {
            *left += right;
        }
        self
    }
}

fn evaluate_prefix_product_sum<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    query: &SumcheckPrefixProductSumQuery<F>,
) -> Result<F, BackendError>
where
    F: Field,
{
    let total_vars = validate_prefix_product_query(context, kernel, query)?;
    if !query.fixed_prefix.is_empty() {
        return evaluate_bound_prefix_product_sum(context, kernel, query);
    }
    if query.suffix_vars >= usize::BITS as usize {
        return context.invalid(format!(
            "suffix variable count {} is too large",
            query.suffix_vars
        ));
    }
    let suffix_count = 1usize << query.suffix_vars;

    (0..suffix_count)
        .into_par_iter()
        .map(|suffix_index| {
            let assignment = PrefixAssignment::new(query, total_vars, suffix_index);
            let stream = assignment.value(0);
            let eq = assignment.reversed_eq(&query.eq_point);
            if eq.is_zero() || query.scale.is_zero() {
                return Ok(F::zero());
            }
            let scale = query.scale * eq;
            if stream.is_zero() {
                evaluate_assignment_product(
                    context,
                    kernel,
                    &assignment,
                    &query.row_weights_at_zero,
                    scale,
                )
            } else if stream.is_one() {
                evaluate_assignment_product(
                    context,
                    kernel,
                    &assignment,
                    &query.row_weights_at_one,
                    scale,
                )
            } else {
                let row_weights = blend_row_weights(
                    stream,
                    &query.row_weights_at_zero,
                    &query.row_weights_at_one,
                );
                evaluate_assignment_product(context, kernel, &assignment, &row_weights, scale)
            }
        })
        .try_reduce(F::zero, |left, right| Ok(left + right))
}

fn validate_prefix_product_query<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    query: &SumcheckPrefixProductSumQuery<F>,
) -> Result<usize, BackendError>
where
    F: Field,
{
    let total_vars = kernel.log_rows() + 1;
    if query.fixed_prefix.len() + query.suffix_vars != total_vars {
        return context.invalid(format!(
            "prefix product query fixes {} variables and leaves {} suffix variables, expected {total_vars} total variables",
            query.fixed_prefix.len(),
            query.suffix_vars
        ));
    }
    if query.eq_point.len() != total_vars {
        return context.invalid(format!(
            "prefix product equality point has {} variables, expected {total_vars}",
            query.eq_point.len()
        ));
    }
    if query.row_weights_at_zero.len() != query.row_weights_at_one.len() {
        return context.invalid(format!(
            "row weight selector lengths differ: zero has {}, one has {}",
            query.row_weights_at_zero.len(),
            query.row_weights_at_one.len()
        ));
    }
    Ok(total_vars)
}

fn can_evaluate_bound_prefix_group<F>(
    queries: &[(usize, &SumcheckPrefixProductSumQuery<F>)],
) -> bool
where
    F: Field,
{
    let Some((_, first)) = queries.first() else {
        return false;
    };
    let fixed_len = first.fixed_prefix.len();
    fixed_len >= 2
        && queries.iter().all(|(_, query)| {
            query.fixed_prefix.len() == fixed_len
                && query.fixed_prefix[..fixed_len - 1] == first.fixed_prefix[..fixed_len - 1]
                && query.suffix_vars == first.suffix_vars
                && query.row_weights_at_zero == first.row_weights_at_zero
                && query.row_weights_at_one == first.row_weights_at_one
        })
}

fn evaluate_bound_prefix_group<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    eq_point: &[F],
    queries: &[(usize, &SumcheckPrefixProductSumQuery<F>)],
) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
where
    F: Field,
{
    let Some(&(_, first)) = queries.first() else {
        return Ok(Vec::new());
    };
    let fixed_len = first.fixed_prefix.len();
    if fixed_len < 2 {
        return context
            .invalid("bound-prefix group must fix stream plus at least one cycle variable");
    }

    for &(query_index, query) in queries {
        let _ = validate_prefix_product_query(context, kernel, query)
            .map_err(|error| with_query_context(error, query_index))?;
        if query.fixed_prefix.len() != fixed_len
            || query.fixed_prefix[..fixed_len - 1] != first.fixed_prefix[..fixed_len - 1]
            || query.row_weights_at_zero != first.row_weights_at_zero
            || query.row_weights_at_one != first.row_weights_at_one
        {
            return Err(with_query_context(
                BackendError::InvalidRequest {
                    backend: context.backend,
                    task: context.task,
                    reason: "bound-prefix group queries do not share the reusable prefix"
                        .to_owned(),
                },
                query_index,
            ));
        }
    }

    if first.row_weights_at_zero.len() != kernel.sparse_row_count() {
        return context.invalid(format!(
            "query has {} row weights, expected {}",
            first.row_weights_at_zero.len(),
            kernel.sparse_row_count()
        ));
    }

    let stream = first.fixed_prefix[0];
    let stream_challenge = eq_point[kernel.log_rows()];
    let stream_eq = eq_factor(stream_challenge, stream);
    let row_weights = blend_row_weights(
        stream,
        &first.row_weights_at_zero,
        &first.row_weights_at_one,
    );
    if fixed_len == 2 {
        return evaluate_single_cycle_bound_prefix_group(
            context,
            kernel,
            eq_point,
            queries,
            &row_weights,
            stream_eq,
        );
    }

    let (mut left, mut right) =
        evaluate_linear_forms_over_boolean_rows(context, kernel, &row_weights)?;

    let mut active_len = kernel.rows();
    let mut shared_fixed_eq = F::one();
    for (fixed_index, &value) in first.fixed_prefix[1..fixed_len - 1].iter().enumerate() {
        let challenge = eq_point[kernel.log_rows() - fixed_index - 1];
        shared_fixed_eq *= eq_factor(challenge, value);
        active_len = bind_low_variable_in_place(&mut left, active_len, value);
        let right_active_len = bind_low_variable_in_place(&mut right, active_len * 2, value);
        debug_assert_eq!(right_active_len, active_len);
    }

    let remaining_vars = kernel.log_rows() + 1 - fixed_len;
    let eq_tensor = TensorEqTable::<F>::new(&eq_point[..remaining_vars]);
    if eq_tensor.len() != active_len / 2 {
        return context.invalid(format!(
            "bound-prefix group equality tensor has {} rows, expected {}",
            eq_tensor.len(),
            active_len / 2
        ));
    }

    let last_fixed_index = fixed_len - 2;
    let last_challenge = eq_point[kernel.log_rows() - last_fixed_index - 1];
    queries
        .iter()
        .map(|&(query_index, query)| {
            let last_value = query.fixed_prefix[fixed_len - 1];
            let query_scale =
                query.scale * stream_eq * shared_fixed_eq * eq_factor(last_challenge, last_value);
            if query_scale.is_zero() {
                return Ok((
                    query_index,
                    SumcheckLinearProductOutput::new(query.slot, F::zero()),
                ));
            }

            let value = sum_bound_pairs(
                &eq_tensor,
                &left[..active_len],
                &right[..active_len],
                last_value,
            );
            Ok((
                query_index,
                SumcheckLinearProductOutput::new(query.slot, query_scale * value),
            ))
        })
        .collect()
}

struct SingleCycleBoundPrefixQuery<F: Field> {
    query_index: usize,
    slot: BackendValueSlot,
    point: F,
    scale: F,
}

struct SingleCycleBoundPrefixAccumulator<F: Field> {
    totals: Vec<F>,
}

impl<F: Field> SingleCycleBoundPrefixAccumulator<F> {
    fn new(query_count: usize) -> Self {
        Self {
            totals: vec![F::zero(); query_count],
        }
    }

    fn scale(mut self, scale: F) -> Self {
        if scale.is_zero() {
            self.totals.fill(F::zero());
        } else {
            for total in &mut self.totals {
                *total *= scale;
            }
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.totals.iter_mut().zip(other.totals) {
            *left += right;
        }
        self
    }
}

fn evaluate_single_cycle_bound_prefix_group<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    eq_point: &[F],
    queries: &[(usize, &SumcheckPrefixProductSumQuery<F>)],
    row_weights: &[F],
    stream_eq: F,
) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
where
    F: Field,
{
    let remaining_vars = kernel.log_rows() - 1;
    let eq_tensor = TensorEqTable::<F>::new(&eq_point[..remaining_vars]);
    if eq_tensor.len() * 2 != kernel.rows() {
        return context.invalid(format!(
            "single-cycle bound-prefix equality tensor has {} row pairs, expected {}",
            eq_tensor.len(),
            kernel.rows() / 2,
        ));
    }
    let last_challenge = eq_point[kernel.log_rows() - 1];
    let compiled = queries
        .iter()
        .map(|&(query_index, query)| SingleCycleBoundPrefixQuery {
            query_index,
            slot: query.slot,
            point: query.fixed_prefix[1],
            scale: query.scale * stream_eq * eq_factor(last_challenge, query.fixed_prefix[1]),
        })
        .collect::<Vec<_>>();
    let (left_form, right_form) = kernel.combine_weighted_forms(context, row_weights)?;

    let values = eq_tensor.par_fold_out_in(
        || SingleCycleBoundPrefixAccumulator::new(compiled.len()),
        |inner, row_index, _x_in, e_in| {
            if e_in.is_zero() {
                return;
            }
            let low_index = 2 * row_index;
            let high_index = low_index + 1;
            let (low_left, low_right) =
                kernel.evaluate_combined_forms_at_boolean_row(&left_form, &right_form, low_index);
            let (high_left, high_right) =
                kernel.evaluate_combined_forms_at_boolean_row(&left_form, &right_form, high_index);
            for (total, query) in inner.totals.iter_mut().zip(&compiled) {
                if query.scale.is_zero() {
                    continue;
                }
                let one_minus_point = F::one() - query.point;
                let left_value = one_minus_point * low_left + query.point * high_left;
                let right_value = one_minus_point * low_right + query.point * high_right;
                *total += e_in * query.scale * left_value * right_value;
            }
        },
        |_x_out, e_out, inner| inner.scale(e_out),
        SingleCycleBoundPrefixAccumulator::merge,
    );

    Ok(values
        .totals
        .into_iter()
        .zip(compiled)
        .map(|(value, query)| {
            (
                query.query_index,
                SumcheckLinearProductOutput::new(query.slot, value),
            )
        })
        .collect())
}

fn evaluate_bound_prefix_product_sum<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    query: &SumcheckPrefixProductSumQuery<F>,
) -> Result<F, BackendError>
where
    F: Field,
{
    if query.row_weights_at_zero.len() != kernel.sparse_row_count() {
        return context.invalid(format!(
            "query has {} row weights, expected {}",
            query.row_weights_at_zero.len(),
            kernel.sparse_row_count()
        ));
    }
    if query.scale.is_zero() {
        return Ok(F::zero());
    }

    let stream = query.fixed_prefix[0];
    let stream_challenge = query.eq_point[kernel.log_rows()];
    let stream_eq = eq_factor(stream_challenge, stream);
    if stream_eq.is_zero() {
        return Ok(F::zero());
    }

    let row_weights = blend_row_weights(
        stream,
        &query.row_weights_at_zero,
        &query.row_weights_at_one,
    );
    let (mut left, mut right) =
        evaluate_linear_forms_over_boolean_rows(context, kernel, &row_weights)?;

    let mut active_len = kernel.rows();
    let mut fixed_eq = F::one();
    for (fixed_index, &value) in query.fixed_prefix[1..].iter().enumerate() {
        let challenge = query.eq_point[kernel.log_rows() - fixed_index - 1];
        fixed_eq *= eq_factor(challenge, value);
        if fixed_eq.is_zero() {
            return Ok(F::zero());
        }
        active_len = bind_low_variable_in_place(&mut left, active_len, value);
        let right_active_len = bind_low_variable_in_place(&mut right, active_len * 2, value);
        debug_assert_eq!(right_active_len, active_len);
    }

    let remaining_vars = kernel.log_rows() + 1 - query.fixed_prefix.len();
    if remaining_vars != query.suffix_vars {
        return context.invalid(format!(
            "bound prefix query leaves {} cycle variables, expected {} suffix variables",
            remaining_vars, query.suffix_vars
        ));
    }
    let eq_tensor = TensorEqTable::<F>::new(&query.eq_point[..remaining_vars]);
    if eq_tensor.len() != active_len {
        return context.invalid(format!(
            "bound prefix equality tensor has {} rows, expected {active_len}",
            eq_tensor.len()
        ));
    }

    let value = sum_bound_rows(&eq_tensor, &left[..active_len], &right);

    Ok(query.scale * stream_eq * fixed_eq * value)
}

fn sum_bound_rows<F: Field>(eq_tensor: &TensorEqTable<F>, left: &[F], right: &[F]) -> F {
    eq_tensor.par_fold_out_in(
        F::zero,
        |inner, row_index, _x_in, e_in| {
            *inner += e_in * left[row_index] * right[row_index];
        },
        |_x_out, e_out, inner| e_out * inner,
        |left, right| left + right,
    )
}

fn sum_bound_pairs<F: Field>(eq_tensor: &TensorEqTable<F>, left: &[F], right: &[F], point: F) -> F {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(eq_tensor.len() * 2, left.len());
    let one_minus_point = F::one() - point;
    eq_tensor.par_fold_out_in(
        F::zero,
        |inner, row_index, _x_in, e_in| {
            let low_index = 2 * row_index;
            let high_index = low_index + 1;
            let left_value = one_minus_point * left[low_index] + point * left[high_index];
            let right_value = one_minus_point * right[low_index] + point * right[high_index];
            *inner += e_in * left_value * right_value;
        },
        |_x_out, e_out, inner| e_out * inner,
        |left, right| left + right,
    )
}

fn sum_bound_pair_endpoints<F>(eq_tensor: &TensorEqTable<F>, left: &[F], right: &[F]) -> (F, F)
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(eq_tensor.len() * 2, left.len());

    let totals = eq_tensor.par_fold_out_in(
        || [<F as WithAccumulator>::Accumulator::default(); 2],
        |inner, row_index, _x_in, e_in| {
            let low_index = 2 * row_index;
            let high_index = low_index + 1;
            let left_low = left[low_index];
            let right_low = right[low_index];
            inner[0].fmadd(e_in, left_low * right_low);
            inner[1].fmadd(
                e_in,
                (left[high_index] - left_low) * (right[high_index] - right_low),
            );
        },
        |_x_out, e_out, inner| {
            let mut out = [<F as WithAccumulator>::Accumulator::default(); 2];
            out[0].fmadd(e_out, inner[0].reduce());
            out[1].fmadd(e_out, inner[1].reduce());
            out
        },
        |mut left, right| {
            left[0].merge(right[0]);
            left[1].merge(right[1]);
            left
        },
    );
    let [q_at_zero, q_at_infinity] = totals.map(|total| total.reduce());
    (q_at_zero, q_at_infinity)
}

fn evaluate_linear_forms_over_boolean_rows<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    row_weights: &[F],
) -> Result<(Vec<F>, Vec<F>), BackendError>
where
    F: Field,
{
    let mut left = vec![F::zero(); kernel.rows()];
    let mut right = vec![F::zero(); kernel.rows()];
    left.par_iter_mut()
        .zip(right.par_iter_mut())
        .enumerate()
        .try_for_each(|(row_index, (left, right))| {
            let (left_value, right_value) =
                kernel.evaluate_linear_forms_at_boolean_row(context, row_index, row_weights)?;
            *left = left_value;
            *right = right_value;
            Ok::<(), BackendError>(())
        })?;
    Ok((left, right))
}

fn evaluate_assignment_product<F>(
    context: CpuKernelContext,
    kernel: &SparseProductKernel<'_, F>,
    assignment: &PrefixAssignment<'_, F>,
    row_weights: &[F],
    scale: F,
) -> Result<F, BackendError>
where
    F: Field,
{
    if assignment.cycle_is_boolean() {
        let row_index = assignment.cycle_row_index();
        kernel.evaluate_linear_product_at_boolean_row(context, row_index, row_weights, scale)
    } else {
        let point = assignment.cycle_point();
        kernel.evaluate_linear_product(context, &point, row_weights, scale)
    }
}

struct PrefixAssignment<'a, F: Field> {
    query: &'a SumcheckPrefixProductSumQuery<F>,
    total_vars: usize,
    suffix_index: usize,
}

impl<'a, F: Field> PrefixAssignment<'a, F> {
    const fn new(
        query: &'a SumcheckPrefixProductSumQuery<F>,
        total_vars: usize,
        suffix_index: usize,
    ) -> Self {
        Self {
            query,
            total_vars,
            suffix_index,
        }
    }

    fn value(&self, position: usize) -> F {
        if position < self.query.fixed_prefix.len() {
            return self.query.fixed_prefix[position];
        }
        let suffix_position = position - self.query.fixed_prefix.len();
        let shift = self.query.suffix_vars - suffix_position - 1;
        F::from_bool(((self.suffix_index >> shift) & 1) == 1)
    }

    fn reversed_eq(&self, eq_point: &[F]) -> F {
        (0..self.total_vars)
            .map(|position| {
                let value = self.value(position);
                let challenge = eq_point[self.total_vars - position - 1];
                challenge * value + (F::one() - challenge) * (F::one() - value)
            })
            .product()
    }

    fn cycle_is_boolean(&self) -> bool {
        (1..self.total_vars).all(|position| {
            let value = self.value(position);
            value.is_zero() || value.is_one()
        })
    }

    fn cycle_row_index(&self) -> usize {
        let mut index = 0usize;
        for position in (1..self.total_vars).rev() {
            index <<= 1;
            if self.value(position).is_one() {
                index |= 1;
            }
        }
        index
    }

    fn cycle_point(&self) -> Vec<F> {
        (1..self.total_vars)
            .rev()
            .map(|position| self.value(position))
            .collect()
    }
}

fn bind_low_variable_in_place<F: Field>(values: &mut [F], active_len: usize, point: F) -> usize {
    debug_assert!(active_len.is_power_of_two());
    let next_len = active_len / 2;
    for index in 0..next_len {
        let lo = values[2 * index];
        let hi = values[2 * index + 1];
        values[index] = lo + point * (hi - lo);
    }
    next_len
}

fn eq_factor<F: Field>(challenge: F, value: F) -> F {
    challenge * value + (F::one() - challenge) * (F::one() - value)
}

fn blend_row_weights<F: Field>(selector: F, zero: &[F], one: &[F]) -> Vec<F> {
    if selector.is_zero() {
        return zero.to_vec();
    }
    if selector.is_one() {
        return one.to_vec();
    }
    zero.iter()
        .zip(one)
        .map(|(&at_zero, &at_one)| at_zero + selector * (at_one - at_zero))
        .collect()
}

fn validate_unique_query_slots(
    backend: &'static str,
    task: &'static str,
    slots: impl Iterator<Item = BackendValueSlot>,
) -> Result<(), BackendError> {
    let mut seen = HashSet::new();
    for slot in slots {
        if !seen.insert(slot) {
            return Err(BackendError::InvalidRequest {
                backend,
                task,
                reason: format!("duplicate value slot {slot:?}"),
            });
        }
    }
    Ok(())
}

fn with_query_context(error: BackendError, query_index: usize) -> BackendError {
    match error {
        BackendError::InvalidRequest {
            backend,
            task,
            reason,
        } => BackendError::InvalidRequest {
            backend,
            task,
            reason: format!("query {query_index}: {reason}"),
        },
        other => other,
    }
}
