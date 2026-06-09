use std::collections::{HashMap, HashSet};

use jolt_field::Field;
use jolt_poly::{boolean_index_msb, EqPolynomial, TensorEqTable};
use rayon::prelude::*;

use crate::{
    BackendError, BackendValueSlot, SumcheckLinearProductOutput, SumcheckLinearProductRequest,
    SumcheckRowProductQuery, SumcheckRowProductRequest,
};

#[derive(Clone, Copy)]
pub(super) struct CpuKernelContext {
    pub(super) backend: &'static str,
    pub(super) task: &'static str,
}

impl CpuKernelContext {
    pub(super) const fn new(backend: &'static str, task: &'static str) -> Self {
        Self { backend, task }
    }

    pub(super) fn invalid<T>(self, reason: impl Into<String>) -> Result<T, BackendError> {
        Err(BackendError::InvalidRequest {
            backend: self.backend,
            task: self.task,
            reason: reason.into(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PowerOfTwoDomain {
    rows: usize,
    log_rows: usize,
}

impl PowerOfTwoDomain {
    fn new(context: CpuKernelContext, rows: usize) -> Result<Self, BackendError> {
        if rows == 0 || !rows.is_power_of_two() {
            return context.invalid(format!(
                "witness polynomial row count {rows} is not a nonzero power of two"
            ));
        }
        Ok(Self {
            rows,
            log_rows: rows.trailing_zeros() as usize,
        })
    }
}

pub(super) struct DenseMleBatch<'a, F: Field> {
    polynomials: &'a [Vec<F>],
    domain: PowerOfTwoDomain,
}

impl<'a, F: Field> DenseMleBatch<'a, F> {
    fn new(context: CpuKernelContext, polynomials: &'a [Vec<F>]) -> Result<Self, BackendError> {
        let Some(first) = polynomials.first() else {
            return context.invalid("request has no witness polynomials");
        };
        let domain = PowerOfTwoDomain::new(context, first.len())?;
        for (index, polynomial) in polynomials.iter().enumerate() {
            if polynomial.len() != domain.rows {
                return context.invalid(format!(
                    "witness polynomial {index} has {} rows, expected {}",
                    polynomial.len(),
                    domain.rows
                ));
            }
        }
        Ok(Self {
            polynomials,
            domain,
        })
    }

    pub(super) const fn log_rows(&self) -> usize {
        self.domain.log_rows
    }

    fn evaluate_all(&self, context: CpuKernelContext, point: &[F]) -> Result<Vec<F>, BackendError> {
        if point.len() != self.domain.log_rows {
            return context.invalid(format!(
                "point has {} variables, expected {}",
                point.len(),
                self.domain.log_rows
            ));
        }
        if let Some(row) = boolean_index_msb(point) {
            return Ok(self
                .polynomials
                .iter()
                .map(|polynomial| polynomial[row])
                .collect());
        }
        let eq_table = EqPolynomial::new(point.to_vec()).evaluations();
        Ok(self
            .polynomials
            .par_iter()
            .map(|polynomial| {
                polynomial
                    .par_iter()
                    .zip(eq_table.par_iter())
                    .map(|(&witness, &eq)| witness * eq)
                    .sum()
            })
            .collect())
    }
}

struct ColumnLayout {
    input_index_by_column: HashMap<usize, usize>,
    constant_column: usize,
}

impl ColumnLayout {
    fn new(
        context: CpuKernelContext,
        input_columns: &[usize],
        constant_column: usize,
        expected_inputs: usize,
    ) -> Result<Self, BackendError> {
        if input_columns.len() != expected_inputs {
            return context.invalid(format!(
                "input column count {} does not match witness polynomial count {expected_inputs}",
                input_columns.len()
            ));
        }
        let mut input_index_by_column = HashMap::with_capacity(input_columns.len());
        let mut seen = HashSet::with_capacity(input_columns.len());
        for (index, &column) in input_columns.iter().enumerate() {
            if column == constant_column {
                return context
                    .invalid(format!("input column {column} aliases the constant column"));
            }
            if !seen.insert(column) {
                return context.invalid(format!("duplicate input column {column}"));
            }
            let _ = input_index_by_column.insert(column, index);
        }
        Ok(Self {
            input_index_by_column,
            constant_column,
        })
    }

    fn compile_term<F: Field>(
        &self,
        context: CpuKernelContext,
        column: usize,
        coefficient: F,
    ) -> Result<CompiledLinearTerm<F>, BackendError> {
        if column == self.constant_column {
            return Ok(CompiledLinearTerm {
                input_index: None,
                coefficient,
            });
        }
        let Some(&input_index) = self.input_index_by_column.get(&column) else {
            return context.invalid(format!("row referenced unsupported input column {column}"));
        };
        Ok(CompiledLinearTerm {
            input_index: Some(input_index),
            coefficient,
        })
    }
}

#[derive(Clone, Copy)]
struct CompiledLinearTerm<F: Field> {
    input_index: Option<usize>,
    coefficient: F,
}

struct CompiledSparseLinearRows<F: Field> {
    rows: Vec<Vec<CompiledLinearTerm<F>>>,
    input_count: usize,
}

impl<F: Field> CompiledSparseLinearRows<F> {
    fn new(
        context: CpuKernelContext,
        layout: &ColumnLayout,
        rows: &[Vec<(usize, F)>],
        input_count: usize,
    ) -> Result<Self, BackendError> {
        let rows = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(column, coefficient)| layout.compile_term(context, column, coefficient))
                    .collect()
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        Ok(Self { rows, input_count })
    }

    fn combine_weighted(
        &self,
        context: CpuKernelContext,
        row_weights: &[F],
    ) -> Result<CombinedLinearForm<F>, BackendError> {
        if row_weights.len() != self.rows.len() {
            return context.invalid(format!(
                "query has {} row weights, expected {}",
                row_weights.len(),
                self.rows.len()
            ));
        }

        let mut coefficients = vec![F::zero(); self.input_count];
        let mut constant = F::zero();
        for (row, &weight) in self.rows.iter().zip(row_weights) {
            if weight.is_zero() {
                continue;
            }
            for term in row {
                let scaled = weight * term.coefficient;
                if let Some(input_index) = term.input_index {
                    coefficients[input_index] += scaled;
                } else {
                    constant += scaled;
                }
            }
        }

        let terms = coefficients
            .into_iter()
            .enumerate()
            .filter_map(|(input_index, coefficient)| {
                (!coefficient.is_zero()).then_some((input_index, coefficient))
            })
            .collect();
        Ok(CombinedLinearForm { constant, terms })
    }

    fn evaluate_weighted_at_evals(
        &self,
        context: CpuKernelContext,
        row_weights: &[F],
        witness_evaluations: &[F],
    ) -> Result<F, BackendError> {
        if row_weights.len() != self.rows.len() {
            return context.invalid(format!(
                "query has {} row weights, expected {}",
                row_weights.len(),
                self.rows.len()
            ));
        }
        if witness_evaluations.len() != self.input_count {
            return context.invalid(format!(
                "query has {} witness evaluations, expected {}",
                witness_evaluations.len(),
                self.input_count
            ));
        }

        Ok(self
            .rows
            .iter()
            .zip(row_weights)
            .filter(|(_, &weight)| !weight.is_zero())
            .map(|(row, &weight)| weight * evaluate_sparse_row_at_evals(row, witness_evaluations))
            .sum())
    }

    fn evaluate_weighted_at_row(
        &self,
        context: CpuKernelContext,
        row_weights: &[F],
        witness_polynomials: &[Vec<F>],
        row_index: usize,
    ) -> Result<F, BackendError> {
        if row_weights.len() != self.rows.len() {
            return context.invalid(format!(
                "query has {} row weights, expected {}",
                row_weights.len(),
                self.rows.len()
            ));
        }

        Ok(self
            .rows
            .iter()
            .zip(row_weights)
            .filter(|(_, &weight)| !weight.is_zero())
            .map(|(row, &weight)| {
                weight * evaluate_sparse_row_at_polynomial_row(row, witness_polynomials, row_index)
            })
            .sum())
    }
}

fn evaluate_sparse_row_at_evals<F: Field>(
    row: &[CompiledLinearTerm<F>],
    witness_evaluations: &[F],
) -> F {
    row.iter().fold(F::zero(), |acc, term| {
        acc + term.coefficient
            * term
                .input_index
                .map_or_else(F::one, |input_index| witness_evaluations[input_index])
    })
}

fn evaluate_sparse_row_at_polynomial_row<F: Field>(
    row: &[CompiledLinearTerm<F>],
    witness_polynomials: &[Vec<F>],
    row_index: usize,
) -> F {
    row.iter().fold(F::zero(), |acc, term| {
        acc + term.coefficient
            * term.input_index.map_or_else(F::one, |input_index| {
                witness_polynomials[input_index][row_index]
            })
    })
}

#[derive(Clone)]
pub(super) struct CombinedLinearForm<F: Field> {
    constant: F,
    terms: Vec<(usize, F)>,
}

impl<F: Field> CombinedLinearForm<F> {
    pub(super) fn evaluate_at_row(&self, witness_polynomials: &[Vec<F>], row_index: usize) -> F {
        self.terms.iter().fold(self.constant, |acc, term| {
            acc + term.1 * witness_polynomials[term.0][row_index]
        })
    }
}

pub(super) struct BatchedLinearForms<F: Field> {
    constants: Vec<F>,
    coeffs_by_input: Vec<Vec<(usize, F)>>,
}

impl<F: Field> BatchedLinearForms<F> {
    pub(super) fn new(forms: &[CombinedLinearForm<F>], input_count: usize) -> Self {
        let mut coeffs_by_input = vec![Vec::new(); input_count];
        for (form_index, form) in forms.iter().enumerate() {
            for &(input_index, coefficient) in &form.terms {
                coeffs_by_input[input_index].push((form_index, coefficient));
            }
        }
        let constants = forms.iter().map(|form| form.constant).collect();
        Self {
            constants,
            coeffs_by_input,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.constants.len()
    }

    pub(super) fn initialize_scratch(&self) -> Vec<F> {
        self.constants.clone()
    }

    pub(super) fn evaluate_at_row_into(
        &self,
        witness_polynomials: &[Vec<F>],
        row_index: usize,
        output: &mut [F],
    ) {
        output.copy_from_slice(&self.constants);
        for (input_index, coefficients) in self.coeffs_by_input.iter().enumerate() {
            if coefficients.is_empty() {
                continue;
            }
            let value = witness_polynomials[input_index][row_index];
            for &(form_index, coefficient) in coefficients {
                output[form_index] += coefficient * value;
            }
        }
    }
}

struct RowProductGroupAccumulator<F: Field> {
    totals: Vec<F>,
    left: Vec<F>,
    right: Vec<F>,
}

impl<F: Field> RowProductGroupAccumulator<F> {
    fn new(left_forms: &BatchedLinearForms<F>, right_forms: &BatchedLinearForms<F>) -> Self {
        Self {
            totals: vec![F::zero(); left_forms.len()],
            left: left_forms.initialize_scratch(),
            right: right_forms.initialize_scratch(),
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

pub(super) struct SparseProductKernel<'a, F: Field> {
    dense: DenseMleBatch<'a, F>,
    left_rows: CompiledSparseLinearRows<F>,
    right_rows: CompiledSparseLinearRows<F>,
}

impl<'a, F: Field> SparseProductKernel<'a, F> {
    pub(super) fn new(
        context: CpuKernelContext,
        witness_polynomials: &'a [Vec<F>],
        input_columns: &'a [usize],
        constant_column: usize,
        left_rows: &'a [Vec<(usize, F)>],
        right_rows: &'a [Vec<(usize, F)>],
    ) -> Result<Self, BackendError> {
        if left_rows.len() != right_rows.len() {
            return context.invalid(format!(
                "left row count {} does not match right row count {}",
                left_rows.len(),
                right_rows.len()
            ));
        }
        let dense = DenseMleBatch::new(context, witness_polynomials)?;
        let layout = ColumnLayout::new(
            context,
            input_columns,
            constant_column,
            witness_polynomials.len(),
        )?;
        let left_rows =
            CompiledSparseLinearRows::new(context, &layout, left_rows, witness_polynomials.len())?;
        let right_rows =
            CompiledSparseLinearRows::new(context, &layout, right_rows, witness_polynomials.len())?;
        Ok(Self {
            dense,
            left_rows,
            right_rows,
        })
    }

    pub(super) fn log_rows(&self) -> usize {
        self.dense.log_rows()
    }

    pub(super) fn rows(&self) -> usize {
        self.dense.domain.rows
    }

    pub(super) fn sparse_row_count(&self) -> usize {
        self.left_rows.rows.len()
    }

    pub(super) fn combine_weighted_forms(
        &self,
        context: CpuKernelContext,
        row_weights: &[F],
    ) -> Result<(CombinedLinearForm<F>, CombinedLinearForm<F>), BackendError> {
        let left = self.left_rows.combine_weighted(context, row_weights)?;
        let right = self.right_rows.combine_weighted(context, row_weights)?;
        Ok((left, right))
    }

    pub(super) fn evaluate_combined_forms_at_boolean_row(
        &self,
        left_form: &CombinedLinearForm<F>,
        right_form: &CombinedLinearForm<F>,
        row_index: usize,
    ) -> (F, F) {
        (
            left_form.evaluate_at_row(self.dense.polynomials, row_index),
            right_form.evaluate_at_row(self.dense.polynomials, row_index),
        )
    }

    pub(super) fn evaluate_sparse_rows_at_boolean_row(
        &self,
        row_index: usize,
        left: &mut [F],
        right: &mut [F],
    ) {
        debug_assert_eq!(left.len(), self.left_rows.rows.len());
        debug_assert_eq!(right.len(), self.right_rows.rows.len());
        for (output, row) in left.iter_mut().zip(&self.left_rows.rows) {
            *output = evaluate_sparse_row_at_polynomial_row(row, self.dense.polynomials, row_index);
        }
        for (output, row) in right.iter_mut().zip(&self.right_rows.rows) {
            *output = evaluate_sparse_row_at_polynomial_row(row, self.dense.polynomials, row_index);
        }
    }

    pub(super) fn evaluate_linear_product(
        &self,
        context: CpuKernelContext,
        point: &[F],
        row_weights: &[F],
        scale: F,
    ) -> Result<F, BackendError> {
        if point.len() != self.log_rows() {
            return context.invalid(format!(
                "point has {} variables, expected {}",
                point.len(),
                self.log_rows()
            ));
        }
        if let Some(row_index) = boolean_index_msb(point) {
            let left = self.left_rows.evaluate_weighted_at_row(
                context,
                row_weights,
                self.dense.polynomials,
                row_index,
            )?;
            let right = self.right_rows.evaluate_weighted_at_row(
                context,
                row_weights,
                self.dense.polynomials,
                row_index,
            )?;
            return Ok(scale * left * right);
        }

        let witness_evaluations = self.dense.evaluate_all(context, point)?;
        let left = self.left_rows.evaluate_weighted_at_evals(
            context,
            row_weights,
            &witness_evaluations,
        )?;
        let right = self.right_rows.evaluate_weighted_at_evals(
            context,
            row_weights,
            &witness_evaluations,
        )?;
        Ok(scale * left * right)
    }

    pub(super) fn evaluate_linear_product_at_boolean_row(
        &self,
        context: CpuKernelContext,
        row_index: usize,
        row_weights: &[F],
        scale: F,
    ) -> Result<F, BackendError> {
        if row_index >= self.dense.domain.rows {
            return context.invalid(format!(
                "row index {row_index} exceeds witness row count {}",
                self.dense.domain.rows
            ));
        }
        let left = self.left_rows.evaluate_weighted_at_row(
            context,
            row_weights,
            self.dense.polynomials,
            row_index,
        )?;
        let right = self.right_rows.evaluate_weighted_at_row(
            context,
            row_weights,
            self.dense.polynomials,
            row_index,
        )?;
        Ok(scale * left * right)
    }

    pub(super) fn evaluate_linear_forms_at_boolean_row(
        &self,
        context: CpuKernelContext,
        row_index: usize,
        row_weights: &[F],
    ) -> Result<(F, F), BackendError> {
        if row_index >= self.dense.domain.rows {
            return context.invalid(format!(
                "row index {row_index} exceeds witness row count {}",
                self.dense.domain.rows
            ));
        }
        let left = self.left_rows.evaluate_weighted_at_row(
            context,
            row_weights,
            self.dense.polynomials,
            row_index,
        )?;
        let right = self.right_rows.evaluate_weighted_at_row(
            context,
            row_weights,
            self.dense.polynomials,
            row_index,
        )?;
        Ok((left, right))
    }

    pub(super) fn evaluate_row_product(
        &self,
        context: CpuKernelContext,
        eq_point: &[F],
        row_weights: &[F],
        scale: F,
    ) -> Result<F, BackendError> {
        if eq_point.len() != self.log_rows() {
            return context.invalid(format!(
                "equality point has {} variables, expected {}",
                eq_point.len(),
                self.log_rows()
            ));
        }
        if row_weights.len() != self.left_rows.rows.len() {
            return context.invalid(format!(
                "query has {} row weights, expected {}",
                row_weights.len(),
                self.left_rows.rows.len()
            ));
        }
        let left_form = self.left_rows.combine_weighted(context, row_weights)?;
        let right_form = self.right_rows.combine_weighted(context, row_weights)?;
        if let Some(row_index) = boolean_index_msb(eq_point) {
            let left = left_form.evaluate_at_row(self.dense.polynomials, row_index);
            let right = right_form.evaluate_at_row(self.dense.polynomials, row_index);
            return Ok(scale * left * right);
        }

        let eq_tensor = TensorEqTable::<F>::new(eq_point);
        if eq_tensor.len() != self.dense.domain.rows {
            return context.invalid(format!(
                "equality tensor has {} rows, expected {}",
                eq_tensor.len(),
                self.dense.domain.rows
            ));
        }
        let value = eq_tensor.par_fold_out_in(
            F::zero,
            |inner, row_index, _x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                let left = left_form.evaluate_at_row(self.dense.polynomials, row_index);
                let right = right_form.evaluate_at_row(self.dense.polynomials, row_index);
                *inner += e_in * left * right;
            },
            |_x_out, e_out, inner| e_out * inner,
            |left, right| left + right,
        );

        Ok(scale * value)
    }

    fn evaluate_row_product_group(
        &self,
        context: CpuKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckRowProductQuery<F>)],
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        if eq_point.len() != self.log_rows() {
            return context.invalid(format!(
                "equality point has {} variables, expected {}",
                eq_point.len(),
                self.log_rows()
            ));
        }

        if let Some(row_index) = boolean_index_msb(eq_point) {
            return queries
                .iter()
                .map(|&(query_index, query)| {
                    let left_form = self
                        .left_rows
                        .combine_weighted(context, &query.row_weights)
                        .map_err(|error| with_query_context(error, query_index))?;
                    let right_form = self
                        .right_rows
                        .combine_weighted(context, &query.row_weights)
                        .map_err(|error| with_query_context(error, query_index))?;
                    let left = left_form.evaluate_at_row(self.dense.polynomials, row_index);
                    let right = right_form.evaluate_at_row(self.dense.polynomials, row_index);
                    Ok((
                        query_index,
                        SumcheckLinearProductOutput::new(query.slot, query.scale * left * right),
                    ))
                })
                .collect();
        }

        let forms = queries
            .iter()
            .map(|&(query_index, query)| {
                let left = self
                    .left_rows
                    .combine_weighted(context, &query.row_weights)
                    .map_err(|error| with_query_context(error, query_index))?;
                let right = self
                    .right_rows
                    .combine_weighted(context, &query.row_weights)
                    .map_err(|error| with_query_context(error, query_index))?;
                Ok((left, right))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;
        let left_forms = forms
            .iter()
            .map(|(left, _right)| CombinedLinearForm {
                constant: left.constant,
                terms: left.terms.clone(),
            })
            .collect::<Vec<_>>();
        let right_forms = forms
            .iter()
            .map(|(_left, right)| CombinedLinearForm {
                constant: right.constant,
                terms: right.terms.clone(),
            })
            .collect::<Vec<_>>();
        let left_forms = BatchedLinearForms::new(&left_forms, self.left_rows.input_count);
        let right_forms = BatchedLinearForms::new(&right_forms, self.right_rows.input_count);

        let eq_tensor = TensorEqTable::<F>::new(eq_point);
        if eq_tensor.len() != self.dense.domain.rows {
            return context.invalid(format!(
                "equality tensor has {} rows, expected {}",
                eq_tensor.len(),
                self.dense.domain.rows
            ));
        }

        let values = eq_tensor.par_fold_out_in(
            || RowProductGroupAccumulator::new(&left_forms, &right_forms),
            |inner, row_index, _x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                left_forms.evaluate_at_row_into(self.dense.polynomials, row_index, &mut inner.left);
                right_forms.evaluate_at_row_into(
                    self.dense.polynomials,
                    row_index,
                    &mut inner.right,
                );
                for ((total, &left), &right) in
                    inner.totals.iter_mut().zip(&inner.left).zip(&inner.right)
                {
                    *total += e_in * left * right;
                }
            },
            |_x_out, e_out, inner| inner.scale(e_out),
            RowProductGroupAccumulator::merge,
        );

        Ok(values
            .totals
            .into_iter()
            .zip(queries)
            .map(|(value, &(query_index, query))| {
                (
                    query_index,
                    SumcheckLinearProductOutput::new(query.slot, query.scale * value),
                )
            })
            .collect())
    }
}

pub(in crate::cpu::sumcheck) fn evaluate_linear_product_queries<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckLinearProductRequest<F>,
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

    request
        .queries
        .par_iter()
        .enumerate()
        .map(|(query_index, query)| {
            kernel
                .evaluate_linear_product(context, &query.point, &query.row_weights, query.scale)
                .map(|value| SumcheckLinearProductOutput::new(query.slot, value))
                .map_err(|error| with_query_context(error, query_index))
        })
        .collect()
}

pub(in crate::cpu::sumcheck) fn evaluate_row_product_queries<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRowProductRequest<F>,
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

    let mut groups = HashMap::<Vec<F>, Vec<(usize, &SumcheckRowProductQuery<F>)>>::new();
    for (query_index, query) in request.queries.iter().enumerate() {
        groups
            .entry(query.eq_point.clone())
            .or_default()
            .push((query_index, query));
    }

    let mut outputs = Vec::with_capacity(request.queries.len());
    outputs.resize_with(request.queries.len(), || None);
    for (eq_point, queries) in groups {
        if queries.len() == 1 {
            let (query_index, query) = queries[0];
            let value = kernel
                .evaluate_row_product(context, &eq_point, &query.row_weights, query.scale)
                .map_err(|error| with_query_context(error, query_index))?;
            outputs[query_index] = Some(SumcheckLinearProductOutput::new(query.slot, value));
        } else {
            for (query_index, output) in
                kernel.evaluate_row_product_group(context, &eq_point, &queries)?
            {
                outputs[query_index] = Some(output);
            }
        }
    }
    outputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| BackendError::InvalidRequest {
            backend,
            task,
            reason: "row product evaluation did not produce every requested slot".to_owned(),
        })
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

#[cfg(test)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{eq_index_msb, MultilinearPoly};

    use super::{CpuKernelContext, SparseProductKernel};

    const BACKEND: &str = "cpu";
    const TASK: &str = "kernel test";

    #[test]
    fn sparse_product_kernel_evaluates_linear_product() -> Result<(), String> {
        let witness = vec![
            vec![
                Fr::from_u64(1),
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(4),
            ],
            vec![
                Fr::from_u64(5),
                Fr::from_u64(6),
                Fr::from_u64(7),
                Fr::from_u64(8),
            ],
        ];
        let left_rows = vec![
            vec![(10, Fr::from_u64(2)), (0, Fr::from_u64(3))],
            vec![(20, Fr::from_u64(1))],
        ];
        let right_rows = vec![
            vec![(20, Fr::from_u64(1)), (0, Fr::from_i64(-1))],
            vec![(10, Fr::from_u64(4))],
        ];
        let input_columns = vec![10, 20];
        let point = vec![Fr::from_u64(13), Fr::from_u64(17)];
        let row_weights = vec![Fr::from_u64(7), Fr::from_u64(11)];
        let scale = Fr::from_u64(19);
        let context = CpuKernelContext::new(BACKEND, TASK);
        let kernel = SparseProductKernel::new(
            context,
            &witness,
            &input_columns,
            0,
            &left_rows,
            &right_rows,
        )
        .map_err(|error| error.to_string())?;

        let value = kernel
            .evaluate_linear_product(context, &point, &row_weights, scale)
            .map_err(|error| error.to_string())?;

        let first = witness[0].as_slice().evaluate(&point);
        let second = witness[1].as_slice().evaluate(&point);
        let left =
            row_weights[0] * (Fr::from_u64(2) * first + Fr::from_u64(3)) + row_weights[1] * second;
        let right =
            row_weights[0] * (second - Fr::from_u64(1)) + row_weights[1] * Fr::from_u64(4) * first;
        assert_eq!(value, scale * left * right);
        Ok(())
    }

    #[test]
    fn sparse_product_kernel_evaluates_row_product() -> Result<(), String> {
        let witness = vec![
            vec![
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(5),
                Fr::from_u64(7),
            ],
            vec![
                Fr::from_u64(11),
                Fr::from_u64(13),
                Fr::from_u64(17),
                Fr::from_u64(19),
            ],
        ];
        let input_columns = vec![1, 2];
        let left_rows = vec![vec![(1, Fr::from_u64(2)), (0, Fr::from_u64(3))]];
        let right_rows = vec![vec![(2, Fr::from_u64(5)), (0, Fr::from_u64(7))]];
        let row_weights = vec![Fr::from_u64(23)];
        let eq_point = vec![Fr::from_u64(29), Fr::from_u64(31)];
        let scale = Fr::from_u64(37);
        let context = CpuKernelContext::new(BACKEND, TASK);
        let kernel = SparseProductKernel::new(
            context,
            &witness,
            &input_columns,
            0,
            &left_rows,
            &right_rows,
        )
        .map_err(|error| error.to_string())?;

        let value = kernel
            .evaluate_row_product(context, &eq_point, &row_weights, scale)
            .map_err(|error| error.to_string())?;

        let expected_inner = (0..4)
            .map(|row| {
                let left = row_weights[0] * (Fr::from_u64(2) * witness[0][row] + Fr::from_u64(3));
                let right = row_weights[0] * (Fr::from_u64(5) * witness[1][row] + Fr::from_u64(7));
                eq_index_msb(&eq_point, row) * left * right
            })
            .sum::<Fr>();

        assert_eq!(value, scale * expected_inner);
        Ok(())
    }
}
