use std::collections::{HashMap, HashSet};

use jolt_field::{
    signed::{S128, S256},
    AdditiveAccumulator, Field, RingAccumulator, SignedProductAccumulator, WithAccumulator,
    WithSignedProductAccumulator,
};
use jolt_poly::TensorEqTable;

use crate::{
    BackendError, BackendValueSlot, SumcheckLinearProductOutput, SumcheckProductUniskipRequest,
    SumcheckProductUniskipRow, SumcheckRowProductQuery, SumcheckRowProductRequest,
};

const JOLT_VM_NAMESPACE: &str = "jolt_vm";
const UNISKIP_RELATION: &str = "spartan_product.uniskip_first_round";
const PRODUCT_INPUT_COUNT: usize = 6;
const PRODUCT_ROW_COUNT: usize = 3;
const PRODUCT_EXTENDED_COEFFS: [[i32; PRODUCT_ROW_COUNT]; 2] = [[3, -3, 1], [1, -3, 3]];

pub(in crate::cpu::sumcheck) fn matches_row_product<F: Field>(
    request: &SumcheckRowProductRequest<F>,
) -> bool {
    request.kernel.relation.is_some_and(|relation| {
        relation.namespace == JOLT_VM_NAMESPACE && relation.name == UNISKIP_RELATION
    })
}

pub(in crate::cpu::sumcheck) fn evaluate_row_products<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckRowProductRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let context = ProductKernelContext::new(backend, task);
    if let Some(kernel) = ProductUniskipKernel::new(context, request)? {
        return kernel.evaluate(context, request);
    }

    super::evaluate_row_product_queries(backend, task, request)
}

pub(in crate::cpu::sumcheck) fn evaluate_product_uniskip_rows<F>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckProductUniskipRequest<F>,
) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    RawProductUniskipKernel::new(ProductKernelContext::new(backend, task), request)?
        .evaluate(ProductKernelContext::new(backend, task), request)
}

#[derive(Clone, Copy)]
struct ProductKernelContext {
    backend: &'static str,
    task: &'static str,
}

impl ProductKernelContext {
    const fn new(backend: &'static str, task: &'static str) -> Self {
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

struct RawProductUniskipKernel<'a> {
    rows: &'a [SumcheckProductUniskipRow],
    row_count: usize,
    log_rows: usize,
}

impl<'a> RawProductUniskipKernel<'a> {
    fn new<F: Field>(
        context: ProductKernelContext,
        request: &'a SumcheckProductUniskipRequest<F>,
    ) -> Result<Self, BackendError> {
        let row_count = request.rows.len();
        if row_count == 0 || !row_count.is_power_of_two() {
            return context.invalid(format!(
                "product uniskip row count {row_count} is not a nonzero power of two"
            ));
        }
        Ok(Self {
            rows: request.rows,
            row_count,
            log_rows: row_count.trailing_zeros() as usize,
        })
    }

    fn evaluate<F: Field>(
        &self,
        context: ProductKernelContext,
        request: &SumcheckProductUniskipRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
    where
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        validate_unique_query_slots(context, request.queries.iter().map(|query| query.slot))?;

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
                reason: "product uniskip row evaluation did not produce every requested slot"
                    .to_owned(),
            })
    }

    fn evaluate_group<F: Field>(
        &self,
        context: ProductKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckRowProductQuery<F>)],
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
    where
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        if eq_point.len() != self.log_rows {
            return context.invalid(format!(
                "equality point has {} variables, expected {}",
                eq_point.len(),
                self.log_rows
            ));
        }
        let compiled = queries
            .iter()
            .map(|&(query_index, query)| {
                ProductQuery::new(context, query_index, query)
                    .map_err(|error| with_query_context(error, context, query_index))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        if let Some(row_index) = boolean_point_index(eq_point) {
            return Ok(compiled
                .iter()
                .map(|query| {
                    let value = self.evaluate_product_at_row(row_index, query);
                    (
                        query.query_index,
                        SumcheckLinearProductOutput::new(query.slot, query.scale * value),
                    )
                })
                .collect());
        }

        let eq_tensor = TensorEqTable::<F>::new(eq_point);
        if eq_tensor.len() != self.row_count {
            return context.invalid(format!(
                "equality tensor has {} rows, expected {}",
                eq_tensor.len(),
                self.row_count
            ));
        }
        let values = if compiled.iter().all(|query| query.extended_coeffs.is_some()) {
            eq_tensor.par_fold_out_in(
                || ProductSignedGroupAccumulator::new(compiled.len()),
                |inner: &mut ProductSignedGroupAccumulator<F>, row_index, _x_in, e_in| {
                    if e_in.is_zero() {
                        return;
                    }
                    let row = &self.rows[row_index];
                    for (total, query) in inner.totals.iter_mut().zip(&compiled) {
                        let Some(coeffs) = query.extended_coeffs else {
                            continue;
                        };
                        total.fmadd_s256(e_in, &raw_extended_product_signed(row, coeffs));
                    }
                },
                |_x_out, e_out, inner: ProductSignedGroupAccumulator<F>| inner.scale(e_out),
                ProductGroupAccumulator::merge,
            )
        } else {
            eq_tensor.par_fold_out_in(
                || ProductGroupAccumulator::new(compiled.len()),
                |inner: &mut ProductGroupAccumulator<F>, row_index, _x_in, e_in| {
                    if e_in.is_zero() {
                        return;
                    }
                    let row = self.row_values(row_index);
                    for (total, query) in inner.totals.iter_mut().zip(&compiled) {
                        total.fmadd(e_in, row.product(query));
                    }
                },
                |_x_out, e_out, inner: ProductGroupAccumulator<F>| inner.scale(e_out),
                ProductGroupAccumulator::merge,
            )
        };

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

    fn evaluate_product_at_row<F: Field>(&self, row_index: usize, query: &ProductQuery<F>) -> F {
        if let Some(coeffs) = query.extended_coeffs {
            return raw_extended_product(&self.rows[row_index], coeffs);
        }
        self.row_values(row_index).product(query)
    }

    fn row_values<F: Field>(&self, row_index: usize) -> ProductRowValues<F> {
        ProductRowValues::from_raw(&self.rows[row_index])
    }
}

#[derive(Clone, Copy)]
struct ProductUniskipColumns {
    left_instruction: usize,
    lookup_output: usize,
    jump_flag: usize,
    right_instruction: usize,
    branch_flag: usize,
    next_is_noop: usize,
}

struct ProductUniskipKernel<'a, F: Field> {
    witness_polynomials: &'a [Vec<F>],
    columns: ProductUniskipColumns,
    rows: usize,
    log_rows: usize,
}

impl<'a, F: Field> ProductUniskipKernel<'a, F> {
    fn new(
        context: ProductKernelContext,
        request: &'a SumcheckRowProductRequest<F>,
    ) -> Result<Option<Self>, BackendError> {
        if !is_product_uniskip_candidate(request) {
            return Ok(None);
        }
        let columns = compile_product_columns(context, request)?;
        let (rows, log_rows) = validate_witness_polynomials(context, request.witness_polynomials)?;
        Ok(Some(Self {
            witness_polynomials: request.witness_polynomials,
            columns,
            rows,
            log_rows,
        }))
    }

    fn evaluate(
        &self,
        context: ProductKernelContext,
        request: &SumcheckRowProductRequest<F>,
    ) -> Result<Vec<SumcheckLinearProductOutput<F>>, BackendError>
    where
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        validate_unique_query_slots(context, request.queries.iter().map(|query| query.slot))?;

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
                reason: "product uniskip evaluation did not produce every requested slot"
                    .to_owned(),
            })
    }

    fn evaluate_group(
        &self,
        context: ProductKernelContext,
        eq_point: &[F],
        queries: &[(usize, &SumcheckRowProductQuery<F>)],
    ) -> Result<Vec<(usize, SumcheckLinearProductOutput<F>)>, BackendError>
    where
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        if eq_point.len() != self.log_rows {
            return context.invalid(format!(
                "equality point has {} variables, expected {}",
                eq_point.len(),
                self.log_rows
            ));
        }
        let compiled = queries
            .iter()
            .map(|&(query_index, query)| {
                ProductQuery::new(context, query_index, query)
                    .map_err(|error| with_query_context(error, context, query_index))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        if let Some(row_index) = boolean_point_index(eq_point) {
            return Ok(compiled
                .iter()
                .map(|query| {
                    let value = self.evaluate_product_at_row(row_index, query);
                    (
                        query.query_index,
                        SumcheckLinearProductOutput::new(query.slot, query.scale * value),
                    )
                })
                .collect());
        }

        let eq_tensor = TensorEqTable::<F>::new(eq_point);
        if eq_tensor.len() != self.rows {
            return context.invalid(format!(
                "equality tensor has {} rows, expected {}",
                eq_tensor.len(),
                self.rows
            ));
        }
        let values = eq_tensor.par_fold_out_in(
            || ProductGroupAccumulator::new(compiled.len()),
            |inner: &mut ProductGroupAccumulator<F>, row_index, _x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                let row = self.row_values(row_index);
                for (total, query) in inner.totals.iter_mut().zip(&compiled) {
                    total.fmadd(e_in, row.product(query));
                }
            },
            |_x_out, e_out, inner: ProductGroupAccumulator<F>| inner.scale(e_out),
            ProductGroupAccumulator::merge,
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

    fn evaluate_product_at_row(&self, row_index: usize, query: &ProductQuery<F>) -> F {
        self.row_values(row_index).product(query)
    }

    fn row_values(&self, row_index: usize) -> ProductRowValues<F> {
        let columns = self.columns;
        ProductRowValues {
            left_instruction: self.witness_polynomials[columns.left_instruction][row_index],
            lookup_output: self.witness_polynomials[columns.lookup_output][row_index],
            jump_flag: self.witness_polynomials[columns.jump_flag][row_index],
            right_instruction: self.witness_polynomials[columns.right_instruction][row_index],
            branch_flag: self.witness_polynomials[columns.branch_flag][row_index],
            not_next_is_noop: F::one() - self.witness_polynomials[columns.next_is_noop][row_index],
        }
    }
}

struct ProductRowValues<F: Field> {
    left_instruction: F,
    lookup_output: F,
    jump_flag: F,
    right_instruction: F,
    branch_flag: F,
    not_next_is_noop: F,
}

impl<F: Field> ProductRowValues<F> {
    fn from_raw(row: &SumcheckProductUniskipRow) -> Self {
        Self {
            left_instruction: F::from_u64(row.left_instruction),
            lookup_output: F::from_u64(row.lookup_output),
            jump_flag: F::from_bool(row.jump_flag),
            right_instruction: F::from_i128(row.right_instruction),
            branch_flag: F::from_bool(row.branch_flag),
            not_next_is_noop: F::from_bool(!row.next_is_noop),
        }
    }

    fn product(&self, query: &ProductQuery<F>) -> F {
        let left = query.product_weight * self.left_instruction
            + query.branch_weight * self.lookup_output
            + query.jump_weight * self.jump_flag;
        let right = query.product_weight * self.right_instruction
            + query.branch_weight * self.branch_flag
            + query.jump_weight * self.not_next_is_noop;
        left * right
    }
}

struct ProductQuery<F: Field> {
    query_index: usize,
    slot: BackendValueSlot,
    scale: F,
    product_weight: F,
    branch_weight: F,
    jump_weight: F,
    extended_coeffs: Option<[i32; PRODUCT_ROW_COUNT]>,
}

impl<F: Field> ProductQuery<F> {
    fn new(
        context: ProductKernelContext,
        query_index: usize,
        query: &SumcheckRowProductQuery<F>,
    ) -> Result<Self, BackendError> {
        let [product_weight, branch_weight, jump_weight] = query.row_weights.as_slice() else {
            return Err(BackendError::InvalidRequest {
                backend: context.backend,
                task: context.task,
                reason: format!(
                    "query has {} row weights, expected {PRODUCT_ROW_COUNT}",
                    query.row_weights.len()
                ),
            });
        };
        Ok(Self {
            query_index,
            slot: query.slot,
            scale: query.scale,
            product_weight: *product_weight,
            branch_weight: *branch_weight,
            jump_weight: *jump_weight,
            extended_coeffs: product_extended_coeffs::<F>([
                *product_weight,
                *branch_weight,
                *jump_weight,
            ]),
        })
    }
}

fn product_extended_coeffs<F: Field>(
    weights: [F; PRODUCT_ROW_COUNT],
) -> Option<[i32; PRODUCT_ROW_COUNT]> {
    PRODUCT_EXTENDED_COEFFS.iter().copied().find(|coeffs| {
        weights
            .iter()
            .zip(coeffs.iter().copied())
            .all(|(&weight, coeff)| weight == F::from_i64(i64::from(coeff)))
    })
}

fn raw_extended_product<F: Field>(
    row: &SumcheckProductUniskipRow,
    coeffs: [i32; PRODUCT_ROW_COUNT],
) -> F {
    signed_256_to_field(raw_extended_product_signed(row, coeffs))
}

fn raw_extended_product_signed(
    row: &SumcheckProductUniskipRow,
    coeffs: [i32; PRODUCT_ROW_COUNT],
) -> S256 {
    let coeffs = coeffs.map(i128::from);
    let left_sum = coeffs[0] * i128::from(row.left_instruction)
        + coeffs[1] * i128::from(row.lookup_output)
        + if row.jump_flag { coeffs[2] } else { 0 };
    let right_sum = coeffs[0] * row.right_instruction
        + if row.branch_flag { coeffs[1] } else { 0 }
        + if row.next_is_noop { 0 } else { coeffs[2] };
    S128::from_i128(left_sum).mul_trunc::<2, 4>(&S128::from_i128(right_sum))
}

fn signed_256_to_field<F: Field>(value: S256) -> F {
    let mut bytes = [0u8; 32];
    for (index, limb) in value.magnitude_limbs().iter().copied().enumerate() {
        bytes[index * 8..(index + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let magnitude = F::from_le_bytes_mod_order(&bytes);
    if value.is_positive {
        magnitude
    } else {
        -magnitude
    }
}

struct ProductGroupAccumulator<F: Field>
where
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    totals: Vec<<F as WithAccumulator>::Accumulator>,
}

impl<F> ProductGroupAccumulator<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn new(len: usize) -> Self {
        Self {
            totals: vec![<F as WithAccumulator>::Accumulator::default(); len],
        }
    }

    fn scale(mut self, scale: F) -> Self {
        if scale.is_zero() {
            self.totals
                .fill(<F as WithAccumulator>::Accumulator::default());
        } else {
            for value in &mut self.totals {
                let total = std::mem::take(value);
                value.fmadd(scale, total.reduce());
            }
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.totals.iter_mut().zip(other.totals) {
            left.merge(right);
        }
        self
    }

    fn into_values(self) -> impl Iterator<Item = F> {
        self.totals.into_iter().map(AdditiveAccumulator::reduce)
    }
}

struct ProductSignedGroupAccumulator<F: Field> {
    totals: Vec<<F as WithSignedProductAccumulator>::SignedProductAccumulator>,
}

impl<F> ProductSignedGroupAccumulator<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    fn new(len: usize) -> Self {
        Self {
            totals: vec![
                <F as WithSignedProductAccumulator>::SignedProductAccumulator::default();
                len
            ],
        }
    }

    fn scale(self, scale: F) -> ProductGroupAccumulator<F> {
        let mut out = ProductGroupAccumulator::new(self.totals.len());
        if scale.is_zero() {
            return out;
        }
        for (target, total) in out.totals.iter_mut().zip(self.totals) {
            target.fmadd(scale, total.reduce());
        }
        out
    }
}

fn is_product_uniskip_candidate<F: Field>(request: &SumcheckRowProductRequest<F>) -> bool {
    request.witness_polynomials.len() == PRODUCT_INPUT_COUNT
        && request.input_columns.len() == PRODUCT_INPUT_COUNT
        && request.left_rows.len() == PRODUCT_ROW_COUNT
        && request.right_rows.len() == PRODUCT_ROW_COUNT
}

fn compile_product_columns<F: Field>(
    context: ProductKernelContext,
    request: &SumcheckRowProductRequest<F>,
) -> Result<ProductUniskipColumns, BackendError> {
    validate_input_columns(context, request.input_columns, request.constant_column)?;

    Ok(ProductUniskipColumns {
        left_instruction: single_unit_input(context, request, &request.left_rows[0])?,
        lookup_output: single_unit_input(context, request, &request.left_rows[1])?,
        jump_flag: single_unit_input(context, request, &request.left_rows[2])?,
        right_instruction: single_unit_input(context, request, &request.right_rows[0])?,
        branch_flag: single_unit_input(context, request, &request.right_rows[1])?,
        next_is_noop: one_minus_input(context, request, &request.right_rows[2])?,
    })
}

fn validate_input_columns(
    context: ProductKernelContext,
    input_columns: &[usize],
    constant_column: usize,
) -> Result<(), BackendError> {
    let mut seen = HashSet::with_capacity(input_columns.len());
    for &column in input_columns {
        if column == constant_column {
            return context.invalid(format!("input column {column} aliases the constant column"));
        }
        if !seen.insert(column) {
            return context.invalid(format!("duplicate input column {column}"));
        }
    }
    Ok(())
}

fn single_unit_input<F: Field>(
    context: ProductKernelContext,
    request: &SumcheckRowProductRequest<F>,
    row: &[(usize, F)],
) -> Result<usize, BackendError> {
    let [(column, coefficient)] = row else {
        return context.invalid("product uniskip row is not a single input term");
    };
    if *coefficient != F::one() {
        return context.invalid("product uniskip input coefficient is not one");
    }
    input_index(context, request, *column)
}

fn one_minus_input<F: Field>(
    context: ProductKernelContext,
    request: &SumcheckRowProductRequest<F>,
    row: &[(usize, F)],
) -> Result<usize, BackendError> {
    if row.len() != 2 {
        return context.invalid("product uniskip jump row is not 1 - next_is_noop");
    }
    let mut saw_constant = false;
    let mut input = None;
    for &(column, coefficient) in row {
        if column == request.constant_column {
            if coefficient != F::one() {
                return context.invalid("product uniskip constant coefficient is not one");
            }
            saw_constant = true;
        } else {
            if coefficient != F::zero() - F::one() {
                return context.invalid("product uniskip next_is_noop coefficient is not -one");
            }
            input = Some(input_index(context, request, column)?);
        }
    }
    if !saw_constant {
        return context.invalid("product uniskip jump row missing constant term");
    }
    let Some(input) = input else {
        return context.invalid("product uniskip jump row missing next_is_noop term");
    };
    Ok(input)
}

fn input_index<F: Field>(
    context: ProductKernelContext,
    request: &SumcheckRowProductRequest<F>,
    column: usize,
) -> Result<usize, BackendError> {
    if column == request.constant_column {
        return context.invalid("product uniskip expected input column, got constant column");
    }
    request
        .input_columns
        .iter()
        .position(|&input_column| input_column == column)
        .ok_or_else(|| BackendError::InvalidRequest {
            backend: context.backend,
            task: context.task,
            reason: format!("row referenced unsupported input column {column}"),
        })
}

fn validate_witness_polynomials<F: Field>(
    context: ProductKernelContext,
    witness_polynomials: &[Vec<F>],
) -> Result<(usize, usize), BackendError> {
    let Some(first) = witness_polynomials.first() else {
        return context.invalid("request has no witness polynomials");
    };
    let rows = first.len();
    if rows == 0 || !rows.is_power_of_two() {
        return context.invalid(format!(
            "witness polynomial row count {rows} is not a nonzero power of two"
        ));
    }
    for (index, polynomial) in witness_polynomials.iter().enumerate() {
        if polynomial.len() != rows {
            return context.invalid(format!(
                "witness polynomial {index} has {} rows, expected {rows}",
                polynomial.len()
            ));
        }
    }
    Ok((rows, rows.trailing_zeros() as usize))
}

fn boolean_point_index<F: Field>(point: &[F]) -> Option<usize> {
    let mut index = 0usize;
    for &value in point {
        index <<= 1;
        if value.is_one() {
            index |= 1;
        } else if !value.is_zero() {
            return None;
        }
    }
    Some(index)
}

fn validate_unique_query_slots(
    context: ProductKernelContext,
    slots: impl Iterator<Item = BackendValueSlot>,
) -> Result<(), BackendError> {
    let mut seen = HashSet::new();
    for slot in slots {
        if !seen.insert(slot) {
            return context.invalid(format!("duplicate value slot {slot:?}"));
        }
    }
    Ok(())
}

fn with_query_context(
    error: BackendError,
    context: ProductKernelContext,
    query_index: usize,
) -> BackendError {
    match error {
        BackendError::InvalidRequest { reason, .. } => BackendError::InvalidRequest {
            backend: context.backend,
            task: context.task,
            reason: format!("query {query_index}: {reason}"),
        },
        other => other,
    }
}
