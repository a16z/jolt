use std::cmp::Ordering;

use jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder;
use jolt_field::{
    signed::{S128, S64},
    AdditiveAccumulator, Field, OptimizedMul, RingAccumulator,
};
use jolt_poly::{EqPolynomial, OneHotIndexOrder};
use jolt_witness::protocols::jolt_vm::JoltVmStage6Row;
use rayon::prelude::*;

const PAR_BOUND_MIN_LEN: usize = 4096;
const SPLIT_EQ_PARALLEL_THRESHOLD: usize = 16;

pub trait CpuCompactScalar: Copy + Ord + Send + Sync {
    fn field_mul<F: Field>(self, n: F) -> F;
    fn to_field<F: Field>(self) -> F;
    fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F;
}

pub enum LinearCombinationInput<'a, F: Field> {
    Dense(&'a [F]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    U128(&'a [u128]),
    I64(&'a [i64]),
    I128(&'a [i128]),
    S128(&'a [S128]),
}

#[derive(Clone, Copy, Debug)]
pub struct OneHotVectorMatrixProductInput<'a, F: Field> {
    pub k: usize,
    pub indices: &'a [Option<u8>],
    pub coefficient: F,
    pub index_order: OneHotIndexOrder,
}

impl<F: Field> LinearCombinationInput<'_, F> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::U128(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::I128(values) => values.len(),
            Self::S128(values) => values.len(),
        }
    }

    fn scaled_coeff(&self, index: usize, coefficient: F) -> F {
        match self {
            Self::Dense(values) => values[index].mul_01_optimized(coefficient),
            Self::U8(values) => values[index].field_mul(coefficient),
            Self::U16(values) => values[index].field_mul(coefficient),
            Self::U32(values) => values[index].field_mul(coefficient),
            Self::U64(values) => values[index].field_mul(coefficient),
            Self::U128(values) => values[index].field_mul(coefficient),
            Self::I64(values) => values[index].field_mul(coefficient),
            Self::I128(values) => values[index].field_mul(coefficient),
            Self::S128(values) => values[index].field_mul(coefficient),
        }
    }
}

macro_rules! impl_unsigned_compact_scalar {
    ($type:ty, $from:ident, $mul:ident) => {
        impl CpuCompactScalar for $type {
            #[inline]
            fn field_mul<F: Field>(self, n: F) -> F {
                n.$mul(self.into())
            }

            #[inline]
            fn to_field<F: Field>(self) -> F {
                F::$from(self)
            }

            #[inline]
            fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F {
                r.$mul(self.abs_diff(other).into())
            }
        }
    };
}

impl CpuCompactScalar for bool {
    #[inline]
    fn field_mul<F: Field>(self, n: F) -> F {
        if self {
            n
        } else {
            F::zero()
        }
    }

    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_bool(self)
    }

    #[inline]
    fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F {
        if self ^ other {
            r
        } else {
            F::zero()
        }
    }
}

impl_unsigned_compact_scalar!(u8, from_u8, mul_u64);
impl_unsigned_compact_scalar!(u16, from_u16, mul_u64);
impl_unsigned_compact_scalar!(u32, from_u32, mul_u64);
impl_unsigned_compact_scalar!(u64, from_u64, mul_u64);

impl CpuCompactScalar for u128 {
    #[inline]
    fn field_mul<F: Field>(self, n: F) -> F {
        n.mul_u128(self)
    }

    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_u128(self)
    }

    #[inline]
    fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F {
        r.mul_u128(self.abs_diff(other))
    }
}

impl CpuCompactScalar for i64 {
    #[inline]
    fn field_mul<F: Field>(self, n: F) -> F {
        if self.is_negative() {
            -n.mul_u64(self.unsigned_abs())
        } else {
            n.mul_u64(self as u64)
        }
    }

    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_i64(self)
    }

    #[inline]
    fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F {
        r.mul_u64(self.abs_diff(other))
    }
}

impl CpuCompactScalar for i128 {
    #[inline]
    fn field_mul<F: Field>(self, n: F) -> F {
        n.mul_i128(self)
    }

    #[inline]
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self)
    }

    #[inline]
    fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F {
        r.mul_u128(self.abs_diff(other))
    }
}

impl CpuCompactScalar for S64 {
    #[inline]
    fn field_mul<F: Field>(self, n: F) -> F {
        let magnitude = self.magnitude.0[0];
        if self.is_positive {
            n.mul_u64(magnitude)
        } else {
            -n.mul_u64(magnitude)
        }
    }

    #[inline]
    fn to_field<F: Field>(self) -> F {
        let magnitude = self.magnitude.0[0];
        if self.is_positive {
            F::from_u64(magnitude)
        } else {
            -F::from_u64(magnitude)
        }
    }

    #[inline]
    fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F {
        let diff = (self.to_i128() - other.to_i128()).unsigned_abs();
        r.mul_u64(diff as u64)
    }
}

impl CpuCompactScalar for S128 {
    #[inline]
    fn field_mul<F: Field>(self, n: F) -> F {
        signed_128_field_mul(self, n)
    }

    #[inline]
    fn to_field<F: Field>(self) -> F {
        if let Some(value) = self.to_i128() {
            F::from_i128(value)
        } else {
            let magnitude = self.magnitude_as_u128();
            if self.is_positive {
                F::from_u128(magnitude)
            } else {
                -F::from_u128(magnitude)
            }
        }
    }

    #[inline]
    fn diff_mul_field<F: Field>(self, other: Self, r: F) -> F {
        if self > other {
            self.field_mul(r) - other.field_mul(r)
        } else {
            other.field_mul(r) - self.field_mul(r)
        }
    }
}

pub fn bind_compact_first_high_to_low<T, F>(coeffs: &[T], r: F) -> Vec<F>
where
    T: CpuCompactScalar,
    F: Field,
{
    let half = validate_first_bind_len(coeffs.len());
    let (left, right) = coeffs.split_at(half);
    left.par_iter()
        .zip(right.par_iter())
        .map(|(&a, &b)| bind_scalar_pair(a, b, r))
        .collect()
}

pub fn bind_compact_first_low_to_high<T, F>(coeffs: &[T], r: F) -> Vec<F>
where
    T: CpuCompactScalar,
    F: Field,
{
    let half = validate_first_bind_len(coeffs.len());
    (0..half)
        .into_par_iter()
        .map(|index| bind_scalar_pair(coeffs[2 * index], coeffs[2 * index + 1], r))
        .collect()
}

pub fn bind_field_high_to_low<F: Field>(values: &mut Vec<F>, r: F) {
    let half = validate_first_bind_len(values.len());
    let (left, right) = values.split_at_mut(half);
    left.par_iter_mut()
        .zip(right.par_iter())
        .with_min_len(PAR_BOUND_MIN_LEN)
        .for_each(|(a, b)| {
            *a += r * (*b - *a);
        });
    values.truncate(half);
}

pub fn bind_field_low_to_high<F: Field>(values: &[F], r: F) -> Vec<F> {
    let half = validate_first_bind_len(values.len());
    values
        .par_chunks_exact(2)
        .with_min_len(PAR_BOUND_MIN_LEN)
        .map(|pair| pair[0] + r * (pair[1] - pair[0]))
        .take(half)
        .collect()
}

pub fn dense_split_eq_evaluate<F: Field>(
    values: &[F],
    r_len: usize,
    eq_one: &[F],
    eq_two: &[F],
) -> F {
    validate_split_eq_len(values.len(), eq_one.len(), eq_two.len());
    if r_len < SPLIT_EQ_PARALLEL_THRESHOLD {
        dense_split_eq_serial(values, eq_one, eq_two)
    } else {
        dense_split_eq_parallel(values, eq_one, eq_two)
    }
}

pub fn compact_split_eq_evaluate<T, F>(coeffs: &[T], r_len: usize, eq_one: &[F], eq_two: &[F]) -> F
where
    T: CpuCompactScalar,
    F: Field,
{
    validate_split_eq_len(coeffs.len(), eq_one.len(), eq_two.len());
    if r_len < SPLIT_EQ_PARALLEL_THRESHOLD {
        compact_split_eq_serial(coeffs, eq_one, eq_two)
    } else {
        compact_split_eq_parallel(coeffs, eq_one, eq_two)
    }
}

pub fn dense_inside_out_evaluate<F: Field>(values: &[F], point: &[F]) -> F {
    validate_inside_out_len(values.len(), point.len());
    if point.len() < SPLIT_EQ_PARALLEL_THRESHOLD {
        dense_inside_out_serial(values, point)
    } else {
        dense_inside_out_parallel(values, point)
    }
}

pub fn compact_inside_out_evaluate<T, F>(coeffs: &[T], point: &[F]) -> F
where
    T: CpuCompactScalar,
    F: Field,
{
    validate_inside_out_len(coeffs.len(), point.len());
    if point.len() < SPLIT_EQ_PARALLEL_THRESHOLD {
        compact_inside_out_serial(coeffs, point)
    } else {
        compact_inside_out_parallel(coeffs, point)
    }
}

pub fn dense_batch_evaluate<F: Field>(polys: &[&[F]], point: &[F]) -> Vec<F> {
    if polys.is_empty() {
        return Vec::new();
    }
    validate_batch_eval_len(polys, point.len());
    let m = point.len() / 2;
    let (r2, r1) = point.split_at(m);
    let (eq_one, eq_two) = rayon::join(
        || EqPolynomial::<F>::evals(r2, None),
        || EqPolynomial::<F>::evals(r1, None),
    );
    dense_batch_split_eq_evaluate(polys, &eq_one, &eq_two)
}

pub fn dense_batch_split_eq_evaluate<F: Field>(
    polys: &[&[F]],
    eq_one: &[F],
    eq_two: &[F],
) -> Vec<F> {
    if polys.is_empty() {
        return Vec::new();
    }
    validate_batch_split_eq_len(polys, eq_one.len(), eq_two.len());
    dense_batch_split_eq_parallel(polys, eq_one, eq_two)
}

pub fn dense_dot_product_low_optimized<F: Field>(left: &[F], right: &[F]) -> F {
    assert_eq!(
        left.len(),
        right.len(),
        "dot product input length mismatch: left={}, right={}",
        left.len(),
        right.len()
    );
    left.par_iter()
        .zip_eq(right.par_iter())
        .map(|(left, right)| left.mul_01_optimized(*right))
        .sum()
}

pub fn linear_combination<F: Field>(
    polynomials: &[LinearCombinationInput<'_, F>],
    coefficients: &[F],
) -> Vec<F> {
    assert_eq!(
        polynomials.len(),
        coefficients.len(),
        "linear combination input mismatch: polynomials={}, coefficients={}",
        polynomials.len(),
        coefficients.len()
    );
    let max_length = polynomials
        .iter()
        .map(LinearCombinationInput::len)
        .max()
        .unwrap_or(0);

    (0..max_length)
        .into_par_iter()
        .map(|index| {
            let mut acc = F::zero();
            for (poly, &coefficient) in polynomials.iter().zip(coefficients.iter()) {
                if index < poly.len() {
                    acc += poly.scaled_coeff(index, coefficient);
                }
            }
            acc
        })
        .collect()
}

pub fn one_hot_evaluate<F: Field>(k: usize, indices: &[Option<u8>], point: &[F]) -> F {
    validate_one_hot_evaluate_inputs(k, indices, point.len());
    let log_t = indices.len().trailing_zeros() as usize;
    let (address_point, cycle_point) = point.split_at(point.len() - log_t);
    let (eq_address, eq_cycle) = rayon::join(
        || EqPolynomial::<F>::evals(address_point, None),
        || EqPolynomial::<F>::evals(cycle_point, None),
    );

    indices
        .par_iter()
        .zip(eq_cycle.par_iter())
        .map(|(address, eq_cycle)| match address {
            Some(address) => eq_address[*address as usize].mul_01_optimized(*eq_cycle),
            None => F::zero(),
        })
        .sum()
}

pub fn one_hot_vector_matrix_product<F: Field>(
    k: usize,
    indices: &[Option<u8>],
    left_vec: &[F],
    coeff: F,
    num_columns: usize,
    index_order: OneHotIndexOrder,
) -> Vec<F> {
    let mut result = vec![F::zero(); num_columns];
    accumulate_one_hot_vector_matrix_product(k, indices, left_vec, coeff, index_order, &mut result);
    result
}

pub fn materialized_rlc_vector_matrix_product<F: Field>(
    dense_rlc: &[F],
    one_hot_rlc: &[OneHotVectorMatrixProductInput<'_, F>],
    left_vec: &[F],
    num_columns: usize,
) -> Vec<F> {
    validate_materialized_rlc_vmp_inputs(dense_rlc, left_vec, num_columns);
    let mut result = dense_vector_matrix_product(dense_rlc, left_vec, num_columns);

    for component in one_hot_rlc {
        accumulate_one_hot_vector_matrix_product(
            component.k,
            component.indices,
            left_vec,
            component.coefficient,
            component.index_order,
            &mut result,
        );
    }

    result
}

pub struct Stage8StreamingRlcVectorMatrixProductInput<'a, F: Field> {
    pub rows: &'a [JoltVmStage6Row],
    pub log_t: usize,
    pub committed_chunk_bits: usize,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub ram_inc_coefficient: F,
    pub rd_inc_coefficient: F,
    pub instruction_coefficients: &'a [F],
    pub bytecode_coefficients: &'a [F],
    pub ram_coefficients: &'a [F],
    pub left_vec: &'a [F],
    pub num_columns: usize,
}

pub fn stage8_streaming_rlc_vector_matrix_product<F: Field>(
    input: Stage8StreamingRlcVectorMatrixProductInput<'_, F>,
) -> Vec<F>
where
    <F as jolt_field::WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    validate_stage8_streaming_rlc_vmp_input(&input);

    if input.trace_polynomial_order == TracePolynomialOrder::AddressMajor {
        return stage8_streaming_rlc_vector_matrix_product_exact(input);
    }

    let trace_rows = 1usize << input.log_t;
    let address_columns = 1usize << input.committed_chunk_bits;
    if trace_rows < input.num_columns || !trace_rows.is_multiple_of(input.num_columns) {
        return stage8_streaming_rlc_vector_matrix_product_exact(input);
    }
    let rows_per_address = trace_rows / input.num_columns;
    if input.left_vec.len() != address_columns * rows_per_address {
        return stage8_streaming_rlc_vector_matrix_product_exact(input);
    }

    let setup = Stage8StreamingRlcVmpSetup::new(&input, rows_per_address, address_columns);
    let num_threads = rayon::current_num_threads().max(1);
    let chunks_per_thread = rows_per_address.div_ceil(num_threads).max(1);

    let (dense_accs, one_hot_accs) = (0..rows_per_address)
        .collect::<Vec<_>>()
        .par_chunks(chunks_per_thread)
        .enumerate()
        .map(|(_, row_offsets)| {
            let (mut dense_accs, mut one_hot_accs) =
                create_stage8_streaming_rlc_accumulators::<F>(input.num_columns);
            for &row_offset in row_offsets {
                let row_weight = input.left_vec[row_offset];
                let scaled_ram_inc = row_weight * input.ram_inc_coefficient;
                let scaled_rd_inc = row_weight * input.rd_inc_coefficient;
                let row_factor = setup.row_factors[row_offset];
                let cycle_start = row_offset * input.num_columns;
                let cycle_end = cycle_start + input.num_columns;
                for (col_index, row) in input.rows[cycle_start..cycle_end].iter().enumerate() {
                    accumulate_stage8_streaming_rlc_folded_row(
                        row,
                        &setup,
                        scaled_ram_inc,
                        scaled_rd_inc,
                        row_factor,
                        &mut dense_accs[col_index],
                        &mut one_hot_accs[col_index],
                    );
                }
            }
            (dense_accs, one_hot_accs)
        })
        .reduce(
            || create_stage8_streaming_rlc_accumulators::<F>(input.num_columns),
            merge_stage8_streaming_rlc_accumulators::<F>,
        );
    finalize_stage8_streaming_rlc_accumulators(dense_accs, one_hot_accs)
}

struct Stage8StreamingRlcVmpSetup<F: Field> {
    row_factors: Vec<F>,
    instruction_tables: Vec<Vec<F>>,
    bytecode_tables: Vec<Vec<F>>,
    ram_tables: Vec<Vec<F>>,
}

impl<F: Field> Stage8StreamingRlcVmpSetup<F> {
    fn new(
        input: &Stage8StreamingRlcVectorMatrixProductInput<'_, F>,
        rows_per_address: usize,
        address_columns: usize,
    ) -> Self {
        let (row_factors, address_factors) =
            compute_stage8_dory_left_factors(input.left_vec, rows_per_address, address_columns);
        Self {
            row_factors,
            instruction_tables: build_stage8_folded_one_hot_tables(
                input.instruction_coefficients,
                &address_factors,
            ),
            bytecode_tables: build_stage8_folded_one_hot_tables(
                input.bytecode_coefficients,
                &address_factors,
            ),
            ram_tables: build_stage8_folded_one_hot_tables(
                input.ram_coefficients,
                &address_factors,
            ),
        }
    }
}

fn compute_stage8_dory_left_factors<F: Field>(
    left_vec: &[F],
    rows_per_address: usize,
    address_columns: usize,
) -> (Vec<F>, Vec<F>) {
    let mut row_factors = vec![F::zero(); rows_per_address];
    let mut address_factors = vec![F::zero(); address_columns];
    for (address, address_factor_slot) in
        address_factors.iter_mut().enumerate().take(address_columns)
    {
        let base = address * rows_per_address;
        let mut address_factor = F::zero();
        for row_offset in 0..rows_per_address {
            let value = left_vec[base + row_offset];
            address_factor += value;
            row_factors[row_offset] += value;
        }
        *address_factor_slot = address_factor;
    }
    (row_factors, address_factors)
}

fn build_stage8_folded_one_hot_tables<F: Field>(coefficients: &[F], factors: &[F]) -> Vec<Vec<F>> {
    coefficients
        .par_iter()
        .map(|&coefficient| {
            if coefficient.is_zero() {
                return vec![F::zero(); factors.len()];
            }
            factors
                .iter()
                .map(|&factor| coefficient * factor)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn accumulate_stage8_streaming_rlc_folded_row<F: Field>(
    row: &JoltVmStage6Row,
    setup: &Stage8StreamingRlcVmpSetup<F>,
    scaled_ram_inc: F,
    scaled_rd_inc: F,
    row_factor: F,
    dense_acc: &mut <F as jolt_field::WithAccumulator>::Accumulator,
    one_hot_acc: &mut <F as jolt_field::WithAccumulator>::Accumulator,
) where
    <F as jolt_field::WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    if row.ram_increment != 0 {
        dense_acc.fmadd(scaled_ram_inc, F::from_i128(row.ram_increment));
    }
    if row.rd_increment != 0 {
        dense_acc.fmadd(scaled_rd_inc, F::from_i128(row.rd_increment));
    }

    let mut one_hot_value = F::zero();
    accumulate_stage8_folded_ra_tables(
        row.instruction_lookup_index,
        &setup.instruction_tables,
        &mut one_hot_value,
    );
    accumulate_stage8_folded_ra_tables(
        row.bytecode_index as u128,
        &setup.bytecode_tables,
        &mut one_hot_value,
    );
    if let Some(address) = row.remapped_ram_address {
        accumulate_stage8_folded_ra_tables(address as u128, &setup.ram_tables, &mut one_hot_value);
    }
    if !one_hot_value.is_zero() {
        one_hot_acc.fmadd(row_factor, one_hot_value);
    }
}

fn accumulate_stage8_folded_ra_tables<F: Field>(
    value: u128,
    tables: &[Vec<F>],
    accumulator: &mut F,
) {
    for (index, table) in tables.iter().enumerate() {
        let hot_column = stage8_ra_chunk(value, index, tables.len(), table.len().ilog2() as usize);
        *accumulator += table[hot_column];
    }
}

fn create_stage8_streaming_rlc_accumulators<F: Field>(
    num_columns: usize,
) -> (
    Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
    Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
) {
    (
        vec![<F as jolt_field::WithAccumulator>::Accumulator::default(); num_columns],
        vec![<F as jolt_field::WithAccumulator>::Accumulator::default(); num_columns],
    )
}

fn merge_stage8_streaming_rlc_accumulators<F: Field>(
    (mut dense_left, mut one_hot_left): (
        Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
        Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
    ),
    (dense_right, one_hot_right): (
        Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
        Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
    ),
) -> (
    Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
    Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
) {
    for (left, right) in dense_left.iter_mut().zip(dense_right) {
        left.merge(right);
    }
    for (left, right) in one_hot_left.iter_mut().zip(one_hot_right) {
        left.merge(right);
    }
    (dense_left, one_hot_left)
}

fn finalize_stage8_streaming_rlc_accumulators<F: Field>(
    dense_accs: Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
    one_hot_accs: Vec<<F as jolt_field::WithAccumulator>::Accumulator>,
) -> Vec<F> {
    dense_accs
        .into_par_iter()
        .zip(one_hot_accs)
        .map(|(dense, one_hot)| dense.reduce() + one_hot.reduce())
        .collect()
}

fn stage8_streaming_rlc_vector_matrix_product_exact<F: Field>(
    input: Stage8StreamingRlcVectorMatrixProductInput<'_, F>,
) -> Vec<F> {
    let trace_rows = 1usize << input.log_t;
    let num_threads = rayon::current_num_threads().max(1);
    let rows_per_thread = trace_rows.div_ceil(num_threads).max(1);

    input
        .rows
        .par_chunks(rows_per_thread)
        .enumerate()
        .map(|(chunk_index, rows)| {
            let mut result = vec![F::zero(); input.num_columns];
            let cycle_start = chunk_index * rows_per_thread;
            for (local_cycle, row) in rows.iter().enumerate() {
                let cycle = cycle_start + local_cycle;
                accumulate_stage8_streaming_rlc_row(cycle, row, &input, trace_rows, &mut result);
            }
            result
        })
        .reduce(
            || vec![F::zero(); input.num_columns],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
}

fn accumulate_stage8_streaming_rlc_row<F: Field>(
    cycle: usize,
    row: &JoltVmStage6Row,
    input: &Stage8StreamingRlcVectorMatrixProductInput<'_, F>,
    trace_rows: usize,
    result: &mut [F],
) {
    let dense = input.ram_inc_coefficient * F::from_i128(row.ram_increment)
        + input.rd_inc_coefficient * F::from_i128(row.rd_increment);
    if !dense.is_zero() {
        let address_columns = 1usize << input.committed_chunk_bits;
        let flat = input.trace_polynomial_order.address_cycle_to_index(
            0,
            cycle,
            address_columns,
            trace_rows,
        );
        accumulate_stage8_flat_value(flat, dense, input.left_vec, result);
    }

    accumulate_stage8_ra_coefficients(
        cycle,
        row.instruction_lookup_index,
        input.instruction_coefficients,
        input.committed_chunk_bits,
        input.trace_polynomial_order,
        trace_rows,
        input.left_vec,
        result,
    );
    accumulate_stage8_ra_coefficients(
        cycle,
        row.bytecode_index as u128,
        input.bytecode_coefficients,
        input.committed_chunk_bits,
        input.trace_polynomial_order,
        trace_rows,
        input.left_vec,
        result,
    );
    if let Some(address) = row.remapped_ram_address {
        accumulate_stage8_ra_coefficients(
            cycle,
            address as u128,
            input.ram_coefficients,
            input.committed_chunk_bits,
            input.trace_polynomial_order,
            trace_rows,
            input.left_vec,
            result,
        );
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "hot coefficient accumulator keeps loop inputs explicit to avoid wrapper churn"
)]
fn accumulate_stage8_ra_coefficients<F: Field>(
    cycle: usize,
    value: u128,
    coefficients: &[F],
    committed_chunk_bits: usize,
    trace_polynomial_order: TracePolynomialOrder,
    trace_rows: usize,
    left_vec: &[F],
    result: &mut [F],
) {
    let address_columns = 1usize << committed_chunk_bits;
    for (index, &coefficient) in coefficients.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let hot_column = stage8_ra_chunk(value, index, coefficients.len(), committed_chunk_bits);
        let flat = trace_polynomial_order.address_cycle_to_index(
            hot_column,
            cycle,
            address_columns,
            trace_rows,
        );
        accumulate_stage8_flat_value(flat, coefficient, left_vec, result);
    }
}

#[inline]
fn accumulate_stage8_flat_value<F: Field>(flat: usize, value: F, left_vec: &[F], result: &mut [F]) {
    let col = flat % result.len();
    let row = flat / result.len();
    result[col] += left_vec[row] * value;
}

#[inline]
fn stage8_ra_chunk(value: u128, index: usize, chunks: usize, committed_chunk_bits: usize) -> usize {
    let remaining = chunks - index - 1;
    let shift = remaining * committed_chunk_bits;
    let mask = (1u128 << committed_chunk_bits) - 1;
    ((value >> shift) & mask) as usize
}

fn dense_vector_matrix_product<F: Field>(
    dense_rlc: &[F],
    left_vec: &[F],
    num_columns: usize,
) -> Vec<F> {
    if dense_rlc.is_empty() {
        return vec![F::zero(); num_columns];
    }

    (0..num_columns)
        .into_par_iter()
        .map(|col_index| {
            dense_rlc
                .iter()
                .skip(col_index)
                .step_by(num_columns)
                .zip(left_vec.iter())
                .map(|(&dense, &left)| dense * left)
                .sum()
        })
        .collect()
}

fn accumulate_one_hot_vector_matrix_product<F: Field>(
    k: usize,
    indices: &[Option<u8>],
    left_vec: &[F],
    coeff: F,
    index_order: OneHotIndexOrder,
    result: &mut [F],
) {
    validate_one_hot_vmp_inputs(k, indices, left_vec, result.len());
    let t = indices.len();
    let num_columns = result.len();

    if index_order == OneHotIndexOrder::ColumnMajor && t >= num_columns {
        assert_eq!(
            t % num_columns,
            0,
            "CycleMajor one-hot VMP requires T divisible by num_columns"
        );
        let rows_per_k = t / num_columns;
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(col_index, dest)| {
                let mut col_dot_product = F::zero();
                for (row_offset, cycle) in (col_index..t).step_by(num_columns).enumerate() {
                    // SAFETY: the upfront validation proves `cycle < indices.len()`, every
                    // present address is `< k`, and `left_vec.len() == k * rows_per_k`.
                    if let Some(address) = unsafe { *indices.get_unchecked(cycle) } {
                        let row_index = usize::from(address) * rows_per_k + row_offset;
                        // SAFETY: `row_offset < rows_per_k` by construction, so the computed
                        // row lies within the validated left-vector matrix shape.
                        col_dot_product += unsafe { *left_vec.get_unchecked(row_index) };
                    }
                }
                *dest += coeff * col_dot_product;
            });
        return;
    }

    for (cycle, address) in indices.iter().enumerate() {
        if let Some(address) = address {
            let global_index = match index_order {
                OneHotIndexOrder::RowMajor => cycle * k + usize::from(*address),
                OneHotIndexOrder::ColumnMajor => usize::from(*address) * t + cycle,
            };
            let row_index = global_index / num_columns;
            let col_index = global_index % num_columns;
            result[col_index] += coeff * left_vec[row_index];
        }
    }
}

fn dense_split_eq_parallel<F: Field>(values: &[F], eq_one: &[F], eq_two: &[F]) -> F {
    (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let partial_sum = (0..eq_two.len())
                .into_par_iter()
                .map(|x2| {
                    let idx = x1 * eq_two.len() + x2;
                    eq_two[x2].mul_01_optimized(values[idx])
                })
                .reduce(F::zero, |acc, val| acc + val);
            eq_one[x1].mul_01_optimized(partial_sum)
        })
        .reduce(F::zero, |acc, val| acc + val)
}

fn dense_split_eq_serial<F: Field>(values: &[F], eq_one: &[F], eq_two: &[F]) -> F {
    (0..eq_one.len())
        .map(|x1| {
            let partial_sum = (0..eq_two.len())
                .map(|x2| {
                    let idx = x1 * eq_two.len() + x2;
                    eq_two[x2].mul_01_optimized(values[idx])
                })
                .fold(F::zero(), |acc, val| acc + val);
            eq_one[x1].mul_01_optimized(partial_sum)
        })
        .fold(F::zero(), |acc, val| acc + val)
}

fn compact_split_eq_parallel<T, F>(coeffs: &[T], eq_one: &[F], eq_two: &[F]) -> F
where
    T: CpuCompactScalar,
    F: Field,
{
    (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let partial_sum = (0..eq_two.len())
                .into_par_iter()
                .map(|x2| {
                    let idx = x1 * eq_two.len() + x2;
                    coeffs[idx].field_mul(eq_two[x2])
                })
                .reduce(F::zero, |acc, val| acc + val);
            partial_sum.mul_01_optimized(eq_one[x1])
        })
        .reduce(F::zero, |acc, val| acc + val)
}

fn compact_split_eq_serial<T, F>(coeffs: &[T], eq_one: &[F], eq_two: &[F]) -> F
where
    T: CpuCompactScalar,
    F: Field,
{
    (0..eq_one.len())
        .map(|x1| {
            let partial_sum = (0..eq_two.len())
                .map(|x2| {
                    let idx = x1 * eq_two.len() + x2;
                    coeffs[idx].field_mul(eq_two[x2])
                })
                .fold(F::zero(), |acc, val| acc + val);
            partial_sum.mul_01_optimized(eq_one[x1])
        })
        .fold(F::zero(), |acc, val| acc + val)
}

fn dense_batch_split_eq_parallel<F: Field>(polys: &[&[F]], eq_one: &[F], eq_two: &[F]) -> Vec<F> {
    let num_polys = polys.len();
    (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];
            let inner_sums = (0..eq_two.len())
                .into_par_iter()
                .map(|x2| {
                    let eq2_val = eq_two[x2];
                    let idx = x1 * eq_two.len() + x2;
                    polys
                        .iter()
                        .map(|poly| eq2_val.mul_01_optimized(poly[idx]))
                        .collect::<Vec<_>>()
                })
                .reduce(
                    || vec![F::zero(); num_polys],
                    |mut acc, item| {
                        for i in 0..num_polys {
                            acc[i] += item[i];
                        }
                        acc
                    },
                );
            inner_sums
                .into_iter()
                .map(|sum| eq1_val.mul_01_optimized(sum))
                .collect::<Vec<_>>()
        })
        .reduce(
            || vec![F::zero(); num_polys],
            |mut acc, item| {
                for i in 0..num_polys {
                    acc[i] += item[i];
                }
                acc
            },
        )
}

fn dense_inside_out_parallel<F: Field>(values: &[F], point: &[F]) -> F {
    let mut current = values.par_iter().copied().collect::<Vec<_>>();
    inside_out_fold(&mut current, point);
    current[0]
}

fn dense_inside_out_serial<F: Field>(values: &[F], point: &[F]) -> F {
    let mut current = values.to_vec();
    inside_out_fold_serial(&mut current, point);
    current[0]
}

fn compact_inside_out_parallel<T, F>(coeffs: &[T], point: &[F]) -> F
where
    T: CpuCompactScalar,
    F: Field,
{
    let mut current = coeffs
        .par_iter()
        .map(|&coeff| coeff.to_field())
        .collect::<Vec<F>>();
    inside_out_fold(&mut current, point);
    current[0]
}

fn compact_inside_out_serial<T, F>(coeffs: &[T], point: &[F]) -> F
where
    T: CpuCompactScalar,
    F: Field,
{
    let mut current = coeffs
        .iter()
        .map(|&coeff| coeff.to_field())
        .collect::<Vec<F>>();
    inside_out_fold_serial(&mut current, point);
    current[0]
}

fn inside_out_fold<F: Field>(current: &mut [F], point: &[F]) {
    let m = point.len();
    for i in (0..m).rev() {
        let stride = 1 << i;
        let r_val = point[m - 1 - i];
        let (evals_left, evals_right) = current.split_at_mut(stride);
        let (evals_right, _) = evals_right.split_at_mut(stride);

        evals_left
            .par_iter_mut()
            .zip(evals_right.par_iter())
            .for_each(|(x, y)| {
                let slope = *y - *x;
                if slope.is_zero() {
                    return;
                }
                if slope.is_one() {
                    *x += r_val;
                } else {
                    *x += r_val * slope;
                }
            });
    }
}

fn inside_out_fold_serial<F: Field>(current: &mut [F], point: &[F]) {
    let m = point.len();
    for i in (0..m).rev() {
        let stride = 1 << i;
        let r_val = point[m - 1 - i];
        for j in 0..stride {
            let f0 = current[j];
            let f1 = current[j + stride];
            let slope = f1 - f0;
            if slope.is_zero() {
                current[j] = f0;
            } else if slope.is_one() {
                current[j] = f0 + r_val;
            } else {
                current[j] = f0 + r_val * slope;
            }
        }
    }
}

#[inline]
fn bind_scalar_pair<T, F>(a: T, b: T, r: F) -> F
where
    T: CpuCompactScalar,
    F: Field,
{
    match a.cmp(&b) {
        Ordering::Equal => a.to_field(),
        Ordering::Less => a.to_field::<F>() + b.diff_mul_field(a, r),
        Ordering::Greater => a.to_field::<F>() - a.diff_mul_field(b, r),
    }
}

fn validate_first_bind_len(len: usize) -> usize {
    assert!(
        len > 1 && len.is_power_of_two(),
        "compact bind input length must be a power of two greater than one, got {len}"
    );
    len / 2
}

fn validate_split_eq_len(values_len: usize, eq_one_len: usize, eq_two_len: usize) {
    assert_eq!(
        values_len,
        eq_one_len * eq_two_len,
        "split-eq input length mismatch: values={values_len}, eq_one={eq_one_len}, eq_two={eq_two_len}"
    );
}

fn validate_batch_eval_len<F: Field>(polys: &[&[F]], point_len: usize) {
    let expected = 1usize << point_len;
    for (index, poly) in polys.iter().enumerate() {
        assert_eq!(
            poly.len(),
            expected,
            "batch evaluate input length mismatch at polynomial {index}: values={}, point={point_len}",
            poly.len()
        );
    }
}

fn validate_batch_split_eq_len<F: Field>(polys: &[&[F]], eq_one_len: usize, eq_two_len: usize) {
    let expected = eq_one_len * eq_two_len;
    for (index, poly) in polys.iter().enumerate() {
        assert_eq!(
            poly.len(),
            expected,
            "batch split-eq input length mismatch at polynomial {index}: values={}, eq_one={eq_one_len}, eq_two={eq_two_len}",
            poly.len()
        );
    }
}

fn validate_inside_out_len(values_len: usize, point_len: usize) {
    assert_eq!(
        values_len,
        1usize << point_len,
        "inside-out input length mismatch: values={values_len}, point={point_len}"
    );
}

fn validate_one_hot_evaluate_inputs(k: usize, indices: &[Option<u8>], point_len: usize) {
    assert!(
        k > 0 && k.is_power_of_two() && k <= 1usize << u8::BITS,
        "one-hot address count must be a power of two <= 256, got {k}"
    );
    assert!(
        !indices.is_empty() && indices.len().is_power_of_two(),
        "one-hot cycle count must be a nonzero power of two, got {}",
        indices.len()
    );
    assert_eq!(
        point_len,
        k.trailing_zeros() as usize + indices.len().trailing_zeros() as usize,
        "one-hot evaluation point length mismatch"
    );
    debug_assert!(
        indices
            .iter()
            .all(|index| index.is_none_or(|index| usize::from(index) < k)),
        "one-hot index out of range for k={k}"
    );
}

fn validate_one_hot_vmp_inputs<F: Field>(
    k: usize,
    indices: &[Option<u8>],
    left_vec: &[F],
    num_columns: usize,
) {
    assert!(
        k > 0 && k.is_power_of_two() && k <= 1usize << u8::BITS,
        "one-hot address count must be a power of two <= 256, got {k}"
    );
    assert!(
        !indices.is_empty() && indices.len().is_power_of_two(),
        "one-hot cycle count must be a nonzero power of two, got {}",
        indices.len()
    );
    let total_len = k * indices.len();
    assert!(
        num_columns > 0 && num_columns.is_power_of_two() && total_len.is_multiple_of(num_columns),
        "one-hot VMP num_columns must be a nonzero power of two dividing total length"
    );
    assert_eq!(
        left_vec.len(),
        total_len / num_columns,
        "one-hot VMP left vector length mismatch"
    );
    debug_assert!(
        indices
            .iter()
            .all(|index| index.is_none_or(|index| usize::from(index) < k)),
        "one-hot index out of range for k={k}"
    );
}

fn validate_materialized_rlc_vmp_inputs<F: Field>(
    dense_rlc: &[F],
    left_vec: &[F],
    num_columns: usize,
) {
    assert!(
        num_columns > 0 && num_columns.is_power_of_two(),
        "RLC VMP num_columns must be a nonzero power of two, got {num_columns}"
    );
    if dense_rlc.is_empty() {
        return;
    }
    assert!(
        dense_rlc.len().is_multiple_of(num_columns),
        "RLC dense component length must be divisible by num_columns"
    );
    assert!(
        left_vec.len() >= dense_rlc.len() / num_columns,
        "RLC VMP left vector is too short for dense component"
    );
}

fn validate_stage8_streaming_rlc_vmp_input<F: Field>(
    input: &Stage8StreamingRlcVectorMatrixProductInput<'_, F>,
) {
    assert!(
        input.committed_chunk_bits > 0
            && input.committed_chunk_bits < u128::BITS as usize
            && input.committed_chunk_bits <= u8::BITS as usize,
        "Stage 8 RLC committed_chunk_bits must be in 1..=8, got {}",
        input.committed_chunk_bits
    );
    let trace_rows = 1usize << input.log_t;
    assert_eq!(
        input.rows.len(),
        trace_rows,
        "Stage 8 RLC row count mismatch"
    );
    let total_len = trace_rows << input.committed_chunk_bits;
    assert!(
        input.num_columns > 0
            && input.num_columns.is_power_of_two()
            && total_len.is_multiple_of(input.num_columns),
        "Stage 8 RLC num_columns must be a nonzero power of two dividing the committed domain"
    );
    assert_eq!(
        input.left_vec.len(),
        total_len / input.num_columns,
        "Stage 8 RLC left vector length mismatch"
    );
}

#[inline]
fn signed_128_field_mul<F: Field>(value: S128, r: F) -> F {
    let magnitude = value.magnitude_as_u128();
    if value.is_positive {
        r.mul_u128(magnitude)
    } else {
        -r.mul_u128(magnitude)
    }
}
