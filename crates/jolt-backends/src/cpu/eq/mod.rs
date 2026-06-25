use jolt_field::Field;
use jolt_poly::{EqPolynomial, TensorEqTable};

pub fn evals<F: Field>(point: &[F], scaling_factor: Option<F>) -> Vec<F> {
    EqPolynomial::<F>::evals(point, scaling_factor)
}

pub fn evals_cached<F: Field>(point: &[F], scaling_factor: Option<F>) -> Vec<Vec<F>> {
    EqPolynomial::<F>::evals_cached(point, scaling_factor)
}

pub fn evals_cached_rev<F: Field>(point: &[F], scaling_factor: Option<F>) -> Vec<Vec<F>> {
    EqPolynomial::<F>::evals_cached_rev(point, scaling_factor)
}

pub fn evals_for_aligned_block<F: Field>(
    point: &[F],
    start_index: usize,
    block_size: usize,
) -> Vec<F> {
    EqPolynomial::<F>::evals_for_aligned_block(point, start_index, block_size)
}

pub fn evals_for_max_aligned_block<F: Field>(
    point: &[F],
    start_index: usize,
    remaining_len: usize,
) -> (usize, Vec<F>) {
    EqPolynomial::<F>::evals_for_max_aligned_block(point, start_index, remaining_len)
}

pub fn tensor_table<F: Field>(point: &[F]) -> TensorEqTable<F> {
    TensorEqTable::new(point)
}
