use jolt_field::Field;
use jolt_poly::lagrange::{
    centered_interpolate_coeffs_array, centered_lagrange_evals_array, centered_lagrange_evaluate,
    centered_lagrange_evaluate_many, centered_lagrange_kernel, CenteredIntegerDomainError,
};

pub fn centered_evals<F: Field, const N: usize>(
    point: F,
) -> Result<[F; N], CenteredIntegerDomainError> {
    centered_lagrange_evals_array::<F, N>(point)
}

pub fn centered_kernel<F: Field>(
    domain_size: usize,
    x: F,
    y: F,
) -> Result<F, CenteredIntegerDomainError> {
    centered_lagrange_kernel(domain_size, x, y)
}

pub fn centered_evaluate<F: Field, const N: usize>(
    values: &[F; N],
    point: F,
) -> Result<F, CenteredIntegerDomainError> {
    centered_lagrange_evaluate(values, point)
}

pub fn centered_evaluate_many<F: Field, const N: usize>(
    values: &[F; N],
    points: &[F],
) -> Result<Vec<F>, CenteredIntegerDomainError> {
    centered_lagrange_evaluate_many(values, points)
}

pub fn centered_interpolate_coeffs<F: Field, const N: usize>(
    values: &[F; N],
) -> Result<[F; N], CenteredIntegerDomainError> {
    centered_interpolate_coeffs_array(values)
}
