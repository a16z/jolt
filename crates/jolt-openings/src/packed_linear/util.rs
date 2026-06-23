use jolt_field::Field;
use jolt_poly::MultilinearPoly;

use crate::OpeningsError;

pub(super) fn checked_domain_size(num_vars: usize) -> Result<usize, OpeningsError> {
    if num_vars >= usize::BITS as usize {
        return Err(invalid_batch(format!(
            "packed linear dimension {num_vars} exceeds usize bit width"
        )));
    }
    Ok(1usize << num_vars)
}

pub(super) fn log2_power_of_two(value: usize, label: &'static str) -> Result<usize, OpeningsError> {
    if value == 0 || !value.is_power_of_two() {
        return Err(invalid_batch(format!(
            "{label} must be a nonzero power of two"
        )));
    }
    Ok(value.trailing_zeros() as usize)
}

pub(super) fn offset_bit(offset: usize, bit: usize) -> bool {
    bit < usize::BITS as usize && ((offset >> bit) & 1) != 0
}

pub(super) fn polynomial_evaluations<F, P>(polynomial: &P) -> Vec<F>
where
    F: Field,
    P: MultilinearPoly<F>,
{
    let mut evals = Vec::with_capacity(1usize << polynomial.num_vars());
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}

pub(super) fn invalid_batch(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}
