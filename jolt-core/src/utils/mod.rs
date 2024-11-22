#![allow(dead_code)]
use crate::field::JoltField;

use ark_std::test_rng;
use rayon::prelude::*;

pub mod errors;
pub mod gaussian_elimination;
pub mod instruction_utils;
pub mod math;
pub mod profiling;
pub mod sol_types;
pub mod thread;
pub mod transcript;

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```ignore
/// use jolt_core::utils::index_to_field_bitvector;
/// # use ark_bn254::Fr;
/// # use ark_std::{One, Zero};
/// let zero = Fr::zero();
/// let one = Fr::one();
///
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 1), vec![one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 3), vec![zero, zero, one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 7), vec![zero, zero, zero, zero, zero, zero, one]);
/// ```
pub fn index_to_field_bitvector<F: JoltField>(value: usize, bits: usize) -> Vec<F> {
    assert!(value < 1 << bits);

    let mut bitvector: Vec<F> = Vec::with_capacity(bits);

    for i in (0..bits).rev() {
        if (value >> i) & 1 == 1 {
            bitvector.push(F::one());
        } else {
            bitvector.push(F::zero());
        }
    }
    bitvector
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| *a_i * *b_i)
        .sum()
}

/// Compute dotproduct optimized for values being 0 / 1
#[tracing::instrument(skip_all)]
pub fn compute_dotproduct_low_optimized<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| mul_0_1_optimized(a_i, b_i))
        .sum()
}

#[inline(always)]
pub fn mul_0_1_optimized<F: JoltField>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        *a * *b
    }
}

#[inline(always)]
pub fn mul_0_optimized<F: JoltField>(likely_zero: &F, x: &F) -> F {
    if likely_zero.is_zero() {
        F::zero()
    } else {
        *likely_zero * *x
    }
}

/// Checks if `num` is a power of 2.
pub fn is_power_of_two(num: usize) -> bool {
    num != 0 && num.is_power_of_two()
}

/// Take the first two `num_bits` chunks of `item` (from the right / LSB) and return them as a tuple `(high_chunk, low_chunk)`.
///
/// If `item` is shorter than `2 * num_bits`, the remaining bits are zero-padded.
///
/// If `item` is longer than `2 * num_bits`, the remaining bits are discarded.
///
/// # Examples
///
/// ```
/// use jolt_core::utils::split_bits;
///
/// assert_eq!(split_bits(0b101000, 2), (0b10, 0b00));
/// assert_eq!(split_bits(0b101000, 3), (0b101, 0b000));
/// assert_eq!(split_bits(0b101000, 4), (0b0010, 0b1000));
/// ```
pub fn split_bits(item: usize, num_bits: usize) -> (usize, usize) {
    let max_value = (1 << num_bits) - 1; // Calculate the maximum value that can be represented with num_bits

    let low_chunk = item & max_value; // Extract the lower bits
    let high_chunk = (item >> num_bits) & max_value; // Shift the item to the right and extract the next set of bits

    (high_chunk, low_chunk)
}

/// Generate a random point with `memory_bits` field elements.
pub fn gen_random_point<F: JoltField>(memory_bits: usize) -> Vec<F> {
    let mut rng = test_rng();
    let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
    for _ in 0..memory_bits {
        r_i.push(F::random(&mut rng));
    }
    r_i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split() {
        assert_eq!(split_bits(0b00_01, 2), (0, 1));
        assert_eq!(split_bits(0b10_01, 2), (2, 1));
    }
}
