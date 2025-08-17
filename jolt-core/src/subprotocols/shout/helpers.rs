use crate::{field::JoltField, poly::eq_poly::EqPolynomial};
use rayon::prelude::*;

/// Constructs the evaluations of the final univariate polynomial for sumcheck in parallel.
///
/// - For `i == 1`: `result[1] = previous_claim - val_claim * univariate_poly_evals[0]`
/// - For other `i`: `result[i] = val_claim * univariate_poly_evals[i]`
///
/// # Arguments
/// - `univariate_poly_evals`: Vector of evaluations of the product polynomial (length `degree`)
/// - `val_claim`: val(r_address) which needs to be nmultiplied to each of the evals
/// - `previous_claim`: Claimed value for this round which is meant to equal result[0] + result[1]
///
/// # Returns
/// A vector `result` of length `degree+1` representing the evaluations of the final univariate polynomial
pub(crate) fn construct_final_sumcheck_evals<F: JoltField>(
    univariate_poly_evals: &[F],
    val_claim: F,
    previous_claim: F,
    degree: usize,
) -> Vec<F> {
    let first_term = val_claim
        * univariate_poly_evals
            .first()
            .expect("univariate_poly_evals must be non-empty");

    let result: Vec<_> = (0..=degree)
        .into_par_iter()
        .map(|i| match i {
            1 => previous_claim - first_term,
            _ => {
                let eval_idx = if i > 1 { i - 1 } else { i };
                val_claim * univariate_poly_evals[eval_idx]
            }
        })
        .collect();

    result
}

pub(crate) fn compute_eq_taus_parallel<F: JoltField>(
    r_address: &[F], // length must be d * log_N = \log K
    d: usize,
    log_n: usize,
) -> Vec<Vec<F>> {
    assert_eq!(r_address.len(), d * log_n);

    (0..d)
        .into_par_iter()
        .map(|j| {
            let start = j * log_n;
            let end = start + log_n;

            let mut tau_bits = r_address[start..end].to_vec();
            tau_bits.reverse(); // BigEndian
            EqPolynomial::evals(&tau_bits)
        })
        .collect()
}

pub(crate) fn compute_eq_taus_serial<F: JoltField>(
    r_address: &[F], // length must be d * log_N = \log K
    d: usize,
    log_n: usize,
) -> Vec<Vec<F>> {
    assert_eq!(r_address.len(), d * log_n);

    (0..d)
        .map(|j| {
            let start = j * log_n;
            let end = start + log_n;

            let mut tau_bits = r_address[start..end].to_vec();
            tau_bits.reverse(); // BigEndian

            EqPolynomial::evals(&tau_bits)
        })
        .collect()
}

/// Extracts the `j`-th digit (0-indexed from the most significant digit) of `addr`
/// when written in base `base (N=K^{1/d})`, assuming the total number of digits is `d`.
///
/// # Arguments
///
/// - `addr`: The integer address to decompose.
/// - `j`: The digit index (0 = most significant, `d - 1` = least significant).
/// - `d`: The total number of digits in the base-`base` representation.
/// - `base`: The numerical base (e.g., `K^{1/d}`).
///
/// # Returns
///
/// The `j`-th digit of `addr` in base `base`.
///
/// # Panics
///
/// Panics if `base.pow(d as u32)` overflows or if `j >= d`.
///
/// # Examples
///
/// ```rust
/// let addr = 10;
/// let base = 4;
/// let d = 2; // because 10 in base 4 is "2 2"
///
/// assert_eq!(digit_j(addr, 0, d, base), 2); // most significant digit
/// assert_eq!(digit_j(addr, 1, d, base), 2); // least significant digit
/// ```
pub(crate) fn digit_j_of(addr: usize, j: usize, d: usize, base: usize) -> usize {
    // Convert from most-significant-first index (0 = most significant)
    let exp = d - 1 - j;
    (addr / base.pow(exp as u32)) % base
}

pub(crate) fn construct_vector_c_in_shout<F: JoltField>(
    table_size: usize,
    read_addresses: &[usize],
    e_star: &Vec<F>,
) -> Vec<F> {
    let c = read_addresses
        .par_iter()
        .zip(e_star.par_iter())
        .fold(
            || vec![F::zero(); table_size],
            |mut acc, (&address, &val)| {
                if address < table_size {
                    acc[address] += val;
                }
                acc
            },
        )
        .reduce(
            || vec![F::zero(); table_size],
            |mut a, b| {
                for (ai, bi) in a.iter_mut().zip(b) {
                    *ai += bi;
                }
                a
            },
        );

    c
}
