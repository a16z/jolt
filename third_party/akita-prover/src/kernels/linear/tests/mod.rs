use super::{
    aligned_i8_tile_width, balanced_digit_abs_bound, decompose_block_i8, fused_split_eq_quotients,
    mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many, mat_vec_mul_digits_i8_block_parallel,
    mat_vec_mul_digits_i8_strided_block_parallel, mat_vec_mul_digits_i8_strided_with_params,
    mat_vec_mul_digits_i8_with_params, mat_vec_mul_i8_dense_single_row_with_params,
    mat_vec_mul_i8_dense_with_params, mat_vec_mul_i8_strided_with_params,
    mat_vec_mul_i8_with_params, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8_dense_single_row,
    mat_vec_mul_ntt_raw_i8_strided, mat_vec_mul_ntt_single_i8_cyclic, mat_vec_mul_unchecked,
    precompute_dense_mat_ntt_with_params,
};
use crate::kernels::crt_ntt::{build_ntt_slot, select_crt_ntt_params, ProtocolCrtNttParams};
use akita_algebra::ntt::{
    tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q64_NUM_PRIMES},
    PrimeWidth,
};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing};
use akita_field::{CanonicalField, FieldCore, Fp64, Prime128Offset275, Prime64Offset59};
use akita_types::layout::FlatMatrix;

fn centered_i32_ring<F: akita_field::CanonicalField, const D: usize>(
    coeffs: &[i32; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(coeffs[idx] as i64)))
}

fn cyclic_product<F: akita_field::FieldCore, const D: usize>(
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let mut out = CyclotomicRing::<F, D>::zero();
    for (i, &a) in lhs.coefficients().iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs.coefficients().iter().enumerate() {
            if !b.is_zero() {
                out.coefficients_mut()[(i + j) % D] += a * b;
            }
        }
    }
    out
}

fn mat_vec_mul_i8_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_with_params(ntt_mat, blocks, num_digits, log_basis, params)
}

fn mat_vec_mul_i8_dense_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_dense_with_params(ntt_mat, blocks, num_digits, log_basis, params)
}

fn mat_vec_mul_i8_strided_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_strided_with_params(
        ntt_mat, coeffs, num_blocks, block_len, num_digits, log_basis, params,
    )
}

fn mat_vec_mul_digits_i8_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params(ntt_mat, blocks, log_basis, params)
}

fn mat_vec_mul_digits_i8_strided_with_params_for_log_basis<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_strided_with_params(
        ntt_mat, coeffs, num_blocks, block_len, log_basis, params,
    )
}

fn quotient_from_cyclic_and_negacyclic<
    F: akita_field::FieldCore + akita_field::HalvingField,
    const D: usize,
>(
    cyclic: &CyclotomicRing<F, D>,
    negacyclic: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc = cyclic.coefficients();
    let neg = negacyclic.coefficients();
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| (cyc[idx] - neg[idx]).half()))
}

fn schoolbook_digit_mat_vec<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    blocks: &[Vec<[i8; D]>],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    blocks
        .iter()
        .map(|block| {
            mat.iter()
                .map(|row| {
                    row.iter().zip(block.iter()).fold(
                        CyclotomicRing::<F, D>::zero(),
                        |mut acc, (lhs, digit)| {
                            let rhs = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                                F::from_i64(i64::from(digit[k]))
                            }));
                            acc += *lhs * rhs;
                            acc
                        },
                    )
                })
                .collect()
        })
        .collect()
}

mod api;
mod chunking;
mod crt_dense;
mod digit_matvec;
mod fused;
mod reduced_profiles;
