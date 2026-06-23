use super::*;
use crate::validation::{is_i8_log_basis, validate_i8_input_log_basis};

#[inline]
pub(super) fn accumulate_pointwise_product_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    lhs: &CyclotomicCrtNtt<W, K, D>,
    rhs: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    acc.add_assign_pointwise_mul_with_params(lhs, rhs, params);
}

#[inline]
pub(super) fn is_zero_plane<const D: usize>(plane: &[i8; D]) -> bool {
    plane.iter().all(|&d| d == 0)
}

#[inline]
pub(super) fn is_zero_centered_row<const D: usize>(row: &[i32; D]) -> bool {
    row.iter().all(|&d| d == 0)
}

pub(super) fn quotient_from_cyclic_and_negacyclic<F: FieldCore + HalvingField, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    negacyclic: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc = cyclic.coefficients();
    let neg = negacyclic.coefficients();
    CyclotomicRing::from_coefficients(from_fn(|k| (cyc[k] - neg[k]).half()))
}

pub(super) fn add_cyclic_product_into<F: FieldCore, const D: usize>(
    acc: &mut CyclotomicRing<F, D>,
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) {
    for (i, &a) in lhs.coefficients().iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs.coefficients().iter().enumerate() {
            if !b.is_zero() {
                acc.coefficients_mut()[(i + j) % D] += a * b;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) const TARGET_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
pub(super) const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;
pub(super) const CENTERED_LUT_MAX_ABS: u32 = (1 << 16) - 1;
// Row-count ceiling for the block-parallel matvec. Commitments up to `n_a == 7`
// still parallelize over blocks through the generic accumulator loop instead of
// falling back to the column-tiled path, which has too few tiles to scale at
// high nv. The block-parallel and column-tiled paths produce identical ring
// output (per-step `reduce_range` accumulation + canonicalizing `to_ring`), so
// raising the cap is a pure performance change.
pub(super) const SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS: usize = 7;
pub(super) const SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS: usize = 16;

#[inline]
pub(super) fn validate_i8_log_basis(log_basis: u32) -> Result<(), AkitaError> {
    validate_i8_input_log_basis(log_basis, "for i8 NTT kernels")
}

#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
#[inline]
pub(super) fn add_ntt_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    other: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    for k in 0..K {
        let prime = params.primes[k];
        for d in 0..D {
            let sum =
                MontCoeff::from_raw(acc.limbs[k][d].raw().wrapping_add(other.limbs[k][d].raw()));
            acc.limbs[k][d] = prime.reduce_range(sum);
        }
    }
}

#[inline]
pub(super) fn balanced_digit_abs_bound(log_basis: u32) -> u64 {
    debug_assert!(is_i8_log_basis(log_basis));
    1u64 << (log_basis - 1)
}

#[inline]
pub(super) fn digit_rows_within_digit_bound<const D: usize>(
    rows: &[[i8; D]],
    len: usize,
    digit_bound: u64,
) -> bool {
    let bound = i16::try_from(digit_bound).unwrap_or(i16::MAX);
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| (-bound..bound).contains(&i16::from(coeff)))
}

#[inline]
pub(super) fn validate_digit_rows_for_log_basis<const D: usize>(
    rows: &[[i8; D]],
    len: usize,
    log_basis: u32,
    context: &str,
) -> Result<(), AkitaError> {
    let bound = 1i16 << (log_basis - 1);
    if rows
        .iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| (-bound..bound).contains(&i16::from(coeff)))
    {
        Ok(())
    } else {
        Err(AkitaError::InvalidInput(format!(
            "predecomposed digit row contains a coefficient outside the balanced log_basis range {context}"
        )))
    }
}

#[inline]
pub(super) fn aligned_i8_tile_width(
    raw_width: usize,
    inner_width: usize,
    num_digits: usize,
) -> usize {
    debug_assert!(inner_width > 0);
    debug_assert!(num_digits > 0);

    if inner_width <= num_digits {
        return inner_width;
    }

    let clamped = raw_width.min(inner_width).max(num_digits);
    ((clamped / num_digits).max(1)) * num_digits
}

#[inline]
pub(super) fn capacity_safe_i8_chunk_width(
    safe_width: usize,
    inner_width: usize,
    num_digits: usize,
) -> usize {
    debug_assert!(safe_width > 0);
    debug_assert!(inner_width > 0);
    debug_assert!(num_digits > 0);

    if safe_width < num_digits {
        safe_width.min(inner_width)
    } else {
        aligned_i8_tile_width(safe_width, inner_width, num_digits).min(safe_width)
    }
}

#[cfg(feature = "parallel")]
#[inline]
pub(super) fn add_ntt_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    other: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    #[cfg(target_arch = "aarch64")]
    if neon::use_neon_ntt() {
        for k in 0..K {
            let prime = params.primes[k];
            unsafe {
                if size_of::<W>() == size_of::<i32>() {
                    neon::add_reduce_i32(
                        acc.limbs[k].as_mut_ptr() as *mut i32,
                        other.limbs[k].as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                    );
                } else {
                    neon::add_reduce_i16(
                        acc.limbs[k].as_mut_ptr() as *mut i16,
                        other.limbs[k].as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                    );
                }
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(mode) = avx::avx_ntt_mode() {
        if size_of::<W>() == size_of::<i16>() && avx::use_avx2_transform_ntt() {
            for k in 0..K {
                let prime = params.primes[k];
                unsafe {
                    avx::add_reduce_i16(
                        acc.limbs[k].as_mut_ptr() as *mut i16,
                        other.limbs[k].as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                    );
                }
            }
            return;
        }
        if size_of::<W>() == size_of::<i32>() {
            for k in 0..K {
                let prime = params.primes[k];
                unsafe {
                    match mode {
                        AvxNttMode::Avx2 => avx::add_reduce_i32(
                            acc.limbs[k].as_mut_ptr() as *mut i32,
                            other.limbs[k].as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                        ),
                        AvxNttMode::Avx512 => avx::add_reduce_i32_avx512(
                            acc.limbs[k].as_mut_ptr() as *mut i32,
                            other.limbs[k].as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                        ),
                    }
                }
            }
            return;
        }
    }

    for k in 0..K {
        let prime = params.primes[k];
        for d in 0..D {
            let sum =
                MontCoeff::from_raw(acc.limbs[k][d].raw().wrapping_add(other.limbs[k][d].raw()));
            acc.limbs[k][d] = prime.reduce_range(sum);
        }
    }
}
