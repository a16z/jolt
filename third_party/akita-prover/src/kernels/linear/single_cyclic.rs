use super::*;

/// Column-tiled mat-vec for a single pre-decomposed i8 digit vector.
///
/// Same tiling strategy as [`mat_vec_mul_ntt_i8`] but for a single
/// input vector of i8 digit planes (already decomposed with `log_basis <= 6`).
/// Tiles the matrix columns to keep each tile in L2, eliminating the full `ntt_vec`
/// materialization of the non-tiled path.
/// Tile width is auto-computed from ring parameters and target L2 cache size.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_single_i8")]
pub fn mat_vec_mul_ntt_single_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    vec: &[[i8; D]],
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    validate_digit_rows_for_log_basis(
        vec,
        num_cols.min(vec.len()),
        log_basis,
        "for single predecomposed digit mat-vec",
    )?;
    Ok(match slot {
        NttSlotCache::Q32 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, log_basis, p)
        }
        NttSlotCache::Q64 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, log_basis, p)
        }
        NttSlotCache::Q128 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, log_basis, p)
        }
    })
}

/// Cyclic-domain variant of [`mat_vec_mul_ntt_single_i8`].
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_single_i8_cyclic")]
pub fn mat_vec_mul_ntt_single_i8_cyclic<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    vec: &[[i8; D]],
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    validate_digit_rows_for_log_basis(
        vec,
        num_cols.min(vec.len()),
        log_basis,
        "for cyclic single predecomposed digit mat-vec",
    )?;
    Ok(match slot {
        NttSlotCache::Q32 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, log_basis, p)
        }
        NttSlotCache::Q64 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, log_basis, p)
        }
        NttSlotCache::Q128 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, log_basis, p)
        }
    })
}

pub(super) fn mat_vec_mul_single_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    vec: &[[i8; D]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let vec_len = vec.len().min(inner_width);
    if vec_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }
    let digit_bound = balanced_digit_abs_bound(log_basis);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    debug_assert!(
        digit_rows_within_digit_bound::<D>(vec, vec_len, digit_bound),
        "single digit vector contains digits outside its log_basis range"
    );
    let chunk_width = safe_crt_chunk_width::<F, W, K, D>(params, vec_len, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    drive_single_chunked_matvec(
        n_a,
        vec_len,
        chunk_width,
        base_tile_width::<W, K, D>(),
        chunk_width,
        params,
        |accs, start, end| {
            for (j, digit) in vec[start..end].iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(acc, &mat_row[start + j], &ntt_d, params);
                }
            }
        },
        |acc, params| acc.to_ring_with_params(params),
    )
}

pub(super) fn mat_vec_mul_single_i8_cyclic_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    vec: &[[i8; D]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let vec_len = vec.len().min(inner_width);
    if vec_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }
    let digit_bound = balanced_digit_abs_bound(log_basis);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    debug_assert!(
        digit_rows_within_digit_bound::<D>(vec, vec_len, digit_bound),
        "single cyclic digit vector contains digits outside its log_basis range"
    );
    let chunk_width = safe_crt_chunk_width::<F, W, K, D>(params, vec_len, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    drive_single_chunked_matvec(
        n_a,
        vec_len,
        chunk_width,
        base_tile_width::<W, K, D>(),
        chunk_width,
        params,
        |accs, start, end| {
            for (j, digit) in vec[start..end].iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut);
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(acc, &mat_row[start + j], &ntt_d, params);
                }
            }
        },
        |acc, params| acc.to_ring_cyclic(params),
    )
}
