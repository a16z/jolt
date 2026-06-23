use super::*;

pub(super) fn mat_vec_mul_digits_i8_with_params<
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
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, true>(ntt_mat, blocks, log_basis, params)
}

pub(super) fn mat_vec_mul_dense_digits_i8_with_params<
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
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, false>(ntt_mat, blocks, log_basis, params)
}

pub(super) fn mat_vec_mul_digits_i8_with_params_impl<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_blocks = blocks.len();
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    debug_assert!(
        blocks
            .iter()
            .all(|block| digit_rows_within_digit_bound::<D>(
                block,
                inner_width.min(block.len()),
                digit_bound
            )),
        "predecomposed digit block contains digits outside its log_basis range"
    );
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width == max_data_width
    {
        if inner_width <= safe_width {
            return mat_vec_mul_digits_i8_block_parallel::<F, W, K, D, CHECK_ZERO>(
                ntt_mat,
                blocks,
                digit_bound,
                params,
            );
        }
        return mat_vec_mul_digits_i8_block_parallel_chunked::<F, W, K, D, CHECK_ZERO>(
            ntt_mat,
            blocks,
            inner_width,
            safe_width,
            digit_bound,
            params,
        );
    }

    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    drive_block_chunked_matvec(
        num_blocks,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accs, start, end| {
            if CHECK_ZERO {
                for (block_idx, block) in blocks.iter().enumerate() {
                    if start >= block.len() {
                        continue;
                    }
                    let block_tile_end = end.min(block.len());
                    let tile = &block[start..block_tile_end];
                    for (i, digit) in tile.iter().enumerate() {
                        if is_zero_plane(digit) {
                            continue;
                        }
                        let col = start + i;
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                        }
                    }
                }
            } else {
                for block_idx in 0..num_blocks {
                    let block = blocks[block_idx];
                    if start >= block.len() {
                        continue;
                    }
                    let block_tile_end = end.min(block.len());
                    let tile = &block[start..block_tile_end];
                    for (i, digit) in tile.iter().enumerate() {
                        let col = start + i;
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                        }
                    }
                }
            }
        },
    )
}

pub(super) fn mat_vec_mul_digits_i8_strided_with_params<
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
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    debug_assert!(
        digit_rows_within_digit_bound::<D>(
            coeffs,
            inner_width.saturating_mul(num_blocks),
            digit_bound
        ),
        "predecomposed strided digit block contains digits outside its log_basis range"
    );
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width <= safe_width
    {
        return mat_vec_mul_digits_i8_strided_block_parallel(
            ntt_mat,
            coeffs,
            num_blocks,
            inner_width,
            digit_bound,
            params,
        );
    }

    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    drive_block_chunked_matvec(
        num_blocks,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accs, start, end| {
            for col in start..end {
                let seq_start = col * num_blocks;
                if seq_start >= coeffs.len() {
                    break;
                }
                let live_blocks = num_blocks.min(coeffs.len() - seq_start);
                let coeffs_for_col = &coeffs[seq_start..seq_start + live_blocks];
                for (block_idx, digit) in coeffs_for_col.iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                }
            }
        },
    )
}
pub(super) fn mat_vec_mul_raw_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    if num_blocks == 0 {
        return Ok(vec![]);
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len);
    if inner_width == 0 || n_a == 0 {
        return Ok(vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks]);
    }

    // Unlike the balanced-digit paths (bound <= 32, always within capacity),
    // the raw signed-i8 bound is read from the witness and can in principle be
    // large enough that even a single CRT term cannot lift exactly. Reject that
    // at this checked boundary rather than panicking on a `Result` path.
    let rhs_bound = strided_i8_abs_bound(coeffs, num_blocks, inner_width);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, rhs_bound)
        .ok_or_else(|| {
            AkitaError::InvalidInput(
                "raw i8 recursive-witness coefficients exceed the CRT lift range for these parameters"
                    .to_string(),
            )
        })?;
    // Recursive-witness commit shapes are small-row (n_a <= 4). Fan out over
    // blocks whenever that exposes at least as much parallelism as the shared
    // driver's column tiles would: the many-block root gets block fanout, and
    // the deeper few-block levels still beat the 1-2 column tiles their narrow
    // widths produce. Only when blocks are scarce but tiles are plentiful do we
    // fall through to the tiled driver. Requires the full width to fit one CRT
    // lift; over-capacity widths still chunk in the driver.
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS && inner_width <= safe_width {
        let num_tiles = inner_width.div_ceil(base_tile_width::<W, K, D>());
        if num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS || num_blocks >= num_tiles {
            return Ok(mat_vec_mul_raw_i8_strided_block_parallel(
                ntt_mat,
                coeffs,
                num_blocks,
                inner_width,
                params,
            ));
        }
    }
    Ok(drive_block_chunked_matvec(
        num_blocks,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accs, start, end| {
            accumulate_raw_i8_strided_range(accs, ntt_mat, coeffs, num_blocks, start, end, params);
        },
    ))
}

fn strided_i8_abs_bound<const D: usize>(
    coeffs: &[[i8; D]],
    num_blocks: usize,
    inner_width: usize,
) -> u64 {
    let mut bound = 0u64;
    for col in 0..inner_width {
        let Some(seq_start) = col.checked_mul(num_blocks) else {
            break;
        };
        if seq_start >= coeffs.len() {
            break;
        }
        let live_blocks = num_blocks.min(coeffs.len() - seq_start);
        for row in &coeffs[seq_start..seq_start + live_blocks] {
            for &coeff in row {
                bound = bound.max(u64::from(coeff.unsigned_abs()));
            }
        }
    }
    bound
}

#[allow(clippy::too_many_arguments)]
fn accumulate_raw_i8_strided_range<W: PrimeWidth, const K: usize, const D: usize>(
    accs: &mut [Vec<CyclotomicCrtNtt<W, K, D>>],
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    tile_start: usize,
    tile_end: usize,
    params: &CrtNttParamSet<W, K, D>,
) {
    for col in tile_start..tile_end {
        let Some(seq_start) = col.checked_mul(num_blocks) else {
            break;
        };
        if seq_start >= coeffs.len() {
            break;
        }
        let live_blocks = num_blocks.min(coeffs.len() - seq_start);
        let coeffs_for_col = &coeffs[seq_start..seq_start + live_blocks];
        for (block_idx, coeff) in coeffs_for_col.iter().enumerate() {
            if is_zero_plane(coeff) {
                continue;
            }
            let ntt_d = CyclotomicCrtNtt::from_i8_with_params(coeff, params);
            for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
            }
        }
    }
}
