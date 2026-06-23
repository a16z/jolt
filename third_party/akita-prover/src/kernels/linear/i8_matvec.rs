use super::*;

pub(super) fn mat_vec_mul_i8_with_params_impl<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_blocks = blocks.len();
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks
        .iter()
        .map(|b| b.len() * num_digits)
        .max()
        .unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width == max_data_width
    {
        if inner_width <= safe_width {
            return if CHECK_ZERO {
                mat_vec_mul_i8_block_parallel_with_params(
                    ntt_mat, blocks, num_digits, log_basis, params,
                )
            } else {
                mat_vec_mul_i8_dense_block_parallel_with_params(
                    ntt_mat, blocks, num_digits, log_basis, params,
                )
            };
        }
        let chunk_width = capacity_safe_i8_chunk_width(safe_width, inner_width, num_digits);
        return mat_vec_mul_i8_block_parallel_chunked_with_params::<F, W, K, D, CHECK_ZERO>(
            ntt_mat,
            blocks,
            inner_width,
            chunk_width,
            num_digits,
            log_basis,
            params,
        );
    }

    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    let tile_width = aligned_i8_tile_width(base_tile_width::<W, K, D>(), inner_width, num_digits);
    let chunk_width = capacity_safe_i8_chunk_width(safe_width, inner_width, num_digits);
    drive_block_chunked_matvec(
        num_blocks,
        n_a,
        inner_width,
        safe_width,
        tile_width,
        chunk_width,
        params,
        |accs, start, end| {
            let ring_start = start / num_digits;
            let ring_end = ((end - 1) / num_digits) + 1;
            let digit_offset = start - ring_start * num_digits;
            let tile_len = end - start;

            for block_idx in 0..num_blocks {
                let block = blocks[block_idx];
                if ring_start >= block.len() {
                    continue;
                }
                let block_ring_end = ring_end.min(block.len());
                let partial_coeffs = &block[ring_start..block_ring_end];
                let all_digits = decompose_block_i8(partial_coeffs, num_digits, log_basis);
                let available = all_digits.len().saturating_sub(digit_offset);
                let n = tile_len.min(available);

                for (j, digit) in all_digits[digit_offset..digit_offset + n]
                    .iter()
                    .enumerate()
                {
                    if CHECK_ZERO && is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[start + j], &ntt_d, params);
                    }
                }
            }
        },
    )
}

pub(super) fn mat_vec_mul_i8_with_params<
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
    mat_vec_mul_i8_with_params_impl::<F, W, K, D, true>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

pub(super) fn mat_vec_mul_i8_dense_with_params<
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
    mat_vec_mul_i8_with_params_impl::<F, W, K, D, false>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mat_vec_mul_i8_strided_with_params<
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
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len.saturating_mul(num_digits));
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width == block_len.saturating_mul(num_digits)
        && inner_width <= safe_width
    {
        return mat_vec_mul_i8_strided_block_parallel_with_params(
            ntt_mat, coeffs, num_blocks, block_len, num_digits, log_basis, params,
        );
    }

    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    let tile_width = aligned_i8_tile_width(base_tile_width::<W, K, D>(), inner_width, num_digits);
    let chunk_width = capacity_safe_i8_chunk_width(safe_width, inner_width, num_digits);
    drive_block_chunked_matvec(
        num_blocks,
        n_a,
        inner_width,
        safe_width,
        tile_width,
        chunk_width,
        params,
        |accs, start, end| {
            let ring_start = start / num_digits;
            let ring_end = ((end - 1) / num_digits) + 1;
            let digit_offset = start - ring_start * num_digits;
            let tile_len = end - start;

            for (block_idx, block_accs) in accs.iter_mut().enumerate() {
                let mut partial_coeffs = Vec::with_capacity(ring_end.saturating_sub(ring_start));
                for col in ring_start..ring_end {
                    let seq = block_idx + col * num_blocks;
                    let Some(coeff) = coeffs.get(seq) else {
                        break;
                    };
                    partial_coeffs.push(*coeff);
                }
                if partial_coeffs.is_empty() {
                    continue;
                }

                let all_digits = decompose_block_i8(&partial_coeffs, num_digits, log_basis);
                let available = all_digits.len().saturating_sub(digit_offset);
                let n = tile_len.min(available);

                for (j, digit) in all_digits[digit_offset..digit_offset + n]
                    .iter()
                    .enumerate()
                {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in block_accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[start + j], &ntt_d, params);
                    }
                }
            }
        },
    )
}
