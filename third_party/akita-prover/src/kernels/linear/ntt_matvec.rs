use super::*;

macro_rules! dispatch_slot {
    ($slot:expr, $num_rows:expr, $num_cols:expr, $func:ident $(, $arg:expr)*) => {{
        let nr: usize = $num_rows;
        let nc: usize = $num_cols;
        match $slot {
            NttSlotCache::Q32 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
            NttSlotCache::Q64 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
            NttSlotCache::Q128 { neg, params: p, .. } => {
                let rows: Vec<&[_]> = (0..nr).map(|i| &neg[i * nc..(i + 1) * nc]).collect();
                $func(&rows, $($arg,)* p)
            }
        }
    }};
}

/// Column-tiled A*x across multiple blocks simultaneously.
///
/// Each rayon thread owns one column tile of `ntt_mat` (sized to fit in L2
/// cache) and iterates over all blocks, accumulating partial NTT results.
/// The matrix is loaded from DRAM exactly once. A final reduction sums
/// partial accumulators across tiles for each block.
///
/// Accepts raw ring-coefficient slices per block. Decomposes to i8 digits
/// on-the-fly per tile to avoid materializing all digits at once.
/// Tile width is auto-computed from ring parameters and target L2 cache size.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8")]
pub fn mat_vec_mul_ntt_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_i8_with_params,
        blocks,
        num_digits,
        log_basis
    ))
}

/// Dense-optimized variant of [`mat_vec_mul_ntt_i8`].
///
/// Skips the full-plane zero scans that are useful for sparse inputs but are
/// almost always wasted work on dense witnesses.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_dense")]
pub fn mat_vec_mul_ntt_i8_dense<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_i8_dense_with_params,
        blocks,
        num_digits,
        log_basis
    ))
}

/// Single-row dense variant of [`mat_vec_mul_ntt_i8_dense`].
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_dense_single_row")]
pub fn mat_vec_mul_ntt_i8_dense_single_row<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_cols: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        1usize,
        num_cols,
        mat_vec_mul_i8_dense_single_row_with_params,
        blocks,
        num_digits,
        log_basis
    ))
}

/// Strided variant of [`mat_vec_mul_ntt_i8`] for recursive witnesses.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_i8_strided")]
pub fn mat_vec_mul_ntt_i8_strided<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_i8_strided_with_params,
        coeffs,
        num_blocks,
        block_len,
        num_digits,
        log_basis
    ))
}

/// Column-tiled A*x across multiple blocks of pre-decomposed i8 digit planes.
///
/// This is the `num_digits_commit = 1` specialization of
/// [`mat_vec_mul_ntt_i8`]. It skips the `CyclotomicRing -> i8 digit plane`
/// decomposition entirely because the caller already holds each coefficient as a
/// balanced digit plane for a validated `log_basis <= 6`.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_digits_i8")]
pub fn mat_vec_mul_ntt_digits_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[[i8; D]]],
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    for block in blocks {
        validate_digit_rows_for_log_basis(
            block,
            num_cols.min(block.len()),
            log_basis,
            "for predecomposed digit mat-vec",
        )?;
    }
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_digits_i8_with_params,
        blocks,
        log_basis
    ))
}

/// Dense-optimized variant of [`mat_vec_mul_ntt_digits_i8`].
///
/// The generic pre-decomposed digit kernel skips all-zero planes, which is
/// profitable for sparse witnesses. Dense witnesses pay that scan on almost
/// every plane, so this variant uses the same math without the zero checks.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_dense_digits_i8")]
pub fn mat_vec_mul_ntt_dense_digits_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[[i8; D]]],
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    for block in blocks {
        validate_digit_rows_for_log_basis(
            block,
            num_cols.min(block.len()),
            log_basis,
            "for dense predecomposed digit mat-vec",
        )?;
    }
    mat_vec_mul_ntt_dense_digits_i8_trusted(slot, num_rows, num_cols, blocks, log_basis)
}

/// Dense pre-decomposed digit mat-vec for caller-owned trusted digit caches.
///
/// Keeps the public [`mat_vec_mul_ntt_dense_digits_i8`] validation boundary
/// intact while letting `DensePoly` reuse its decomposer-produced cache without
/// rescanning every digit plane on each commit.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_dense_digits_i8_trusted")]
pub(crate) fn mat_vec_mul_ntt_dense_digits_i8_trusted<
    F: FieldCore + CanonicalField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    blocks: &[&[[i8; D]]],
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_dense_digits_i8_with_params,
        blocks,
        log_basis
    ))
}

/// Strided variant of [`mat_vec_mul_ntt_digits_i8`] for recursive witnesses.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_digits_i8_strided")]
pub fn mat_vec_mul_ntt_digits_i8_strided<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    let used = num_cols.min(block_len).saturating_mul(num_blocks);
    validate_digit_rows_for_log_basis(
        coeffs,
        used.min(coeffs.len()),
        log_basis,
        "for strided predecomposed digit mat-vec",
    )?;
    Ok(dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_digits_i8_strided_with_params,
        coeffs,
        num_blocks,
        block_len,
        log_basis
    ))
}

/// Strided direct-signed-i8 variant for recursive witnesses.
///
/// Unlike [`mat_vec_mul_ntt_digits_i8_strided`], this path does not assume the
/// input rows are balanced gadget digits for `log_basis`. It is used for
/// `num_digits_commit = 1`, where the recursive witness is already the
/// committed signed-i8 coefficient stream.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_raw_i8_strided")]
pub fn mat_vec_mul_ntt_raw_i8_strided<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    dispatch_slot!(
        slot,
        num_rows,
        num_cols,
        mat_vec_mul_raw_i8_strided_with_params,
        coeffs,
        num_blocks,
        block_len
    )
}
