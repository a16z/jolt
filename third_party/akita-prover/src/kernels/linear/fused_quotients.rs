use super::*;
use std::mem::size_of;

/// Minimum number of Rayon work-units for the fused one-shot kernel.
const MIN_FUSED_TILES: usize = 30;
#[cfg(target_arch = "aarch64")]
const FUSED_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
const FUSED_L2_CACHE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy)]
struct CenteredRhsBounds {
    capacity: u64,
    lut: u64,
}

/// Fused column-tiled kernel for the three split-eq mat-vec products.
///
/// Replaces three separate NTT-cached mat-vec calls (D-cyclic, B-cyclic,
/// A-quotient) with a single pass over the shared NTT cache. Within each
/// column tile, cache entries are loaded once and reused across all three
/// products with their exact row bounds, eliminating redundant DRAM reads.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn fused_split_eq_quotients_with_params<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    d_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    b_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    a_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    let d_width = d_cyc_rows.first().map_or(0, |r| r.len());
    let b_width = b_cyc_rows.first().map_or(0, |r| r.len());
    let a_width = a_cyc_rows.first().map_or(0, |r| r.len());
    let w_len = e_hat.len().min(d_width);
    let t_len = t_hat.len().min(b_width);
    let z_len = z_folded_rings.len().min(a_width);
    let max_col = w_len.max(t_len).max(z_len);

    if max_col == 0 {
        return Ok((
            vec![CyclotomicRing::<F, D>::zero(); n_d],
            vec![CyclotomicRing::<F, D>::zero(); n_b],
            vec![CyclotomicRing::<F, D>::zero(); n_a],
        ));
    }

    // CRT chunking keeps the caller's full-witness bound, while LUT sizing can
    // use the exact segment bound to avoid oversized centered tables.
    let actual_z_abs_bound = centered_rows_abs_bound(z_folded_rings, z_len);
    let z_bounds = CenteredRhsBounds {
        capacity: u64::from(z_folded_max_abs).max(actual_z_abs_bound),
        lut: actual_z_abs_bound,
    };
    if !digit_rows_within_digit_bound::<D>(e_hat, w_len, w_digit_abs_bound) {
        return Err(AkitaError::InvalidInput(
            "fused quotient e_hat contains digits outside its log_basis range".to_string(),
        ));
    }
    if !digit_rows_within_digit_bound::<D>(t_hat, t_len, t_digit_abs_bound) {
        return Err(AkitaError::InvalidInput(
            "fused quotient t_hat contains digits outside its log_basis range".to_string(),
        ));
    }
    debug_assert!(
        centered_rows_within_bound(z_folded_rings, z_len, z_bounds.capacity),
        "fused quotient centered RHS bound is smaller than the actual max"
    );
    let w_safe = w_len == 0
        || safe_crt_chunk_width::<F, W, K, D>(params, w_len, w_digit_abs_bound) == Some(w_len);
    let t_safe = t_len == 0
        || safe_crt_chunk_width::<F, W, K, D>(params, t_len, t_digit_abs_bound) == Some(t_len);
    let z_safe = z_len == 0
        || z_bounds.capacity == 0
        || safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_bounds.capacity) == Some(z_len);
    if w_safe && t_safe && z_safe {
        return Ok(fused_split_eq_quotients_one_shot(
            d_cyc_rows,
            b_cyc_rows,
            a_cyc_rows,
            neg_rows,
            n_d,
            n_b,
            n_a,
            e_hat,
            t_hat,
            z_folded_rings,
            z_bounds.lut,
            w_digit_abs_bound,
            t_digit_abs_bound,
            max_col,
            w_len,
            t_len,
            z_len,
            params,
        ));
    }

    let d_result =
        accumulate_cyclic_i8_rows(d_cyc_rows, n_d, e_hat, w_len, w_digit_abs_bound, params);
    let b_result =
        accumulate_cyclic_i8_rows(b_cyc_rows, n_b, t_hat, t_len, t_digit_abs_bound, params);
    let a_result = accumulate_centered_quotient_rows(
        neg_rows,
        a_cyc_rows,
        n_a,
        z_folded_rings,
        z_len,
        z_bounds,
        params,
    );

    Ok((d_result, b_result, a_result))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_one_shot<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    d_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    b_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    a_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_lut_abs_bound: u64,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
    max_col: usize,
    w_len: usize,
    t_len: usize,
    z_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let digit_bound = w_digit_abs_bound.max(t_digit_abs_bound);
    let digit_lut = (w_len != 0 || t_len != 0)
        .then(|| DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound));
    let centered_lut = (z_len != 0 && z_lut_abs_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, z_lut_abs_bound as i32));
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let tw = base_tw.min(max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let (d_accs, b_accs, a_neg_accs, a_cyc_accs) = cfg_fold_reduce!(
        0..num_tiles,
        || (
            vec![zero.clone(); n_d],
            vec![zero.clone(); n_b],
            vec![zero.clone(); n_a],
            vec![zero.clone(); n_a],
        ),
        |mut accs: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(max_col);

            for j in tile_start..tile_end {
                if j < w_len && !is_zero_plane(&e_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_w = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&e_hat[j], params, lut);
                    for (acc_d, cyc_row) in accs.0.iter_mut().zip(d_cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_d, &cyc_row[j], &ntt_w, params);
                    }
                }

                if j < t_len && !is_zero_plane(&t_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_t = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, lut);
                    for (acc_b, cyc_row) in accs.1.iter_mut().zip(b_cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_b, &cyc_row[j], &ntt_t, params);
                    }
                }

                if j < z_len && !is_zero_centered_row(&z_folded_rings[j]) {
                    let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                        unsafe {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                                &z_folded_rings[j],
                                params,
                                lut,
                            )
                        }
                    } else {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(
                            &z_folded_rings[j],
                            params,
                        )
                    };
                    for ((acc_neg, acc_cyc), (neg_row, cyc_row)) in accs
                        .2
                        .iter_mut()
                        .zip(accs.3.iter_mut())
                        .zip(neg_rows.iter().zip(a_cyc_rows.iter()))
                    {
                        accumulate_pointwise_product_into(acc_neg, &neg_row[j], &ntt_z_neg, params);
                        accumulate_pointwise_product_into(acc_cyc, &cyc_row[j], &ntt_z_cyc, params);
                    }
                }
            }
            accs
        },
        |mut a: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         b| {
            for r in 0..n_d {
                add_ntt_into(&mut a.0[r], &b.0[r], params);
            }
            for r in 0..n_b {
                add_ntt_into(&mut a.1[r], &b.1[r], params);
            }
            for r in 0..n_a {
                add_ntt_into(&mut a.2[r], &b.2[r], params);
                add_ntt_into(&mut a.3[r], &b.3[r], params);
            }
            a
        }
    );

    let d_result = d_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();
    let b_result = b_accs
        .into_iter()
        .map(|acc| acc.to_ring_cyclic(params))
        .collect();
    let a_result = a_neg_accs
        .into_iter()
        .zip(a_cyc_accs)
        .map(|(neg_acc, cyc_acc)| {
            let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
            let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
            quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring)
        })
        .collect();

    (d_result, b_result, a_result)
}

fn accumulate_cyclic_i8_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    rhs: &[[i8; D]],
    rhs_len: usize,
    rhs_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    if num_rows == 0 {
        return vec![];
    }
    if rhs_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let chunk_width = safe_crt_chunk_width::<F, W, K, D>(params, rhs_len, rhs_abs_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if rhs_len <= chunk_width {
        let (rows, _, _) = fused_split_eq_quotients_one_shot(
            cyc_rows,
            &[],
            &[],
            &[],
            num_rows,
            0,
            0,
            rhs,
            &[],
            &[],
            0,
            rhs_abs_bound,
            0,
            rhs_len,
            rhs_len,
            0,
            0,
            params,
        );
        return rows;
    }

    let num_chunks = rhs_len.div_ceil(chunk_width);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, rhs_abs_bound);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(rhs_len);
            let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk_start..chunk_end {
                if is_zero_plane(&rhs[j]) {
                    continue;
                }
                let ntt_rhs = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&rhs[j], params, &lut);
                for (acc, row) in accs.iter_mut().zip(cyc_rows.iter()) {
                    accumulate_pointwise_product_into(acc, &row[j], &ntt_rhs, params);
                }
            }

            for (dst, acc) in out.iter_mut().zip(accs) {
                *dst += acc.to_ring_cyclic(params);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn centered_rows_within_bound<const D: usize>(rows: &[[i32; D]], len: usize, bound: u64) -> bool {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| u64::from(coeff.unsigned_abs()) <= bound)
}

fn centered_rows_abs_bound<const D: usize>(rows: &[[i32; D]], len: usize) -> u64 {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .map(|&coeff| u64::from(coeff.unsigned_abs()))
        .max()
        .unwrap_or(0)
}

fn centered_i32_ring<F: CanonicalField, const D: usize>(coeffs: &[i32; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(from_fn(|k| F::from_i64(coeffs[k] as i64)))
}

fn accumulate_centered_quotient_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_len: usize,
    z_bounds: CenteredRhsBounds,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    if num_rows == 0 {
        return vec![];
    }
    if z_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    if z_bounds.lut == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let Some(chunk_width) = safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_bounds.capacity)
    else {
        return accumulate_centered_quotient_rows_field(
            neg_rows,
            cyc_rows,
            num_rows,
            z_folded_rings,
            z_len,
            params,
        );
    };
    if z_len <= chunk_width {
        let (_, _, rows) = fused_split_eq_quotients_one_shot(
            &[],
            &[],
            cyc_rows,
            neg_rows,
            0,
            0,
            num_rows,
            &[],
            &[],
            z_folded_rings,
            z_bounds.lut,
            0,
            0,
            z_len,
            0,
            0,
            z_len,
            params,
        );
        return rows;
    }

    let centered_lut = (z_bounds.lut <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, z_bounds.lut as i32));
    let num_chunks = z_len.div_ceil(chunk_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(z_len);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk_start..chunk_end {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                    unsafe {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                            &z_folded_rings[j],
                            params,
                            lut,
                        )
                    }
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_folded_rings[j], params)
                };
                for ((neg_acc, cyc_acc), (neg_row, cyc_row)) in neg_accs
                    .iter_mut()
                    .zip(cyc_accs.iter_mut())
                    .zip(neg_rows.iter().zip(cyc_rows.iter()))
                {
                    accumulate_pointwise_product_into(neg_acc, &neg_row[j], &ntt_z_neg, params);
                    accumulate_pointwise_product_into(cyc_acc, &cyc_row[j], &ntt_z_cyc, params);
                }
            }

            for ((dst, neg_acc), cyc_acc) in out.iter_mut().zip(neg_accs).zip(cyc_accs) {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
                let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
                *dst += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn accumulate_centered_quotient_rows_field<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..num_rows)
        .map(|row_idx| {
            let mut out = CyclotomicRing::<F, D>::zero();
            for j in 0..z_len {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let z = centered_i32_ring::<F, D>(&z_folded_rings[j]);
                let neg_lhs: CyclotomicRing<F, D> =
                    neg_rows[row_idx][j].to_ring_with_params(params);
                let cyc_lhs: CyclotomicRing<F, D> = cyc_rows[row_idx][j].to_ring_cyclic(params);
                let neg_product = neg_lhs * z;
                let mut cyc_product = CyclotomicRing::<F, D>::zero();
                add_cyclic_product_into(&mut cyc_product, &cyc_lhs, &z);
                out += quotient_from_cyclic_and_negacyclic(&cyc_product, &neg_product);
            }
            out
        })
        .collect()
}

/// Fused split-eq quotient kernel dispatching over [`NttSlotCache`] variants.
///
/// Computes three NTT-cached mat-vec products in a single tiled pass:
/// - D-cyclic: `cyc[0..n_d] · e_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix, but each role uses
/// its own packed row width.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "fused_split_eq_quotients")]
#[cfg(test)]
pub(crate) fn fused_split_eq_quotients<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    fused_split_eq_quotients_with_digit_bound(
        slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        balanced_digit_abs_bound(6),
        balanced_digit_abs_bound(6),
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fused_split_eq_quotients_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    validate_i8_log_basis(log_basis)?;
    let digit_bound = balanced_digit_abs_bound(log_basis);
    fused_split_eq_quotients_with_digit_bound(
        slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        digit_bound,
        digit_bound,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_with_digit_bound<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    let d_width = e_hat.len();
    let b_width = t_hat.len();
    let a_width = z_folded_rings.len();
    match slot {
        NttSlotCache::Q32 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &neg[i * a_width..(i + 1) * a_width])
                .collect();
            let d_rows: Vec<&[_]> = (0..n_d)
                .map(|i| &cyc[i * d_width..(i + 1) * d_width])
                .collect();
            let b_rows: Vec<&[_]> = (0..n_b)
                .map(|i| &cyc[i * b_width..(i + 1) * b_width])
                .collect();
            let a_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &cyc[i * a_width..(i + 1) * a_width])
                .collect();
            fused_split_eq_quotients_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
                &neg_rows,
                n_d,
                n_b,
                n_a,
                e_hat,
                t_hat,
                z_folded_rings,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                p,
            )
        }
        NttSlotCache::Q64 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &neg[i * a_width..(i + 1) * a_width])
                .collect();
            let d_rows: Vec<&[_]> = (0..n_d)
                .map(|i| &cyc[i * d_width..(i + 1) * d_width])
                .collect();
            let b_rows: Vec<&[_]> = (0..n_b)
                .map(|i| &cyc[i * b_width..(i + 1) * b_width])
                .collect();
            let a_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &cyc[i * a_width..(i + 1) * a_width])
                .collect();
            fused_split_eq_quotients_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
                &neg_rows,
                n_d,
                n_b,
                n_a,
                e_hat,
                t_hat,
                z_folded_rings,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                p,
            )
        }
        NttSlotCache::Q128 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &neg[i * a_width..(i + 1) * a_width])
                .collect();
            let d_rows: Vec<&[_]> = (0..n_d)
                .map(|i| &cyc[i * d_width..(i + 1) * d_width])
                .collect();
            let b_rows: Vec<&[_]> = (0..n_b)
                .map(|i| &cyc[i * b_width..(i + 1) * b_width])
                .collect();
            let a_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &cyc[i * a_width..(i + 1) * a_width])
                .collect();
            fused_split_eq_quotients_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
                &neg_rows,
                n_d,
                n_b,
                n_a,
                e_hat,
                t_hat,
                z_folded_rings,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                p,
            )
        }
    }
}
