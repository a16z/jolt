use super::*;

/// L2-cache-sized column tile width for the one-shot CRT accumulation path.
#[inline]
pub(super) fn base_tile_width<W: PrimeWidth, const K: usize, const D: usize>() -> usize {
    (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1)
}

/// Shared one-shot/chunked driver for block-shaped CRT matvecs that produce
/// `num_blocks` groups of `n_a` ring rows.
///
/// When `inner_width <= safe_width` the whole accumulation fits a single CRT
/// lift, so columns are tiled at `tile_width` for cache locality and Rayon
/// parallelism and reconstructed once. Otherwise columns are partitioned into
/// `chunk_width` chunks, each reconstructed independently and summed in the
/// native field. `accumulate` adds the products for column range
/// `[start, end)` into the per-block CRT accumulators; it is the only
/// kernel-specific piece.
#[allow(clippy::too_many_arguments)]
pub(super) fn drive_block_chunked_matvec<F, W, const K: usize, const D: usize, Acc>(
    num_blocks: usize,
    n_a: usize,
    inner_width: usize,
    safe_width: usize,
    tile_width: usize,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
    accumulate: Acc,
) -> Vec<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    Acc: Fn(&mut [Vec<CyclotomicCrtNtt<W, K, D>>], usize, usize) + Sync,
{
    if inner_width <= safe_width {
        let num_tiles = inner_width.div_ceil(tile_width);
        let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
            |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
                let tile_start = tile_idx * tile_width;
                let tile_end = (tile_start + tile_width).min(inner_width);
                accumulate(&mut accs, tile_start, tile_end);
                accs
            },
            |mut a: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, b| {
                for block_idx in 0..num_blocks {
                    for row in 0..n_a {
                        add_ntt_into(&mut a[block_idx][row], &b[block_idx][row], params);
                    }
                }
                a
            }
        );

        return cfg_into_iter!(final_accs)
            .map(|row_accs| {
                row_accs
                    .into_iter()
                    .map(|acc| acc.to_ring_with_params(params))
                    .collect()
            })
            .collect();
    }

    let num_chunks = inner_width.div_ceil(chunk_width);
    cfg_fold_reduce!(
        0..num_chunks,
        || vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks],
        |mut out: Vec<Vec<CyclotomicRing<F, D>>>, chunk_idx| {
            let tile_start = chunk_idx * chunk_width;
            let tile_end = (tile_start + chunk_width).min(inner_width);
            let mut accs = vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks];
            accumulate(&mut accs, tile_start, tile_end);
            for (out_block, acc_block) in out.iter_mut().zip(accs) {
                for (dst, acc) in out_block.iter_mut().zip(acc_block) {
                    *dst += acc.to_ring_with_params(params);
                }
            }
            out
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for (a_block, b_block) in a.iter_mut().zip(b) {
                for (dst, src) in a_block.iter_mut().zip(b_block) {
                    *dst += src;
                }
            }
            a
        }
    )
}

/// Shared one-shot/chunked driver for single-vector CRT matvecs producing
/// `n_a` ring rows.
///
/// Same safe-width policy as [`drive_block_chunked_matvec`]. `accumulate` adds
/// the products for column range `[start, end)` into the `n_a` CRT
/// accumulators, and `finalize` reconstructs one accumulator to a ring
/// (negacyclic vs cyclic differs only here and inside `accumulate`).
#[allow(clippy::too_many_arguments)]
pub(super) fn drive_single_chunked_matvec<F, W, const K: usize, const D: usize, Acc, Fin>(
    n_a: usize,
    inner_width: usize,
    safe_width: usize,
    tile_width: usize,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
    accumulate: Acc,
    finalize: Fin,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    Acc: Fn(&mut [CyclotomicCrtNtt<W, K, D>], usize, usize) + Sync,
    Fin: Fn(&CyclotomicCrtNtt<W, K, D>, &CrtNttParamSet<W, K, D>) -> CyclotomicRing<F, D> + Sync,
{
    if inner_width <= safe_width {
        let num_tiles = inner_width.div_ceil(tile_width);
        let final_accs: Vec<CyclotomicCrtNtt<W, K, D>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a],
            |mut accs: Vec<CyclotomicCrtNtt<W, K, D>>, tile_idx| {
                let tile_start = tile_idx * tile_width;
                let tile_end = (tile_start + tile_width).min(inner_width);
                accumulate(&mut accs, tile_start, tile_end);
                accs
            },
            |mut a: Vec<CyclotomicCrtNtt<W, K, D>>, b| {
                for row in 0..n_a {
                    add_ntt_into(&mut a[row], &b[row], params);
                }
                a
            }
        );

        return final_accs.iter().map(|acc| finalize(acc, params)).collect();
    }

    let num_chunks = inner_width.div_ceil(chunk_width);
    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); n_a],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let tile_start = chunk_idx * chunk_width;
            let tile_end = (tile_start + chunk_width).min(inner_width);
            let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            accumulate(&mut accs, tile_start, tile_end);
            for (dst, acc) in out.iter_mut().zip(&accs) {
                *dst += finalize(acc, params);
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
