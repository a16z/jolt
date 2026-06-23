use super::*;

#[cfg(feature = "parallel")]
const DENSE_PARALLEL_PAIR_THRESHOLD: usize = 1 << 14;

pub(crate) fn accumulate_dense_round<E: FieldCore + HasUnreducedOps>(
    witness_evals: &[E],
    factor_evals: &[E],
    coeff: E,
) -> (E, E) {
    let _span = tracing::trace_span!(
        "dense_extension_reduction_accumulate_round",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    if coeff == E::zero() {
        return (E::zero(), E::zero());
    }

    // Sum the wide products in `E::ProductAccum` only when the field has proven
    // that delayed reduction is exact for these batch sizes; otherwise reduce
    // each product immediately so the coefficients stay byte-identical to
    // per-term `Mul` (the `DELAYED_PRODUCT_SUM_IS_EXACT` contract).
    let (constant, quadratic) = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        accumulate_dense_round_with::<E, DelayedDeg2<E>>(witness_evals, factor_evals)
    } else {
        accumulate_dense_round_with::<E, DirectDeg2<E>>(witness_evals, factor_evals)
    };
    (coeff * constant, coeff * quadratic)
}

fn accumulate_dense_round_with<E, A>(witness_evals: &[E], factor_evals: &[E]) -> (E, E)
where
    E: FieldCore + HasUnreducedOps,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;

    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            return (0..half)
                .into_par_iter()
                .fold(A::zero, |mut acc, i| {
                    let w0 = witness_evals[2 * i];
                    let w1 = witness_evals[2 * i + 1];
                    let a0 = factor_evals[2 * i];
                    let a1 = factor_evals[2 * i + 1];

                    acc.add_constant_product(w0, a0);
                    acc.add_quadratic_product(w1 - w0, a1 - a0);
                    acc
                })
                .reduce(A::zero, A::merge)
                .finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..half {
        let w0 = witness_evals[2 * i];
        let w1 = witness_evals[2 * i + 1];
        let a0 = factor_evals[2 * i];
        let a1 = factor_evals[2 * i + 1];

        acc.add_constant_product(w0, a0);
        acc.add_quadratic_product(w1 - w0, a1 - a0);
    }
    acc.finish()
}

pub(crate) fn fold_dense_reduction_tables_in_place<E: HasUnreducedOps + HasOptimizedFold>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) {
    let _span = tracing::trace_span!(
        "fold_dense_reduction_tables_in_place",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    fold_evals_in_place(witness_evals, r_round);
    fold_evals_in_place(factor_evals, r_round);
}

/// Fold both tables by one variable AND pre-compute the next round's
/// `(constant, quadratic)` accumulation in a single pass over the data.
pub(crate) fn fused_fold_and_accumulate<E: HasUnreducedOps + HasOptimizedFold>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) -> (E, E) {
    let _span = tracing::trace_span!("fused_fold_and_accumulate", table_len = witness_evals.len())
        .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 4);

    // The fold itself (`E::fold_one`) is always exact; only the product
    // accumulation respects `DELAYED_PRODUCT_SUM_IS_EXACT`, matching
    // `accumulate_dense_round`.
    if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        fused_fold_and_accumulate_with::<E, DelayedDeg2<E>>(witness_evals, factor_evals, r_round)
    } else {
        fused_fold_and_accumulate_with::<E, DirectDeg2<E>>(witness_evals, factor_evals, r_round)
    }
}

fn fused_fold_and_accumulate_with<E, A>(
    witness_evals: &mut Vec<E>,
    factor_evals: &mut Vec<E>,
    r_round: E,
) -> (E, E)
where
    E: FieldCore + HasUnreducedOps + HasOptimizedFold,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;
    let quarter = half / 2;
    let ctx = E::precompute_fold(r_round);

    #[cfg(feature = "parallel")]
    {
        if quarter >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let mut folded_w = Vec::<E>::with_capacity(half);
            let mut folded_f = Vec::<E>::with_capacity(half);
            // SAFETY: both vectors are allocated with capacity `half`. `half` is
            // even (table length is a power of two >= 4), so the `par_chunks_mut(2)`
            // loop below yields exactly `quarter` chunks of length 2 and writes all
            // `half` slots before the first read (`*witness_evals = folded_w`).
            // `E: FieldCore` is `Copy` with a trivial drop, so overwriting the
            // uninitialized slots is sound.
            unsafe {
                folded_w.set_len(half);
                folded_f.set_len(half);
            }

            let acc = {
                let input_w: &[E] = witness_evals;
                let input_f: &[E] = factor_evals;

                folded_w
                    .par_chunks_mut(2)
                    .zip(folded_f.par_chunks_mut(2))
                    .enumerate()
                    .fold(A::zero, |mut acc, (i, (w_out, f_out))| {
                        let fw0 = E::fold_one(&ctx, input_w[4 * i], input_w[4 * i + 1]);
                        let fw1 = E::fold_one(&ctx, input_w[4 * i + 2], input_w[4 * i + 3]);
                        let fa0 = E::fold_one(&ctx, input_f[4 * i], input_f[4 * i + 1]);
                        let fa1 = E::fold_one(&ctx, input_f[4 * i + 2], input_f[4 * i + 3]);

                        acc.add_constant_product(fw0, fa0);
                        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

                        w_out[0] = fw0;
                        w_out[1] = fw1;
                        f_out[0] = fa0;
                        f_out[1] = fa1;

                        acc
                    })
                    .reduce(A::zero, A::merge)
            };

            *witness_evals = folded_w;
            *factor_evals = folded_f;
            return acc.finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..quarter {
        let fw0 = E::fold_one(&ctx, witness_evals[4 * i], witness_evals[4 * i + 1]);
        let fw1 = E::fold_one(&ctx, witness_evals[4 * i + 2], witness_evals[4 * i + 3]);
        let fa0 = E::fold_one(&ctx, factor_evals[4 * i], factor_evals[4 * i + 1]);
        let fa1 = E::fold_one(&ctx, factor_evals[4 * i + 2], factor_evals[4 * i + 3]);

        acc.add_constant_product(fw0, fa0);
        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

        witness_evals[2 * i] = fw0;
        witness_evals[2 * i + 1] = fw1;
        factor_evals[2 * i] = fa0;
        factor_evals[2 * i + 1] = fa1;
    }
    witness_evals.truncate(half);
    factor_evals.truncate(half);
    acc.finish()
}
