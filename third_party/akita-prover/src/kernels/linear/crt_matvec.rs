#[cfg(all(test, not(feature = "zk")))]
use super::*;

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn mat_vec_mul_unchecked<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(mat.len());
    for row in mat {
        debug_assert_eq!(row.len(), vec.len());
        let mut acc = CyclotomicRing::<F, D>::zero();
        for (a, x) in row.iter().zip(vec.iter()) {
            acc += *a * *x;
        }
        out.push(acc);
    }
    out
}

#[cfg(all(test, not(feature = "zk")))]
pub(super) fn precompute_dense_mat_ntt_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicCrtNtt<W, K, D>>> {
    cfg_iter!(mat)
        .map(|row| {
            row.iter()
                .map(|a| CyclotomicCrtNtt::from_ring_with_params(a, params))
                .collect()
        })
        .collect()
}

#[cfg(all(test, not(feature = "zk")))]
fn mat_vec_mul_dense_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
        .iter()
        .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
        .collect();

    mat.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), ntt_vec.len());
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            for (a, x_ntt) in row.iter().zip(ntt_vec.iter()) {
                let a_ntt = CyclotomicCrtNtt::from_ring_with_params(a, params);
                accumulate_pointwise_product_into(&mut acc, &a_ntt, x_ntt, params);
            }
            acc.to_ring_with_params(params)
        })
        .collect()
}

#[cfg(all(test, not(feature = "zk")))]
fn mat_vec_mul_dense_many_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let ntt_mat = precompute_dense_mat_ntt_with_params(mat, params);
    vecs.iter()
        .map(|vec| {
            let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
                .iter()
                .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
                .collect();

            ntt_mat
                .iter()
                .map(|row_ntt| {
                    debug_assert_eq!(row_ntt.len(), ntt_vec.len());
                    let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                    for (a_ntt, x_ntt) in row_ntt.iter().zip(ntt_vec.iter()) {
                        accumulate_pointwise_product_into(&mut acc, a_ntt, x_ntt, params);
                    }
                    acc.to_ring_with_params(params)
                })
                .collect()
        })
        .collect()
}

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn mat_vec_mul_crt_ntt<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_with_params(mat, vec, p),
    };
    Ok(out)
}

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn mat_vec_mul_crt_ntt_many<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
    };
    Ok(out)
}
