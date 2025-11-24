use crate::field::JoltField;
use rayon::prelude::*;
use std::iter::zip;

/// Inverses all `src` values into `dst` using Montgomery batch inversion.
///
/// # Panics
///
/// Panics if of of the values is zero.
pub fn batch_inverse<F: JoltField>(src: &[F], dst: &mut [F]) {
    let mut acc = F::one();
    zip(src, &mut *dst).for_each(|(src_v, dst_v)| {
        *dst_v = acc;
        acc *= *src_v;
    });
    let mut acc_inv = acc.inverse().unwrap();
    zip(src, &mut *dst).rev().for_each(|(src_v, dst_v)| {
        *dst_v = acc_inv * *dst_v;
        acc_inv *= *src_v;
    });
}

pub fn par_batch_inverse<F: JoltField>(src: &[F], dst: &mut [F]) {
    (src.par_chunks(4096), dst.par_chunks_mut(4096))
        .into_par_iter()
        .for_each(|(src_chunk, dst_chunk)| batch_inverse(src_chunk, dst_chunk));
}

#[cfg(test)]
mod tests {
    use super::batch_inverse;
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use ark_ff::AdditiveGroup;
    use std::array;

    #[test]
    fn test_batch_inverse() {
        let vals: [Fr; 10] = array::from_fn(|i| Fr::from(i as i32 + 1));

        let mut inv_vals = [Fr::ZERO; 10];
        batch_inverse(&vals, &mut inv_vals);

        assert_eq!(inv_vals, vals.map(|v| v.inverse().unwrap()));
    }
}
