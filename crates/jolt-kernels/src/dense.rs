use jolt_field::Field;
use rayon::prelude::*;

pub(crate) const DENSE_BIND_PAR_THRESHOLD: usize = 1024;

#[inline]
pub(crate) fn bind_dense_evals_reuse<F: Field>(
    values: &mut Vec<F>,
    scratch: &mut Vec<F>,
    challenge: F,
) {
    let half = values.len() / 2;
    scratch.resize(half, F::zero());
    if half >= DENSE_BIND_PAR_THRESHOLD {
        scratch
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let low = values[index << 1];
                let high = values[(index << 1) + 1];
                *output = low + challenge * (high - low);
            });
    } else {
        for (index, output) in scratch.iter_mut().enumerate() {
            let low = values[index << 1];
            let high = values[(index << 1) + 1];
            *output = low + challenge * (high - low);
        }
    }
    std::mem::swap(values, scratch);
    scratch.clear();
}

#[inline]
pub(crate) fn bind_dense_evals_reuse_serial<F: Field>(
    values: &mut Vec<F>,
    scratch: &mut Vec<F>,
    challenge: F,
) {
    let half = values.len() / 2;
    scratch.resize(half, F::zero());
    for (index, output) in scratch.iter_mut().enumerate() {
        let low = values[index << 1];
        let high = values[(index << 1) + 1];
        *output = low + challenge * (high - low);
    }
    std::mem::swap(values, scratch);
    scratch.clear();
}

#[cfg(feature = "cuda")]
use jolt_field::arkworks::cuda::{CudaError, CudaFieldContext, DeviceFrVec};
#[cfg(feature = "cuda")]
use jolt_field::Fr;

#[cfg(feature = "cuda")]
#[expect(clippy::todo)]
#[cfg_attr(not(test), expect(dead_code))]
pub fn bind_dense_evals_reuse_cuda(
    ctx: &CudaFieldContext,
    values: &mut DeviceFrVec,
    scratch: &mut DeviceFrVec,
    challenge: Fr,
) -> Result<(), CudaError> {
    let _ = (ctx, values, scratch, challenge);
    todo!()
}

#[cfg(all(test, feature = "cuda"))]
#[expect(clippy::unwrap_used)]
mod cuda_tests {
    use super::*;
    use jolt_field::Field;
    use proptest::prelude::*;

    fn fr_strategy() -> impl Strategy<Value = Fr> {
        any::<[u8; 32]>().prop_map(|bytes| Fr::from_bytes(&bytes))
    }

    proptest! {
        #[test]
        #[ignore = "bind_dense_evals_reuse_cuda body is todo!()"]
        fn bind_dense_evals_reuse_cuda_matches_serial(
            half in 1usize..1000,
            challenge in fr_strategy(),
        ) {
            let values: Vec<Fr> = (0..half * 2).map(|i| Fr::from_u64(i as u64)).collect();

            let mut expected = values.clone();
            let mut scratch = Vec::new();
            bind_dense_evals_reuse_serial(&mut expected, &mut scratch, challenge);

            let ctx = CudaFieldContext::new(0).unwrap();
            let mut values_dev = ctx.upload(&values).unwrap();
            let mut scratch_dev = ctx.upload(&[]).unwrap();
            bind_dense_evals_reuse_cuda(&ctx, &mut values_dev, &mut scratch_dev, challenge).unwrap();

            prop_assert_eq!(values_dev.to_host().unwrap(), expected);
        }
    }
}
