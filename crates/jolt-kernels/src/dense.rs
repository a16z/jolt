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
