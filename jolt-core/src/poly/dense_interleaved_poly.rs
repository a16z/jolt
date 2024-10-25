use crate::{field::JoltField, utils::thread::unsafe_allocate_zero_vec};
use rayon::{prelude::*, slice::Chunks};

#[cfg(test)]
use super::dense_mlpoly::DensePolynomial;

#[derive(Default, Debug, Clone)]
pub struct DenseInterleavedPolynomial<F: JoltField> {
    pub(crate) coeffs: Vec<F>,
    len: usize,
    binding_scratch_space: Vec<F>,
}

impl<F: JoltField> DenseInterleavedPolynomial<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len() % 2 == 0);
        let len = coeffs.len();
        Self {
            coeffs,
            len,
            binding_scratch_space: unsafe_allocate_zero_vec(len),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.coeffs[..self.len].iter()
    }

    pub fn par_chunks(&self, chunk_size: usize) -> Chunks<'_, F> {
        self.coeffs[..self.len].par_chunks(chunk_size)
    }

    #[cfg(test)]
    pub fn interleave(left: &Vec<F>, right: &Vec<F>) -> Self {
        assert_eq!(left.len(), right.len());
        let mut interleaved = vec![];
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        Self::new(interleaved)
    }

    #[cfg(test)]
    pub fn uninterleave(&self) -> (Vec<F>, Vec<F>) {
        let left: Vec<_> = self.coeffs[..self.len]
            .to_vec()
            .into_iter()
            .step_by(2)
            .collect();
        let mut right: Vec<_> = self.coeffs[..self.len]
            .to_vec()
            .into_iter()
            .skip(1)
            .step_by(2)
            .collect();
        if right.len() < left.len() {
            right.resize(left.len(), F::zero());
        }
        (left, right)
    }

    pub fn bind(&mut self, r: F) {
        let padded_len = self.len.next_multiple_of(4);
        self.binding_scratch_space
            .par_chunks_mut(2)
            .zip(self.coeffs[..self.len].par_chunks(4))
            .for_each(|(bound_chunk, unbound_chunk)| {
                let unbound_chunk = [
                    *unbound_chunk.get(0).unwrap_or(&F::zero()),
                    *unbound_chunk.get(1).unwrap_or(&F::zero()),
                    *unbound_chunk.get(2).unwrap_or(&F::zero()),
                    *unbound_chunk.get(3).unwrap_or(&F::zero()),
                ];

                bound_chunk[0] = unbound_chunk[0] + r * (unbound_chunk[2] - unbound_chunk[0]);
                bound_chunk[1] = unbound_chunk[1] + r * (unbound_chunk[3] - unbound_chunk[1]);
            });

        self.len = padded_len / 2;
        std::mem::swap(&mut self.coeffs, &mut self.binding_scratch_space);
    }
}

#[cfg(test)]
pub fn bind_left_and_right<F: JoltField>(left: &mut Vec<F>, right: &mut Vec<F>, r: F) {
    if left.len() % 2 != 0 {
        left.push(F::zero())
    }
    if right.len() % 2 != 0 {
        right.push(F::zero())
    }
    let mut left_poly = DensePolynomial::new_padded(left.clone());
    let mut right_poly = DensePolynomial::new_padded(right.clone());
    left_poly.bound_poly_var_bot(&r);
    right_poly.bound_poly_var_bot(&r);

    *left = left_poly.Z[..left.len() / 2].to_vec();
    *right = right_poly.Z[..right.len() / 2].to_vec();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use itertools::Itertools;

    #[test]
    fn interleave_uninterleave() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        const BATCH_SIZE: [usize; 5] = [2, 3, 4, 5, 6];

        for (num_vars, batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let left: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(batch_size << num_vars)
                .collect();
            let right: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(batch_size << num_vars)
                .collect();

            let interleaved = DenseInterleavedPolynomial::interleave(&left, &right);
            assert_eq!(interleaved.uninterleave(), (left, right));
        }
    }

    #[test]
    fn uninterleave_interleave() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        const BATCH_SIZE: [usize; 5] = [2, 3, 4, 5, 6];

        for (num_vars, batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let coeffs: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(2 * (batch_size << num_vars))
                .collect();
            let interleaved = DenseInterleavedPolynomial::new(coeffs);
            let (left, right) = interleaved.uninterleave();

            assert_eq!(
                interleaved.iter().collect::<Vec<_>>(),
                DenseInterleavedPolynomial::interleave(&left, &right)
                    .iter()
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn bind() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        const BATCH_SIZE: [usize; 5] = [2, 3, 4, 5, 6];

        for (num_vars, batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let mut left: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(batch_size << num_vars)
                .collect();
            let mut right: Vec<_> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(batch_size << num_vars)
                .collect();

            let mut interleaved = DenseInterleavedPolynomial::interleave(&left, &right);

            let r = Fr::random(&mut rng);
            interleaved.bind(r);
            bind_left_and_right(&mut left, &mut right, r);

            assert_eq!(
                interleaved.iter().collect::<Vec<_>>(),
                DenseInterleavedPolynomial::interleave(&left, &right)
                    .iter()
                    .collect::<Vec<_>>()
            );
        }
    }
}
