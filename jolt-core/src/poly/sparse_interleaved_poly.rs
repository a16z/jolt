use super::dense_mlpoly::DensePolynomial;
use crate::field::{JoltField, OptimizedMul};
use rayon::prelude::*;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct SparseCoefficient<F: JoltField> {
    pub(crate) index: usize,
    pub(crate) value: F,
}

impl<F: JoltField> From<(usize, F)> for SparseCoefficient<F> {
    fn from(x: (usize, F)) -> Self {
        Self {
            index: x.0,
            value: x.1,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct SparseInterleavedPolynomial<F: JoltField> {
    pub(crate) coeffs: Vec<SparseCoefficient<F>>,
    pub(crate) dense_len: usize,
    // TODO(moodlezoup): Is it possible to "collect_into_vec"
    // for a non-indexed rayon ParallelIterator if we know that
    // the destination has sufficient capacity for the collect?
    // If so, we can reuse the same "scratch space" vector across
    // bindings. par_extend + clear/set_len?
    // binding_scratch_space: Vec<SparseCoefficient<F>>
}

impl<F: JoltField> SparseInterleavedPolynomial<F> {
    pub fn new(coeffs: Vec<SparseCoefficient<F>>, dense_len: usize) -> Self {
        assert!(dense_len.is_power_of_two());
        Self { dense_len, coeffs }
    }

    pub fn to_dense(&self) -> DensePolynomial<F> {
        let mut dense_layer = vec![F::one(); self.dense_len];
        for coeff in &self.coeffs {
            dense_layer[coeff.index] = coeff.value;
        }
        DensePolynomial::new(dense_layer)
    }

    #[cfg(test)]
    pub fn interleave(left: &DensePolynomial<F>, right: &DensePolynomial<F>) -> Self {
        assert_eq!(left.len(), right.len());
        let mut interleaved = vec![];
        for i in 0..left.len() {
            if !left[i].is_one() {
                interleaved.push((2 * i, left[i]).into());
            }
            if !right[i].is_one() {
                interleaved.push((2 * i + 1, right[i]).into());
            }
        }
        Self::new(interleaved, left.len() + right.len())
    }

    #[cfg(test)]
    pub fn uninterleave(&self) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let mut left = vec![F::one(); self.dense_len / 2];
        let mut right = vec![F::one(); self.dense_len / 2];
        for coeff in &self.coeffs {
            if coeff.index % 2 == 0 {
                left[coeff.index / 2] = coeff.value;
            } else {
                right[coeff.index / 2] = coeff.value;
            }
        }
        let left_poly = DensePolynomial::new(left);
        let right_poly = DensePolynomial::new(right);
        (left_poly, right_poly)
    }

    pub fn parallelizable_slices(&self) -> Vec<&[SparseCoefficient<F>]> {
        let num_threads = rayon::current_num_threads();
        let target_slice_length = self.coeffs.len() / num_threads;
        if target_slice_length == 0 {
            return vec![&self.coeffs];
        }

        let mut boundary_indices: Vec<_> = (target_slice_length..self.coeffs.len())
            .step_by(target_slice_length)
            .collect();

        for boundary in boundary_indices.iter_mut() {
            while *boundary < self.coeffs.len() - 1 {
                let current = &self.coeffs[*boundary];
                let next = &self.coeffs[*boundary + 1];
                if next.index / 4 > current.index / 4 {
                    *boundary += 1;
                    break;
                }
                *boundary += 1;
            }
            if *boundary == self.coeffs.len() - 1 {
                *boundary += 1;
            }
        }
        boundary_indices.dedup();

        let mut slices = vec![&self.coeffs[..boundary_indices[0]]];
        for boundaries in boundary_indices.windows(2) {
            slices.push(&self.coeffs[boundaries[0]..boundaries[1]])
        }
        let last_boundary = *boundary_indices.last().unwrap();
        if last_boundary != self.coeffs.len() {
            slices.push(&self.coeffs[last_boundary..]);
        }

        slices
    }

    pub fn par_blocks(
        &self,
        block_size: usize,
    ) -> impl ParallelIterator<Item = &[SparseCoefficient<F>]> {
        if block_size != 2 && block_size != 4 {
            panic!("unsupported block_size: {}", block_size);
        }
        self.coeffs
            .par_chunk_by(move |x, y| x.index / block_size == y.index / block_size)
    }

    pub fn bind_par_blocks(&mut self, r: F) {
        self.coeffs = self
            .par_blocks(4)
            .flat_map(|sparse_block| {
                let mut bound: [Option<SparseCoefficient<F>>; 2] = [None, None];

                let block_index = sparse_block[0].index / 4;
                let mut dense_block = [F::one(), F::one(), F::one(), F::one()];
                for coeff in sparse_block {
                    if coeff.index / 4 == block_index {
                        dense_block[coeff.index % 4] = coeff.value;
                    }
                }
                let left = dense_block[0] + r.mul_0_optimized(dense_block[2] - dense_block[0]);
                let right = dense_block[1] + r.mul_0_optimized(dense_block[3] - dense_block[1]);
                if !left.is_one() {
                    let left_index = 2 * block_index;
                    bound[0] = Some((left_index, left).into());
                }
                if !right.is_one() {
                    let right_index = 2 * block_index + 1;
                    bound[1] = Some((right_index, right).into());
                }

                bound
            })
            .filter_map(|bound_value| bound_value)
            .collect();

        self.dense_len /= 2;
    }

    // TODO(moodlezoup): Dynamic density
    pub fn bind(&mut self, r: F) {
        let last_index = self.coeffs.last().unwrap().index;
        self.coeffs.push((last_index + 1, F::one()).into());
        self.coeffs.push((last_index + 2, F::one()).into());
        self.coeffs.push((last_index + 3, F::one()).into());

        // TODO(moodlezoup): Is it more efficient to filter first?
        self.coeffs = self
            .coeffs
            .par_windows(4)
            .flat_map(|window| {
                let mut bound = vec![];

                if window[0].index % 4 == 0 {
                    let block_index = window[0].index / 4;
                    let mut block = [F::one(), F::one(), F::one(), F::one()];
                    for coeff in window {
                        if coeff.index / 4 == block_index {
                            block[coeff.index % 4] = coeff.value;
                        }
                    }
                    let left = block[0] + r.mul_0_optimized(block[2] - block[0]);
                    let right = block[1] + r.mul_0_optimized(block[3] - block[1]);
                    if !left.is_one() {
                        let left_index = 2 * block_index;
                        bound.push((left_index, left).into());
                    }
                    if !right.is_one() {
                        let right_index = 2 * block_index + 1;
                        bound.push((right_index, right).into());
                    }
                }

                if window[1].index / 4 > window[0].index / 4 && window[1].index % 4 != 0 {
                    let block_index = window[1].index / 4;
                    let mut block = [F::one(), F::one(), F::one(), F::one()];
                    for coeff in window {
                        if coeff.index / 4 == block_index {
                            block[coeff.index % 4] = coeff.value;
                        }
                    }
                    let left = block[0] + r.mul_0_optimized(block[2] - block[0]);
                    let right = block[1] + r.mul_0_optimized(block[3] - block[1]);
                    if !left.is_one() {
                        let left_index = 2 * block_index;
                        bound.push((left_index, left).into());
                    }
                    if !right.is_one() {
                        let right_index = 2 * block_index + 1;
                        bound.push((right_index, right).into());
                    }
                }

                bound
            })
            .collect();

        // TODO(moodlezoup): Can avoid inserts
        if self.coeffs[0].index % 4 != 0 {
            // Process block starting with self.coeffs[0]
            let block_index = self.coeffs[0].index / 4;
            let mut block = [F::one(), F::one(), F::one(), F::one()];
            for coeff in &self.coeffs[..4] {
                if coeff.index / 4 == block_index {
                    block[coeff.index % 4] = coeff.value;
                }
            }
            let left = block[0] + r.mul_0_optimized(block[2] - block[0]);
            let right = block[1] + r.mul_0_optimized(block[3] - block[1]);
            if !right.is_one() {
                let right_index = 2 * block_index + 1;
                self.coeffs.insert(0, (right_index, right).into());
            }
            if !left.is_one() {
                let left_index = 2 * block_index;
                self.coeffs.insert(0, (left_index, left).into());
            }
        }

        // self.dense_len /= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, test_rng, One};
    use itertools::Itertools;

    fn random_sparse_vector(rng: &mut impl Rng, len: usize, density: f64) -> Vec<Fr> {
        std::iter::repeat_with(|| {
            if rng.gen_bool(density) {
                Fr::random(rng)
            } else {
                Fr::one()
            }
        })
        .take(len)
        .collect()
    }

    #[test]
    fn interleave_uninterleave() {
        let mut rng = test_rng();
        for (num_vars, density) in (0..10).cartesian_product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
            let left = random_sparse_vector(&mut rng, 1 << num_vars, density);
            let left = DensePolynomial::new(left);
            let right = random_sparse_vector(&mut rng, 1 << num_vars, density);
            let right = DensePolynomial::new(right);

            let interleaved = SparseInterleavedPolynomial::interleave(&left, &right);

            assert_eq!(interleaved.uninterleave(), (left, right));
        }
    }

    #[test]
    fn uninterleave_interleave() {
        let mut rng = test_rng();
        for (num_vars, density) in (0..10).cartesian_product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
            let mut coeffs = vec![];
            for i in 0..(2 << num_vars) {
                if rng.gen_bool(density) {
                    coeffs.push((i, Fr::random(&mut rng)).into())
                }
            }
            let interleaved = SparseInterleavedPolynomial::new(coeffs, 2 << num_vars);
            let (left, right) = interleaved.uninterleave();

            assert_eq!(
                interleaved.coeffs,
                SparseInterleavedPolynomial::interleave(&left, &right).coeffs
            );
        }
    }

    #[test]
    fn bind_par_blocks() {
        let mut rng = test_rng();
        const NUM_VARS: usize = 10;
        let left = random_sparse_vector(&mut rng, 1 << NUM_VARS, 0.2);
        let mut left = DensePolynomial::new(left);
        let right = random_sparse_vector(&mut rng, 1 << NUM_VARS, 0.2);
        let mut right = DensePolynomial::new(right);

        let mut interleaved = SparseInterleavedPolynomial::interleave(&left, &right);

        for _ in 0..NUM_VARS {
            let r = Fr::random(&mut rng);
            interleaved.bind_par_blocks(r);
            left.bound_poly_var_bot(&r);
            right.bound_poly_var_bot(&r);

            assert_eq!(
                interleaved.coeffs,
                SparseInterleavedPolynomial::interleave(&left, &right).coeffs
            );
        }
    }

    #[test]
    fn bind() {
        let mut rng = test_rng();
        const NUM_VARS: usize = 10;
        let left = random_sparse_vector(&mut rng, 1 << NUM_VARS, 0.2);
        let mut left = DensePolynomial::new(left);
        let right = random_sparse_vector(&mut rng, 1 << NUM_VARS, 0.2);
        let mut right = DensePolynomial::new(right);

        let mut interleaved = SparseInterleavedPolynomial::interleave(&left, &right);

        for _ in 0..NUM_VARS {
            let r = Fr::random(&mut rng);
            interleaved.bind(r);
            left.bound_poly_var_bot(&r);
            right.bound_poly_var_bot(&r);

            assert_eq!(
                interleaved.coeffs,
                SparseInterleavedPolynomial::interleave(&left, &right).coeffs
            );
        }
    }

    #[test]
    fn par_blocks() {
        let mut rng = test_rng();
        for (num_vars, density) in (0..10).cartesian_product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
            let mut coeffs = vec![];
            for i in 0..(2 << num_vars) {
                if rng.gen_bool(density) {
                    coeffs.push((i, Fr::random(&mut rng)).into())
                }
            }

            let poly = SparseInterleavedPolynomial::new(coeffs, 2 << num_vars);

            // Block size = 4
            let blocks: Vec<_> = poly.par_blocks(4).collect();
            for block in blocks.iter() {
                assert!(block.len() <= 4);
            }
            let concatenated_blocks = blocks.concat();
            // Check that blocks comprise the full polynomial
            assert_eq!(concatenated_blocks, poly.coeffs);

            // Block size = 2
            let blocks: Vec<_> = poly.par_blocks(2).collect();
            for block in blocks.iter() {
                assert!(block.len() <= 2);
            }
            let concatenated_blocks = blocks.concat();
            // Check that blocks comprise the full polynomial
            assert_eq!(concatenated_blocks, poly.coeffs);
        }
    }

    #[test]
    fn parallelizable_slices() {
        let mut rng = test_rng();
        for (num_vars, density) in (0..10).cartesian_product([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
            let mut coeffs = vec![];
            for i in 0..(2 << num_vars) {
                if rng.gen_bool(density) {
                    coeffs.push((i, Fr::random(&mut rng)).into())
                }
            }

            let poly = SparseInterleavedPolynomial::new(coeffs, 2 << num_vars);
            let slices = poly.parallelizable_slices();

            for (slice_index, slice) in slices.iter().enumerate().skip(1) {
                let prev_slice = slices[slice_index - 1];
                if !slice.is_empty() && !prev_slice.is_empty() {
                    // Check that slices do not split siblings
                    assert!(slice[0].index / 4 != prev_slice[0].index / 4);
                }
            }
            let concatenated_slices = slices.concat();
            // Check that slices comprise the full polynomial
            assert_eq!(concatenated_slices, poly.coeffs);
        }
    }
}
