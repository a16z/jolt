use super::dense_mlpoly::DensePolynomial;
use crate::field::{JoltField, OptimizedMul};
use rayon::prelude::*;
use smallvec::SmallVec;

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
    pub coeffs: Vec<SparseCoefficient<F>>,
    pub dense_len: usize,
    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> SparseInterleavedPolynomial<F> {
    pub fn new(coeffs: Vec<SparseCoefficient<F>>, dense_len: usize) -> Self {
        assert!(dense_len.is_power_of_two());
        let sparse_len = coeffs.len();
        Self {
            dense_len,
            coeffs,
            binding_scratch_space: vec![SparseCoefficient::default(); sparse_len], // TODO(moodlezoup): can optimize
        }
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

    pub fn parallelizable_slices(coeffs: &[SparseCoefficient<F>]) -> Vec<&[SparseCoefficient<F>]> {
        let num_slices = 4 * rayon::current_num_threads();
        let target_slice_length = coeffs.len() / num_slices;
        if target_slice_length == 0 {
            return vec![&coeffs];
        }

        let mut boundary_indices: Vec<_> = (target_slice_length..coeffs.len())
            .step_by(target_slice_length)
            .collect();

        for boundary in boundary_indices.iter_mut() {
            while *boundary < coeffs.len() - 1 {
                let current = &coeffs[*boundary];
                let next = &coeffs[*boundary + 1];
                if next.index / 4 > current.index / 4 {
                    *boundary += 1;
                    break;
                }
                *boundary += 1;
            }
            if *boundary == coeffs.len() - 1 {
                *boundary += 1;
            }
        }
        boundary_indices.dedup();

        let mut slices = vec![&coeffs[..boundary_indices[0]]];
        for boundaries in boundary_indices.windows(2) {
            slices.push(&coeffs[boundaries[0]..boundaries[1]])
        }
        let last_boundary = *boundary_indices.last().unwrap();
        if last_boundary != coeffs.len() {
            slices.push(&coeffs[last_boundary..]);
        }

        slices
    }

    pub fn bind_slices(&mut self, r: F) {
        unsafe {
            self.binding_scratch_space.set_len(0);
        }
        let slices = Self::parallelizable_slices(&self.coeffs);
        let binding_iter = slices.par_iter().flat_map_iter(|slice| {
            let mut bound: Vec<SparseCoefficient<F>> = Vec::with_capacity(slice.len());

            let mut next_left_node_to_process = 0usize;
            let mut next_right_node_to_process = 0usize;

            for j in 0..slice.len() {
                let current = &slice[j];
                if current.index % 2 == 0 && current.index < next_left_node_to_process {
                    // This left node was already bound with its sibling in a previous iteration
                    continue;
                }
                if current.index % 2 == 1 && current.index < next_right_node_to_process {
                    // This right node was already bound with its sibling in a previous iteration
                    continue;
                }

                let neighbors = [
                    slice
                        .get(j + 1)
                        .cloned()
                        .unwrap_or((current.index + 1, F::one()).into()),
                    slice
                        .get(j + 2)
                        .cloned()
                        .unwrap_or((current.index + 2, F::one()).into()),
                ];
                let find_neighbor = |query_index: usize| {
                    neighbors
                        .iter()
                        .find_map(|neighbor| {
                            if neighbor.index == query_index {
                                Some(neighbor.value)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(F::one())
                };

                match current.index % 4 {
                    0 => {
                        // Find sibling left node
                        let sibling_value: F = find_neighbor(current.index + 2);
                        bound.push(
                            (
                                current.index / 2,
                                current.value + r * (sibling_value - current.value),
                            )
                                .into(),
                        );
                        next_left_node_to_process = current.index + 4;
                    }
                    1 => {
                        // Edge case: If this right node's neighbor is not 1 and has _not_
                        // been bound yet, we need to bind the neighbor first to preserve
                        // the monotonic ordering of the bound layer.
                        if next_left_node_to_process <= current.index + 1 {
                            let left_neighbor: F = find_neighbor(current.index + 1);
                            if !left_neighbor.is_one() {
                                bound.push(
                                    (current.index / 2, F::one() + r * (left_neighbor - F::one()))
                                        .into(),
                                );
                            }
                            next_left_node_to_process = current.index + 3;
                        }

                        // Find sibling right node
                        let sibling_value: F = find_neighbor(current.index + 2);
                        bound.push(
                            (
                                current.index / 2 + 1,
                                current.value + r * (sibling_value - current.value),
                            )
                                .into(),
                        );
                        next_right_node_to_process = current.index + 4;
                    }
                    2 => {
                        // Sibling left node wasn't encountered in previous iteration,
                        // so sibling must have value 1.
                        bound.push(
                            (
                                current.index / 2 - 1,
                                F::one() + r * (current.value - F::one()),
                            )
                                .into(),
                        );
                        next_left_node_to_process = current.index + 2;
                    }
                    3 => {
                        // Sibling right node wasn't encountered in previous iteration,
                        // so sibling must have value 1.
                        bound.push(
                            (current.index / 2, F::one() + r * (current.value - F::one())).into(),
                        );
                        next_right_node_to_process = current.index + 2;
                    }
                    _ => unreachable!("?_?"),
                }
            }

            bound.into_iter()
        });

        self.binding_scratch_space.par_extend(binding_iter);
        std::mem::swap(&mut self.coeffs, &mut self.binding_scratch_space);

        self.dense_len /= 2;
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

    pub fn bind(&mut self, r: F) {
        unsafe {
            self.binding_scratch_space.set_len(0);
        }

        let binding_iter = self
            .coeffs
            .par_chunk_by(|x, y| x.index / 4 == y.index / 4)
            .flat_map_iter(|sparse_block| {
                let mut bound = SmallVec::<[SparseCoefficient<F>; 2]>::new();

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
                    bound.push((left_index, left).into());
                }
                if !right.is_one() {
                    let right_index = 2 * block_index + 1;
                    bound.push((right_index, right).into());
                }
                bound.into_iter()
            });

        self.binding_scratch_space.par_extend(binding_iter);
        std::mem::swap(&mut self.coeffs, &mut self.binding_scratch_space);

        self.dense_len /= 2;
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
            let slices = SparseInterleavedPolynomial::parallelizable_slices(&poly.coeffs);

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
