use crate::field::JoltField;
use rayon::prelude::*;

#[cfg(test)]
use super::dense_mlpoly::DensePolynomial;

#[derive(Default, Debug, Clone, PartialEq)]
struct SparseCoefficient<F: JoltField> {
    index: usize,
    value: F,
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
    // num_vars: usize,
    pub(crate) coeffs: Vec<SparseCoefficient<F>>,
    #[cfg(test)]
    dense_len: usize,
    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> SparseInterleavedPolynomial<F> {
    pub fn new(coeffs: Vec<SparseCoefficient<F>>, _dense_len: usize) -> Self {
        assert!(_dense_len.is_power_of_two());
        Self {
            #[cfg(test)]
            dense_len: _dense_len,
            coeffs,
            binding_scratch_space: vec![],
        }
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

    pub fn bind(&mut self, r: F) {
        let last_index = self.coeffs.last().unwrap().index;
        self.coeffs.push((last_index + 1, F::one()).into());
        self.coeffs.push((last_index + 2, F::one()).into());
        self.coeffs.push((last_index + 3, F::one()).into());

        // TODO(moodlezoup): Is it more efficient to filter first?
        self.binding_scratch_space = self
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
                    let left = block[0] + r * (block[2] - block[0]);
                    let right = block[1] + r * (block[3] - block[1]);
                    let left_index = 2 * block_index;
                    let right_index = 2 * block_index + 1;
                    if !left.is_one() {
                        bound.push((left_index, left).into());
                    }
                    if !right.is_one() {
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
                    let left = block[0] + r * (block[2] - block[0]);
                    let right = block[1] + r * (block[3] - block[1]);
                    let left_index = 2 * block_index;
                    let right_index = 2 * block_index + 1;
                    if !left.is_one() {
                        bound.push((left_index, left).into());
                    }
                    if !right.is_one() {
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
            let left = block[0] + r * (block[2] - block[0]);
            let right = block[1] + r * (block[3] - block[1]);
            let left_index = 2 * block_index;
            let right_index = 2 * block_index + 1;
            if !right.is_one() {
                self.binding_scratch_space
                    .insert(0, (right_index, right).into());
            }
            if !left.is_one() {
                self.binding_scratch_space
                    .insert(0, (left_index, left).into());
            }
        }

        std::mem::swap(&mut self.coeffs, &mut self.binding_scratch_space);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, test_rng, One};

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
        const NUM_VARS: usize = 10;
        let left = random_sparse_vector(&mut rng, 1 << NUM_VARS, 0.2);
        let left = DensePolynomial::new(left);
        let right = random_sparse_vector(&mut rng, 1 << NUM_VARS, 0.2);
        let right = DensePolynomial::new(right);

        let interleaved = SparseInterleavedPolynomial::interleave(&left, &right);

        assert_eq!(interleaved.uninterleave(), (left, right));
    }

    #[test]
    fn uninterleave_interleave() {
        let mut rng = test_rng();
        const NUM_VARS: usize = 10;
        let mut coeffs = vec![];
        for i in 0..(2 << NUM_VARS) {
            if rng.gen_bool(0.5) {
                coeffs.push((i, Fr::random(&mut rng)).into())
            }
        }
        let interleaved = SparseInterleavedPolynomial::new(coeffs, 2 << NUM_VARS);
        let (left, right) = interleaved.uninterleave();

        assert_eq!(
            interleaved.coeffs,
            SparseInterleavedPolynomial::interleave(&left, &right).coeffs
        );
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
}
