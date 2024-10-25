use super::{dense_interleaved_poly::DenseInterleavedPolynomial, dense_mlpoly::DensePolynomial};
use crate::field::JoltField;
use rayon::prelude::*;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
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
    pub(crate) coeffs: Vec<Vec<SparseCoefficient<F>>>,
    pub(crate) coalesced: Option<DenseInterleavedPolynomial<F>>,
    pub(crate) dense_len: usize,
}

impl<F: JoltField> PartialEq for SparseInterleavedPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.dense_len != other.dense_len {
            return false;
        }
        if self.coalesced.is_some() != other.coalesced.is_some() {
            return false;
        }

        if self.coalesced.is_some() {
            self.coalesced == other.coalesced
        } else {
            self.coeffs == other.coeffs
        }
    }
}

impl<F: JoltField> SparseInterleavedPolynomial<F> {
    pub fn new(coeffs: Vec<Vec<SparseCoefficient<F>>>, dense_len: usize) -> Self {
        let batch_size = coeffs.len();
        assert!((dense_len / batch_size).is_power_of_two());
        if (dense_len / batch_size) <= 2 {
            // Coalesce
            let mut coalesced = vec![F::one(); dense_len];
            coeffs
                .iter()
                .flatten()
                .for_each(|sparse_coeff| coalesced[sparse_coeff.index] = sparse_coeff.value);
            Self {
                dense_len,
                coeffs: vec![vec![]; batch_size],
                coalesced: Some(DenseInterleavedPolynomial::new(coalesced)),
            }
        } else {
            Self {
                dense_len,
                coeffs,
                coalesced: None,
            }
        }
    }

    pub fn batch_size(&self) -> usize {
        self.coeffs.len()
    }

    pub fn to_dense(&self) -> DensePolynomial<F> {
        if let Some(coalesced) = &self.coalesced {
            DensePolynomial::new_padded(coalesced.coeffs[..coalesced.len()].to_vec())
        } else {
            let mut dense_layer = vec![F::one(); self.dense_len];
            for coeff in self.coeffs.iter().flatten() {
                dense_layer[coeff.index] = coeff.value;
            }
            DensePolynomial::new_padded(dense_layer)
        }
    }

    #[cfg(test)]
    pub fn interleave(left: &Vec<F>, right: &Vec<F>, batch_size: usize) -> Self {
        use itertools::Itertools;
        assert_eq!(left.len(), right.len());

        if left.len() <= batch_size {
            // Coalesced
            let coalesced: Vec<F> = left
                .into_iter()
                .interleave(right.into_iter())
                .cloned()
                .collect();
            let dense_len = coalesced.len();
            return Self {
                coeffs: vec![vec![]; batch_size],
                coalesced: Some(DenseInterleavedPolynomial::new(coalesced)),
                dense_len,
            };
        }

        let mut coeffs = vec![];
        let mut index_offset = 0usize;
        for (left_chunk, right_chunk) in left
            .chunks(left.len() / batch_size)
            .zip(right.chunks(right.len() / batch_size))
        {
            coeffs.push(
                left_chunk
                    .iter()
                    .interleave(right_chunk)
                    .enumerate()
                    .filter_map(|(index, coeff)| {
                        if coeff.is_one() {
                            None
                        } else {
                            Some((index_offset + index, *coeff).into())
                        }
                    })
                    .collect(),
            );
            index_offset += left_chunk.len() + right_chunk.len();
        }

        Self::new(coeffs, left.len() + right.len())
    }

    pub fn uninterleave(&self) -> (Vec<F>, Vec<F>) {
        if let Some(coalesced) = &self.coalesced {
            coalesced.uninterleave()
        } else {
            let mut left = vec![F::one(); self.dense_len / 2];
            let mut right = vec![F::one(); self.dense_len / 2];

            self.coeffs.iter().flatten().for_each(|coeff| {
                if coeff.index % 2 == 0 {
                    left[coeff.index / 2] = coeff.value;
                } else {
                    right[coeff.index / 2] = coeff.value;
                }
            });
            (left, right)
        }
    }

    pub fn par_blocks(&self) -> impl ParallelIterator<Item = &[SparseCoefficient<F>]> {
        self.coeffs
            .par_iter()
            .flat_map(|segment| segment.par_chunk_by(|x, y| x.index / 4 == y.index / 4))
    }

    #[tracing::instrument(skip_all, name = "SparseInterleavedPolynomial::bind")]
    pub fn bind(&mut self, r: F) {
        if let Some(coalesced) = &mut self.coalesced {
            let padded_len = self.dense_len.next_multiple_of(4);
            coalesced.bind(r);
            self.dense_len = padded_len / 2;
            return;
        }

        self.coeffs
            .par_iter_mut()
            .for_each(|segment: &mut Vec<SparseCoefficient<F>>| {
                let mut next_left_node_to_process = 0;
                let mut next_right_node_to_process = 0;
                let mut bound_index = 0;

                for j in 0..segment.len() {
                    let current = segment[j];
                    if current.index % 2 == 0 && current.index < next_left_node_to_process {
                        // This left node was already bound with its sibling in a previous iteration
                        continue;
                    }
                    if current.index % 2 == 1 && current.index < next_right_node_to_process {
                        // This right node was already bound with its sibling in a previous iteration
                        continue;
                    }

                    let neighbors = [
                        segment
                            .get(j + 1)
                            .cloned()
                            .unwrap_or((current.index + 1, F::one()).into()),
                        segment
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
                            segment[bound_index] = (
                                current.index / 2,
                                current.value + r * (sibling_value - current.value),
                            )
                                .into();
                            next_left_node_to_process = current.index + 4;
                        }
                        1 => {
                            // Edge case: If this right node's neighbor is not 1 and has _not_
                            // been bound yet, we need to bind the neighbor first to preserve
                            // the monotonic ordering of the bound layer.
                            if next_left_node_to_process <= current.index + 1 {
                                let left_neighbor: F = find_neighbor(current.index + 1);
                                if !left_neighbor.is_one() {
                                    segment[bound_index] = (
                                        current.index / 2,
                                        F::one() + r * (left_neighbor - F::one()),
                                    )
                                        .into();
                                    bound_index += 1;
                                }
                                next_left_node_to_process = current.index + 3;
                            }

                            // Find sibling right node
                            let sibling_value: F = find_neighbor(current.index + 2);
                            segment[bound_index] = (
                                current.index / 2 + 1,
                                current.value + r * (sibling_value - current.value),
                            )
                                .into();
                            next_right_node_to_process = current.index + 4;
                        }
                        2 => {
                            // Sibling left node wasn't encountered in previous iteration,
                            // so sibling must have value 1.
                            segment[bound_index] = (
                                current.index / 2 - 1,
                                F::one() + r * (current.value - F::one()),
                            )
                                .into();
                            next_left_node_to_process = current.index + 2;
                        }
                        3 => {
                            // Sibling right node wasn't encountered in previous iteration,
                            // so sibling must have value 1.
                            segment[bound_index] =
                                (current.index / 2, F::one() + r * (current.value - F::one()))
                                    .into();
                            next_right_node_to_process = current.index + 2;
                        }
                        _ => unreachable!("?_?"),
                    }
                    bound_index += 1;
                }
                segment.truncate(bound_index);
            });

        self.dense_len /= 2;
        if (self.dense_len / self.batch_size()) == 2 {
            // Coalesce
            let mut coalesced = vec![F::one(); self.dense_len];
            self.coeffs
                .iter()
                .flatten()
                .for_each(|sparse_coeff| coalesced[sparse_coeff.index] = sparse_coeff.value);
            self.coalesced = Some(DenseInterleavedPolynomial::new(coalesced));
        }
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
    use crate::utils::math::Math;

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
        const NUM_VARS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        let mut rng = test_rng();
        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let left = random_sparse_vector(&mut rng, batch_size * (1 << num_vars), density);
            let right = random_sparse_vector(&mut rng, batch_size * (1 << num_vars), density);

            let interleaved = SparseInterleavedPolynomial::interleave(&left, &right, batch_size);

            assert_eq!(interleaved.uninterleave(), (left, right));
        }
    }

    #[test]
    fn uninterleave_interleave() {
        const NUM_VARS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        let mut rng = test_rng();
        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let coeffs = (0..batch_size)
                .map(|batch_index| {
                    let mut coeffs: Vec<SparseCoefficient<Fr>> = vec![];
                    for i in 0..(1 << num_vars) {
                        if rng.gen_bool(density) {
                            coeffs.push(
                                (batch_index * (1 << num_vars) + i, Fr::random(&mut rng)).into(),
                            )
                        }
                    }
                    coeffs
                })
                .collect();
            let interleaved = SparseInterleavedPolynomial::new(coeffs, batch_size << num_vars);
            let (left, right) = interleaved.uninterleave();

            assert_eq!(
                interleaved,
                SparseInterleavedPolynomial::interleave(&left, &right, batch_size)
            );
        }
    }

    #[test]
    fn bind() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];
        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let coeffs = (0..batch_size)
                .map(|batch_index| {
                    let mut coeffs: Vec<SparseCoefficient<Fr>> = vec![];
                    for i in 0..(1 << num_vars) {
                        if rng.gen_bool(density) {
                            coeffs.push(
                                (batch_index * (1 << num_vars) + i, Fr::random(&mut rng)).into(),
                            )
                        }
                    }
                    coeffs
                })
                .collect();
            let mut interleaved = SparseInterleavedPolynomial::new(coeffs, batch_size << num_vars);
            let (mut left, mut right) = interleaved.uninterleave();
            assert_eq!(
                interleaved,
                SparseInterleavedPolynomial::interleave(&left, &right, batch_size)
            );

            for _ in 0..num_vars + batch_size.log_2() - 1 {
                let r = Fr::random(&mut rng);
                interleaved.bind(r);
                bind_left_and_right(&mut left, &mut right, r);

                assert_eq!(
                    interleaved,
                    SparseInterleavedPolynomial::interleave(&left, &right, batch_size)
                );
            }
        }
    }
}
