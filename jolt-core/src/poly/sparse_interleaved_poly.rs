use super::{
    dense_interleaved_poly::DenseInterleavedPolynomial, dense_mlpoly::DensePolynomial,
    split_eq_poly::SplitEqPolynomial, unipoly::UniPoly,
};
use crate::{
    field::{JoltField, OptimizedMul},
    subprotocols::{
        grand_product::BatchedGrandProductLayer,
        sumcheck::{BatchedCubicSumcheck, Bindable},
    },
    utils::math::Math,
};
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

    pub fn coalesce(&self) -> Vec<F> {
        if let Some(coalesced) = &self.coalesced {
            coalesced.coeffs.clone()
        } else {
            let mut coalesced = vec![F::one(); self.dense_len];
            self.coeffs
                .iter()
                .flatten()
                .for_each(|sparse_coeff| coalesced[sparse_coeff.index] = sparse_coeff.value);
            coalesced
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

    pub fn layer_output(&self) -> Self {
        if let Some(coalesced) = &self.coalesced {
            Self {
                dense_len: self.dense_len / 2,
                coeffs: vec![vec![]; self.batch_size()],
                coalesced: Some(coalesced.layer_output()),
            }
        } else {
            let coeffs: Vec<Vec<_>> = self
                .coeffs
                .par_iter()
                .map(|segment| {
                    segment
                        .par_chunk_by(move |x, y| x.index / 2 == y.index / 2)
                        .map(|sparse_block| {
                            let mut dense_block = [F::one(); 2];
                            for coeff in sparse_block {
                                dense_block[coeff.index % 2] = coeff.value;
                            }

                            let output_index = sparse_block[0].index / 2;
                            let output_value = dense_block[0].mul_1_optimized(dense_block[1]);
                            (output_index, output_value).into()
                        })
                        .collect()
                })
                .collect();

            Self::new(coeffs, self.dense_len / 2)
        }
    }
}

impl<F: JoltField> Bindable<F> for SparseInterleavedPolynomial<F> {
    /// Incrementally binds a variable of this batched layer's polynomials.
    /// If `self` is dense, we bind as in `BatchedDenseGrandProductLayer`,
    /// processing nodes 4 at a time to preserve the interleaved order:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    /// If `self` is sparse, we basically do the same thing but with more
    /// cases to check 😬
    #[tracing::instrument(skip_all, name = "SparseInterleavedPolynomial::bind")]
    fn bind(&mut self, r: F) {
        #[cfg(test)]
        let (mut left_before_binding, mut right_before_binding) = self.uninterleave();

        if let Some(coalesced) = &mut self.coalesced {
            let padded_len = self.dense_len.next_multiple_of(4);
            coalesced.bind(r);
            self.dense_len = padded_len / 2;
        } else {
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
                self.coalesced = Some(DenseInterleavedPolynomial::new(self.coalesce()));
            }
        }

        #[cfg(test)]
        {
            let (left_after_binding, right_after_binding) = self.uninterleave();
            bind_left_and_right(&mut left_before_binding, &mut right_before_binding, r);

            assert_eq!(
                *self,
                Self::interleave(
                    &left_before_binding,
                    &right_before_binding,
                    self.batch_size()
                )
            );
            assert_eq!(left_after_binding, left_before_binding);
            assert_eq!(right_after_binding, right_before_binding);
        }
    }
}

impl<F: JoltField> BatchedGrandProductLayer<F> for SparseInterleavedPolynomial<F> {}
impl<F: JoltField> BatchedCubicSumcheck<F> for SparseInterleavedPolynomial<F> {
    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &SplitEqPolynomial<F>, round_claim: F) {
        let merged_eq = eq_poly.merge();
        let (left, right) = self.uninterleave();
        let expected: F = left
            .iter()
            .zip(right.iter())
            .zip(merged_eq.evals_ref().iter())
            .map(|((l, r), eq)| *eq * l * r)
            .sum();
        assert_eq!(expected, round_claim);
    }

    /// We want to compute the evaluations of the following univariate cubic polynomial at
    /// points {0, 1, 2, 3}:
    ///     Σ eq(r, x) * left(x) * right(x)
    /// where the inner summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent coefficients of
    /// `eq`, `left`, and `right`.
    /// If `self` is dense, we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    /// If `self` is sparse, we basically do the same thing but with some fancy optimizations and
    /// more cases to check 😬
    #[tracing::instrument(skip_all, name = "BatchedSparseGrandProductLayer::compute_cubic")]
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        if let Some(coalesced) = &self.coalesced {
            return coalesced.compute_cubic(eq_poly, previous_round_claim);
        }

        let cubic_evals = if eq_poly.E1_len == 1 {
            let eq_evals: Vec<(F, F, F)> = eq_poly
                .E2
                .par_chunks(2)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            // TODO(moodlezoup): Can more efficiently compute these
            let eq_eval_sums: (F, F, F) = eq_evals
                .par_iter()
                .fold(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            let deltas: Vec<(F, F, F)> = self
                .coeffs
                .par_iter()
                .flat_map(|segment| {
                    segment
                        .par_chunk_by(|x, y| x.index / 4 == y.index / 4)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 4;
                            let mut block = [F::one(); 4];
                            for coeff in sparse_block {
                                block[coeff.index % 4] = coeff.value;
                            }

                            let left = (block[0], block[2]);
                            let right = (block[1], block[3]);

                            let m_left = left.1 - left.0;
                            let m_right = right.1 - right.0;

                            let left_eval_2 = left.1 + m_left;
                            let left_eval_3 = left_eval_2 + m_left;

                            let right_eval_2 = right.1 + m_right;
                            let right_eval_3 = right_eval_2 + m_right;

                            let E2_eval = eq_poly.E2[block_index];
                            // TODO(moodlezoup): Can save a multiplication here
                            (
                                E2_eval * (left.0 * right.0 - F::one()),
                                E2_eval * (left_eval_2 * right_eval_2 - F::one()),
                                E2_eval * (left_eval_3 * right_eval_3 - F::one()),
                            )
                        })
                })
                .collect();

            deltas.into_par_iter().reduce(
                || eq_eval_sums,
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            )
        } else {
            let deltas: Vec<(F, F, F, usize)> = self
                .coeffs
                .par_iter()
                .flat_map(|segment| {
                    segment
                        .par_chunk_by(|x, y| x.index / 4 == y.index / 4)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 4;
                            let mut block = [F::one(); 4];
                            for coeff in sparse_block {
                                block[coeff.index % 4] = coeff.value;
                            }

                            let left = (block[0], block[2]);
                            let right = (block[1], block[3]);

                            let m_left = left.1 - left.0;
                            let m_right = right.1 - right.0;

                            let left_eval_2 = left.1 + m_left;
                            let left_eval_3 = left_eval_2 + m_left;

                            let right_eval_2 = right.1 + m_right;
                            let right_eval_3 = right_eval_2 + m_right;

                            let num_x1_bits = eq_poly.E1_len.log_2() - 1;
                            let x1_bitmask = (1 << num_x1_bits) - 1;
                            let x1 = block_index & x1_bitmask;
                            let x2 = block_index >> num_x1_bits;

                            let E2_eval = eq_poly.E2[x2];
                            // TODO(moodlezoup): Can save a multiplication here
                            (
                                E2_eval * (left.0 * right.0 - F::one()),
                                E2_eval * (left_eval_2 * right_eval_2 - F::one()),
                                E2_eval * (left_eval_3 * right_eval_3 - F::one()),
                                x1,
                            )
                        })
                })
                .collect();

            let mut inner_sums: Vec<(F, F, F)> =
                vec![(F::one(), F::one(), F::one()); eq_poly.E1_len / 2];
            for delta in deltas.iter() {
                let x1 = delta.3;
                inner_sums[x1].0 += delta.0;
                inner_sums[x1].1 += delta.1;
                inner_sums[x1].2 += delta.2;
            }

            // Correct for the fact that the batch size is padded to a power of two
            // with all-0 circuits.
            // TODO(moodlezoup): optimize this
            for x in (self.dense_len..self.dense_len.next_power_of_two()).step_by(4) {
                let block_index = x / 4;
                let num_x1_bits = eq_poly.E1_len.log_2() - 1;
                let x1_bitmask = (1 << num_x1_bits) - 1;
                let x1 = block_index & x1_bitmask;
                let x2 = block_index >> num_x1_bits;
                let E2_eval = eq_poly.E2[x2];

                inner_sums[x1].0 -= E2_eval;
                inner_sums[x1].1 -= E2_eval;
                inner_sums[x1].2 -= E2_eval;
            }

            eq_poly.E1[..eq_poly.E1_len]
                .par_chunks(2)
                .zip(inner_sums.par_iter())
                .map(|(E1_chunk, inner_sum)| {
                    let E1_evals = {
                        let eval_point_0 = E1_chunk[0];
                        let m_eq = E1_chunk[1] - E1_chunk[0];
                        let eval_point_2 = E1_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };
                    (
                        E1_evals.0 * inner_sum.0,
                        E1_evals.1 * inner_sum.1,
                        E1_evals.2 * inner_sum.2,
                    )
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
        };

        let cubic_evals = [
            cubic_evals.0,
            previous_round_claim - cubic_evals.0,
            cubic_evals.1,
            cubic_evals.2,
        ];

        let cubic = UniPoly::from_evals(&cubic_evals);

        #[cfg(test)]
        {
            let dense = DenseInterleavedPolynomial::new(self.coalesce());
            let dense_cubic = dense.compute_cubic(eq_poly, previous_round_claim);
            assert_eq!(cubic, dense_cubic);
        }

        cubic
    }

    fn final_claims(&self) -> (F, F) {
        assert_eq!(self.dense_len, 2);
        let dense = self.to_dense();
        (dense[0], dense[1])
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
