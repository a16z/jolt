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
    utils::{math::Math, transcript::Transcript},
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

/// Represents a single layer of a sparse grand product circuit.
///
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      /  \
///   L0   R0  L1   R1  L2   R2  L3   R3   <- This is layer would be represented as [L0, R0, L1, R1, L2, R2, L3, R3]
///                                           (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
///
/// Where SparseInterleavedPolynomial differs from DenseInterleavedPolynomial
/// is that many of the coefficients are expected to be 1s, so the circuit may
/// look something like this:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      /  \
///    1   R0   1   1   L2   1    1    1
///
/// Instead of materializing all the 1s, we use a sparse vector to represent the layer,
/// where each element of the vector contains the index and value of a non-one coefficient.
/// So the above layer would be represented by:
///   vec![(1, R0), (4, L2)]        (except with `SparseCoefficient` structs, not tuples)
///
/// In the context of a batched grand product (see sparse_grand_product.rs), there
/// are k of these sparse vectors, where k is the batch size.
/// For the first log2(n) rounds of binding, these k vectors can be processed in parallel.
/// After that, they are "coalesced" into a single DenseInterleavedPolynomial for the
/// remaining rounds of binding.
#[derive(Default, Debug, Clone)]
pub struct SparseInterleavedPolynomial<F: JoltField> {
    /// A vector of sparse vectors representing the coefficients in a batched grand product
    /// layer, where batch size = coeffs.len().
    pub(crate) coeffs: Vec<Vec<SparseCoefficient<F>>>,
    /// Once `coeffs` cannot be bound further (i.e. binding would require processing values
    /// in different vectors), we switch to using `coalesced` to represent the grand product
    /// layer. See `SparseInterleavedPolynomial::coalesce()`.
    pub(crate) coalesced: Option<DenseInterleavedPolynomial<F>>,
    /// The length of the layer if it were represented by a single dense vector.
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
                // The batch size is implied by coeffs.len(), so we must initialize this
                // vector:
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

    /// Converts a `SparseInterleavedPolynomial` into the equivalent `DensePolynomial`.
    pub fn to_dense(&self) -> DensePolynomial<F> {
        DensePolynomial::new_padded(self.coalesce())
    }

    #[tracing::instrument(skip_all, name = "SparseInterleavedPolynomial::coalesce")]
    /// Coalesces a `SparseInterleavedPolynomial` into a `DenseInterleavedPolynomial`.
    pub fn coalesce(&self) -> Vec<F> {
        if let Some(coalesced) = &self.coalesced {
            coalesced.coeffs[..coalesced.len()].to_vec()
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

    /// Uninterleaves a `SparseInterleavedPolynomial` into two vectors
    /// containing the left and right coefficients.
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

    /// Computes the grand product layer output by this one.
    ///     L0'      R0'      L1'      R1'     <- Output layer
    ///      Λ        Λ        Λ        Λ
    ///     / \      / \      / \      /  \
    ///   L0   R0  L1   R1  L2   R2  L3   R3   <- This layer
    #[tracing::instrument(skip_all, name = "SparseInterleavedPolynomial::layer_output")]
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
                    let mut output_segment: Vec<SparseCoefficient<F>> =
                        Vec::with_capacity(segment.len());
                    let mut next_index_to_process = 0usize;
                    for (j, coeff) in segment.iter().enumerate() {
                        if coeff.index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if coeff.index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = segment
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((coeff.index + 1, F::one()).into());
                            if right.index == coeff.index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_segment
                                    .push((coeff.index / 2, right.value * coeff.value).into());
                            } else {
                                // Corresponding right node not found, so it must be 1
                                output_segment.push((coeff.index / 2, coeff.value).into());
                            }
                            next_index_to_process = coeff.index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration, so it must have value 1
                            output_segment.push((coeff.index / 2, coeff.value).into());
                            next_index_to_process = coeff.index + 1;
                        }
                    }
                    output_segment
                })
                .collect();

            Self::new(coeffs, self.dense_len / 2)
        }
    }
}

impl<F: JoltField> Bindable<F> for SparseInterleavedPolynomial<F> {
    /// Incrementally binds a variable of the interleaved left and right polynomials.
    /// If `self` is coalesced, we invoke `DenseInterleavedPolynomial::bind`,
    /// processing nodes 4 at a time to preserve the interleaved order:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    ///
    /// If `self` is not coalesced, we basically do the same thing but with the
    /// sparse vectors in `self.coeffs`, and many more cases to check 😬
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

impl<F: JoltField, ProofTranscript: Transcript> BatchedGrandProductLayer<F, ProofTranscript>
    for SparseInterleavedPolynomial<F>
{
}
impl<F: JoltField, ProofTranscript: Transcript> BatchedCubicSumcheck<F, ProofTranscript>
    for SparseInterleavedPolynomial<F>
{
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
    /// If `self` is coalesced, we invoke `DenseInterleavedPolynomial::compute_cubic`, processing
    /// 4 values at a time:
    ///                 coeffs = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    ///
    /// If `self` is not coalesced, we basically do the same thing but with with the
    /// sparse vectors in `self.coeffs`, some fancy optimizations, and many more cases to check 😬
    #[tracing::instrument(skip_all, name = "SparseInterleavedPolynomial::compute_cubic")]
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        if let Some(coalesced) = &self.coalesced {
            return BatchedCubicSumcheck::<F, ProofTranscript>::compute_cubic(
                coalesced,
                eq_poly,
                previous_round_claim,
            );
        }

        // We use the Dao-Thaler optimization for the EQ polynomial, so there are two cases we
        // must handle. For details, refer to Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
        let cubic_evals = if eq_poly.E1_len == 1 {
            // If `eq_poly.E1` has been fully bound, we compute the cubic polynomial as we
            // would without the Dao-Thaler optimization, using the standard linear-time
            // sumcheck algorithm with optimizations for sparsity.

            let eq_evals: Vec<(F, F, F)> = eq_poly
                .E2
                .par_chunks(2)
                .take(self.dense_len / 4)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            // This is what Σ eq(r, x) * left(x) * right(x) would be if
            // `left` and `right` were both all ones.
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
            // Now we compute the deltas, correcting `eq_eval_sums` for the
            // elements of `left` and `right` that aren't ones.
            let deltas: (F, F, F) = self
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

                            let eq_evals = eq_evals[block_index];
                            (
                                eq_evals
                                    .0
                                    .mul_0_optimized(left.0.mul_1_optimized(right.0) - F::one()),
                                eq_evals.1 * (left_eval_2 * right_eval_2 - F::one()),
                                eq_evals.2 * (left_eval_3 * right_eval_3 - F::one()),
                            )
                        })
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            (
                eq_eval_sums.0 + deltas.0,
                eq_eval_sums.1 + deltas.1,
                eq_eval_sums.2 + deltas.2,
            )
        } else {
            // This is a more complicated version of the `else` case in
            // `DenseInterleavedPolynomial::compute_cubic`. Read that one first.

            // We start by computing the E1 evals:
            // (1 - j) * E1[0, x1] + j * E1[1, x1]
            let E1_evals: Vec<_> = eq_poly.E1[..eq_poly.E1_len]
                .par_chunks(2)
                .map(|E1_chunk| {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            // Now compute \sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1])
            let E1_eval_sums: (F, F, F) = E1_evals
                .par_iter()
                .fold(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            let num_x1_bits = eq_poly.E1_len.log_2() - 1;
            let x1_bitmask = (1 << num_x1_bits) - 1;

            // Iterate over the non-one coefficients and compute the deltas (relative to
            // what the cubic would be if all the coefficients were ones).
            let deltas = self
                .coeffs
                .par_iter()
                .flat_map(|segment| {
                    segment
                        .par_chunk_by(|a, b| {
                            // Group by x2
                            let a_x2 = (a.index / 4) >> num_x1_bits;
                            let b_x2 = (b.index / 4) >> num_x1_bits;
                            a_x2 == b_x2
                        })
                        .map(|chunk| {
                            let mut inner_sum = (F::zero(), F::zero(), F::zero());
                            for sparse_block in chunk.chunk_by(|x, y| x.index / 4 == y.index / 4) {
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

                                let x1 = block_index & x1_bitmask;
                                let delta = (
                                    E1_evals[x1].0.mul_0_optimized(
                                        left.0.mul_1_optimized(right.0) - F::one(),
                                    ),
                                    E1_evals[x1].1 * (left_eval_2 * right_eval_2 - F::one()),
                                    E1_evals[x1].2 * (left_eval_3 * right_eval_3 - F::one()),
                                );
                                inner_sum.0 += delta.0;
                                inner_sum.1 += delta.1;
                                inner_sum.2 += delta.2;
                            }

                            let x2 = (chunk[0].index / 4) >> num_x1_bits;
                            (
                                eq_poly.E2[x2] * inner_sum.0,
                                eq_poly.E2[x2] * inner_sum.1,
                                eq_poly.E2[x2] * inner_sum.2,
                            )
                        })
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            // The cubic evals assuming all the coefficients are ones is affected by the
            // `dense_len`, since we implicitly 0-pad the `dense_len` to a power of 2.
            //
            // As a refresher, the cubic evals we're computing are:
            //
            // \sum_x2 E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
            let evals_assuming_all_ones = if self.dense_len.is_power_of_two() {
                // If `dense_len` is a power of 2, there is no 0-padding.
                //
                // So we have:
                // \sum_x2 (E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * 1))
                //   = \sum_x2 (E2[x2] * \sum_x1 E1_evals[x1])
                //   = (\sum_x2 E2[x2]) * (\sum_x1 E1_evals[x1])
                //   = 1 * E1_eval_sums
                E1_eval_sums
            } else {
                let chunk_size = self.dense_len.next_power_of_two() / eq_poly.E2_len;
                let num_all_one_chunks = self.dense_len / chunk_size;
                let E2_sum: F = eq_poly.E2[..num_all_one_chunks].iter().sum();
                if self.dense_len % chunk_size == 0 {
                    // If `dense_len` isn't a power of 2 but evenly divides `chunk_size`,
                    // that means that for the last values of x2, we have:
                    //   (1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)) = 0
                    // due to the 0-padding.
                    //
                    // This makes the entire inner sum 0 for those values of x2.
                    // So we can simply sum over E2 for the _other_ values of x2, and
                    // multiply by `E1_eval_sums`.
                    (
                        E2_sum * E1_eval_sums.0,
                        E2_sum * E1_eval_sums.1,
                        E2_sum * E1_eval_sums.2,
                    )
                } else {
                    // If `dense_len` isn't a power of 2 and doesn't divide `chunk_size`,
                    // the last nonzero "chunk" will have (self.dense_len % chunk_size) ones,
                    // followed by (chunk_size - self.dense_len % chunk_size) zeros,
                    // e.g. 1 1 1 1 1 1 1 1 0 0 0 0
                    //
                    // This handles this last chunk:
                    let last_chunk_evals = E1_evals[..(self.dense_len % chunk_size) / 4]
                        .par_iter()
                        .fold(
                            || (F::zero(), F::zero(), F::zero()),
                            |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                        )
                        .reduce(
                            || (F::zero(), F::zero(), F::zero()),
                            |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                        );
                    (
                        E2_sum * E1_eval_sums.0
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals.0,
                        E2_sum * E1_eval_sums.1
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals.1,
                        E2_sum * E1_eval_sums.2
                            + eq_poly.E2[num_all_one_chunks] * last_chunk_evals.2,
                    )
                }
            };

            (
                evals_assuming_all_ones.0 + deltas.0,
                evals_assuming_all_ones.1 + deltas.1,
                evals_assuming_all_ones.2 + deltas.2,
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
            let dense_cubic = BatchedCubicSumcheck::<F, ProofTranscript>::compute_cubic(
                &dense,
                eq_poly,
                previous_round_claim,
            );
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
