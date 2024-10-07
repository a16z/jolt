use super::grand_product::{
    BatchedGrandProduct, BatchedGrandProductLayer, BatchedGrandProductLayerProof,
};
use super::sumcheck::BatchedCubicSumcheck;
use crate::field::{JoltField, OptimizedMul};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::ProofTranscript;
use rayon::prelude::*;

/// Represents a single layer of a single grand product circuit.
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   R0  L1   R1  L2   R2  L3   R3   <- This is layer would be represented as [L0, R0, L1, R1, L2, R2, L3, R3]
///                                           (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
pub type DenseGrandProductLayer<F> = Vec<F>;

/// Represents a single layer of a single grand product circuit using a sparse vector,
/// i.e. a vector containing (index, value) pairs.
/// Nodes with value 1 are omitted from the sparse vector.
/// Like `DenseGrandProductLayer`, a `SparseGrandProductLayer` is assumed to be
/// arranged in "interleaved" order:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   1   L1   R1   1   1   L3   1   <- This is layer would be represented as [(0, L0), (2, L1), (3, R1), (6, L3)]
pub type SparseGrandProductLayer<F> = Vec<(usize, F)>;

/// A "dynamic density" grand product layer can switch from sparse representation
/// to dense representation once it's no longer sparse (after binding).
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicDensityGrandProductLayer<F: JoltField> {
    Sparse(SparseGrandProductLayer<F>),
    Dense(DenseGrandProductLayer<F>),
}

impl<F: JoltField> DynamicDensityGrandProductLayer<F> {
    fn to_dense(&self, size: usize) -> DenseGrandProductLayer<F> {
        match self {
            DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                let mut dense_layer = vec![F::one(); size];
                for (index, value) in sparse_layer {
                    dense_layer[*index] = *value;
                }
                dense_layer
            }
            DynamicDensityGrandProductLayer::Dense(dense_layer) => dense_layer[..size].to_vec(),
        }
    }
}

/// This constant determines:
///     - whether the `layer_output` of a `DynamicDensityGrandProductLayer` is dense
///       or sparse
///     - when to switch from sparse to dense representation during the binding of a
///       `DynamicDensityGrandProductLayer`
/// If the layer has >DENSIFICATION_THRESHOLD fraction of non-1 values, it'll switch
/// to the dense representation. Value tuned experimentally.
const DENSIFICATION_THRESHOLD: f64 = 0.8;

impl<F: JoltField> DynamicDensityGrandProductLayer<F> {
    /// Computes the grand product layer that is output by this layer.
    ///     L0'      R0'      L1'      R1'     <- output layer
    ///      Λ        Λ        Λ        Λ
    ///     / \      / \      / \      / \
    ///   L0   R0  L1   R1  L2   R2  L3   R3   <- this layer
    ///
    /// If the current layer is dense, the output layer will be dense.
    /// If the current layer is sparse, but already not very sparse (as parametrized by
    /// `DENSIFICATION_THRESHOLD`), the output layer will be dense.
    /// Otherwise, the output layer will be sparse.
    pub fn layer_output(&self, output_len: usize) -> Self {
        match self {
            DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                #[cfg(test)]
                let product: F = sparse_layer.iter().map(|(_, value)| value).product();

                if (sparse_layer.len() as f64 / (output_len * 2) as f64) > DENSIFICATION_THRESHOLD {
                    // Current layer is already not very sparse, so make the next layer dense
                    let mut output_layer: DenseGrandProductLayer<F> = vec![F::one(); output_len];
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::one()));
                            if right.0 == index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_layer[index / 2] = right.1 * *value;
                            } else {
                                // Corresponding right node not found, so it must be 1
                                output_layer[index / 2] = *value;
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration, so it must have value 1
                            output_layer[index / 2] = *value;
                            next_index_to_process = index + 1;
                        }
                    }
                    #[cfg(test)]
                    {
                        let output_product: F = output_layer.iter().product();
                        assert_eq!(product, output_product);
                    }
                    DynamicDensityGrandProductLayer::Dense(output_layer)
                } else {
                    // Current layer is still pretty sparse, so make the next layer sparse
                    let mut output_layer: SparseGrandProductLayer<F> =
                        Vec::with_capacity(output_len);
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::one()));
                            if right.0 == index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_layer.push((index / 2, right.1 * *value));
                            } else {
                                // Corresponding right node not found, so it must be 1
                                output_layer.push((index / 2, *value));
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration, so it must have value 1
                            output_layer.push((index / 2, *value));
                            next_index_to_process = index + 1;
                        }
                    }
                    #[cfg(test)]
                    {
                        let output_product: F =
                            output_layer.iter().map(|(_, value)| value).product();
                        assert_eq!(product, output_product);
                    }
                    DynamicDensityGrandProductLayer::Sparse(output_layer)
                }
            }
            DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                #[cfg(test)]
                let product: F = dense_layer.iter().product();

                // If current layer is dense, next layer should also be dense.
                let output_layer: DenseGrandProductLayer<F> = (0..output_len)
                    .map(|i| {
                        let (left, right) = (dense_layer[2 * i], dense_layer[2 * i + 1]);
                        left * right
                    })
                    .collect();
                #[cfg(test)]
                {
                    let output_product: F = output_layer.iter().product();
                    assert_eq!(product, output_product);
                }
                DynamicDensityGrandProductLayer::Dense(output_layer)
            }
        }
    }
}

/// Represents a batch of `DynamicDensityGrandProductLayer`, all of which have the same
/// size `layer_len`. Note that within a single batch, some layers may be represented by
/// sparse vectors and others by dense vectors.
#[derive(Debug, Clone)]
pub struct BatchedSparseGrandProductLayer<F: JoltField> {
    pub layer_len: usize,
    pub batched_layer_len: usize,
    pub values: Vec<DynamicDensityGrandProductLayer<F>>,
    /// At some point in sumcheck, the polynomials represented by `values` will be bound
    /// to the extent that each element in the `Vec` will only contain a pair of values
    /// (a "left" node and a "right" node). For subsequent sumcheck iterations, we transition
    /// to using this "coalesced" layer, which is simply flattens `values` into a single
    /// dense vector.
    coalesced_layer: Option<DenseGrandProductLayer<F>>,
}

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedSparseGrandProductLayer<F> {}
impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedSparseGrandProductLayer<F> {
    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &DensePolynomial<F>, round_claim: F) {
        if let Some(coalesced) = &self.coalesced_layer {
            let left = coalesced.iter().step_by(2);
            let right = coalesced.iter().skip(1).step_by(2);
            let eq = eq_poly.evals_ref().iter();
            let expected: F = left
                .zip(right)
                .zip(eq)
                .map(|((l, r), eq)| *eq * l * r)
                .sum();
            assert_eq!(expected, round_claim);
        } else {
            let dense: Vec<_> = self
                .values
                .iter()
                .flat_map(|layer| layer.to_dense(self.layer_len))
                .collect();
            let left = dense.iter().step_by(2);
            let right = dense.iter().skip(1).step_by(2);

            let eq = eq_poly.evals_ref().iter();
            let expected: F = left
                .zip(right)
                .zip(eq)
                .map(|((l, r), eq)| *eq * l * r)
                .sum();
            assert_eq!(expected, round_claim);
        }
    }
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
    #[tracing::instrument(skip_all, name = "BatchedSparseGrandProductLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        if let Some(coalesced_layer) = &mut self.coalesced_layer {
            let mut bound_layer = vec![F::zero(); coalesced_layer.len() / 2];
            for i in 0..bound_layer.len() / 2 {
                // left
                bound_layer[2 * i] = coalesced_layer[4 * i]
                    + *r * (coalesced_layer[4 * i + 2] - coalesced_layer[4 * i]);
                // right
                bound_layer[2 * i + 1] = coalesced_layer[4 * i + 1]
                    + *r * (coalesced_layer[4 * i + 3] - coalesced_layer[4 * i + 1]);
            }
            self.coalesced_layer = Some(bound_layer);
            eq_poly.bound_poly_var_bot(r);
            self.batched_layer_len /= 2;
            return;
        } else if self.layer_len == 2 {
            let mut coalesced = vec![F::zero(); self.values.len().next_power_of_two()];
            for (bound, unbound) in coalesced.chunks_mut(2).zip(self.values.chunks(2)) {
                let children = [
                    unbound
                        .get(0)
                        .unwrap_or(&DynamicDensityGrandProductLayer::Dense(vec![F::zero(); 2]))
                        .to_dense(2),
                    unbound
                        .get(1)
                        .unwrap_or(&DynamicDensityGrandProductLayer::Dense(vec![F::zero(); 2]))
                        .to_dense(2),
                ];
                // left
                bound[0] = children[0][0] + *r * (children[1][0] - children[0][0]);
                // right
                bound[1] = children[0][1] + *r * (children[1][1] - children[0][1]);
            }

            self.coalesced_layer = Some(coalesced);
            eq_poly.bound_poly_var_bot(r);
            self.batched_layer_len /= 2;
            return;
        } else {
            debug_assert!(self.layer_len % 4 == 0);
        }

        rayon::join(
            || {
                self.values.par_iter_mut().for_each(|layer| match layer {
                    DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                        let mut dense_bound_layer = if (sparse_layer.len() as f64
                            / self.layer_len as f64)
                            > DENSIFICATION_THRESHOLD
                        {
                            // Current layer is already not very sparse, so make the next layer dense
                            Some(vec![F::one(); self.layer_len / 2])
                        } else {
                            None
                        };

                        let mut num_bound = 0usize;
                        let mut push_to_bound_layer =
                            |sparse_layer: &mut Vec<(usize, F)>, dense_index: usize, value: F| {
                                match &mut dense_bound_layer {
                                    Some(ref mut dense_vec) => {
                                        debug_assert_eq!(dense_vec[dense_index], F::one());
                                        dense_vec[dense_index] = value;
                                    }
                                    None => {
                                        sparse_layer[num_bound] = (dense_index, value);
                                    }
                                };
                                num_bound += 1;
                            };

                        let mut next_left_node_to_process = 0usize;
                        let mut next_right_node_to_process = 0usize;

                        for j in 0..sparse_layer.len() {
                            let (index, value) = sparse_layer[j];
                            if index % 2 == 0 && index < next_left_node_to_process {
                                // This left node was already bound with its sibling in a previous iteration
                                continue;
                            }
                            if index % 2 == 1 && index < next_right_node_to_process {
                                // This right node was already bound with its sibling in a previous iteration
                                continue;
                            }

                            let neighbors = [
                                sparse_layer
                                    .get(j + 1)
                                    .cloned()
                                    .unwrap_or((index + 1, F::one())),
                                sparse_layer
                                    .get(j + 2)
                                    .cloned()
                                    .unwrap_or((index + 2, F::one())),
                            ];
                            let find_neighbor = |query_index: usize| {
                                neighbors
                                    .iter()
                                    .find_map(|(neighbor_index, neighbor_value)| {
                                        if *neighbor_index == query_index {
                                            Some(neighbor_value)
                                        } else {
                                            None
                                        }
                                    })
                                    .cloned()
                                    .unwrap_or(F::one())
                            };

                            match index % 4 {
                                0 => {
                                    // Find sibling left node
                                    let sibling_value: F = find_neighbor(index + 2);
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2,
                                        value + *r * (sibling_value - value),
                                    );
                                    next_left_node_to_process = index + 4;
                                }
                                1 => {
                                    // Edge case: If this right node's neighbor is not 1 and has _not_
                                    // been bound yet, we need to bind the neighbor first to preserve
                                    // the monotonic ordering of the bound layer.
                                    if next_left_node_to_process <= index + 1 {
                                        let left_neighbor: F = find_neighbor(index + 1);
                                        if !left_neighbor.is_one() {
                                            push_to_bound_layer(
                                                sparse_layer,
                                                index / 2,
                                                F::one() + *r * (left_neighbor - F::one()),
                                            );
                                        }
                                        next_left_node_to_process = index + 3;
                                    }

                                    // Find sibling right node
                                    let sibling_value: F = find_neighbor(index + 2);
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2 + 1,
                                        value + *r * (sibling_value - value),
                                    );
                                    next_right_node_to_process = index + 4;
                                }
                                2 => {
                                    // Sibling left node wasn't encountered in previous iteration,
                                    // so sibling must have value 1.
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2 - 1,
                                        F::one() + *r * (value - F::one()),
                                    );
                                    next_left_node_to_process = index + 2;
                                }
                                3 => {
                                    // Sibling right node wasn't encountered in previous iteration,
                                    // so sibling must have value 1.
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2,
                                        F::one() + *r * (value - F::one()),
                                    );
                                    next_right_node_to_process = index + 2;
                                }
                                _ => unreachable!("?_?"),
                            }
                        }
                        if let Some(dense_vec) = dense_bound_layer {
                            *layer = DynamicDensityGrandProductLayer::Dense(dense_vec);
                        } else {
                            sparse_layer.truncate(num_bound);
                        }
                    }
                    DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                        // If current layer is dense, next layer should also be dense.
                        let n = self.layer_len / 4;
                        for i in 0..n {
                            // left
                            dense_layer[2 * i] = dense_layer[4 * i]
                                + *r * (dense_layer[4 * i + 2] - dense_layer[4 * i]);
                            // right
                            dense_layer[2 * i + 1] = dense_layer[4 * i + 1]
                                + *r * (dense_layer[4 * i + 3] - dense_layer[4 * i + 1]);
                        }
                    }
                })
            },
            || eq_poly.bound_poly_var_bot(r),
        );
        self.layer_len /= 2;
        self.batched_layer_len /= 2;
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
    fn compute_cubic(&self, eq_poly: &DensePolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        debug_assert_eq!(self.batched_layer_len, 2 * eq_poly.len());

        let eq_evals: Vec<(F, F, F)> = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eval_point_0 = eq_poly[2 * i];
                let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
                let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        if let Some(coalesced_layer) = &self.coalesced_layer {
            // Computes:
            //     Σ eq_evals[i] * left[i] * right[i]
            // for the evaluation points {0, 2, 3}
            let evals = eq_evals
                .iter()
                .zip(coalesced_layer.chunks_exact(4))
                .map(|(eq_evals, chunk)| {
                    let left = (chunk[0], chunk[2]);
                    let right = (chunk[1], chunk[3]);

                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right.1 + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    (
                        eq_evals.0 * left.0 * right.0,
                        eq_evals.1 * left_eval_2 * right_eval_2,
                        eq_evals.2 * left_eval_3 * right_eval_3,
                    )
                })
                .fold(
                    (F::zero(), F::zero(), F::zero()),
                    |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                );

            let cubic_evals = [evals.0, previous_round_claim - evals.0, evals.1, evals.2];
            return UniPoly::from_evals(&cubic_evals);
        } else if self.layer_len == 2 {
            let evals = eq_evals
                .iter()
                .zip(self.values.chunks(2))
                .map(|(eq, vals)| {
                    let lo = vals
                        .get(0)
                        .unwrap_or(&DynamicDensityGrandProductLayer::Dense(vec![F::zero(); 2]))
                        .to_dense(2);
                    let hi = vals
                        .get(1)
                        .unwrap_or(&DynamicDensityGrandProductLayer::Dense(vec![F::zero(); 2]))
                        .to_dense(2);

                    let left = [lo[0], hi[0]];
                    let right = [lo[1], hi[1]];

                    let m_left = left[1] - left[0];
                    let m_right = right[1] - right[0];

                    let left_eval_2 = left[1] + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right[1] + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    let temp = (
                        eq.0 * left[0] * right[0],
                        eq.1 * left_eval_2 * right_eval_2,
                        eq.2 * left_eval_3 * right_eval_3,
                    );
                    temp
                })
                .fold(
                    (F::zero(), F::zero(), F::zero()),
                    |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                );
            let cubic_evals = [evals.0, previous_round_claim - evals.0, evals.1, evals.2];
            return UniPoly::from_evals(&cubic_evals);
        }

        let evals: Vec<(F, F, F)> = self
            .values
            .par_iter()
            .enumerate()
            .map(|(layer_index, layer)| {
                let eq_chunk_size = self.layer_len / 4;
                let eq_chunk =
                    &eq_evals[layer_index * eq_chunk_size..(layer_index + 1) * eq_chunk_size];
                match layer {
                    // If sparse, we use the pre-emptively computed `eq_eval_sums` as a starting point:
                    //     eq_eval_sum := Σ eq_evals[i]
                    // What we ultimately want to compute:
                    //     Σ eq_evals[i] * left[i] * right[i])
                    // Note that if left[i] and right[i] are all 1s, the inner sum is:
                    //     Σ eq_evals[i] = eq_eval_sum
                    // To recover the actual inner sum, we find all the non-1
                    // left[i] and right[i] terms and compute the delta:
                    //     ∆ := Σ eq_evals[j] * (left[j] * right[j] - 1)    ∀j where left[j] ≠ 1 or right[j] ≠ 1
                    // Then we can compute:
                    //    eq_eval_sum + ∆ = Σ eq_evals[i] + Σ eq_evals[j] * (left[j] * right[j] - 1)
                    //                    = Σ eq_evals[j] * left[j] * right[j]
                    // ...which is exactly the summand we want.
                    DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                        let mut evals = eq_chunk
                            .iter()
                            .fold((F::zero(), F::zero(), F::zero()), |acc, evals| {
                                (acc.0 + evals.0, acc.1 + evals.1, acc.2 + evals.2)
                            });

                        let mut next_index_to_process = 0usize;
                        for (j, (index, value)) in sparse_layer.iter().enumerate() {
                            if *index < next_index_to_process {
                                // This node was already processed in a previous iteration
                                continue;
                            }
                            let neighbors = [
                                sparse_layer
                                    .get(j + 1)
                                    .cloned()
                                    .unwrap_or((index + 1, F::one())),
                                sparse_layer
                                    .get(j + 2)
                                    .cloned()
                                    .unwrap_or((index + 2, F::one())),
                                sparse_layer
                                    .get(j + 3)
                                    .cloned()
                                    .unwrap_or((index + 3, F::one())),
                            ];

                            let find_neighbor = |query_index: usize| {
                                neighbors
                                    .iter()
                                    .find_map(|(neighbor_index, neighbor_value)| {
                                        if *neighbor_index == query_index {
                                            Some(neighbor_value)
                                        } else {
                                            None
                                        }
                                    })
                                    .cloned()
                                    .unwrap_or(F::one())
                            };

                            // Recall that in the dense case, we process four values at a time:
                            //                  layer = [L, R, L, R, L, R, ...]
                            //                           |  |  |  |
                            //    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
                            //     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
                            //
                            // In the sparse case, we do something similar, but some of the four
                            // values may be omitted from the sparse vector.
                            // We match on `index % 4` to determine which of the four values are
                            // present in the sparse vector, and infer the rest are 1.
                            let (left, right) = match index % 4 {
                                0 => {
                                    let left = (*value, find_neighbor(index + 2));
                                    let right =
                                        (find_neighbor(index + 1), find_neighbor(index + 3));
                                    next_index_to_process = index + 4;
                                    (left, right)
                                }
                                1 => {
                                    let left = (F::one(), find_neighbor(index + 1));
                                    let right = (*value, find_neighbor(index + 2));
                                    next_index_to_process = index + 3;
                                    (left, right)
                                }
                                2 => {
                                    let left = (F::one(), *value);
                                    let right = (F::one(), find_neighbor(index + 1));
                                    next_index_to_process = index + 2;
                                    (left, right)
                                }
                                3 => {
                                    let left = (F::one(), F::one());
                                    let right = (F::one(), *value);
                                    next_index_to_process = index + 1;
                                    (left, right)
                                }
                                _ => unreachable!("?_?"),
                            };

                            let m_left = left.1 - left.0;
                            let m_right = right.1 - right.0;

                            let left_eval_2 = left.1 + m_left;
                            let left_eval_3 = left_eval_2 + m_left;

                            let right_eval_2 = right.1 + m_right;
                            let right_eval_3 = right_eval_2 + m_right;

                            let (eq_eval_0, eq_eval_2, eq_eval_3) = eq_chunk[index / 4];
                            evals.0 += eq_eval_0
                                .mul_0_optimized(left.0.mul_1_optimized(right.0) - F::one());
                            evals.1 += eq_eval_2.mul_0_optimized(
                                left_eval_2.mul_1_optimized(right_eval_2) - F::one(),
                            );
                            evals.2 += eq_eval_3.mul_0_optimized(
                                left_eval_3.mul_1_optimized(right_eval_3) - F::one(),
                            );
                        }

                        evals
                    }
                    // If dense, we just compute
                    //     Σ eq_evals[i] * left[i] * right[i]
                    // directly in `self.compute_cubic`, without using `eq_eval_sums`.
                    DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                        // Computes:
                        //     Σ eq_evals[i] * left[i] * right[i]
                        // for the evaluation points {0, 2, 3}
                        eq_chunk
                            .iter()
                            .zip(dense_layer.chunks_exact(4))
                            .map(|(eq_evals, chunk)| {
                                let left = (chunk[0], chunk[2]);
                                let right = (chunk[1], chunk[3]);

                                let m_left = left.1 - left.0;
                                let m_right = right.1 - right.0;

                                let left_eval_2 = left.1 + m_left;
                                let left_eval_3 = left_eval_2 + m_left;

                                let right_eval_2 = right.1 + m_right;
                                let right_eval_3 = right_eval_2 + m_right;

                                (
                                    eq_evals.0 * left.0 * right.0,
                                    eq_evals.1 * left_eval_2 * right_eval_2,
                                    eq_evals.2 * left_eval_3 * right_eval_3,
                                )
                            })
                            .fold(
                                (F::zero(), F::zero(), F::zero()),
                                |(sum_0, sum_2, sum_3), (a, b, c)| {
                                    (sum_0 + a, sum_2 + b, sum_3 + c)
                                },
                            )
                    }
                }
            })
            .collect();

        let evals_combined_0 = evals.iter().map(|eval| eval.0).sum();
        let evals_combined_2 = evals.iter().map(|eval| eval.1).sum();
        let evals_combined_3 = evals.iter().map(|eval| eval.2).sum();

        let cubic_evals = [
            evals_combined_0,
            previous_round_claim - evals_combined_0,
            evals_combined_2,
            evals_combined_3,
        ];
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (F, F) {
        assert_eq!(self.layer_len, 2);
        if let Some(coalesced) = &self.coalesced_layer {
            assert_eq!(coalesced.len(), 2);
            (coalesced[0], coalesced[1])
        } else {
            panic!("Layer should be coalesced by now");
        }
    }
}

/// A special bottom layer of a grand product, where boolean flags are used to
/// toggle the other inputs (fingerprints) going into the rest of the tree.
/// Note that the gates for this layer are *not* simple multiplication gates.
/// ```ignore
///
///      …            …
///    /    \       /    \     the rest of the tree, which is now sparse (lots of 1s)
///   o      o     o      o                          ↑
///  / \    / \   / \    / \    –––––––––––––––––––––––––––––––––––––––––––
/// 🏴  o  🏳️ o  🏳️ o  🏴  o    toggle layer        ↓
#[derive(Debug)]
struct BatchedGrandProductToggleLayer<F: JoltField> {
    /// The list of non-zero flag indices for each layer in the batch.
    flag_indices: Vec<Vec<usize>>,
    /// The list of non-zero flag values for each layer in the batch.
    /// Before the first binding iteration of sumcheck, this will be empty
    /// (we know that all non-zero, unbound flag values are 1).
    flag_values: Vec<Vec<F>>,
    fingerprints: Vec<Vec<F>>,

    coalesced_flags: Option<Vec<F>>,
    coalesced_fingerprints: Option<Vec<F>>,

    layer_len: usize,
    batched_layer_len: usize,
}

impl<F: JoltField> BatchedGrandProductToggleLayer<F> {
    #[cfg(test)]
    fn to_dense(&self) -> (DensePolynomial<F>, DensePolynomial<F>) {
        if let Some(coalesced_flags) = &self.coalesced_flags {
            let coalesced_fingerprints = self.coalesced_fingerprints.as_ref().unwrap();
            (
                DensePolynomial::new_padded(coalesced_flags.clone()),
                DensePolynomial::new_padded(coalesced_fingerprints.clone()),
            )
        } else if self.flag_values.is_empty() {
            let fingerprints: Vec<_> = self.fingerprints.concat();
            let mut flags = vec![F::zero(); fingerprints.len() / 2];
            for (batch_index, flag_indices) in self.flag_indices.iter().enumerate() {
                for flag_index in flag_indices {
                    flags[batch_index * (self.layer_len / 2) + flag_index] = F::one();
                }
            }

            (
                DensePolynomial::new_padded(flags),
                DensePolynomial::new_padded(fingerprints),
            )
        } else {
            let fingerprints: Vec<_> = self
                .fingerprints
                .iter()
                .flat_map(|f| f[..self.layer_len / 2].iter())
                .cloned()
                .collect();
            let mut flags = vec![F::zero(); fingerprints.len() / 2];
            for (batch_index, (flag_indices, flag_values)) in self
                .flag_indices
                .iter()
                .zip(self.flag_values.iter())
                .enumerate()
            {
                for (flag_index, flag_value) in flag_indices.iter().zip(flag_values) {
                    flags[batch_index * (self.layer_len / 2) + flag_index] = *flag_value;
                }
            }

            (
                DensePolynomial::new_padded(flags),
                DensePolynomial::new_padded(fingerprints),
            )
        }
    }
}

impl<F: JoltField> BatchedGrandProductToggleLayer<F> {
    fn new(flag_indices: Vec<Vec<usize>>, fingerprints: Vec<Vec<F>>) -> Self {
        let layer_len = 2 * fingerprints[0].len();
        let batched_layer_len = fingerprints.len().next_power_of_two() * layer_len;
        Self {
            flag_indices,
            // While flags remain unbound, all values are boolean, so we can assume any flag that appears in `flag_indices` has value 1.
            flag_values: vec![],
            fingerprints,
            layer_len,
            batched_layer_len,
            coalesced_flags: None,
            coalesced_fingerprints: None,
        }
    }

    fn layer_output(&self) -> BatchedSparseGrandProductLayer<F> {
        let output_layers = self
            .fingerprints
            .par_iter()
            .enumerate()
            .map(|(batch_index, fingerprints)| {
                let flag_indices = &self.flag_indices[batch_index / 2];
                let mut sparse_layer = vec![];
                for i in flag_indices {
                    sparse_layer.push((*i, fingerprints[*i]));
                }
                DynamicDensityGrandProductLayer::Sparse(sparse_layer)
            })
            .collect();
        BatchedSparseGrandProductLayer {
            layer_len: self.layer_len / 2,
            batched_layer_len: self.batched_layer_len / 2,
            values: output_layers,
            coalesced_layer: None,
        }
    }
}

impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedGrandProductToggleLayer<F> {
    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &DensePolynomial<F>, round_claim: F) {
        let (flags, fingerprints) = self.to_dense();
        let expected: F = flags
            .evals_ref()
            .iter()
            .zip(fingerprints.evals_ref().iter())
            .zip(eq_poly.evals_ref().iter())
            .map(|((flag, fingerprint), eq)| *eq * (*flag * fingerprint + F::one() - flag))
            .sum();
        // assert_eq!(expected, round_claim);
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// Similar to `BatchedSparseGrandProductLayer::bind`, in that fingerprints use
    /// a sparse representation, but different in a couple of key ways:
    /// - flags use two separate vectors (for indices and values) rather than
    ///   a single vector of (index, value) pairs
    /// - The left and right nodes in this layer are flags and fingerprints, respectively.
    ///   They are represented by *separate* vectors, so they are *not* interleaved. This
    ///   means we process 2 flag values at a time, rather than 4.
    /// - In `BatchedSparseGrandProductLayer`, the absence of a node implies that it has
    ///   value 1. For our sparse representation of flags, the absence of a node implies
    ///   that it has value 0. In other words, a flag with value 1 will be present in both
    ///   `self.flag_indices` and `self.flag_values`.
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        #[cfg(test)]
        let (mut flags_before_binding, mut fingerprints_before_binding) = self.to_dense();

        if let Some(coalesced_flags) = &mut self.coalesced_flags {
            if coalesced_flags.len() > 1 {
                let mut bound_flags = vec![F::zero(); coalesced_flags.len() / 2];
                for i in 0..bound_flags.len() {
                    bound_flags[i] = coalesced_flags[2 * i]
                        + *r * (coalesced_flags[2 * i + 1] - coalesced_flags[2 * i]);
                }
                self.coalesced_flags = Some(bound_flags);
            }

            let coalesced_fingerpints = self.coalesced_fingerprints.as_mut().unwrap();
            let mut bound_fingerprints = vec![F::zero(); coalesced_fingerpints.len() / 2];
            for i in 0..bound_fingerprints.len() {
                bound_fingerprints[i] = coalesced_fingerpints[2 * i]
                    + *r * (coalesced_fingerpints[2 * i + 1] - coalesced_fingerpints[2 * i]);
            }
            self.coalesced_fingerprints = Some(bound_fingerprints);

            eq_poly.bound_poly_var_bot(r);
            self.batched_layer_len /= 2;

            return;
        }
        debug_assert!(self.layer_len % 4 == 0);

        self.fingerprints
            .par_iter_mut()
            .for_each(|layer: &mut Vec<F>| {
                let n = self.layer_len / 4;
                for i in 0..n {
                    // TODO(moodlezoup): Try mul_0_optimized here
                    layer[i] = layer[2 * i] + *r * (layer[2 * i + 1] - layer[2 * i]);
                }
            });

        rayon::join(
            || {
                let is_first_bind = self.flag_values.is_empty();
                if is_first_bind {
                    self.flag_values = vec![vec![]; self.flag_indices.len()];
                }

                self.flag_indices
                    .par_iter_mut()
                    .zip(self.flag_values.par_iter_mut())
                    .for_each(|(flag_indices, flag_values)| {
                        let mut next_index_to_process = 0usize;

                        let mut bound_index = 0usize;
                        for j in 0..flag_indices.len() {
                            let index = flag_indices[j];
                            if index < next_index_to_process {
                                // This flag was already bound with its sibling in the previous iteration.
                                continue;
                            }

                            // Bind indices in place
                            flag_indices[bound_index] = index / 2;

                            if index % 2 == 0 {
                                let neighbor = flag_indices.get(j + 1).cloned().unwrap_or(0);
                                if neighbor == index + 1 {
                                    // Neighbor is flag's sibling

                                    if is_first_bind {
                                        // For first bind, all non-zero flag values are 1.
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        //                = 1 - r * (1 - 1)
                                        //                = 1
                                        flag_values.push(F::one());
                                    } else {
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        flag_values[bound_index] = flag_values[j]
                                            + *r * (flag_values[j + 1] - flag_values[j]);
                                    };
                                } else {
                                    // This flag's sibling wasn't found, so it must have value 0.

                                    if is_first_bind {
                                        // For first bind, all non-zero flag values are 1.
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        //                = flags[2 * i] - r * flags[2 * i]
                                        //                = 1 - r
                                        flag_values.push(F::one() - *r);
                                    } else {
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        //                = flags[2 * i] - r * flags[2 * i]
                                        flag_values[bound_index] =
                                            flag_values[j] - *r * flag_values[j];
                                    };
                                }
                                next_index_to_process = index + 2;
                            } else {
                                // This flag's sibling wasn't encountered in a previous iteration,
                                // so it must have had value 0.

                                if is_first_bind {
                                    // For first bind, all non-zero flag values are 1.
                                    // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                    //                = r * flags[2 * i + 1]
                                    //                = r
                                    flag_values.push(*r);
                                } else {
                                    // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                    //                = r * flags[2 * i + 1]
                                    flag_values[bound_index] = *r * flag_values[j];
                                };
                                next_index_to_process = index + 1;
                            }

                            bound_index += 1;
                        }

                        flag_indices.truncate(bound_index);
                        // We only ever use `flag_indices.len()`, so no need to truncate `flag_values`
                        // flag_values.truncate(bound_index);
                    });
            },
            || eq_poly.bound_poly_var_bot(r),
        );
        self.layer_len /= 2;
        self.batched_layer_len /= 2;

        #[cfg(test)]
        {
            let (bound_flags, bound_fingerprints) = self.to_dense();
            flags_before_binding.bound_poly_var_bot(r);
            fingerprints_before_binding.bound_poly_var_bot(r);
            assert_eq!(
                bound_flags.Z[..bound_flags.len()],
                flags_before_binding.Z[..flags_before_binding.len()]
            );
            assert_eq!(
                bound_fingerprints.Z[..bound_fingerprints.len()],
                fingerprints_before_binding.Z[..fingerprints_before_binding.len()]
            );
        }

        if self.layer_len == 2 {
            assert!(self.coalesced_fingerprints.is_none());
            assert!(self.coalesced_flags.is_none());
            let mut coalesced_fingerprints: Vec<F> =
                self.fingerprints.iter().map(|f| f[0]).collect::<Vec<_>>();
            coalesced_fingerprints
                .resize(coalesced_fingerprints.len().next_power_of_two(), F::zero());

            let mut coalesced_flags: Vec<_> = self
                .flag_indices
                .iter()
                .zip(self.flag_values.iter())
                .flat_map(|(indices, values)| {
                    let mut coalesced = [F::zero()];
                    for (index, value) in indices.iter().zip(values.iter()) {
                        coalesced[*index] = *value;
                    }
                    coalesced
                })
                .collect();
            coalesced_flags.resize(coalesced_flags.len().next_power_of_two(), F::zero());

            self.coalesced_fingerprints = Some(coalesced_fingerprints);
            self.coalesced_flags = Some(coalesced_flags);

            #[cfg(test)]
            {
                let (bound_flags, bound_fingerprints) = self.to_dense();
                assert_eq!(
                    bound_flags.Z[..bound_flags.len()],
                    flags_before_binding.Z[..flags_before_binding.len()]
                );
                assert_eq!(
                    bound_fingerprints.Z[..bound_fingerprints.len()],
                    fingerprints_before_binding.Z[..fingerprints_before_binding.len()]
                );
            }
        }
    }

    /// Similar to `BatchedSparseGrandProductLayer::compute_cubic`, but with changes to
    /// accomodate the differences between `BatchedSparseGrandProductLayer` and
    /// `BatchedGrandProductToggleLayer`. These differences are described in the doc comments
    /// for `BatchedGrandProductToggleLayer::bind`.
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::compute_cubic")]
    fn compute_cubic(&self, eq_poly: &DensePolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        let eq_evals: Vec<(F, F, F)> = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eval_point_0 = eq_poly[2 * i];
                let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
                let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        if let Some(coalesced_flags) = &self.coalesced_flags {
            let coalesced_fingerpints = self.coalesced_fingerprints.as_ref().unwrap();
            if coalesced_flags.len() == 1 {
                assert_eq!(eq_evals.len(), 1);
                let eq_eval = eq_evals[0];
                assert_eq!(coalesced_fingerpints.len(), 2);
                let m_fingerprint = coalesced_fingerpints[1] - coalesced_fingerpints[0];
                let fingerprint_eval_2 = coalesced_fingerpints[1] + m_fingerprint;
                let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                let evals = (
                    eq_eval.0
                        * (coalesced_flags[0] * coalesced_fingerpints[0] + F::one()
                            - coalesced_flags[0]),
                    eq_eval.1
                        * (coalesced_flags[0] * fingerprint_eval_2 + F::one() - coalesced_flags[0]),
                    eq_eval.2
                        * (coalesced_flags[0] * fingerprint_eval_3 + F::one() - coalesced_flags[0]),
                );
                let cubic_evals = [evals.0, previous_round_claim - evals.0, evals.1, evals.2];
                return UniPoly::from_evals(&cubic_evals);
            }

            let evals = eq_evals
                .iter()
                .zip(coalesced_flags.chunks(2))
                .zip(coalesced_fingerpints.chunks(4))
                .map(|((eq, flags), fingerprints)| {
                    let read_fingerprints = &fingerprints[..2];
                    let write_fingerprints = &fingerprints[2..4];

                    let m_flag = *flags.get(1).unwrap_or(&F::zero()) - flags[0];
                    let m_read_fingerprint = read_fingerprints[1] - read_fingerprints[0];
                    let m_write_fingerprint = write_fingerprints[1] - write_fingerprints[0];

                    let flag_eval_2 = *flags.get(1).unwrap_or(&F::zero()) + m_flag;
                    let flag_eval_3 = flag_eval_2 + m_flag;

                    let read_fingerprint_eval_2 = read_fingerprints[1] + m_read_fingerprint;
                    let read_fingerprint_eval_3 = read_fingerprint_eval_2 + m_read_fingerprint;

                    let write_fingerprint_eval_2 = write_fingerprints[1] + m_write_fingerprint;
                    let write_fingerprint_eval_3 = write_fingerprint_eval_2 + m_write_fingerprint;

                    (
                        eq.0 * (flags[0] * read_fingerprints[0] + F::one() - flags[0])
                            + eq.0 * (flags[0] * write_fingerprints[0] + F::one() - flags[0]),
                        eq.1 * (flag_eval_2 * read_fingerprint_eval_2 + F::one() - flag_eval_2)
                            + eq.1
                                * (flag_eval_2 * write_fingerprint_eval_2 + F::one() - flag_eval_2),
                        eq.2 * (flag_eval_3 * read_fingerprint_eval_3 + F::one() - flag_eval_3)
                            + eq.2
                                * (flag_eval_3 * write_fingerprint_eval_3 + F::one() - flag_eval_3),
                    )
                })
                .fold(
                    (F::zero(), F::zero(), F::zero()),
                    |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                );

            let cubic_evals = [evals.0, previous_round_claim - evals.0, evals.1, evals.2];
            return UniPoly::from_evals(&cubic_evals);
        }
        debug_assert!(self.layer_len % 4 == 0);

        let eq_chunk_size = self.layer_len / 4;
        assert!(eq_chunk_size != 0);

        // This is what the cubic evals would be if a layer's flags were *all 0*
        // We pre-emptively compute these sums as a starting point:
        //     eq_eval_sum := Σ eq_evals[i]
        // What we ultimately want to compute:
        //     Σ eq_evals[i] * (flag[i] * fingerprint[i] + 1 - flag[i])
        // Note that if flag[i] is all 1s, the inner sum is:
        //     Σ eq_evals[i] = eq_eval_sum
        // To recover the actual inner sum, we find all the non-zero flag[i] terms
        // computes the delta:
        //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j]))    ∀j where flag[j] ≠ 0
        // Then we can compute:
        //    eq_eval_sum + ∆ = Σ eq_evals[i] + Σ eq_evals[i] * (flag[i] * fingerprint[i] - flag[i]))
        //                    = Σ eq_evals[j] * (flag[i] * fingerprint[i] + 1 - flag[i])
        // ...which is exactly the summand we want.
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

        let deltas: Vec<(F, F, F)> = (0..self.fingerprints.len())
            .into_par_iter()
            .map(|batch_index| {
                // Computes:
                //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j])    ∀j where flag[j] ≠ 0
                // for the evaluation points {0, 2, 3}

                let fingerprints = &self.fingerprints[batch_index];
                let flag_indices = &self.flag_indices[batch_index / 2];

                let unbound = self.flag_values.is_empty();
                let mut delta = (F::zero(), F::zero(), F::zero());

                let mut next_index_to_process = 0usize;
                for (j, index) in flag_indices.iter().enumerate() {
                    if *index < next_index_to_process {
                        // This node was already processed in a previous iteration
                        continue;
                    }

                    let (flags, fingerprints) = if index % 2 == 0 {
                        let neighbor = flag_indices.get(j + 1).cloned().unwrap_or(0);
                        let flags = if neighbor == index + 1 {
                            // Neighbor is flag's sibling
                            if unbound {
                                (F::one(), F::one())
                            } else {
                                (
                                    self.flag_values[batch_index / 2][j],
                                    self.flag_values[batch_index / 2][j + 1],
                                )
                            }
                        } else {
                            // This flag's sibling wasn't found, so it must have value 0.
                            if unbound {
                                (F::one(), F::zero())
                            } else {
                                (self.flag_values[batch_index / 2][j], F::zero())
                            }
                        };
                        let fingerprints = (fingerprints[*index], fingerprints[index + 1]);

                        next_index_to_process = index + 2;
                        (flags, fingerprints)
                    } else {
                        // This flag's sibling wasn't encountered in a previous iteration,
                        // so it must have had value 0.
                        let flags = if unbound {
                            (F::zero(), F::one())
                        } else {
                            (F::zero(), self.flag_values[batch_index / 2][j])
                        };
                        let fingerprints = (fingerprints[index - 1], fingerprints[*index]);

                        next_index_to_process = index + 1;
                        (flags, fingerprints)
                    };

                    let m_flag = flags.1 - flags.0;
                    let m_fingerprint = fingerprints.1 - fingerprints.0;

                    // If flags are still unbound, flag evals will mostly be 0s and 1s
                    // Bound flags are still mostly 0s, so flag evals will mostly be 0s.
                    let flag_eval_2 = flags.1 + m_flag;
                    let flag_eval_3 = flag_eval_2 + m_flag;

                    let fingerprint_eval_2 = fingerprints.1 + m_fingerprint;
                    let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                    let (eq_eval_0, eq_eval_2, eq_eval_3) =
                        eq_evals[batch_index * eq_chunk_size + index / 2];
                    delta.0 += eq_eval_0
                        .mul_0_optimized(flags.0.mul_01_optimized(fingerprints.0) - flags.0);
                    delta.1 += eq_eval_2.mul_0_optimized(
                        flag_eval_2.mul_01_optimized(fingerprint_eval_2) - flag_eval_2,
                    );
                    delta.2 += eq_eval_3.mul_0_optimized(
                        flag_eval_3.mul_01_optimized(fingerprint_eval_3) - flag_eval_3,
                    );
                }

                // eq_eval_sum + ∆ = Σ eq_evals[i] + Σ eq_evals[i] * (flag[i] * fingerprint[i] - flag[i]))
                //                 = Σ eq_evals[j] * (flag[i] * fingerprint[i] + 1 - flag[i])
                (delta.0, delta.1, delta.2)
            })
            .collect();

        let evals_combined_0 = eq_eval_sums.0 + deltas.iter().map(|eval| eval.0).sum::<F>();
        let evals_combined_2 = eq_eval_sums.1 + deltas.iter().map(|eval| eval.1).sum::<F>();
        let evals_combined_3 = eq_eval_sums.2 + deltas.iter().map(|eval| eval.2).sum::<F>();

        let cubic_evals = [
            evals_combined_0,
            previous_round_claim - evals_combined_0,
            evals_combined_2,
            evals_combined_3,
        ];
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (F, F) {
        assert_eq!(self.layer_len, 2);
        let flags = self.coalesced_flags.as_ref().unwrap();
        let fingerprints = self.coalesced_fingerprints.as_ref().unwrap();

        (flags[0], fingerprints[0])
    }
}

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedGrandProductToggleLayer<F> {
    fn prove_layer(
        &mut self,
        claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedGrandProductLayerProof<F> {
        let mut eq_poly = DensePolynomial::new(EqPolynomial::<F>::evals(r_grand_product));

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(&claim, &mut eq_poly, transcript);

        drop_in_background_thread(eq_poly);

        let (left_claim, right_claim) = sumcheck_claims;
        transcript.append_scalar(&left_claim);
        transcript.append_scalar(&right_claim);

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claim,
            right_claim,
        }
    }
}

pub struct ToggledBatchedGrandProduct<F: JoltField> {
    toggle_layer: BatchedGrandProductToggleLayer<F>,
    sparse_layers: Vec<BatchedSparseGrandProductLayer<F>>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> BatchedGrandProduct<F, PCS>
    for ToggledBatchedGrandProduct<PCS::Field>
{
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<F>>); // (flags, fingerprints)
    type Config = ();

    #[tracing::instrument(skip_all, name = "ToggledBatchedGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (flags, fingerprints) = leaves;
        let num_layers = fingerprints[0].len().log_2();

        let toggle_layer = BatchedGrandProductToggleLayer::new(flags, fingerprints);
        let mut layers: Vec<BatchedSparseGrandProductLayer<F>> = Vec::with_capacity(num_layers);
        layers.push(toggle_layer.layer_output());

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            let new_layers = previous_layers
                .values
                .par_iter()
                .map(|previous_layer| previous_layer.layer_output(len))
                .collect();
            layers.push(BatchedSparseGrandProductLayer {
                layer_len: len,
                batched_layer_len: previous_layers.batched_layer_len / 2,
                values: new_layers,
                coalesced_layer: None,
            });
        }

        Self {
            toggle_layer,
            sparse_layers: layers,
        }
    }

    fn num_layers(&self) -> usize {
        self.sparse_layers.len() + 1
    }

    fn claimed_outputs(&self) -> Vec<F> {
        let last_layers = &self.sparse_layers.last().unwrap();
        last_layers
            .values
            .par_iter()
            .map(|layer| match layer {
                DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                    match sparse_layer.len() {
                        0 => F::one(), // Neither left nor right claim is present, so they must both be 1
                        1 => sparse_layer[0].1, // Only one claim is present
                        2 => sparse_layer[0].1 * sparse_layer[1].1, // Both left and right claim are present
                        _ => panic!("Sparse layer length > 2"),
                    }
                }
                DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                    assert_eq!(dense_layer.len(), 2);
                    dense_layer[0] * dense_layer[1]
                }
            })
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>> {
        [&mut self.toggle_layer as &mut dyn BatchedGrandProductLayer<F>]
            .into_iter()
            .chain(
                self.sparse_layers
                    .iter_mut()
                    .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F>),
            )
            .rev()
    }

    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedGrandProductLayerProof<F>],
        layer_index: usize,
        sumcheck_claim: F,
        eq_eval: F,
        grand_product_claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) {
        let layer_proof = &layer_proofs[layer_index];
        if layer_index != layer_proofs.len() - 1 {
            // Normal grand product layer (multiplication gates)
            let expected_sumcheck_claim: F =
                layer_proof.left_claim * layer_proof.right_claim * eq_eval;

            assert_eq!(expected_sumcheck_claim, sumcheck_claim);

            // produce a random challenge to condense two claims into a single claim
            let r_layer = transcript.challenge_scalar();

            *grand_product_claim = layer_proof.left_claim
                + r_layer * (layer_proof.right_claim - layer_proof.left_claim);

            r_grand_product.push(r_layer);
        } else {
            // Grand product toggle layer: layer_proof.left_claim is flag,
            // layer_proof.right_claim is fingerprint
            let expected_sumcheck_claim: F = eq_eval
                * (layer_proof.left_claim * layer_proof.right_claim + F::one()
                    - layer_proof.left_claim);

            assert_eq!(expected_sumcheck_claim, sumcheck_claim);

            // flag * fingerprint + 1 - flag
            *grand_product_claim = layer_proof.left_claim * layer_proof.right_claim + F::one()
                - layer_proof.left_claim;
        }
    }

    fn construct_with_config(leaves: Self::Leaves, _config: Self::Config) -> Self {
        <Self as BatchedGrandProduct<F, PCS>>::construct(leaves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::commitment::zeromorph::Zeromorph,
        subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedDenseGrandProductLayer},
    };
    use ark_bn254::{Bn254, Fr};
    use ark_std::{test_rng, One};
    use num_integer::Integer;
    use rand_core::RngCore;

    fn condense(sparse_layers: BatchedSparseGrandProductLayer<Fr>) -> Vec<Fr> {
        if let Some(coalesced) = sparse_layers.coalesced_layer {
            return coalesced;
        }
        sparse_layers
            .values
            .iter()
            .map(|layer| match layer {
                DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                    let mut densified =
                        DenseGrandProductLayer::from(vec![Fr::one(); sparse_layers.layer_len]);
                    for (index, value) in sparse_layer {
                        densified[*index] = *value;
                    }
                    densified
                }
                DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                    dense_layer[..sparse_layers.layer_len].to_vec()
                }
            })
            .flatten()
            .collect::<Vec<_>>()
    }

    #[test]
    fn dense_sparse_bind_parity() {
        const LAYER_SIZE: usize = 1 << 10;
        const BATCH_SIZE: usize = 5;
        let mut rng = test_rng();

        let dense_layers: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| {
                if rng.next_u32() % 4 == 0 {
                    Fr::random(&mut rng)
                } else {
                    Fr::one()
                }
            })
            .take(LAYER_SIZE)
            .collect()
        })
        .take(BATCH_SIZE)
        .collect();

        let mut batched_dense_layer = BatchedDenseGrandProductLayer::new(dense_layers.concat());

        let sparse_layers: Vec<DynamicDensityGrandProductLayer<Fr>> = dense_layers
            .iter()
            .map(|dense_layer| {
                let mut sparse_layer = vec![];
                for (i, val) in dense_layer.iter().enumerate() {
                    if !val.is_one() {
                        sparse_layer.push((i, *val));
                    }
                }
                DynamicDensityGrandProductLayer::Sparse(sparse_layer)
            })
            .collect();
        let mut batched_sparse_layer: BatchedSparseGrandProductLayer<Fr> =
            BatchedSparseGrandProductLayer {
                layer_len: LAYER_SIZE,
                batched_layer_len: (BATCH_SIZE * LAYER_SIZE).next_power_of_two(),
                values: sparse_layers,
                coalesced_layer: None,
            };

        for (dense, sparse) in batched_dense_layer
            .iter()
            .zip(condense(batched_sparse_layer.clone()).iter())
        {
            assert_eq!(dense, sparse);
        }

        for _ in 0..LAYER_SIZE.log_2() - 1 {
            let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(4)
                .collect::<Vec<_>>();
            let mut eq_poly_dense = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_eq));
            let mut eq_poly_sparse = eq_poly_dense.clone();

            let r = Fr::random(&mut rng);
            batched_dense_layer.bind(&mut eq_poly_dense, &r);
            batched_sparse_layer.bind(&mut eq_poly_sparse, &r);

            assert_eq!(eq_poly_dense, eq_poly_sparse);

            for (dense, sparse) in batched_dense_layer
                .iter()
                .zip(condense(batched_sparse_layer.clone()).iter())
            {
                assert_eq!(dense, sparse);
            }
        }
    }

    #[test]
    fn dense_sparse_compute_cubic_parity() {
        const LAYER_SIZE: usize = 1 << 10;
        const BATCH_SIZE: usize = 5;
        let mut rng = test_rng();

        let dense_layers: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            let layer: DenseGrandProductLayer<Fr> = std::iter::repeat_with(|| {
                if rng.next_u32() % 4 == 0 {
                    Fr::random(&mut rng)
                } else {
                    Fr::one()
                }
            })
            .take(LAYER_SIZE)
            .collect::<Vec<_>>();
            layer
        })
        .take(BATCH_SIZE)
        .collect();

        let batched_dense_layer = BatchedDenseGrandProductLayer::new(dense_layers.concat());

        let sparse_layers: Vec<DynamicDensityGrandProductLayer<Fr>> = dense_layers
            .iter()
            .map(|dense_layer| {
                let mut sparse_layer = vec![];
                for (i, val) in dense_layer.iter().enumerate() {
                    if !val.is_one() {
                        sparse_layer.push((i, *val));
                    }
                }
                DynamicDensityGrandProductLayer::Sparse(sparse_layer)
            })
            .collect();

        let batched_sparse_layer: BatchedSparseGrandProductLayer<Fr> =
            BatchedSparseGrandProductLayer {
                layer_len: LAYER_SIZE,
                batched_layer_len: (BATCH_SIZE * LAYER_SIZE).next_power_of_two(),
                values: sparse_layers,
                coalesced_layer: None,
            };

        for (dense, sparse) in batched_dense_layer
            .iter()
            .zip(condense(batched_sparse_layer.clone()).iter())
        {
            assert_eq!(dense, sparse);
        }

        let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take((BATCH_SIZE * LAYER_SIZE).next_power_of_two().log_2() - 1)
            .collect::<Vec<_>>();
        let eq_poly = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_eq));
        let claim = Fr::random(&mut rng);

        let dense_evals = batched_dense_layer.compute_cubic(&eq_poly, claim);
        let sparse_evals = batched_sparse_layer.compute_cubic(&eq_poly, claim);
        assert_eq!(dense_evals, sparse_evals);
    }

    #[test]
    fn sparse_prove_verify() {
        const LAYER_SIZE: usize = 1 << 5;
        const BATCH_SIZE: usize = 2;
        let mut rng = test_rng();

        let fingerprints: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            let layer: DenseGrandProductLayer<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect::<Vec<_>>();
            layer
        })
        .take(BATCH_SIZE)
        .collect();

        let flags: Vec<Vec<usize>> = std::iter::repeat_with(|| {
            let mut layer = vec![];
            for i in 0..LAYER_SIZE {
                if rng.next_u32().is_even() {
                    layer.push(i);
                }
            }
            layer
        })
        .take(BATCH_SIZE / 2)
        .collect();

        let mut circuit = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::construct((flags, fingerprints));

        // let dense_leaves = circuit.sparse_layers[0]
        //     .values
        //     .iter()
        //     .flat_map(|layer| layer.to_dense(LAYER_SIZE))
        //     .collect();
        // let mut dense_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        //     Fr,
        //     Zeromorph<Bn254>,
        // >>::construct((dense_leaves, BATCH_SIZE));
        // let dense_claims = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        //     Fr,
        //     Zeromorph<Bn254>,
        // >>::claimed_outputs(&dense_circuit);

        // let mut dense_transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        // let _ = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        //     Fr,
        //     Zeromorph<Bn254>,
        // >>::prove_grand_product(
        //     &mut dense_circuit, &mut dense_transcript, None
        // );

        let claims = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::claimed_outputs(&circuit);

        // assert_eq!(dense_claims, claims);

        let mut prover_transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        // prover_transcript.compare_to(dense_transcript);
        let (proof, r_prover) = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::prove_grand_product(
            &mut circuit, None, &mut prover_transcript, None
        );

        let mut verifier_transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let (_, r_verifier) = ToggledBatchedGrandProduct::verify_grand_product(
            &proof,
            &claims,
            None,
            &mut verifier_transcript,
            None,
        );
        assert_eq!(r_prover, r_verifier);
    }

    #[test]
    fn sparse_construct() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 3;
        let mut rng = test_rng();

        let fingerprints: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            let layer: DenseGrandProductLayer<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect::<Vec<_>>();
            layer
        })
        .take(BATCH_SIZE * 2)
        .collect();

        let flag_indices: Vec<Vec<usize>> = std::iter::repeat_with(|| {
            let mut layer = vec![];
            for i in 0..LAYER_SIZE {
                if rng.next_u32().is_even() {
                    layer.push(i);
                }
            }
            layer
        })
        .take(BATCH_SIZE)
        .collect();

        let mut expected_outputs: Vec<Fr> = vec![];
        for (indices, fingerprints) in flag_indices.iter().zip(fingerprints.chunks(2)) {
            let read_fingerprints = &fingerprints[0];
            let write_fingerprints = &fingerprints[1];

            expected_outputs.push(
                indices
                    .iter()
                    .map(|index| read_fingerprints[*index])
                    .product(),
            );
            expected_outputs.push(
                indices
                    .iter()
                    .map(|index| write_fingerprints[*index])
                    .product(),
            );
        }

        let circuit = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::construct((flag_indices, fingerprints));

        let mut layer_size = LAYER_SIZE;
        for layers in &circuit.sparse_layers {
            for (layer, expected_product) in layers.values.iter().zip(expected_outputs.iter()) {
                let actual_product: Fr = layer.to_dense(layer_size).iter().product();
                assert_eq!(*expected_product, actual_product);
            }
            layer_size /= 2;
        }

        let claimed_outputs: Vec<Fr> = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::claimed_outputs(&circuit);

        assert!(claimed_outputs == expected_outputs);
    }
}
