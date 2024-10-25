use crate::{
    field::JoltField,
    subprotocols::{
        grand_product::BatchedGrandProductLayer,
        sumcheck::{BatchedCubicSumcheck, Bindable},
    },
    utils::thread::unsafe_allocate_zero_vec,
};
use rayon::{prelude::*, slice::Chunks};

#[cfg(test)]
use super::dense_mlpoly::DensePolynomial;
use super::{split_eq_poly::SplitEqPolynomial, unipoly::UniPoly};

#[derive(Default, Debug, Clone)]
pub struct DenseInterleavedPolynomial<F: JoltField> {
    pub(crate) coeffs: Vec<F>,
    len: usize,
    binding_scratch_space: Vec<F>,
}

impl<F: JoltField> PartialEq for DenseInterleavedPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            false
        } else {
            self.coeffs[..self.len] == other.coeffs[..other.len]
        }
    }
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
}

impl<F: JoltField> Bindable<F> for DenseInterleavedPolynomial<F> {
    /// Incrementally binds a variable of this batched layer's polynomials.
    /// Even though each layer is backed by a single Vec<F>, it represents two polynomials
    /// one for the left nodes in the circuit, one for the right nodes in the circuit.
    /// These two polynomials' coefficients are interleaved into one Vec<F>. To preserve
    /// this interleaved order, we bind values like this:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    #[tracing::instrument(skip_all, name = "DenseInterleavedPolynomial::bind")]
    fn bind(&mut self, r: F) {
        #[cfg(test)]
        let (mut left_before_binding, mut right_before_binding) = self.uninterleave();

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

        #[cfg(test)]
        {
            let (left_after_binding, right_after_binding) = self.uninterleave();
            bind_left_and_right(&mut left_before_binding, &mut right_before_binding, r);

            assert_eq!(
                *self,
                Self::interleave(&left_before_binding, &right_before_binding)
            );
            assert_eq!(left_after_binding, left_before_binding);
            assert_eq!(right_after_binding, right_before_binding);
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

/// Represents a single layer of a batched grand product circuit.
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   R0  L1   R1  L2   R2  L3   R3   <- This is layer would be represented as [L0, R0, L1, R1, L2, R2, L3, R3]
///                                           (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
impl<F: JoltField> BatchedGrandProductLayer<F> for DenseInterleavedPolynomial<F> {}
impl<F: JoltField> BatchedCubicSumcheck<F> for DenseInterleavedPolynomial<F> {
    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &SplitEqPolynomial<F>, round_claim: F) {
        let (left, right) = self.uninterleave();
        let merged_eq = eq_poly.merge();
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
    /// Recall that the `left` and `right` polynomials are interleaved in each layer of `self.layers`,
    /// so we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProductLayer::compute_cubic")]
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        let cubic_evals = if eq_poly.E1_len == 1 {
            self.par_chunks(4)
                .zip(eq_poly.E2.par_chunks(2))
                .map(|(layer_chunk, eq_chunk)| {
                    let eq_evals = {
                        let eval_point_0 = eq_chunk[0];
                        let m_eq = eq_chunk[1] - eq_chunk[0];
                        let eval_point_2 = eq_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };
                    let left = (
                        *layer_chunk.get(0).unwrap_or(&F::zero()),
                        *layer_chunk.get(2).unwrap_or(&F::zero()),
                    );
                    let right = (
                        *layer_chunk.get(1).unwrap_or(&F::zero()),
                        *layer_chunk.get(3).unwrap_or(&F::zero()),
                    );

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
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
        } else {
            let num_E1_chunks = eq_poly.E1_len / 2;

            let mut evals = (F::zero(), F::zero(), F::zero());
            for (x1, E1_chunk) in eq_poly.E1[..eq_poly.E1_len].chunks(2).enumerate() {
                let E1_evals = {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                };
                let inner_sums = eq_poly.E2[..eq_poly.E2_len]
                    .par_iter()
                    .zip(self.par_chunks(4).skip(x1).step_by(num_E1_chunks))
                    .map(|(E2_eval, P_x1)| {
                        let left = (
                            *P_x1.get(0).unwrap_or(&F::zero()),
                            *P_x1.get(2).unwrap_or(&F::zero()),
                        );
                        let right = (
                            *P_x1.get(1).unwrap_or(&F::zero()),
                            *P_x1.get(3).unwrap_or(&F::zero()),
                        );

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        // TODO(moodlezoup): can save a mult by E2_eval here
                        (
                            *E2_eval * left.0 * right.0,
                            *E2_eval * left_eval_2 * right_eval_2,
                            *E2_eval * left_eval_3 * right_eval_3,
                        )
                    })
                    .reduce(
                        || (F::zero(), F::zero(), F::zero()),
                        |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                    );

                evals.0 += E1_evals.0 * inner_sums.0;
                evals.1 += E1_evals.1 * inner_sums.1;
                evals.2 += E1_evals.2 * inner_sums.2;
            }
            evals
        };

        let cubic_evals = [
            cubic_evals.0,
            previous_round_claim - cubic_evals.0,
            cubic_evals.1,
            cubic_evals.2,
        ];
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (F, F) {
        assert_eq!(self.len(), 2);
        let left_claim = self.coeffs[0];
        let right_claim = self.coeffs[1];
        (left_claim, right_claim)
    }
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
