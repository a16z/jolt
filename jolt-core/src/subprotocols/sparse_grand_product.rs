use super::grand_product::{
    BatchedGrandProduct, BatchedGrandProductLayer, BatchedGrandProductLayerProof,
};
use super::sumcheck::{BatchedCubicSumcheck, Bindable};
use crate::field::{JoltField, OptimizedMul};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::sparse_interleaved_poly::SparseInterleavedPolynomial;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::ProofTranscript;
use rayon::prelude::*;

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
                DensePolynomial::new(coalesced_flags.clone()),
                DensePolynomial::new(coalesced_fingerprints.clone()),
            )
        } else if self.flag_values.is_empty() {
            let fingerprints: Vec<_> = self.fingerprints.concat();
            let mut flags = vec![F::zero(); fingerprints.len()];
            for (batch_index, flag_indices) in self.flag_indices.iter().enumerate() {
                for flag_index in flag_indices {
                    flags[batch_index * self.layer_len + flag_index] = F::one();
                    flags[batch_index * self.layer_len + self.layer_len / 2 + flag_index] =
                        F::one();
                }
            }
            flags.resize(flags.len().next_power_of_two(), F::one());

            (
                DensePolynomial::new(flags),
                DensePolynomial::new_padded(fingerprints),
            )
        } else {
            let fingerprints: Vec<_> = self
                .fingerprints
                .iter()
                .flat_map(|f| f[..self.layer_len / 2].iter())
                .cloned()
                .collect();
            let mut flags = vec![F::zero(); fingerprints.len()];
            for (batch_index, (flag_indices, flag_values)) in self
                .flag_indices
                .iter()
                .zip(self.flag_values.iter())
                .enumerate()
            {
                for (flag_index, flag_value) in flag_indices.iter().zip(flag_values) {
                    flags[batch_index * self.layer_len + flag_index] = *flag_value;
                    flags[batch_index * self.layer_len + self.layer_len / 2 + flag_index] =
                        *flag_value;
                }
            }
            flags.resize(flags.len().next_power_of_two(), F::one());

            (
                DensePolynomial::new(flags),
                DensePolynomial::new_padded(fingerprints),
            )
        }
    }
}

impl<F: JoltField> BatchedGrandProductToggleLayer<F> {
    fn new(flag_indices: Vec<Vec<usize>>, fingerprints: Vec<Vec<F>>) -> Self {
        let layer_len = 2 * fingerprints[0].len();
        let batched_layer_len = fingerprints.len() * layer_len;
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

    fn layer_output(&self) -> SparseInterleavedPolynomial<F> {
        let values: Vec<_> = self
            .fingerprints
            .par_iter()
            .enumerate()
            .map(|(batch_index, fingerprints)| {
                let flag_indices = &self.flag_indices[batch_index / 2];
                let mut sparse_coeffs = vec![];
                for i in flag_indices {
                    sparse_coeffs
                        .push((batch_index * self.layer_len / 2 + i, fingerprints[*i]).into());
                }
                sparse_coeffs
            })
            .collect();

        let sparse_poly = SparseInterleavedPolynomial::new(values, self.batched_layer_len / 2);
        sparse_poly
        // #[cfg(test)]
        // let dense_poly = DenseInterleavedPolynomial::new(sparse_poly.coalesce());
        // BatchedSparseGrandProductLayer {
        //     values: sparse_poly,
        //     #[cfg(test)]
        //     reference: dense_poly,
        // }
    }
}

impl<F: JoltField> Bindable<F> for BatchedGrandProductToggleLayer<F> {
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
    fn bind(&mut self, r: F) {
        #[cfg(test)]
        let (mut flags_before_binding, mut fingerprints_before_binding) = self.to_dense();

        if let Some(coalesced_flags) = &mut self.coalesced_flags {
            let mut bound_flags = vec![F::one(); coalesced_flags.len() / 2];
            for i in 0..bound_flags.len() {
                bound_flags[i] = coalesced_flags[2 * i]
                    + r * (coalesced_flags[2 * i + 1] - coalesced_flags[2 * i]);
            }
            self.coalesced_flags = Some(bound_flags);

            let coalesced_fingerpints = self.coalesced_fingerprints.as_mut().unwrap();
            let mut bound_fingerprints = vec![F::zero(); coalesced_fingerpints.len() / 2];
            for i in 0..bound_fingerprints.len() {
                bound_fingerprints[i] = coalesced_fingerpints[2 * i]
                    + r * (coalesced_fingerpints[2 * i + 1] - coalesced_fingerpints[2 * i]);
            }
            self.coalesced_fingerprints = Some(bound_fingerprints);
            self.batched_layer_len /= 2;

            #[cfg(test)]
            {
                let (bound_flags, bound_fingerprints) = self.to_dense();
                flags_before_binding.bound_poly_var_bot(&r);
                fingerprints_before_binding.bound_poly_var_bot(&r);
                assert_eq!(
                    bound_flags.Z[..bound_flags.len()],
                    flags_before_binding.Z[..flags_before_binding.len()]
                );
                assert_eq!(
                    bound_fingerprints.Z[..bound_fingerprints.len()],
                    fingerprints_before_binding.Z[..fingerprints_before_binding.len()]
                );
            }

            return;
        }

        debug_assert!(self.layer_len % 4 == 0);

        self.fingerprints
            .par_iter_mut()
            .for_each(|layer: &mut Vec<F>| {
                let n = self.layer_len / 4;
                for i in 0..n {
                    // TODO(moodlezoup): Try mul_0_optimized here
                    layer[i] = layer[2 * i] + r * (layer[2 * i + 1] - layer[2 * i]);
                }
            });

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
                                flag_values[bound_index] =
                                    flag_values[j] + r * (flag_values[j + 1] - flag_values[j]);
                            };
                        } else {
                            // This flag's sibling wasn't found, so it must have value 0.

                            if is_first_bind {
                                // For first bind, all non-zero flag values are 1.
                                // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                //                = flags[2 * i] - r * flags[2 * i]
                                //                = 1 - r
                                flag_values.push(F::one() - r);
                            } else {
                                // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                //                = flags[2 * i] - r * flags[2 * i]
                                flag_values[bound_index] = flag_values[j] - r * flag_values[j];
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
                            flag_values.push(r);
                        } else {
                            // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                            //                = r * flags[2 * i + 1]
                            flag_values[bound_index] = r * flag_values[j];
                        };
                        next_index_to_process = index + 1;
                    }

                    bound_index += 1;
                }

                flag_indices.truncate(bound_index);
                // We only ever use `flag_indices.len()`, so no need to truncate `flag_values`
                // flag_values.truncate(bound_index);
            });
        self.layer_len /= 2;
        self.batched_layer_len /= 2;

        #[cfg(test)]
        {
            let (bound_flags, bound_fingerprints) = self.to_dense();
            flags_before_binding.bound_poly_var_bot(&r);
            fingerprints_before_binding.bound_poly_var_bot(&r);
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
                    debug_assert!(indices.len() <= 1);
                    let mut coalesced = [F::zero(), F::zero()];
                    for (index, value) in indices.iter().zip(values.iter()) {
                        assert_eq!(*index, 0);
                        coalesced[0] = *value;
                        coalesced[1] = *value;
                    }
                    coalesced
                })
                .collect();
            coalesced_flags.resize(coalesced_flags.len().next_power_of_two(), F::one());

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
}

impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedGrandProductToggleLayer<F> {
    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &SplitEqPolynomial<F>, round_claim: F) {
        let (flags, fingerprints) = self.to_dense();
        let merged_eq = eq_poly.merge();
        let expected: F = flags
            .evals_ref()
            .iter()
            .zip(fingerprints.evals_ref().iter())
            .zip(merged_eq.evals_ref().iter())
            .map(|((flag, fingerprint), eq)| *eq * (*flag * fingerprint + F::one() - flag))
            .sum();
        assert_eq!(expected, round_claim);
    }

    /// Similar to `BatchedSparseGrandProductLayer::compute_cubic`, but with changes to
    /// accomodate the differences between `BatchedSparseGrandProductLayer` and
    /// `BatchedGrandProductToggleLayer`. These differences are described in the doc comments
    /// for `BatchedGrandProductToggleLayer::bind`.
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::compute_cubic")]
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        let eq_evals: Vec<(F, F, F)> = todo!();
        // let eq_evals: Vec<(F, F, F)> = (0..eq_poly.len() / 2)
        //     .into_par_iter()
        //     .map(|i| {
        //         let eval_point_0 = eq_poly[2 * i];
        //         let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
        //         let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
        //         let eval_point_3 = eval_point_2 + m_eq;
        //         (eval_point_0, eval_point_2, eval_point_3)
        //     })
        //     .collect();

        if let Some(coalesced_flags) = &self.coalesced_flags {
            let coalesced_fingerpints = self.coalesced_fingerprints.as_ref().unwrap();

            let evals = eq_evals
                .iter()
                .zip(coalesced_flags.chunks(2))
                .zip(coalesced_fingerpints.chunks(2))
                .map(|((eq, flags), fingerprints)| {
                    let m_flag = flags[1] - flags[0];
                    let m_fingerprint = fingerprints[1] - fingerprints[0];

                    let flag_eval_2 = flags[1] + m_flag;
                    let flag_eval_3 = flag_eval_2 + m_flag;

                    let fingerprint_eval_2 = fingerprints[1] + m_fingerprint;
                    let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                    (
                        eq.0 * (flags[0] * fingerprints[0] + F::one() - flags[0]),
                        eq.1 * (flag_eval_2 * fingerprint_eval_2 + F::one() - flag_eval_2),
                        eq.2 * (flag_eval_3 * fingerprint_eval_3 + F::one() - flag_eval_3),
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
        let eq_eval_sums: (F, F, F) = eq_evals[..eq_chunk_size * self.fingerprints.len()]
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
        let mut eq_poly = SplitEqPolynomial::new(r_grand_product);

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
    sparse_layers: Vec<SparseInterleavedPolynomial<F>>,
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
        let mut layers: Vec<SparseInterleavedPolynomial<F>> = Vec::with_capacity(num_layers);
        layers.push(toggle_layer.layer_output());

        for i in 0..num_layers - 1 {
            let previous_layer = &layers[i];
            layers.push(previous_layer.layer_output());
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
        let last_layer = self.sparse_layers.last().unwrap();
        let (left, right) = last_layer.uninterleave();
        left.iter().zip(right.iter()).map(|(l, r)| *l * r).collect()
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
    use crate::poly::{
        commitment::zeromorph::Zeromorph, dense_interleaved_poly::DenseInterleavedPolynomial,
    };
    use ark_bn254::{Bn254, Fr};
    use ark_std::{rand::Rng, test_rng, One};
    use itertools::Itertools;
    use num_integer::Integer;
    use rand_core::RngCore;

    fn condense(sparse_layer: SparseInterleavedPolynomial<Fr>) -> Vec<Fr> {
        sparse_layer.to_dense().Z
    }

    #[test]
    fn dense_sparse_bind_parity() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 7] = [1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let layer_size = 1 << num_vars;
            let dense_layers: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                std::iter::repeat_with(|| {
                    if rng.gen_bool(density) {
                        Fr::random(&mut rng)
                    } else {
                        Fr::one()
                    }
                })
                .take(layer_size)
                .collect()
            })
            .take(batch_size)
            .collect();
            let mut dense_poly = DenseInterleavedPolynomial::new(dense_layers.concat());

            let sparse_coeffs: Vec<_> = dense_layers
                .iter()
                .enumerate()
                .map(|(i, dense_layer)| {
                    let mut sparse_layer = vec![];
                    for (j, val) in dense_layer.iter().enumerate() {
                        if !val.is_one() {
                            sparse_layer.push((i * layer_size + j, *val).into());
                        }
                    }
                    sparse_layer
                })
                .collect();
            let mut sparse_poly =
                SparseInterleavedPolynomial::new(sparse_coeffs, batch_size * layer_size);

            for (dense, sparse) in dense_poly.iter().zip(condense(sparse_poly.clone()).iter()) {
                assert_eq!(dense, sparse);
            }

            for _ in 0..(batch_size * layer_size).log_2() - 1 {
                let r = Fr::random(&mut rng);
                dense_poly.bind(r);
                sparse_poly.bind(r);

                for (dense, sparse) in dense_poly.iter().zip(condense(sparse_poly.clone()).iter()) {
                    assert_eq!(dense, sparse);
                }
            }
        }
    }

    #[test]
    fn dense_sparse_compute_cubic_parity() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 7] = [1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let layer_size = 1 << num_vars;
            let dense_layers: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                let layer: Vec<Fr> = std::iter::repeat_with(|| {
                    if rng.gen_bool(density) {
                        Fr::random(&mut rng)
                    } else {
                        Fr::one()
                    }
                })
                .take(layer_size)
                .collect::<Vec<_>>();
                layer
            })
            .take(batch_size)
            .collect();
            let dense_poly = DenseInterleavedPolynomial::new(dense_layers.concat());

            let sparse_coeffs: Vec<_> = dense_layers
                .iter()
                .enumerate()
                .map(|(i, dense_layer)| {
                    let mut sparse_layer = vec![];
                    for (j, val) in dense_layer.iter().enumerate() {
                        if !val.is_one() {
                            sparse_layer.push((i * layer_size + j, *val).into());
                        }
                    }
                    sparse_layer
                })
                .collect();
            let sparse_poly =
                SparseInterleavedPolynomial::new(sparse_coeffs, batch_size * layer_size);

            for (dense, sparse) in dense_poly.iter().zip(condense(sparse_poly.clone()).iter()) {
                assert_eq!(dense, sparse);
            }

            let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take((batch_size * layer_size).next_power_of_two().log_2() - 1)
                .collect::<Vec<_>>();
            let eq_poly = SplitEqPolynomial::new(&r_eq);
            let r = Fr::random(&mut rng);

            let dense_evals = dense_poly.compute_cubic(&eq_poly, r);
            let sparse_evals = sparse_poly.compute_cubic(&eq_poly, r);
            assert_eq!(dense_evals, sparse_evals);
        }
    }

    #[test]
    fn sparse_prove_verify() {
        const LAYER_SIZE: usize = 1 << 2;
        const BATCH_SIZE: usize = 2;
        let mut rng = test_rng();

        let fingerprints: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            let layer: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
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

        let claims = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::claimed_outputs(&circuit);

        let mut prover_transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
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
        let mut rng = test_rng();
        const NUM_VARS: [usize; 7] = [1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let layer_size = 1 << num_vars;
            let fingerprints: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                let layer: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
                    .take(layer_size)
                    .collect::<Vec<_>>();
                layer
            })
            .take(batch_size)
            .collect();

            let flag_indices: Vec<Vec<usize>> = std::iter::repeat_with(|| {
                let mut layer = vec![];
                for i in 0..layer_size {
                    if rng.gen_bool(density) {
                        layer.push(i);
                    }
                }
                layer
            })
            .take(batch_size / 2)
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

            for layers in &circuit.sparse_layers {
                let dense = layers.to_dense();
                let chunk_size = layers.dense_len / batch_size;
                for (chunk, expected_product) in
                    dense.Z.chunks(chunk_size).zip(expected_outputs.iter())
                {
                    let actual_product: Fr = chunk.iter().product();
                    assert_eq!(*expected_product, actual_product);
                }
            }

            let claimed_outputs: Vec<Fr> = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::claimed_outputs(&circuit);

            assert!(claimed_outputs == expected_outputs);
        }
    }
}
