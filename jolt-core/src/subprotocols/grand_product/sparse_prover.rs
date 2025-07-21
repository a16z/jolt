use crate::field::{JoltField, OptimizedMul};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::sparse_interleaved_poly::SparseInterleavedPolynomial;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::grand_product::quark::QuarkGrandProductBase;
use crate::subprotocols::grand_product::sparse::{
    BatchedGrandProductToggleLayer, SparseGrandProductConfig, ToggledBatchedGrandProduct,
};
use crate::subprotocols::grand_product::{
    BatchedGrandProductLayer, BatchedGrandProductLayerProof, BatchedGrandProductProof,
    BatchedGrandProductProver,
};
use crate::subprotocols::sumcheck::{BatchedCubicSumcheck, Bindable};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use rayon::prelude::*;

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
            // Fingerprints are padded with 0s, flags are padded with 1s
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
            // Fingerprints are padded with 0s, flags are padded with 1s
            flags.resize(flags.len().next_power_of_two(), F::one());

            (
                DensePolynomial::new(flags),
                DensePolynomial::new_padded(fingerprints),
            )
        }
    }
}

impl<F: JoltField> BatchedGrandProductToggleLayer<F> {
    pub(crate) fn new(flag_indices: Vec<Vec<usize>>, fingerprints: Vec<Vec<F>>) -> Self {
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

    /// Computes the grand product layer output by this one.
    /// Since this is a toggle layer, most of the output values are 1s, so
    /// the return type is a SparseInterleavedPolynomial
    ///   o      o     o      o    <-  output layer
    ///  / \    / \   / \    / \
    /// 🏴  o  🏳️ o  🏳️ o  🏴  o  <- toggle layer
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::layer_output")]
    pub(crate) fn layer_output(&self) -> SparseInterleavedPolynomial<F> {
        let values: Vec<_> = self
            .fingerprints
            .par_iter()
            .enumerate()
            .map(|(batch_index, fingerprints)| {
                let flag_indices = &self.flag_indices[batch_index / 2];
                let mut sparse_coeffs = Vec::with_capacity(self.layer_len);
                for i in flag_indices {
                    sparse_coeffs
                        .push((batch_index * self.layer_len / 2 + i, fingerprints[*i]).into());
                }
                sparse_coeffs
            })
            .collect();

        SparseInterleavedPolynomial::new(values, self.batched_layer_len / 2)
    }

    /// Coalesces flags and fingerprints into one (dense) vector each.
    /// After a certain number of bindings, we can no longer process the k
    /// circuits in the batch in independently, at which point we coalesce.
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::coalesce")]
    fn coalesce(&mut self) {
        let mut coalesced_fingerprints: Vec<F> =
            self.fingerprints.iter().map(|f| f[0]).collect::<Vec<_>>();
        coalesced_fingerprints.resize(coalesced_fingerprints.len().next_power_of_two(), F::zero());

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
        // Fingerprints are padded with 0s, flags are padded with 1s
        coalesced_flags.resize(coalesced_flags.len().next_power_of_two(), F::one());

        self.coalesced_fingerprints = Some(coalesced_fingerprints);
        self.coalesced_flags = Some(coalesced_flags);
    }
}

impl<F: JoltField> Bindable<F> for BatchedGrandProductToggleLayer<F> {
    /// Incrementally binds a variable of the flag and fingerprint polynomials.
    /// Similar to `SparseInterleavedPolynomial::bind`, in that flags use
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
            // Polynomials have already been coalesced, so bind the coalesced vectors.
            let mut bound_flags = vec![F::one(); coalesced_flags.len() / 2];
            for i in 0..bound_flags.len() {
                bound_flags[i] = coalesced_flags[2 * i]
                    + r * (coalesced_flags[2 * i + 1] - coalesced_flags[2 * i]);
            }
            self.coalesced_flags = Some(bound_flags);

            let coalesced_fingerprints = self.coalesced_fingerprints.as_mut().unwrap();
            let mut bound_fingerprints = vec![F::zero(); coalesced_fingerprints.len() / 2];
            for i in 0..bound_fingerprints.len() {
                bound_fingerprints[i] = coalesced_fingerprints[2 * i]
                    + r * (coalesced_fingerprints[2 * i + 1] - coalesced_fingerprints[2 * i]);
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

        debug_assert!(self.layer_len.is_multiple_of(4));

        // Bind the fingerprints
        self.fingerprints
            .par_iter_mut()
            .for_each(|layer: &mut Vec<F>| {
                let n = self.layer_len / 4;
                for i in 0..n {
                    layer[i] = layer[2 * i] + r.mul_0_optimized(layer[2 * i + 1] - layer[2 * i]);
                }
            });

        let is_first_bind = self.flag_values.is_empty();
        if is_first_bind {
            self.flag_values = vec![vec![]; self.flag_indices.len()];
        }

        // Bind the flags
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

                    if index.is_multiple_of(2) {
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
            // Time to coalesce
            assert!(self.coalesced_fingerprints.is_none());
            assert!(self.coalesced_flags.is_none());
            self.coalesce();

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

impl<F: JoltField, ProofTranscript: Transcript> BatchedCubicSumcheck<F, ProofTranscript>
    for BatchedGrandProductToggleLayer<F>
{
    /// Similar to `SparseInterleavedPolynomial::compute_cubic`, but with changes to
    /// accommodate the differences between `SparseInterleavedPolynomial` and
    /// `BatchedGrandProductToggleLayer`. These differences are described in the doc comments
    /// for `BatchedGrandProductToggleLayer::bind`.
    ///
    /// Since we are using the Dao-Thaler EQ optimization, there are four cases to handle:
    /// 1. Flags/fingerprints are coalesced, and E1 is fully bound
    /// 2. Flags/fingerprints are coalesced, and E1 isn't fully bound
    /// 3. Flags/fingerprints aren't coalesced, and E1 is fully bound
    /// 4. Flags/fingerprints aren't coalesced, and E1 isn't fully bound
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::compute_cubic")]
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        if let Some(coalesced_flags) = &self.coalesced_flags {
            let coalesced_fingerprints = self.coalesced_fingerprints.as_ref().unwrap();

            let cubic_evals = if eq_poly.E1_len == 1 {
                // 1. Flags/fingerprints are coalesced, and E1 is fully bound
                // This is similar to the if case of `DenseInterleavedPolynomial::compute_cubic`
                coalesced_flags
                    .par_chunks(2)
                    .zip(coalesced_fingerprints.par_chunks(2))
                    .zip(eq_poly.E2.par_chunks(2))
                    .map(|((flags, fingerprints), eq_chunk)| {
                        let eq_evals = {
                            let eval_point_0 = eq_chunk[0];
                            let m_eq = eq_chunk[1] - eq_chunk[0];
                            let eval_point_2 = eq_chunk[1] + m_eq;
                            let eval_point_3 = eval_point_2 + m_eq;
                            (eval_point_0, eval_point_2, eval_point_3)
                        };
                        let m_flag = flags[1] - flags[0];
                        let m_fingerprint = fingerprints[1] - fingerprints[0];

                        let flag_eval_2 = flags[1] + m_flag;
                        let flag_eval_3 = flag_eval_2 + m_flag;

                        let fingerprint_eval_2 = fingerprints[1] + m_fingerprint;
                        let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                        (
                            eq_evals.0 * (flags[0] * fingerprints[0] + F::one() - flags[0]),
                            eq_evals.1
                                * (flag_eval_2 * fingerprint_eval_2 + F::one() - flag_eval_2),
                            eq_evals.2
                                * (flag_eval_3 * fingerprint_eval_3 + F::one() - flag_eval_3),
                        )
                    })
                    .reduce(
                        || (F::zero(), F::zero(), F::zero()),
                        |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                    )
            } else {
                // 2. Flags/fingerprints are coalesced, and E1 isn't fully bound
                // This is similar to the else case of `DenseInterleavedPolynomial::compute_cubic`
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

                let flag_chunk_size = coalesced_flags.len().next_power_of_two() / eq_poly.E2_len;
                let fingerprint_chunk_size =
                    coalesced_fingerprints.len().next_power_of_two() / eq_poly.E2_len;

                eq_poly.E2[..eq_poly.E2_len]
                    .par_iter()
                    .zip(coalesced_flags.par_chunks(flag_chunk_size))
                    .zip(coalesced_fingerprints.par_chunks(fingerprint_chunk_size))
                    .map(|((E2_eval, flag_x2), fingerprint_x2)| {
                        let mut inner_sum = (F::zero(), F::zero(), F::zero());
                        for ((E1_evals, flag_chunk), fingerprint_chunk) in E1_evals
                            .iter()
                            .zip(flag_x2.chunks(2))
                            .zip(fingerprint_x2.chunks(2))
                        {
                            let m_flag = flag_chunk[1] - flag_chunk[0];
                            let m_fingerprint = fingerprint_chunk[1] - fingerprint_chunk[0];

                            let flag_eval_2 = flag_chunk[1] + m_flag;
                            let flag_eval_3 = flag_eval_2 + m_flag;

                            let fingerprint_eval_2 = fingerprint_chunk[1] + m_fingerprint;
                            let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                            inner_sum.0 += E1_evals.0
                                * (flag_chunk[0] * fingerprint_chunk[0] + F::one() - flag_chunk[0]);
                            inner_sum.1 += E1_evals.1
                                * (flag_eval_2 * fingerprint_eval_2 + F::one() - flag_eval_2);
                            inner_sum.2 += E1_evals.2
                                * (flag_eval_3 * fingerprint_eval_3 + F::one() - flag_eval_3);
                        }

                        (
                            *E2_eval * inner_sum.0,
                            *E2_eval * inner_sum.1,
                            *E2_eval * inner_sum.2,
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
            return UniPoly::from_evals(&cubic_evals);
        }

        let cubic_evals = if eq_poly.E1_len == 1 {
            // 3. Flags/fingerprints aren't coalesced, and E1 is fully bound
            // This is similar to the if case of `SparseInterleavedPolynomial::compute_cubic`
            let eq_evals: Vec<(F, F, F)> = eq_poly.E2[..eq_poly.E2_len]
                .par_chunks(2)
                .take(self.batched_layer_len / 4)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
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

            let deltas: (F, F, F) = (0..self.fingerprints.len())
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

                        let (flags, fingerprints) = if index.is_multiple_of(2) {
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

                        let block_index = (self.layer_len * batch_index) / 4 + index / 2;
                        let eq_evals = eq_evals[block_index];

                        delta.0 += eq_evals
                            .0
                            .mul_0_optimized(flags.0.mul_01_optimized(fingerprints.0) - flags.0);
                        delta.1 += eq_evals.1.mul_0_optimized(
                            flag_eval_2.mul_01_optimized(fingerprint_eval_2) - flag_eval_2,
                        );
                        delta.2 += eq_evals.2.mul_0_optimized(
                            flag_eval_3.mul_01_optimized(fingerprint_eval_3) - flag_eval_3,
                        );
                    }

                    (delta.0, delta.1, delta.2)
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );
            // eq_eval_sum + ∆ = Σ eq_evals[i] + Σ eq_evals[i] * (flag[i] * fingerprint[i] - flag[i]))
            //                 = Σ eq_evals[j] * (flag[i] * fingerprint[i] + 1 - flag[i])
            (
                eq_eval_sums.0 + deltas.0,
                eq_eval_sums.1 + deltas.1,
                eq_eval_sums.2 + deltas.2,
            )
        } else {
            // 4. Flags/fingerprints aren't coalesced, and E1 isn't fully bound
            // This is similar to the else case of `SparseInterleavedPolynomial::compute_cubic`
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

            let deltas = (0..self.fingerprints.len())
                .into_par_iter()
                .map(|batch_index| {
                    // Computes:
                    //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j])    ∀j where flag[j] ≠ 0
                    // for the evaluation points {0, 2, 3}

                    let fingerprints = &self.fingerprints[batch_index];
                    let flag_indices = &self.flag_indices[batch_index / 2];

                    let unbound = self.flag_values.is_empty();
                    let mut delta = (F::zero(), F::zero(), F::zero());
                    let mut inner_sum = (F::zero(), F::zero(), F::zero());
                    let mut prev_x2 = 0;

                    let mut next_index_to_process = 0usize;
                    for (j, index) in flag_indices.iter().enumerate() {
                        if *index < next_index_to_process {
                            // This node was already processed in a previous iteration
                            continue;
                        }

                        let (flags, fingerprints) = if index.is_multiple_of(2) {
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

                        let block_index = (self.layer_len * batch_index) / 4 + index / 2;
                        let x2 = block_index >> num_x1_bits;
                        if x2 != prev_x2 {
                            delta.0 += eq_poly.E2[prev_x2] * inner_sum.0;
                            delta.1 += eq_poly.E2[prev_x2] * inner_sum.1;
                            delta.2 += eq_poly.E2[prev_x2] * inner_sum.2;
                            inner_sum = (F::zero(), F::zero(), F::zero());
                            prev_x2 = x2;
                        }

                        let x1 = block_index & x1_bitmask;
                        inner_sum.0 += E1_evals[x1]
                            .0
                            .mul_0_optimized(flags.0.mul_01_optimized(fingerprints.0) - flags.0);
                        inner_sum.1 += E1_evals[x1].1.mul_0_optimized(
                            flag_eval_2.mul_01_optimized(fingerprint_eval_2) - flag_eval_2,
                        );
                        inner_sum.2 += E1_evals[x1].2.mul_0_optimized(
                            flag_eval_3.mul_01_optimized(fingerprint_eval_3) - flag_eval_3,
                        );
                    }

                    delta.0 += eq_poly.E2[prev_x2] * inner_sum.0;
                    delta.1 += eq_poly.E2[prev_x2] * inner_sum.1;
                    delta.2 += eq_poly.E2[prev_x2] * inner_sum.2;

                    delta
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            // The cubic evals assuming all the coefficients are ones is affected by the
            // `batched_layer_len`, since we implicitly pad the `batched_layer_len` to a power of 2.
            // By pad here we mean that flags are padded with 1s, and fingerprints are
            // padded with 0s.
            //
            // As a refresher, the cubic evals we're computing are:
            //
            // \sum_x2 E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * \prod_k ((1 - j) * P_k(0 || x1 || x2) + j * P_k(1 || x1 || x2)))
            let evals_assuming_all_ones = if self.batched_layer_len.is_power_of_two() {
                // If `batched_layer_len` is a power of 2, there is no 0-padding.
                //
                // So we have:
                // \sum_x2 (E2[x2] * (\sum_x1 ((1 - j) * E1[0, x1] + j * E1[1, x1]) * 1))
                //   = \sum_x2 (E2[x2] * \sum_x1 E1_evals[x1])
                //   = (\sum_x2 E2[x2]) * (\sum_x1 E1_evals[x1])
                //   = 1 * E1_eval_sums
                E1_eval_sums
            } else {
                let chunk_size = self.batched_layer_len.next_power_of_two() / eq_poly.E2_len;
                let num_all_one_chunks = self.batched_layer_len / chunk_size;
                let E2_sum: F = eq_poly.E2[..num_all_one_chunks].iter().sum();
                if self.batched_layer_len.is_multiple_of(chunk_size) {
                    // If `batched_layer_len` isn't a power of 2 but evenly divides `chunk_size`,
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
                    // If `batched_layer_len` isn't a power of 2 and doesn't divide `chunk_size`,
                    // the last nonzero "chunk" will have (self.dense_len % chunk_size) ones,
                    // followed by (chunk_size - self.dense_len % chunk_size) zeros,
                    // e.g. 1 1 1 1 1 1 1 1 0 0 0 0
                    //
                    // This handles this last chunk:
                    let last_chunk_evals = E1_evals[..(self.batched_layer_len % chunk_size) / 4]
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

        #[cfg(test)]
        {
            let (dense_flags, dense_fingerprints) = self.to_dense();
            let eq_merged = eq_poly.merge();
            let dense_cubic_evals = dense_flags
                .evals()
                .par_chunks(2)
                .zip(dense_fingerprints.evals().par_chunks(2))
                .zip(eq_merged.evals().par_chunks(2))
                .map(|((flag_chunk, fingerprint_chunk), eq_chunk)| {
                    let eq_evals = {
                        let eval_point_0 = eq_chunk[0];
                        let m_eq = eq_chunk[1] - eq_chunk[0];
                        let eval_point_2 = eq_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };
                    let flags = (
                        *flag_chunk.first().unwrap_or(&F::one()),
                        *flag_chunk.get(1).unwrap_or(&F::one()),
                    );
                    let fingerprints = (
                        *fingerprint_chunk.first().unwrap_or(&F::zero()),
                        *fingerprint_chunk.get(1).unwrap_or(&F::zero()),
                    );

                    let m_flag = flags.1 - flags.0;
                    let m_fingerprint = fingerprints.1 - fingerprints.0;

                    let flag_eval_2 = flags.1 + m_flag;
                    let flag_eval_3 = flag_eval_2 + m_flag;

                    let fingerprint_eval_2 = fingerprints.1 + m_fingerprint;
                    let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                    (
                        eq_evals.0 * (flags.0 * fingerprints.0 + F::one() - flags.0),
                        eq_evals.1 * (flag_eval_2 * fingerprint_eval_2 + F::one() - flag_eval_2),
                        eq_evals.2 * (flag_eval_3 * fingerprint_eval_3 + F::one() - flag_eval_3),
                    )
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );
            let dense_cubic_evals = [
                dense_cubic_evals.0,
                previous_round_claim - dense_cubic_evals.0,
                dense_cubic_evals.1,
                dense_cubic_evals.2,
            ];
            assert_eq!(dense_cubic_evals, cubic_evals);
        }

        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (F, F) {
        assert_eq!(self.layer_len, 2);
        let flags = self.coalesced_flags.as_ref().unwrap();
        let fingerprints = self.coalesced_fingerprints.as_ref().unwrap();

        (flags[0], fingerprints[0])
    }

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
}

impl<F: JoltField, ProofTranscript: Transcript> BatchedGrandProductLayer<F, ProofTranscript>
    for BatchedGrandProductToggleLayer<F>
{
    fn prove_layer(
        &mut self,
        claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedGrandProductLayerProof<F, ProofTranscript> {
        let mut eq_poly = SplitEqPolynomial::new(r_grand_product);

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(claim, &mut eq_poly, transcript);

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

impl<F, PCS, ProofTranscript> BatchedGrandProductProver<F, PCS, ProofTranscript>
    for ToggledBatchedGrandProduct<F>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<F>>); // (flags, fingerprints)
    type Config = SparseGrandProductConfig;

    #[tracing::instrument(skip_all, name = "ToggledBatchedGrandProduct::construct_with_config")]
    fn construct_with_config(leaves: Self::Leaves, config: Self::Config) -> Self {
        let (flags, fingerprints) = leaves;
        let batch_size = fingerprints.len();
        let tree_depth = fingerprints[0].len().log_2();
        let crossover = config.hybrid_layer_depth.get_crossover_depth();

        let uses_quarks = tree_depth - 1 > crossover;
        let num_sparse_layers = if uses_quarks {
            crossover
        } else {
            tree_depth - 1
        };

        let toggle_layer = BatchedGrandProductToggleLayer::new(flags, fingerprints);
        let mut layers: Vec<_> = Vec::with_capacity(1 + num_sparse_layers);
        layers.push(toggle_layer.layer_output());

        for i in 0..num_sparse_layers {
            let previous_layer = &layers[i];
            layers.push(previous_layer.layer_output());
        }

        // Set the Quark polynomial only if the number of layers exceeds the crossover depth
        let quark_poly = if uses_quarks {
            Some(layers.pop().unwrap().coalesce())
        } else {
            None
        };

        Self {
            batch_size,
            toggle_layer,
            sparse_layers: layers,
            quark_poly,
        }
    }

    fn num_layers(&self) -> usize {
        self.sparse_layers.len() + 1 + self.quark_poly.is_some() as usize
    }

    fn claimed_outputs(&self) -> Vec<F> {
        // If there's a quark poly, then that's the claimed output
        if let Some(quark_poly) = &self.quark_poly {
            let chunk_size = quark_poly.len() / self.batch_size;
            quark_poly
                .par_chunks(chunk_size)
                .map(|chunk| chunk.iter().product())
                .collect()
        } else {
            let last_layer = self.sparse_layers.last().unwrap();
            let (left, right) = last_layer.uninterleave();
            left.iter().zip(right.iter()).map(|(l, r)| *l * r).collect()
        }
    }

    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F, ProofTranscript>> {
        [&mut self.toggle_layer as &mut dyn BatchedGrandProductLayer<F, ProofTranscript>]
            .into_iter()
            .chain(
                self.sparse_layers
                    .iter_mut()
                    .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F, ProofTranscript>),
            )
            .rev()
    }

    /// Computes a batched grand product proof, layer by layer.
    #[tracing::instrument(skip_all, name = "ToggledBatchedGrandProduct::prove_grand_product")]
    fn prove_grand_product(
        &mut self,
        opening_accumulator: Option<&mut ProverOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
        setup: Option<&PCS::ProverSetup>,
    ) -> (BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>) {
        QuarkGrandProductBase::prove_quark_grand_product(
            self,
            opening_accumulator,
            transcript,
            setup,
        )
    }
    fn quark_poly(&self) -> Option<&[F]> {
        self.quark_poly.as_deref()
    }
}
