//! Sampled sparse challenges for one folding round.

use crate::SparseChallenge;
use akita_error::AkitaError;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{ExtField, Field, Ring};

#[cfg(feature = "parallel")]
// Tuned with `benches/sparse_challenge.rs::bench_sparse_evaluation` on an
// Apple M4 Max (16 cores, 64 GiB). Near the crossover, two to four coarse
// leaves amortize Rayon dispatch.
const PARALLEL_COARSE_LEAF_TERMS: usize = 1 << 15;
#[cfg(feature = "parallel")]
// Larger batches expose enough work to keep finer leaves busy.
const PARALLEL_FINE_LEAF_TERMS: usize = 1 << 13;
#[cfg(feature = "parallel")]
// The fine tier starts above the largest measured coarse-leaf sweet spot.
const PARALLEL_FINE_TOTAL_TERMS: usize = 1 << 17;

/// Stage-1 fold challenges in claim-major block order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Challenges {
    /// Per-(claim, block) sparse challenges.
    challenges: std::sync::Arc<[SparseChallenge]>,
    /// Exact number of live blocks packed into one claim.
    num_live_blocks_per_claim: usize,
    /// Number of claims represented by this vector.
    num_claims: usize,
}

impl Challenges {
    /// Construct challenges from a sampled vector and its claim/block layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector length does not match the layout.
    pub fn from_sparse(
        challenges: Vec<SparseChallenge>,
        num_live_blocks_per_claim: usize,
        num_claims: usize,
    ) -> Result<Self, AkitaError> {
        let expected = num_live_blocks_per_claim
            .checked_mul(num_claims)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".to_string()))?;
        if challenges.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.len(),
            });
        }
        Ok(Self {
            challenges: challenges.into(),
            num_live_blocks_per_claim,
            num_claims,
        })
    }

    /// Return the sparse challenges in claim-major block order.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[SparseChallenge] {
        &self.challenges
    }

    /// Number of block challenges represented by this value.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.challenges.len()
    }

    /// Whether this challenge set is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.challenges.is_empty()
    }

    /// Number of claims represented by this challenge set.
    #[inline]
    #[must_use]
    pub fn num_claims(&self) -> usize {
        self.num_claims
    }

    /// Number of logical block challenges per claim.
    #[inline]
    #[must_use]
    pub fn num_live_blocks_per_claim(&self) -> usize {
        self.num_live_blocks_per_claim
    }

    /// Evaluate one challenge at the precomputed `alpha` powers.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range or the challenge is invalid.
    pub fn eval_at_pows<F, E>(&self, index: usize, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        F: Field + Ring,
        E: Field + ExtField<F>,
    {
        self.challenges
            .get(index)
            .ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "challenge index {index} out of range for {} challenges",
                    self.challenges.len()
                ))
            })?
            .eval_at_pows::<F, E>(alpha_pows)
    }

    /// Evaluate every challenge at the precomputed `alpha` powers.
    ///
    /// # Errors
    ///
    /// Returns an error if the powers or any challenge are invalid.
    pub fn evals_at_pows<F, E>(&self, alpha_pows: &[E]) -> Result<Vec<E>, AkitaError>
    where
        F: Field + Ring,
        E: Field + ExtField<F>,
    {
        let evaluate = |challenge: &SparseChallenge| challenge.eval_at_pows::<F, E>(alpha_pows);
        #[cfg(feature = "parallel")]
        let parallel_candidate = {
            // Same M4 Max benchmark above: smaller batches never amortized
            // the initial Rayon dispatch, independent of sparse weight.
            const MIN_PARALLEL_CHALLENGES: usize = 1 << 9;
            self.challenges.len() >= MIN_PARALLEL_CHALLENGES
        };
        #[cfg(feature = "parallel")]
        if parallel_candidate {
            let sparse_terms = self.challenges.iter().try_fold(0usize, |sum, challenge| {
                sum.checked_add(challenge.positions.len()).ok_or_else(|| {
                    AkitaError::InvalidInput("sparse challenge term count overflow".into())
                })
            })?;
            if sparse_terms <= PARALLEL_COARSE_LEAF_TERMS {
                return self.challenges.iter().map(evaluate).collect();
            }
            let leaf_terms = if sparse_terms > PARALLEL_FINE_TOTAL_TERMS {
                PARALLEL_FINE_LEAF_TERMS
            } else {
                PARALLEL_COARSE_LEAF_TERMS
            };
            // Production batches have a uniform sparse weight. Pricing leaves
            // from the exact average also keeps arbitrary valid batches linear
            // without rescanning their terms at every recursive fork.
            let average_terms = sparse_terms.div_ceil(self.challenges.len());
            let leaf_challenges = (leaf_terms / average_terms).max(1);
            let mut evaluations = Vec::new();
            evaluations
                .try_reserve_exact(self.challenges.len())
                .map_err(|_| {
                    AkitaError::InvalidInput("challenge evaluation allocation failed".into())
                })?;
            evaluations.resize(self.challenges.len(), E::zero());
            evaluations
                .par_chunks_mut(leaf_challenges)
                .zip(self.challenges.par_chunks(leaf_challenges))
                .try_for_each(|(evaluation_chunk, challenge_chunk)| {
                    for (evaluation, challenge) in evaluation_chunk.iter_mut().zip(challenge_chunk)
                    {
                        *evaluation = challenge.eval_at_pows::<F, E>(alpha_pows)?;
                    }
                    Ok::<(), AkitaError>(())
                })?;
            return Ok(evaluations);
        }

        self.challenges.iter().map(evaluate).collect()
    }

    /// Select complete claims in the requested order.
    ///
    /// # Errors
    ///
    /// Returns an error if any claim index is out of range.
    pub fn select_claims(&self, claim_indices: &[usize]) -> Result<Self, AkitaError> {
        let capacity = claim_indices
            .len()
            .checked_mul(self.num_live_blocks_per_claim)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".to_string()))?;
        let mut selected = Vec::with_capacity(capacity);
        for &claim_index in claim_indices {
            let start = claim_index
                .checked_mul(self.num_live_blocks_per_claim)
                .ok_or_else(|| AkitaError::InvalidSetup("challenge offset overflow".to_string()))?;
            let end = start
                .checked_add(self.num_live_blocks_per_claim)
                .ok_or_else(|| AkitaError::InvalidSetup("challenge offset overflow".to_string()))?;
            selected.extend_from_slice(self.challenges.get(start..end).ok_or(
                AkitaError::InvalidSize {
                    expected: end,
                    actual: self.challenges.len(),
                },
            )?);
        }
        Self::from_sparse(
            selected,
            self.num_live_blocks_per_claim,
            claim_indices.len(),
        )
    }

    /// Embed subring challenge positions into an ambient A ring.
    ///
    /// Coefficients and claim/block order are preserved exactly. Only each
    /// position `j` changes to `embedding_stride * j`.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent, a source challenge is
    /// malformed, or an embedded position overflows or leaves the A ring.
    pub fn embed_subring_positions(
        &self,
        challenge_subring_dimension: usize,
        embedding_stride: usize,
        ambient_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if challenge_subring_dimension == 0
            || embedding_stride == 0
            || challenge_subring_dimension
                .checked_mul(embedding_stride)
                .filter(|&dimension| dimension == ambient_ring_dimension)
                .is_none()
        {
            return Err(AkitaError::InvalidInput(
                "subring challenge embedding dimensions are inconsistent".into(),
            ));
        }
        let embed = |challenge: &SparseChallenge| {
            challenge.validate_dyn(challenge_subring_dimension)?;
            let mut positions = crate::SparseChallengePositions::new();
            positions
                .try_reserve_exact(challenge.positions.len())
                .map_err(|_| {
                    AkitaError::InvalidInput("subring challenge position allocation failed".into())
                })?;
            for &position in &challenge.positions {
                let embedded_position = usize::try_from(position)
                    .ok()
                    .and_then(|position| position.checked_mul(embedding_stride))
                    .filter(|&position| position < ambient_ring_dimension)
                    .and_then(|position| u32::try_from(position).ok())
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "subring challenge embedded position overflow".into(),
                        )
                    })?;
                positions.push(embedded_position);
            }
            Ok::<_, AkitaError>(SparseChallenge {
                positions,
                coeffs: challenge.coeffs.clone(),
            })
        };
        #[cfg(feature = "parallel")]
        let embedded = {
            let work = self
                .challenges
                .len()
                .checked_mul(
                    self.challenges
                        .first()
                        .map_or(0, |challenge| challenge.positions.len()),
                )
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring challenge embedding work overflow".into())
                })?;
            const PARALLEL_THRESHOLD: usize = 1 << 14;
            if work >= PARALLEL_THRESHOLD {
                self.challenges
                    .par_iter()
                    .map(embed)
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                self.challenges
                    .iter()
                    .map(embed)
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        #[cfg(not(feature = "parallel"))]
        let embedded = self
            .challenges
            .iter()
            .map(embed)
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_sparse(embedded, self.num_live_blocks_per_claim, self.num_claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn batch_evaluation_preserves_challenge_order() {
        let challenges = (0..4096)
            .map(|index| SparseChallenge {
                positions: (0..32).collect::<Vec<_>>().into(),
                coeffs: (0..32)
                    .map(|position| if (index + position) % 2 == 0 { 1 } else { -1 })
                    .collect::<Vec<_>>()
                    .into(),
            })
            .collect::<Vec<_>>();
        let batch = Challenges::from_sparse(challenges, 4096, 1).unwrap();
        let powers = (0..32)
            .map(|index| F::from_u64(index + 1))
            .collect::<Vec<_>>();
        let actual = batch.evals_at_pows::<F, F>(&powers).unwrap();
        let expected = batch
            .as_slice()
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, F>(&powers).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn subring_embedding_preserves_coefficients_and_batch_order() {
        let source = Challenges::from_sparse(
            vec![
                SparseChallenge {
                    positions: vec![0, 3, 7].into(),
                    coeffs: vec![1, -2, 1].into(),
                },
                SparseChallenge {
                    positions: vec![1, 6].into(),
                    coeffs: vec![-1, 2].into(),
                },
            ],
            1,
            2,
        )
        .unwrap();
        let embedded = source.embed_subring_positions(8, 4, 32).unwrap();
        assert_eq!(embedded.num_claims(), 2);
        assert_eq!(embedded.num_live_blocks_per_claim(), 1);
        assert_eq!(embedded.as_slice()[0].positions.as_slice(), &[0, 12, 28]);
        assert_eq!(embedded.as_slice()[0].coeffs, source.as_slice()[0].coeffs);
        assert_eq!(embedded.as_slice()[1].positions.as_slice(), &[4, 24]);
        assert_eq!(embedded.as_slice()[1].coeffs, source.as_slice()[1].coeffs);
        assert!(source.embed_subring_positions(8, 0, 32).is_err());
        assert!(source.embed_subring_positions(8, 2, 32).is_err());
    }
}
