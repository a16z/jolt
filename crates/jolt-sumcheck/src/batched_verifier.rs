//! Batched sumcheck verification: reduces multiple claims into one via random
//! linear combination.
//!
//! Supports claims with **different** `num_vars` and `degree` bounds via
//! front-loaded batching: shorter instances are active only in the last
//! `num_vars` rounds and are padded with constant dummy polynomials in
//! earlier rounds. Each claim is scaled by $2^{N - n_i}$ where $N$ is the
//! maximum `num_vars` across all claims.

use ark_serialize::CanonicalSerialize;
use jolt_field::Field;
use jolt_transcript::FsTranscript;

use crate::append_sumcheck_claim;
use crate::claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
use crate::committed::{CommittedSumcheckConsistency, CommittedSumcheckProof};
use crate::domain::{BooleanHypercube, SumcheckDomain};
use crate::error::SumcheckError;
use crate::proof::{ClearProof, CompressedSumcheckProof, SumcheckProof};
use crate::round_proof::ClearRound;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedEvaluationClaim<F: Field> {
    pub reduction: EvaluationClaim<F>,
    pub batching_coefficients: Vec<F>,
    pub max_num_vars: usize,
    pub max_degree: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedCommittedSumcheckConsistency<F: Field, C> {
    pub consistency: CommittedSumcheckConsistency<F, C>,
    pub batching_coefficients: Vec<F>,
    pub max_num_vars: usize,
    pub max_degree: usize,
}

impl<F: Field> BatchedEvaluationClaim<F> {
    #[must_use]
    pub const fn round_offset(&self, num_vars: usize) -> usize {
        self.max_num_vars - num_vars
    }

    /// Returns the front-padding offset for an instance with `num_vars`.
    ///
    /// Batched verification front-loads dummy rounds for smaller instances, so
    /// an instance with `num_vars` is evaluated on the suffix beginning at
    /// `max_num_vars - num_vars`.
    pub fn try_round_offset(&self, num_vars: usize) -> Result<usize, SumcheckError<F>> {
        self.max_num_vars
            .checked_sub(num_vars)
            .ok_or(SumcheckError::BatchedPointOutOfRange {
                offset: 0,
                num_vars,
                total: self.reduction.point.len(),
            })
    }

    #[must_use]
    pub fn instance_point(&self, num_vars: usize) -> &[F] {
        let offset = self.round_offset(num_vars);
        &self.reduction.point.as_slice()[offset..offset + num_vars]
    }

    /// Returns the suffix challenge slice for an instance with `num_vars`.
    pub fn try_instance_point(&self, num_vars: usize) -> Result<&[F], SumcheckError<F>> {
        self.try_instance_point_at(self.try_round_offset(num_vars)?, num_vars)
    }

    /// Returns a challenge slice starting at `offset`.
    ///
    /// This is useful for protocols whose instance point is embedded inside the
    /// batched challenge vector but not necessarily at the canonical suffix
    /// offset.
    pub fn try_instance_point_at(
        &self,
        offset: usize,
        num_vars: usize,
    ) -> Result<&[F], SumcheckError<F>> {
        let end = offset
            .checked_add(num_vars)
            .ok_or(SumcheckError::BatchedPointRangeOverflow { offset, num_vars })?;
        self.reduction.point.as_slice().get(offset..end).ok_or(
            SumcheckError::BatchedPointOutOfRange {
                offset,
                num_vars,
                total: self.reduction.point.len(),
            },
        )
    }
}

impl<F: Field, C> BatchedCommittedSumcheckConsistency<F, C> {
    #[must_use]
    pub const fn round_offset(&self, num_vars: usize) -> usize {
        self.max_num_vars - num_vars
    }

    /// Returns the front-padding offset for an instance with `num_vars`.
    ///
    /// Batched committed verification uses the same front-loaded dummy-round
    /// layout as clear batched verification, but it exposes only transcript
    /// challenges and commitments, not scalar evaluation claims.
    pub fn try_round_offset(&self, num_vars: usize) -> Result<usize, SumcheckError<F>> {
        self.max_num_vars
            .checked_sub(num_vars)
            .ok_or(SumcheckError::BatchedPointOutOfRange {
                offset: 0,
                num_vars,
                total: self.consistency.rounds.len(),
            })
    }

    pub fn challenges(&self) -> Vec<F>
    where
        F: Copy,
    {
        self.consistency
            .rounds
            .iter()
            .map(|round| round.challenge)
            .collect()
    }

    /// Returns the suffix challenge vector for an instance with `num_vars`.
    pub fn try_instance_point(&self, num_vars: usize) -> Result<Vec<F>, SumcheckError<F>>
    where
        F: Copy,
    {
        self.try_instance_point_at(self.try_round_offset(num_vars)?, num_vars)
    }

    /// Returns a challenge vector starting at `offset`.
    ///
    /// This is useful for protocols whose instance point is embedded inside the
    /// batched challenge vector but not necessarily at the canonical suffix
    /// offset.
    pub fn try_instance_point_at(
        &self,
        offset: usize,
        num_vars: usize,
    ) -> Result<Vec<F>, SumcheckError<F>>
    where
        F: Copy,
    {
        let end = offset
            .checked_add(num_vars)
            .ok_or(SumcheckError::BatchedPointRangeOverflow { offset, num_vars })?;
        self.consistency
            .rounds
            .get(offset..end)
            .ok_or(SumcheckError::BatchedPointOutOfRange {
                offset,
                num_vars,
                total: self.consistency.rounds.len(),
            })
            .map(|rounds| rounds.iter().map(|round| round.challenge).collect())
    }
}

/// Batched sumcheck verifier.
///
/// Recomputes the combined claim with the same scaling and batching
/// coefficients as the prover, then delegates to the single-instance
/// verifier.
pub struct BatchedSumcheckVerifier;

impl BatchedSumcheckVerifier {
    /// Verifies a batched sumcheck proof over `domain`.
    ///
    /// Returns an [`EvaluationClaim`] `{ point: r, value: v }` on success,
    /// where `v` is the combined final evaluation and `r` is the full
    /// challenge vector of length `max(num_vars)`.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if verification fails.
    #[tracing::instrument(skip_all, name = "BatchedSumcheckVerifier::verify")]
    pub fn verify<F, T, P, D>(
        claims: &[SumcheckClaim<F>],
        round_proofs: &[P],
        domain: D,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field,
        T: FsTranscript<F>,
        P: ClearRound<F>,
        D: SumcheckDomain<F>,
    {
        let (first, rest) = claims.split_first().ok_or(SumcheckError::EmptyClaims)?;
        let max_num_vars = rest
            .iter()
            .fold(first.num_vars, |acc, c| acc.max(c.num_vars));
        let max_degree = rest.iter().fold(first.degree, |acc, c| acc.max(c.degree));

        // Fiat-Shamir: absorb claimed sums (must match prover).
        for claim in claims {
            transcript.absorb_field(&claim.claimed_sum);
        }

        let alpha: F = transcript.challenge();

        // Running power of alpha: alpha^j for j = 0, 1, 2, …
        let mut alpha_pow = F::one();
        let mut combined_sum = F::zero();
        for claim in claims {
            let scaled = domain.scale_padding(claim.claimed_sum, max_num_vars - claim.num_vars)?;
            combined_sum += alpha_pow * scaled;
            alpha_pow *= alpha;
        }

        let combined_claim = SumcheckClaim {
            num_vars: max_num_vars,
            degree: max_degree,
            claimed_sum: combined_sum,
        };

        crate::verifier::SumcheckVerifier::verify(&combined_claim, round_proofs, domain, transcript)
    }

    #[tracing::instrument(skip_all, name = "BatchedSumcheckVerifier::verify_compressed")]
    pub fn verify_compressed<F, T>(
        claims: &[SumcheckClaim<F>],
        proof: &CompressedSumcheckProof<F>,
        transcript: &mut T,
    ) -> Result<BatchedEvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field,
        T: FsTranscript<F>,
    {
        let statement = Self::batch_claim_statement(claims)?;
        let max_num_vars = statement.num_vars;
        let max_degree = statement.degree;

        for claim in claims {
            append_sumcheck_claim(transcript, &claim.claimed_sum);
        }
        let batching_coefficients = Self::batching_coefficients(claims.len(), transcript);

        let claimed_sum = claims
            .iter()
            .zip(&batching_coefficients)
            .map(|(claim, coefficient)| {
                claim.claimed_sum.mul_pow_2(max_num_vars - claim.num_vars) * *coefficient
            })
            .sum();

        let combined_claim = SumcheckClaim {
            num_vars: max_num_vars,
            degree: max_degree,
            claimed_sum,
        };
        let reduction = crate::verifier::SumcheckVerifier::verify_compressed(
            &combined_claim,
            proof,
            BooleanHypercube,
            transcript,
        )?;

        Ok(BatchedEvaluationClaim {
            reduction,
            batching_coefficients,
            max_num_vars,
            max_degree,
        })
    }

    #[tracing::instrument(skip_all, name = "BatchedSumcheckVerifier::verify_compressed_boolean")]
    pub fn verify_compressed_boolean<F, C, T>(
        claims: &[SumcheckClaim<F>],
        proof: &SumcheckProof<F, C>,
        transcript: &mut T,
    ) -> Result<BatchedEvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field,
        C: Clone + CanonicalSerialize,
        T: FsTranscript<F>,
    {
        match proof {
            SumcheckProof::Clear(ClearProof::Compressed(proof)) => {
                Self::verify_compressed(claims, proof, transcript)
            }
            SumcheckProof::Clear(ClearProof::Full(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "compressed clear",
                got: "full clear",
            }),
            SumcheckProof::Committed(_) => Err(SumcheckError::WrongProofEncoding {
                expected: "compressed clear",
                got: "committed",
            }),
        }
    }

    /// Checks batched committed-proof consistency through the transcript.
    ///
    /// This path is used when BlindFold verifies the hidden claim relations. It
    /// takes only public statements, not [`SumcheckClaim`] values, so it never
    /// absorbs `claim.claimed_sum` into the transcript. The caller must absorb
    /// commitments to those claim scalars before calling this method; batching
    /// coefficients are squeezed first here, and without that prior binding the
    /// prover could choose claim openings after seeing the batching challenge.
    pub fn verify_committed_consistency<F, C, T>(
        statements: &[SumcheckStatement],
        proof: &SumcheckProof<F, C>,
        transcript: &mut T,
    ) -> Result<BatchedCommittedSumcheckConsistency<F, C>, SumcheckError<F>>
    where
        F: Field,
        C: Clone + CanonicalSerialize,
        T: FsTranscript<F>,
    {
        match proof {
            SumcheckProof::Committed(proof) => {
                Self::verify_committed_consistency_for_proof(statements, proof, transcript)
            }
            SumcheckProof::Clear(ClearProof::Full(_)) => Err(SumcheckError::WrongProofEncoding {
                expected: "committed",
                got: "full clear",
            }),
            SumcheckProof::Clear(ClearProof::Compressed(_)) => {
                Err(SumcheckError::WrongProofEncoding {
                    expected: "committed",
                    got: "compressed clear",
                })
            }
        }
    }

    fn verify_committed_consistency_for_proof<F, C, T>(
        statements: &[SumcheckStatement],
        proof: &CommittedSumcheckProof<C>,
        transcript: &mut T,
    ) -> Result<BatchedCommittedSumcheckConsistency<F, C>, SumcheckError<F>>
    where
        F: Field,
        C: Clone + CanonicalSerialize,
        T: FsTranscript<F>,
    {
        let statement = Self::batch_statement(statements)?;
        let batching_coefficients = Self::batching_coefficients(statements.len(), transcript);
        let consistency = proof.verify_committed_consistency(statement, transcript)?;

        Ok(BatchedCommittedSumcheckConsistency {
            consistency,
            batching_coefficients,
            max_num_vars: statement.num_vars,
            max_degree: statement.degree,
        })
    }

    fn batch_claim_statement<F: Field>(
        claims: &[SumcheckClaim<F>],
    ) -> Result<SumcheckStatement, SumcheckError<F>> {
        let statements = claims
            .iter()
            .map(SumcheckClaim::statement)
            .collect::<Vec<_>>();
        Self::batch_statement(&statements)
    }

    fn batch_statement<F: Field>(
        statements: &[SumcheckStatement],
    ) -> Result<SumcheckStatement, SumcheckError<F>> {
        let (first, rest) = statements.split_first().ok_or(SumcheckError::EmptyClaims)?;
        let max_num_vars = rest
            .iter()
            .fold(first.num_vars, |acc, statement| acc.max(statement.num_vars));
        let max_degree = rest
            .iter()
            .fold(first.degree, |acc, statement| acc.max(statement.degree));

        Ok(SumcheckStatement::new(max_num_vars, max_degree))
    }

    fn batching_coefficients<F, T>(count: usize, transcript: &mut T) -> Vec<F>
    where
        F: Field,
        T: FsTranscript<F>,
    {
        (0..count)
            .map(|_| transcript.challenge_scalar())
            .collect::<Vec<_>>()
    }
}
