use jolt_field::Field;
use jolt_poly::UnivariatePolynomial;
use jolt_transcript::{Encoding, FieldTranscript};

use crate::claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
use crate::committed::{
    CommittedRound, CommittedSumcheckConsistency, CommittedSumcheckProof, VerifiedCommittedRound,
};
use crate::domain::{BooleanHypercube, SumcheckDomain};
use crate::error::SumcheckError;
use crate::pow::RoundNonces;
use crate::proof::CompressedSumcheckProof;
use crate::round_proof::{ClearRound, RoundMessage};

pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verifies a sumcheck proof.
    ///
    /// For each round $i = 0, \ldots, n-1$:
    /// 1. The degree bound is enforced against `claim.degree`.
    /// 2. The round sum is checked against the running sum in `domain`.
    /// 3. The round message is absorbed into the transcript.
    /// 4. The round's proof-of-work nonce from `pow_nonces` is checked
    ///    (one per round when the difficulty for `claim.degree` is positive,
    ///    none otherwise).
    /// 5. A challenge $r_i$ is squeezed from the transcript.
    /// 6. The running sum is updated to the round polynomial at $r_i$.
    ///
    /// On success, returns an [`EvaluationClaim`] `{ point: r, value: v }`
    /// where `v` is the final evaluation and `r = (r_1, ..., r_n)` is the
    /// challenge vector.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if any round check fails, a degree bound
    /// is exceeded, a proof-of-work nonce is missing or wrong, or the proof
    /// has the wrong number of rounds or nonces.
    ///
    /// # Soundness
    ///
    /// When `claim.num_vars == 0`, this function performs no transcript
    /// interaction and no checks: it returns
    /// `EvaluationClaim { point: Point::default(), value: claim.claimed_sum }`.
    /// Sumcheck trivially reduces to a single oracle query at that point,
    /// so the caller MUST verify `claim.claimed_sum` against the
    /// commitment/oracle layer to retain soundness.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify")]
    pub fn verify<F, T, R, D>(
        claim: &SumcheckClaim<F>,
        round_proofs: &[R],
        pow_nonces: &[u32],
        domain: D,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field,
        T: FieldTranscript<F>,
        R: ClearRound<F>,
        D: SumcheckDomain<F>,
    {
        if round_proofs.len() != claim.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: claim.num_vars,
                got: round_proofs.len(),
            });
        }
        let pow = RoundNonces::new::<F, T>(claim.degree, claim.num_vars, pow_nonces)?;

        let mut running_sum = claim.claimed_sum;
        let mut challenges = Vec::with_capacity(claim.num_vars);

        for (round, round_proof) in round_proofs.iter().enumerate() {
            if round_proof.degree() > claim.degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: round_proof.degree(),
                    max: claim.degree,
                });
            }
            domain.check_round_sum(round, running_sum, round_proof)?;
            round_proof.append_to_transcript(transcript);
            pow.check(round, transcript)?;
            let r: F = transcript.challenge()?;
            running_sum = round_proof.evaluate(r);
            challenges.push(r);
        }

        Ok(EvaluationClaim::new(challenges, running_sum))
    }

    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify_compressed")]
    pub fn verify_compressed<F, T>(
        claim: &SumcheckClaim<F>,
        proof: &CompressedSumcheckProof<F>,
        domain: BooleanHypercube,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field + Encoding,
        T: FieldTranscript<F>,
    {
        if proof.round_polynomials.len() != claim.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: claim.num_vars,
                got: proof.round_polynomials.len(),
            });
        }
        let pow = RoundNonces::new::<F, T>(claim.degree, claim.num_vars, &proof.pow_nonces)?;
        let BooleanHypercube = domain;

        let mut running_sum = claim.claimed_sum;
        let mut challenges = Vec::with_capacity(claim.num_vars);

        for (round, round_proof) in proof.round_polynomials.iter().enumerate() {
            if round_proof.degree() > claim.degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: round_proof.degree(),
                    max: claim.degree,
                });
            }
            let coeffs = round_proof.coeffs_except_linear_term();
            if coeffs.is_empty() {
                return Err(SumcheckError::CompressedPolynomialTooShort { round, got: 0 });
            }

            // The count word: the prover chooses the degree up to the bound.
            transcript.append(&(coeffs.len() as u64));
            transcript.append_values(coeffs);
            pow.check(round, transcript)?;
            let r: F = transcript.challenge()?;
            running_sum = round_proof.evaluate_with_hint(running_sum, r);
            challenges.push(r);
        }

        Ok(EvaluationClaim::new(challenges, running_sum))
    }

    /// Checks committed sumcheck rounds and returns the transcript-derived data.
    #[tracing::instrument(
        skip_all,
        name = "SumcheckVerifier::verify_committed_round_consistency"
    )]
    pub fn verify_committed_round_consistency<F, T, C>(
        statement: SumcheckStatement,
        round_proofs: &[CommittedRound<C>],
        pow_nonces: &[u32],
        transcript: &mut T,
    ) -> Result<CommittedSumcheckConsistency<F, C>, SumcheckError<F>>
    where
        F: Field,
        T: FieldTranscript<F>,
        C: Clone + Encoding,
    {
        if round_proofs.len() != statement.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: statement.num_vars,
                got: round_proofs.len(),
            });
        }
        // Priced at the statement's degree bound, not the per-round degree the
        // prover declares in each `CommittedRound`.
        let pow = RoundNonces::new::<F, T>(statement.degree, statement.num_vars, pow_nonces)?;

        let mut rounds = Vec::with_capacity(statement.num_vars);
        for (round, round_proof) in round_proofs.iter().enumerate() {
            if round_proof.degree() > statement.degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: round_proof.degree(),
                    max: statement.degree,
                });
            }

            round_proof.append_to_transcript(transcript);
            pow.check(round, transcript)?;
            rounds.push(VerifiedCommittedRound {
                commitment: round_proof.commitment.clone(),
                degree: round_proof.degree,
                challenge: transcript.challenge()?,
            });
        }

        Ok(CommittedSumcheckConsistency { rounds })
    }
}

impl<F> CompressedSumcheckProof<F>
where
    F: Field + Encoding,
{
    pub fn verify<T>(
        &self,
        claim: &SumcheckClaim<F>,
        domain: BooleanHypercube,
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        T: FieldTranscript<F>,
    {
        SumcheckVerifier::verify_compressed(claim, self, domain, transcript)
    }
}

impl<C> CommittedSumcheckProof<C> {
    /// Checks committed-proof consistency through the Fiat-Shamir transcript.
    ///
    /// This checks the round count, degree bounds and proof-of-work nonces,
    /// derives the round challenges, and absorbs the committed output claims
    /// after the round transcript.
    pub fn verify_committed_consistency<F, T>(
        &self,
        statement: SumcheckStatement,
        transcript: &mut T,
    ) -> Result<CommittedSumcheckConsistency<F, C>, SumcheckError<F>>
    where
        F: Field,
        T: FieldTranscript<F>,
        C: Clone + Encoding,
    {
        let consistency = SumcheckVerifier::verify_committed_round_consistency(
            statement,
            &self.rounds,
            &self.pow_nonces,
            transcript,
        )?;
        transcript.append(&self.output_claims);
        Ok(consistency)
    }
}
