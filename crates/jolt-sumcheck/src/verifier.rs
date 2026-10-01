//! Sumcheck verifier: reads round polynomials from the proof and checks them
//! against the running sum.

use jolt_field::{CanonicalDecode, CanonicalEncoding, Field};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
use crate::committed::{
    CommittedOutputClaims, CommittedSumcheckConsistency, VerifiedCommittedRound,
};
use crate::domain::SumcheckDomain;
use crate::error::SumcheckError;
use crate::round_proof::{receive_compressed_round, receive_full_round};

/// Stateless sumcheck verifier engine.
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verifies full-coefficient rounds over `domain`.
    ///
    /// For each round $i = 0, \ldots, n-1$:
    /// 1. The round polynomial's `claim.degree + 1` coefficients are received.
    /// 2. Its sum over `domain` is checked against the running sum.
    /// 3. A challenge $r_i$ is drawn from the small challenge set.
    /// 4. The running sum becomes the round polynomial at $r_i$.
    ///
    /// On success, returns an [`EvaluationClaim`] `{ point: r, value: v }`
    /// where `v` is the final evaluation and `r = (r_1, ..., r_n)`.
    ///
    /// # Soundness
    ///
    /// When `claim.num_vars == 0`, this reads nothing and returns
    /// `EvaluationClaim { point: Point::default(), value: claim.claimed_sum }`.
    /// The caller MUST verify `claim.claimed_sum` against the oracle layer.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify")]
    pub fn verify<F, H, D>(
        claim: &SumcheckClaim<F>,
        domain: D,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field + CanonicalEncoding,
        H: Sponge,
        D: SumcheckDomain<F>,
    {
        let mut running_sum = claim.claimed_sum;
        let mut challenges = Vec::with_capacity(claim.num_vars);
        for round in 0..claim.num_vars {
            let round_poly = receive_full_round(claim.degree, transcript)?;
            domain.check_round_sum(round, running_sum, &round_poly)?;
            let r: F = transcript.challenge_small();
            running_sum = round_poly.evaluate(r);
            challenges.push(r);
        }
        Ok(EvaluationClaim::new(challenges, running_sum))
    }

    /// Verifies compressed Boolean-hypercube rounds, recovering each round's
    /// linear coefficient from the running sum. Soundness as for
    /// [`verify`](Self::verify).
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify_compressed")]
    pub fn verify_compressed<F, H>(
        claim: &SumcheckClaim<F>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field + CanonicalEncoding,
        H: Sponge,
    {
        let mut running_sum = claim.claimed_sum;
        let mut challenges = Vec::with_capacity(claim.num_vars);
        for _ in 0..claim.num_vars {
            let round_poly = receive_compressed_round(claim.degree, transcript)?;
            let r: F = transcript.challenge_small();
            running_sum = round_poly.evaluate_with_hint(running_sum, r);
            challenges.push(r);
        }
        Ok(EvaluationClaim::new(challenges, running_sum))
    }

    /// Reads committed rounds of degree bound `statement.degree`, then the
    /// `num_output_commitments` output-claim row commitments.
    ///
    /// Each round commitment is followed by its challenge. Committed proofs
    /// reveal no claim scalars, so the round relations are left to BlindFold;
    /// this returns the commitments, degrees, and challenges it binds.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify_committed")]
    #[expect(
        clippy::type_complexity,
        reason = "the round consistency and the output-claim commitments are read together"
    )]
    pub fn verify_committed<F, H, C>(
        statement: SumcheckStatement,
        num_output_commitments: usize,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(CommittedSumcheckConsistency<F, C>, CommittedOutputClaims<C>), SumcheckError<F>>
    where
        F: Field + CanonicalEncoding,
        H: Sponge,
        C: CanonicalDecode,
    {
        let mut rounds = Vec::with_capacity(statement.num_vars);
        for _ in 0..statement.num_vars {
            let commitment = transcript.receive()?;
            rounds.push(VerifiedCommittedRound {
                commitment,
                degree: statement.degree,
                challenge: transcript.challenge_small(),
            });
        }
        let output_claims = CommittedOutputClaims {
            commitments: transcript.receive_n(num_output_commitments)?,
        };
        Ok((CommittedSumcheckConsistency { rounds }, output_claims))
    }
}
