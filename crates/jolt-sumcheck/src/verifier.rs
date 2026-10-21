//! Sumcheck verifier: reads round polynomials from the proof and checks them
//! against the running sum.

use jolt_algebra::{CanonicalEncoding, Field};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::claim::{EvaluationClaim, SumcheckClaim};
use crate::domain::SumcheckDomain;
use crate::error::SumcheckError;
use crate::round_proof::{receive_compressed_round, receive_full_round};
use crate::schedule::DegreeSchedule;

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
        Self::verify_compressed_scheduled(
            claim.claimed_sum,
            &DegreeSchedule::uniform(claim.num_vars, claim.degree),
            transcript,
        )
    }

    /// [`verify_compressed`](Self::verify_compressed) with each round read at
    /// its own degree from `degrees`: round `j` receives `degrees[j]`
    /// coefficients, so a round is never longer than its schedule allows.
    pub fn verify_compressed_scheduled<F, H>(
        claimed_sum: F,
        degrees: &DegreeSchedule,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field + CanonicalEncoding,
        H: Sponge,
    {
        let mut running_sum = claimed_sum;
        let mut challenges = Vec::with_capacity(degrees.num_rounds());
        for &degree in degrees.degrees() {
            let round_poly = receive_compressed_round(degree, transcript)?;
            let r: F = transcript.challenge_small();
            running_sum = round_poly.evaluate_with_hint(running_sum, r);
            challenges.push(r);
        }
        Ok(EvaluationClaim::new(challenges, running_sum))
    }
}
