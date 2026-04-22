//! Sumcheck verifier: checks round polynomials against the claimed sum.

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};
use jolt_verifier_backend::{helpers::univariate_horner, FieldBackend};

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::round::RoundVerifier;

/// Stateless sumcheck verifier engine.
///
/// Replays the Fiat-Shamir transcript and checks each round against
/// the running sum, ultimately producing the final evaluation point
/// and expected value for an oracle query.
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verifies a sumcheck proof.
    ///
    /// For each round $i = 0, \ldots, n-1$:
    /// 1. The round verifier absorbs proof data into the transcript and
    ///    checks consistency (clear mode verifies `s_i(0) + s_i(1) == running_sum`;
    ///    committed mode defers to BlindFold).
    /// 2. A challenge $r_i$ is squeezed from the transcript.
    /// 3. The running sum is updated to $s_i(r_i)$.
    ///
    /// On success, returns `(v, r)` where `v` is the final evaluation
    /// and `r = (r_1, ..., r_n)` is the challenge vector.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if any round check fails, a degree bound
    /// is exceeded, or the proof has the wrong number of rounds.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify")]
    pub fn verify<F, T, V>(
        claim: &SumcheckClaim<F>,
        round_proofs: &[V::RoundProof],
        transcript: &mut T,
        verifier: &V,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
        V: RoundVerifier<F>,
    {
        if round_proofs.len() != claim.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: claim.num_vars,
                got: round_proofs.len(),
            });
        }

        let mut running_sum = claim.claimed_sum;
        let mut challenges = Vec::with_capacity(claim.num_vars);

        for (round, round_proof) in round_proofs.iter().enumerate() {
            verifier.absorb_and_check(round_proof, running_sum, claim.degree, round, transcript)?;
            let r: F = transcript.challenge();
            running_sum = verifier.next_running_sum(round_proof, r);
            challenges.push(r);
        }

        Ok((running_sum, challenges))
    }

    /// Verifies a single-instance cleartext sumcheck proof through a
    /// [`FieldBackend`].
    ///
    /// Mirrors the [`ClearRoundVerifier::with_label_compressed`](crate::round::ClearRoundVerifier::with_label_compressed)
    /// transcript shape — every absorption goes through the backend, but
    /// the bytes appended to `transcript` are the same as the native path
    /// so Fiat-Shamir stays in sync with the prover.
    ///
    /// Returns `(final_eval_w, challenges_w, challenges_f)`:
    ///
    /// - `final_eval_w` is the running sum after the last round, expressed
    ///   as a backend scalar (suitable for chaining into a downstream
    ///   `assert_eq` against the sumcheck's output composition).
    /// - `challenges_w` is the round-by-round challenge vector wrapped
    ///   through the backend.
    /// - `challenges_f` is the same vector as raw [`B::F`] values; needed
    ///   by callers that drive the Fiat-Shamir transcript or feed legacy
    ///   (non-backend) helpers downstream.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if any round's degree bound is exceeded
    /// or the consistency check `s(0) + s(1) == running_sum` fails. The
    /// backend's [`assert_eq`](FieldBackend::assert_eq) is the consistency
    /// gate, so backends that record (Tracing) rather than enforce
    /// (Native) will return `Ok` even when the underlying field values
    /// disagree — replay catches the violation.
    #[expect(clippy::type_complexity, reason = "triple-return for downstream wiring")]
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify_with_backend")]
    pub fn verify_with_backend<B, T>(
        backend: &mut B,
        claim: &SumcheckClaim<B::F>,
        round_polys: &[UnivariatePoly<B::F>],
        running_sum_w: B::Scalar,
        transcript: &mut T,
        label: Option<&'static [u8]>,
        compressed: bool,
    ) -> Result<(B::Scalar, Vec<B::Scalar>, Vec<B::F>), SumcheckError>
    where
        B: FieldBackend,
        T: Transcript<Challenge = B::F>,
    {
        if round_polys.len() != claim.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: claim.num_vars,
                got: round_polys.len(),
            });
        }

        let zero_w = backend.const_zero();
        let one_w = backend.const_one();

        let mut running_w = running_sum_w;
        let mut challenges_w: Vec<B::Scalar> = Vec::with_capacity(claim.num_vars);
        let mut challenges_f: Vec<B::F> = Vec::with_capacity(claim.num_vars);

        for (round, round_proof) in round_polys.iter().enumerate() {
            if round_proof.degree() > claim.degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: round_proof.degree(),
                    max: claim.degree,
                });
            }

            let coeffs_f = round_proof.coefficients();
            let coeffs_w: Vec<B::Scalar> = coeffs_f
                .iter()
                .map(|c| backend.wrap_proof(*c, "round_poly_coeff"))
                .collect();

            let s_at_zero = univariate_horner(backend, &coeffs_w, &zero_w);
            let s_at_one = univariate_horner(backend, &coeffs_w, &one_w);
            let sum_w = backend.add(&s_at_zero, &s_at_one);

            backend
                .assert_eq(&sum_w, &running_w, "sumcheck round consistency")
                .map_err(|e| SumcheckError::RoundCheckFailed {
                    round,
                    expected: format!("running_sum (round {round})"),
                    actual: e.to_string(),
                })?;

            // Mirror ClearRoundVerifier transcript absorption (compressed
            // skips coeff index 1; LabelWithCount gets the on-wire length).
            if compressed {
                let compressed_len = coeffs_f.len().saturating_sub(1);
                if let Some(lbl) = label {
                    transcript.append(&LabelWithCount(lbl, compressed_len as u64));
                }
                if let Some(c0) = coeffs_f.first() {
                    c0.append_to_transcript(transcript);
                }
                for c in coeffs_f.iter().skip(2) {
                    c.append_to_transcript(transcript);
                }
            } else {
                if let Some(lbl) = label {
                    transcript.append(&LabelWithCount(lbl, coeffs_f.len() as u64));
                }
                for c in coeffs_f {
                    c.append_to_transcript(transcript);
                }
            }

            let r_f: B::F = transcript.challenge();
            let r_w = backend.wrap_challenge(r_f, "sumcheck_r");

            running_w = univariate_horner(backend, &coeffs_w, &r_w);
            challenges_w.push(r_w);
            challenges_f.push(r_f);
        }

        Ok((running_w, challenges_w, challenges_f))
    }
}
