//! Batched sumcheck: reduces multiple claims into one via random linear
//! combination.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_transcript::Transcript;

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::proof::BatchedSumcheckProof;
use crate::prover::SumcheckWitness;

/// Batched sumcheck prover that combines $m$ independent claims.
///
/// Given claims $C_0, \ldots, C_{m-1}$ over polynomials $g_0, \ldots, g_{m-1}$,
/// draws a batching coefficient $\alpha$ from the transcript and proves the
/// combined claim:
///
/// $$\sum_{x \in \{0,1\}^n} \sum_{j=0}^{m-1} \alpha^j \cdot g_j(x) = \sum_{j} \alpha^j \cdot C_j$$
///
/// All claims must share the same `num_vars` and `degree`.
pub struct BatchedSumcheckProver;

impl BatchedSumcheckProver {
    /// Proves a batch of sumcheck claims.
    ///
    /// # Parameters
    ///
    /// * `claims` -- the individual sumcheck claims to batch.
    /// * `witnesses` -- mutable witnesses, one per claim.
    /// * `transcript` -- Fiat-Shamir transcript.
    /// * `challenge_fn` -- converts transcript challenges to field elements.
    ///
    /// # Panics
    ///
    /// Panics if `claims` is empty, or if `claims` and `witnesses` have
    /// different lengths, or if not all claims share the same `num_vars`
    /// and `degree`.
    pub fn prove<F, T>(
        claims: &[SumcheckClaim<F>],
        witnesses: &mut [Box<dyn SumcheckWitness<F>>],
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
    ) -> BatchedSumcheckProof<F>
    where
        F: Field,
        T: Transcript,
    {
        assert!(!claims.is_empty(), "must have at least one claim");
        assert_eq!(
            claims.len(),
            witnesses.len(),
            "claims and witnesses must have the same length"
        );
        let num_vars = claims[0].num_vars;
        let degree = claims[0].degree;
        for c in claims {
            assert_eq!(c.num_vars, num_vars, "all claims must share num_vars");
            assert_eq!(c.degree, degree, "all claims must share degree");
        }

        // Draw batching coefficient alpha
        let alpha = challenge_fn(transcript.challenge());

        // Compute combined claimed sum: sum_j alpha^j * C_j
        let combined_sum = claims
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (j, claim)| {
                acc + pow(alpha, j) * claim.claimed_sum
            });

        let combined_claim = SumcheckClaim {
            num_vars,
            degree,
            claimed_sum: combined_sum,
        };

        let mut batched_witness = BatchedWitness {
            witnesses,
            alpha,
            degree,
        };

        let proof = crate::prover::SumcheckProver::prove(
            &combined_claim,
            &mut batched_witness,
            transcript,
            challenge_fn,
        );

        BatchedSumcheckProof { proof }
    }
}

/// Batched sumcheck verifier.
pub struct BatchedSumcheckVerifier;

impl BatchedSumcheckVerifier {
    /// Verifies a batched sumcheck proof.
    ///
    /// Recomputes the combined claim from the individual claims and the
    /// batching coefficient $\alpha$ squeezed from the transcript, then
    /// delegates to the single-instance verifier.
    ///
    /// Returns `(v, \mathbf{r})$ on success, where $v$ is the combined
    /// final evaluation and $\mathbf{r}$ is the challenge vector.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if the underlying single-instance
    /// verification fails.
    pub fn verify<F, T>(
        claims: &[SumcheckClaim<F>],
        proof: &BatchedSumcheckProof<F>,
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript,
    {
        assert!(!claims.is_empty(), "must have at least one claim");
        let num_vars = claims[0].num_vars;
        let degree = claims[0].degree;

        let alpha = challenge_fn(transcript.challenge());

        let combined_sum = claims
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (j, claim)| {
                acc + pow(alpha, j) * claim.claimed_sum
            });

        let combined_claim = SumcheckClaim {
            num_vars,
            degree,
            claimed_sum: combined_sum,
        };

        crate::verifier::SumcheckVerifier::verify(
            &combined_claim,
            &proof.proof,
            transcript,
            challenge_fn,
        )
    }
}

/// Witness adapter that linearly combines multiple witnesses with powers
/// of $\alpha$.
struct BatchedWitness<'a, F: Field> {
    witnesses: &'a mut [Box<dyn SumcheckWitness<F>>],
    alpha: F,
    degree: usize,
}

impl<F: Field> SumcheckWitness<F> for BatchedWitness<'_, F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        // Evaluate each witness's round polynomial at points 0, 1, ..., degree
        // and combine with powers of alpha.
        let num_points = self.degree + 1;
        let mut combined_evals = vec![F::zero(); num_points];

        let mut alpha_power = F::one();
        for witness in self.witnesses.iter() {
            let poly = witness.round_polynomial();
            for (t, combined) in combined_evals.iter_mut().enumerate() {
                *combined += alpha_power * poly.evaluate(F::from_u64(t as u64));
            }
            alpha_power *= self.alpha;
        }

        // Interpolate the combined evaluations
        let points: Vec<(F, F)> = combined_evals
            .into_iter()
            .enumerate()
            .map(|(t, y)| (F::from_u64(t as u64), y))
            .collect();

        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: F) {
        for witness in self.witnesses.iter_mut() {
            witness.bind(challenge);
        }
    }
}

// SAFETY: The inner witnesses are already Send + Sync via the trait bound.
// The BatchedWitness just holds a mutable reference to them plus Copy fields.
unsafe impl<F: Field> Send for BatchedWitness<'_, F> {}
// SAFETY: Same reasoning as Send — alpha and degree are Copy, witnesses
// are behind a mutable reference to a slice of Send+Sync trait objects.
unsafe impl<F: Field> Sync for BatchedWitness<'_, F> {}

/// Computes $\text{base}^{\text{exp}}$ by repeated squaring.
#[inline]
fn pow<F: Field>(base: F, exp: usize) -> F {
    if exp == 0 {
        return F::one();
    }
    let mut result = F::one();
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result *= b;
        }
        b = b.square();
        e >>= 1;
    }
    result
}
