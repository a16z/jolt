//! Prover pipeline: stage loop and sumcheck orchestration.
//!
//! [`prove_stages`] drives the proving pipeline over a sequence of
//! [`ProverStage`](crate::stage::ProverStage) implementations.

use jolt_field::Field;
use jolt_openings::ProverClaim;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::handler::RoundHandler;
use jolt_sumcheck::{BatchedSumcheckProver, ClearRoundHandler, SumcheckProof};
use jolt_transcript::Transcript;

use crate::stage::ProverStage;

/// Wrapper handler that captures Fiat-Shamir challenges alongside the proof.
///
/// Delegates all round handling to [`ClearRoundHandler`] but records each
/// challenge via [`on_challenge`](RoundHandler::on_challenge).
struct CaptureHandler<F: Field> {
    inner: ClearRoundHandler<F>,
    challenges: Vec<F>,
}

impl<F: Field> CaptureHandler<F> {
    fn new(capacity: usize) -> Self {
        Self {
            inner: ClearRoundHandler::with_capacity(capacity),
            challenges: Vec::with_capacity(capacity),
        }
    }
}

impl<F: Field> RoundHandler<F> for CaptureHandler<F> {
    type Proof = (SumcheckProof<F>, Vec<F>);

    fn absorb_round_poly(&mut self, poly: &UnivariatePoly<F>, transcript: &mut impl Transcript) {
        self.inner.absorb_round_poly(poly, transcript);
    }

    fn on_challenge(&mut self, challenge: F) {
        self.challenges.push(challenge);
    }

    fn finalize(self) -> (SumcheckProof<F>, Vec<F>) {
        (self.inner.finalize(), self.challenges)
    }
}

/// Drives the proving pipeline, returning per-stage sumcheck proofs and
/// the accumulated opening claims.
///
/// For each stage:
/// 1. [`build()`](ProverStage::build) — construct claims and witnesses
/// 2. [`BatchedSumcheckProver::prove_with_handler`] — run the sumcheck
/// 3. [`extract_claims()`](ProverStage::extract_claims) — produce opening claims
///
/// Opening claims accumulate across stages and feed into each subsequent
/// stage's `build()` method, threading data through the pipeline.
pub fn prove_stages<F, T>(
    stages: &mut [Box<dyn ProverStage<F, T>>],
    transcript: &mut T,
    challenge_fn: impl Fn(T::Challenge) -> F + Copy,
) -> (Vec<SumcheckProof<F>>, Vec<ProverClaim<F>>)
where
    F: Field,
    T: Transcript,
{
    let mut all_opening_claims: Vec<ProverClaim<F>> = Vec::new();
    let mut stage_proofs: Vec<SumcheckProof<F>> = Vec::new();

    for stage in stages.iter_mut() {
        let mut batch = stage.build(&all_opening_claims, transcript);

        let max_rounds = batch
            .claims
            .iter()
            .map(|c| c.num_vars)
            .max()
            .unwrap_or(0);

        let handler = CaptureHandler::new(max_rounds);

        let (proof, challenges) = BatchedSumcheckProver::prove_with_handler(
            &batch.claims,
            &mut batch.witnesses,
            transcript,
            challenge_fn,
            handler,
        );

        // The final evaluation is the last round polynomial evaluated at
        // the last challenge.
        let final_eval = if let (Some(last_poly), Some(&last_challenge)) =
            (proof.round_polynomials.last(), challenges.last())
        {
            last_poly.evaluate(last_challenge)
        } else {
            F::zero()
        };

        let new_claims = stage.extract_claims(&challenges, final_eval);
        all_opening_claims.extend(new_claims);
        stage_proofs.push(proof);
    }

    (stage_proofs, all_opening_claims)
}
