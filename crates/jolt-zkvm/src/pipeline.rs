//! Prover pipeline: stage loop and sumcheck orchestration.
//!
//! [`prove_stages`] drives the proving pipeline over a sequence of
//! [`ProverStage`] implementations.

use jolt_field::Field;
use jolt_openings::ProverClaim;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::handler::RoundHandler;
use jolt_sumcheck::{BatchedSumcheckProver, ClearRoundHandler, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::proof::SumcheckStageProof;
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

/// Creates a tracing span whose name in Perfetto matches the stage's
/// `name()` string. Since `tracing::info_span!` requires a string literal,
/// we dispatch through a match on known stage names.
macro_rules! stage_span {
    ($name:expr, $claims:expr, $rounds:expr) => {
        match $name {
            "S2_ra_virtual" => {
                tracing::info_span!("S2_ra_virtual", claims = $claims, rounds = $rounds)
            }
            "S3_claim_reductions" => {
                tracing::info_span!("S3_claim_reductions", claims = $claims, rounds = $rounds)
            }
            "S4_rw_checking" => {
                tracing::info_span!("S4_rw_checking", claims = $claims, rounds = $rounds)
            }
            "S4_ram_rw_checking" => {
                tracing::info_span!("S4_ram_rw_checking", claims = $claims, rounds = $rounds)
            }
            "S5_ram_checking" => {
                tracing::info_span!("S5_ram_checking", claims = $claims, rounds = $rounds)
            }
            "S6_booleanity" => {
                tracing::info_span!("S6_booleanity", claims = $claims, rounds = $rounds)
            }
            "S7_hamming_reduction" => {
                tracing::info_span!("S7_hamming_reduction", claims = $claims, rounds = $rounds)
            }
            "S2_product_virtual" => {
                tracing::info_span!("S2_product_virtual", claims = $claims, rounds = $rounds)
            }
            other => tracing::info_span!("stage", name = other, claims = $claims, rounds = $rounds),
        }
    };
}

/// Drives the proving pipeline, returning per-stage sumcheck proofs (with
/// evaluations) and the accumulated opening claims.
///
/// For each stage:
/// 1. [`build()`](ProverStage::build) — construct claims and witnesses
/// 2. [`BatchedSumcheckProver::prove_with_handler`] — run the sumcheck
/// 3. [`extract_claims()`](ProverStage::extract_claims) — produce opening claims
///    and capture per-polynomial evaluations for the proof
///
/// Opening claims accumulate across stages and feed into each subsequent
/// stage's `build()` method, threading data through the pipeline.
#[tracing::instrument(skip_all, name = "prove_stages")]
pub fn prove_stages<F, T>(
    stages: &mut [Box<dyn ProverStage<F, T>>],
    transcript: &mut T,
    challenge_fn: impl Fn(T::Challenge) -> F + Copy,
) -> (Vec<SumcheckStageProof<F>>, Vec<ProverClaim<F>>)
where
    F: Field,
    T: Transcript,
{
    let mut all_opening_claims: Vec<ProverClaim<F>> = Vec::new();
    let mut stage_proofs: Vec<SumcheckStageProof<F>> = Vec::new();

    for stage in stages.iter_mut() {
        let stage_name = stage.name();

        let mut batch = {
            let _build_span = tracing::info_span!("build", stage = stage_name).entered();
            stage.build(&all_opening_claims, transcript)
        };

        let max_rounds = batch.claims.iter().map(|c| c.num_vars).max().unwrap_or(0);
        let num_claims = batch.claims.len();
        let degrees: Vec<usize> = batch.claims.iter().map(|c| c.degree).collect();

        let stage_span = stage_span!(stage_name, num_claims, max_rounds);
        let _stage_guard = stage_span.enter();

        tracing::info!(claims = num_claims, max_rounds, ?degrees, "sumcheck start");

        let handler = CaptureHandler::new(max_rounds);

        let (proof, challenges) = {
            let _sc_span = tracing::info_span!("sumcheck", rounds = max_rounds).entered();
            BatchedSumcheckProver::prove_with_handler(
                &batch.claims,
                &mut batch.witnesses,
                transcript,
                challenge_fn,
                handler,
            )
        };

        let final_eval = if let (Some(last_poly), Some(&last_challenge)) =
            (proof.round_polynomials.last(), challenges.last())
        {
            last_poly.evaluate(last_challenge)
        } else {
            F::zero()
        };

        let new_claims = {
            let _extract_span = tracing::info_span!("extract_claims").entered();
            stage.extract_claims(&challenges, final_eval)
        };

        let evaluations: Vec<F> = new_claims.iter().map(|c| c.eval).collect();

        tracing::info!(opening_claims = new_claims.len(), "stage complete");

        // Fiat-Shamir: absorb opening claim evaluations before the next
        // stage derives its challenges. Matches the old pipeline's
        // `opening_accumulator.flush_to_transcript(transcript)`.
        for claim in &new_claims {
            claim.eval.append_to_transcript(transcript);
        }

        all_opening_claims.extend(new_claims);
        stage_proofs.push(SumcheckStageProof {
            sumcheck_proof: proof,
            evaluations,
        });
    }

    (stage_proofs, all_opening_claims)
}
