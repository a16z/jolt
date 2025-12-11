use crate::poly::unipoly::UniPoly;
use crate::transcripts::Transcript;

use crate::{
    field::{JoltField, MaybeAllocative},
    poly::opening_proof::ProverOpeningAccumulator,
};

pub trait SumcheckInstanceProver<F: JoltField, T: Transcript>:
    Send + Sync + MaybeAllocative
{
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize;

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize;

    /// Returns the initial claim of this sumcheck instance.
    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F;

    /// Computes the prover's message for a specific round of the sumcheck protocol.
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F>;

    /// Ingest the verifier's challenge for a sumcheck round.
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize);

    /// Caches polynomial opening claims needed after the sumcheck protocol completes.
    /// These openings will later be proven using either an opening proof or another sumcheck.
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    );

    /// Returns trusted advice dimensions if this is a trusted advice polynomial.
    /// Returns `Some((log_rows, log_columns))` for trusted advice, `None` otherwise.
    /// For trusted advice polynomials, binding happens in two separate phases:
    /// - First phase: bind the row variables (last `log_rows` of the row rounds)
    /// - Second phase: bind the column variables (last `log_columns` of the column rounds)
    fn trusted_advice_dimensions(&self) -> Option<(usize, usize)> {
        None
    }

    /// Returns a debug name for this sumcheck instance (for logging purposes).
    fn debug_name(&self) -> String {
        "unknown".to_string()
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder);
}

/// Trait for a single-round instance of univariate skip
/// We make a number of assumptions for the usage of this trait currently:
/// 1. There is only one univariate skip round, which happens at the beginning of a sumcheck stage
/// 2. We do not bind anything after this round. Instead during the remaining sumcheck, we
///    will stream from the trace again to initialize.
/// 3. We assume that the domain is symmetric around zero, and the prover sends the entire
///    (univariate) polynomial for this round
pub trait UniSkipFirstRoundInstanceProver<F: JoltField, T: Transcript>:
    Send + Sync + MaybeAllocative
{
    /// Returns the initial claim of this univariate skip round, i.e.
    /// input_claim = \sum_{-floor(N/2) <= z <= ceil(N/2)} \sum_{x \in \{0, 1}^n} P(z, x)
    /// where N is the domain size (one more than the degree of univariate skip)
    fn input_claim(&self) -> F;

    /// Computes the full univariate polynomial to be sent in the uni-skip round.
    /// Returns a degree-bounded `UniPoly` with exactly `DEGREE_BOUND + 1` coefficients.
    fn compute_poly(&mut self) -> UniPoly<F>;
}
