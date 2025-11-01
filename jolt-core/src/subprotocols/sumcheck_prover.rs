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
    /// Returns the evaluations of the sumcheck polynomial at 0, 2, 3, ..., degree.
    /// The point evaluation at 1 can be interpolated using the previous round's claim.
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F>;

    /// Binds this sumcheck instance to the verifier's challenge from a specific round.
    /// This updates the internal state to prepare for the next round.
    fn bind(&mut self, r_j: F::Challenge, round: usize);

    /// Caches polynomial opening claims needed after the sumcheck protocol completes.
    /// These openings will later be proven using either an opening proof or another sumcheck.
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    );

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
    /// The degree of the sum-check
    const DEGREE_BOUND: usize;

    /// The domain size of the sum-check. Canonically instantiated to the domain
    /// [-floor(DOMAIN_SIZE/2), ceil(DOMAIN_SIZE)/2]
    const DOMAIN_SIZE: usize;

    /// Returns the initial claim of this univariate skip round, i.e.
    /// input_claim = \sum_{-floor(S/2) <= z <= ceil(S/2)} \sum_{x \in \{0, 1}^n} P(z, x)
    /// where S = DOMAIN_SIZE
    fn input_claim(&self) -> F;

    /// Computes the full univariate polynomial to be sent in the uni-skip round.
    /// Returns a degree-bounded `UniPoly` with exactly `DEGREE_BOUND + 1` coefficients.
    fn compute_poly(&mut self) -> UniPoly<F>;

    // TODO: add flamegraph support
    // #[cfg(feature = "allocative")]
    // fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder);
}
