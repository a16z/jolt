use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;

use crate::{
    field::{JoltField, MaybeAllocative},
    poly::opening_proof::ProverOpeningAccumulator,
};

pub trait SumcheckInstanceProver<F: JoltField, T: Transcript>:
    Send + Sync + MaybeAllocative
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        unimplemented!(
            "If get_params is unimplemented, degree, num_rounds, and \
            input_claim should be implemented directly"
        )
    }

    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize {
        self.get_params().degree()
    }

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize {
        self.get_params().num_rounds()
    }

    /// Returns the initial claim of this sumcheck instance.
    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.get_params().input_claim(accumulator)
    }

    /// Computes the prover's message for a specific round of the sumcheck protocol.
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F>;

    /// Ingest the verifier's challenge for a sumcheck round.
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize);

    /// Finalize prover state after the last challenge has been ingested, but before
    /// [`Self::cache_openings`] is called.
    ///
    /// This hook is useful for sumcheck implementations that need to perform
    /// end-of-protocol work (e.g., flushing delayed/streaming bindings or releasing
    /// intermediate state) once all challenges are known.
    ///
    /// Default implementation is a no-op.
    fn finalize(&mut self) {}

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
