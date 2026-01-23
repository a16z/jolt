use crate::poly::opening_proof::{OpeningAccumulator, OpeningPoint, BIG_ENDIAN};
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use crate::transcripts::Transcript;

use crate::{field::JoltField, poly::opening_proof::VerifierOpeningAccumulator};

pub trait SumcheckInstanceVerifier<F: JoltField, T: Transcript> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F>;
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize {
        self.get_params().degree()
    }

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize {
        self.get_params().num_rounds()
    }

    /// Returns the global round offset (0-based) at which this instance becomes active in a
    /// batched sumcheck of `max_num_rounds` total rounds.
    ///
    /// Default preserves existing "front-loaded" batching behavior.
    fn round_offset(&self, max_num_rounds: usize) -> usize {
        max_num_rounds - self.num_rounds()
    }

    /// Returns the initial claim of this sumcheck instance.
    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.get_params().input_claim(accumulator)
    }

    /// Expected final claim after binding to the provided instance-local r slice.
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F;

    /// Enqueue any openings needed after sumcheck completes.
    /// r is the instance-local slice; instance normalizes internally.
    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    );
}

pub trait SumcheckInstanceParams<F: JoltField> {
    fn degree(&self) -> usize;

    fn num_rounds(&self) -> usize;

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F>;

    fn input_claim_constraint(&self) -> InputClaimConstraint;

    fn input_constraint_challenge_values(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F>;

    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint>;
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F>;
}
