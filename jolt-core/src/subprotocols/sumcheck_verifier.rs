use crate::poly::opening_proof::{OpeningAccumulator, OpeningPoint, BIG_ENDIAN};
use crate::subprotocols::blindfold::OutputClaimConstraint;
use crate::transcripts::Transcript;

use crate::{field::JoltField, poly::opening_proof::VerifierOpeningAccumulator};

pub trait SumcheckInstanceVerifier<F: JoltField, T: Transcript> {
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

    /// Returns the output claim constraint for this sumcheck instance.
    /// This describes how the final sumcheck claim relates to polynomial evaluations.
    /// Returns None if this instance has no output constraint (e.g., round verification only).
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        None
    }

    /// Returns the challenge values needed to evaluate the output constraint.
    /// These are the values for Challenge(0), Challenge(1), etc. in the constraint.
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        Vec::new()
    }
}

pub trait SumcheckInstanceParams<F: JoltField> {
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize;

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize;

    /// Returns the initial claim of this sumcheck instance.
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F>;
}
