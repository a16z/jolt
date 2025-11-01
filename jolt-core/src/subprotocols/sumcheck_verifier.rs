use crate::transcripts::Transcript;

use crate::{field::JoltField, poly::opening_proof::VerifierOpeningAccumulator};

pub trait SumcheckInstanceVerifier<F: JoltField, T: Transcript> {
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize;

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize;

    /// Returns the initial claim of this sumcheck instance.
    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F;

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
