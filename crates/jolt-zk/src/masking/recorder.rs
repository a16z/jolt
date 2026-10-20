//! The prover's masking sumcheck recorder.

use jolt_field::{CanonicalEncoding, Field};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{compressed_wire, padded_coefficients, SumcheckError, SumcheckRecorder};
use jolt_transcript::{Channel, PrivateChannel, ProverTranscript, Sponge, TranscriptError};

use super::CommittedPads;

/// Records a sumcheck masked: every coefficient and output claim the clear
/// recorder would send goes out as m̂ = m + p under the next pads; input
/// claims, which the verifier derives, bind nothing.
pub struct MaskingRecorder<'p, F, O> {
    pads: &'p mut CommittedPads<F, O>,
}

impl<'p, F, O> MaskingRecorder<'p, F, O> {
    pub fn new(pads: &'p mut CommittedPads<F, O>) -> Self {
        Self { pads }
    }
}

impl<F: Field + CanonicalEncoding, O> SumcheckRecorder<F> for MaskingRecorder<'_, F, O> {
    type Witness = ();
    type Error = SumcheckError<F>;

    fn absorb_input_claims<C>(
        &mut self,
        _input_claims: &[F],
        _channel: &mut C,
    ) -> Result<(), SumcheckError<F>>
    where
        C: PrivateChannel<Error = TranscriptError>,
    {
        Ok(())
    }

    fn absorb_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, SumcheckError<F>> {
        let _ = self.pads.send(&compressed_wire(round_poly, degree)?, transcript)?;
        Ok(transcript.challenge_small())
    }

    fn absorb_full_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, SumcheckError<F>> {
        let _ = self
            .pads
            .send(&padded_coefficients(round_poly, degree)?, transcript)?;
        Ok(transcript.challenge_small())
    }

    fn finish<H: Sponge>(
        self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<F>> {
        let _ = self.pads.send(output_claim_values, transcript)?;
        Ok(())
    }
}
