use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    append_sumcheck_claim, CompressedLabeledRoundPoly, CompressedSumcheckProof, RoundMessage,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use crate::error::ProverError;

/// Records batched sumcheck rounds into a verifier-visible proof artifact.
pub trait SumcheckProofRecorder<F: Field> {
    type Proof;

    fn absorb_input_claims<T: Transcript<Challenge = F>>(
        &mut self,
        claims: &[F],
        transcript: &mut T,
    );

    fn absorb_round<T: Transcript<Challenge = F>>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError<F>>;

    fn finish(&mut self) -> Self::Proof;
}

/// Clear batched sumcheck proof using Jolt's compressed Boolean-hypercube wire format.
#[derive(Clone, Debug, Default)]
pub struct ClearCompressedRecorder<F: Field> {
    round_polynomials: Vec<jolt_poly::CompressedPoly<F>>,
}

impl<F: Field> ClearCompressedRecorder<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Field> SumcheckProofRecorder<F> for ClearCompressedRecorder<F> {
    type Proof = CompressedSumcheckProof<F>;

    fn absorb_input_claims<T: Transcript<Challenge = F>>(
        &mut self,
        claims: &[F],
        transcript: &mut T,
    ) {
        for claim in claims {
            append_sumcheck_claim(transcript, claim);
        }
    }

    fn absorb_round<T: Transcript<Challenge = F>>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError<F>> {
        let compressed =
            CompressedLabeledRoundPoly::new(round_poly, SUMCHECK_ROUND_TRANSCRIPT_LABEL);
        <CompressedLabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(
            &compressed,
            transcript,
        );
        self.round_polynomials.push(round_poly.compress());
        Ok(transcript.challenge_scalar())
    }

    fn finish(&mut self) -> Self::Proof {
        CompressedSumcheckProof {
            round_polynomials: std::mem::take(&mut self.round_polynomials),
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests may unwrap on assertion failures")]

    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::UnivariatePoly;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use super::*;

    #[test]
    fn recorder_matches_labeled_absorb_path() {
        let poly = UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(5)]);
        let mut recorder = ClearCompressedRecorder::<Fr>::new();
        let mut transcript = Blake2bTranscript::new(b"recorder-test");

        let challenge = recorder.absorb_round(&poly, &mut transcript).unwrap();
        let proof = {
            let mut recorder = recorder;
            recorder.finish()
        };

        let mut expected = Blake2bTranscript::new(b"recorder-test");
        let compressed = CompressedLabeledRoundPoly::new(&poly, SUMCHECK_ROUND_TRANSCRIPT_LABEL);
        <CompressedLabeledRoundPoly<'_, Fr> as RoundMessage>::append_to_transcript(
            &compressed,
            &mut expected,
        );
        let expected_challenge = expected.challenge_scalar();

        assert_eq!(challenge, expected_challenge);
        assert_eq!(proof.round_polynomials.len(), 1);
        assert_eq!(proof.round_polynomials[0], poly.compress());
    }
}
