//! The prover's masking sumcheck recorder.

use jolt_algebra::Field;

use jolt_poly::UnivariatePoly;

use jolt_sumcheck::{
    check_uniskip_round, compressed_wire, padded_coefficients, RecordedSumcheck, SumcheckError,
};
use jolt_sumcheck::{CompressedSumcheckProof, SumcheckRecorder};
use jolt_transcript::{Encoding, PadSlots, Transcript};

use jolt_transcript::{FieldTranscript, ProverTranscript};

use super::{CommittedPads, MaskedMessages};

/// The masking [`SumcheckRecorder`]: every round goes out as its fixed-width
/// compressed coefficients masked by the next pads, followed by the round's
/// proof of work and challenge; the output claims go out masked; the input
/// claims bind nothing, since the masked messages they derive from already
/// do. Every masked value and nonce lands on the proof's [`MaskedMessages`]
/// tape, so the recorded per-stage proof is empty.
pub struct MaskingRecorder<'p, F, O> {
    pads: &'p mut CommittedPads<F, O>,
    tape: &'p mut MaskedMessages<F>,
    rounds: usize,
}

impl<'p, F, O> MaskingRecorder<'p, F, O> {
    pub fn new(pads: &'p mut CommittedPads<F, O>, tape: &'p mut MaskedMessages<F>) -> Self {
        Self {
            pads,
            tape,
            rounds: 0,
        }
    }
}

impl<F: Field + Encoding + 'static, O> SumcheckRecorder<F> for MaskingRecorder<'_, F, O> {
    type Proof = CompressedSumcheckProof<F>;

    fn absorb_input_claims<T: Transcript>(&mut self, _input_claims: &[F], _transcript: &mut T) {}

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree_bound: usize,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        let round = self.rounds;
        let wire = compressed_wire(round_poly, degree_bound)?;
        let mut channel = self.pads.channel(self.tape, transcript);
        let _: PadSlots = channel.send_private(&wire).map_err(SumcheckError::from)?;
        channel
            .grind_round(round, degree_bound)
            .map_err(SumcheckError::from)?;
        self.rounds += 1;
        Ok(transcript.challenge()?)
    }

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Proof>, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        let _: PadSlots = self
            .pads
            .send(output_claim_values, self.tape, transcript)
            .map_err(SumcheckError::from)?;
        Ok(RecordedSumcheck {
            proof: CompressedSumcheckProof::default(),
            committed_witness: None,
        })
    }
}

/// A masked uni-skip first round: its challenge and the output claim, which
/// went out masked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskedUniskip<F> {
    pub challenge: F,
    pub output_claim: F,
}

/// Proves a uni-skip first round masked, mirroring the clear
/// `prove_uniskip_clear`: the self-check, all `degree + 1` coefficients
/// masked, the round's proof of work, the challenge, then the output claim
/// masked.
///
/// # Errors
///
/// A round that fails its self-check, exhausted pads, or a failed nonce
/// search.
pub fn prove_uniskip_masked<F, O, T>(
    pads: &mut CommittedPads<F, O>,
    tape: &mut MaskedMessages<F>,
    round_poly: &UnivariatePoly<F>,
    input_claim: F,
    degree: usize,
    domain_size: usize,
    transcript: &mut T,
) -> Result<MaskedUniskip<F>, SumcheckError<F>>
where
    F: Field + Encoding + 'static,
    T: FieldTranscript<F> + ProverTranscript,
{
    check_uniskip_round(round_poly, input_claim, degree, domain_size)?;
    let coefficients = padded_coefficients(round_poly, degree)?;
    let mut channel = pads.channel(tape, transcript);
    let _: PadSlots = channel
        .send_private(&coefficients)
        .map_err(SumcheckError::from)?;
    channel
        .grind_round(0, degree)
        .map_err(SumcheckError::from)?;
    let challenge = transcript.challenge()?;
    let output_claim = round_poly.evaluate(challenge);
    let _: PadSlots = pads
        .send(&[output_claim], tape, transcript)
        .map_err(SumcheckError::from)?;
    Ok(MaskedUniskip {
        challenge,
        output_claim,
    })
}
