//! The prover's masking sumcheck recorder.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    check_uniskip_round, compressed_wire, padded_coefficients, CompressedSumcheckProof, PadSlots,
    ProverRoundChannel, RecordedSumcheck, SumcheckError, SumcheckRecorder,
};
use jolt_transcript::{Encoding, FieldTranscript, ProverTranscript, Transcript};

use super::{CommittedPads, MaskedMessages, ZkError};

/// How a masked batch binds its members' input claims, which the clear
/// recorder absorbs, each as its own private message, before the first
/// round. A batch's claimed sum is only bound if every input claim is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputClaims {
    /// Each input claim is a function of masked values already sent, of
    /// challenges and of public data, so the residual relation pins it and
    /// nothing is sent. The caller asserts this: a claim that depends on a
    /// value sent after the batch, as stage 4's RAM value check depends on
    /// openings stage 4 itself outputs, is free to the prover, who picks
    /// that value's mask after the batch's challenges.
    Derived,
    /// Each input claim goes out masked, as its own private message, where
    /// the clear recorder absorbs it.
    Send,
}

/// The masking [`SumcheckRecorder`]: the input claims go out as `claims`
/// says; every round goes out as its fixed-width compressed coefficients
/// masked by the next pads, followed by the round's proof of work and
/// challenge; the output claims go out masked. Every masked value and nonce
/// lands on the proof's [`MaskedMessages`] tape, so the recorded per-stage
/// proof is empty, and a masked verifier must reject a stage whose clear
/// proof is not.
pub struct MaskingRecorder<'p, F, O> {
    pads: &'p mut CommittedPads<F, O>,
    tape: &'p mut MaskedMessages<F>,
    claims: InputClaims,
    rounds: usize,
    /// The failure of [`absorb_input_claims`](SumcheckRecorder::absorb_input_claims),
    /// which cannot return it, reported by the next round or by `finish`.
    failed: Option<ZkError>,
}

impl<'p, F, O> MaskingRecorder<'p, F, O> {
    pub fn new(
        pads: &'p mut CommittedPads<F, O>,
        tape: &'p mut MaskedMessages<F>,
        claims: InputClaims,
    ) -> Self {
        Self {
            pads,
            tape,
            claims,
            rounds: 0,
            failed: None,
        }
    }

    fn check_input_claims(&mut self) -> Result<(), ZkError> {
        self.failed.take().map_or(Ok(()), Err)
    }
}

impl<F: Field + Encoding + 'static, O> SumcheckRecorder<F> for MaskingRecorder<'_, F, O> {
    type Proof = CompressedSumcheckProof<F>;

    fn absorb_input_claims<T: Transcript>(&mut self, input_claims: &[F], transcript: &mut T) {
        if self.claims == InputClaims::Derived || self.failed.is_some() {
            return;
        }
        for claim in input_claims {
            if let Err(error) = self.pads.send(&[*claim], self.tape, transcript) {
                self.failed = Some(error);
                return;
            }
        }
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree_bound: usize,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        self.check_input_claims()?;
        let round = self.rounds;
        let mut wire = compressed_wire(round_poly, degree_bound)?;
        let mut channel = self.pads.channel(self.tape, transcript);
        channel.send_round(round, &mut wire)?;
        let challenge = channel.round_challenge(round, degree_bound)?;
        self.rounds += 1;
        Ok(challenge)
    }

    fn finish<T>(
        mut self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<F, Self::Proof>, SumcheckError<F>>
    where
        T: FieldTranscript<F> + ProverTranscript,
    {
        self.check_input_claims()?;
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
    let mut coefficients = padded_coefficients(round_poly, degree)?;
    let mut channel = pads.channel(tape, transcript);
    channel.send_round(0, &mut coefficients)?;
    let challenge = channel.round_challenge(0, degree)?;
    let output_claim = round_poly.evaluate(challenge);
    let _: PadSlots = pads
        .send(&[output_claim], tape, transcript)
        .map_err(SumcheckError::from)?;
    Ok(MaskedUniskip {
        challenge,
        output_claim,
    })
}
