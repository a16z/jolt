//! The clear/ZK recording seam for sumcheck proving.
//!
//! A batched sumcheck's round loop is identical in clear and ZK mode; only the
//! per-message encoding differs (cleartext values vs. values masked by
//! committed pads in `jolt-zk`) and whether input claims enter the transcript. [`SumcheckRecorder`] captures exactly that difference so the engine
//! and the generated per-stage drivers are written once, generic over the
//! recorder. Whether claims are absorbed or sent is decided by the recorder
//! **type**; there is no runtime mode boolean to drift.
//!
//! The generated `begin_batch` drivers (`#[derive(SumcheckBatch)]` in
//! `jolt-verifier`) call [`absorb_input_claims`](SumcheckRecorder::absorb_input_claims);
//! the prove-side round loop calls [`absorb_round`](SumcheckRecorder::absorb_round)
//! per round and [`finish`](SumcheckRecorder::finish) once. The clear verifier
//! also runs `begin_batch` (with [`ClearSumcheckRecorder`]) so the two sides
//! share the head's Fiat-Shamir sequence structurally.

use std::marker::PhantomData;

use jolt_algebra::{CanonicalEncoding, Field};
use jolt_poly::UnivariatePoly;
use jolt_transcript::{
    Channel, PadSlots, PrivateChannel, ProverTranscript, Sponge, TranscriptError,
};

use crate::error::SumcheckError;
use crate::round_proof::{send_compressed_round, send_full_round};

/// Records one sumcheck's proof material into the prover transcript,
/// abstracting over clear vs. masked (ZK) recording: `absorb_input_claims`
/// once (from `begin_batch`), `absorb_round` per round (returning the
/// Fiat-Shamir challenge), then `finish` with the flattened output-claim values.
///
/// The clear recorder is written against the private-message seam
/// ([`PrivateChannel`]): input claims are derived claims, round coefficients
/// and output claims are private messages.
pub trait SumcheckRecorder<F: Field> {
    /// What the prover retains after recording.
    type Witness;

    /// Why recording failed: a [`SumcheckError`] for the clear and masking
    /// recorders, which report pad exhaustion as
    /// [`SumcheckError::PrivateChannel`].
    type Error;

    /// Absorb the batch's per-member input claims (present members, in
    /// declaration order). Clear: bound as derived claims. Masked: nothing,
    /// since the masked messages the claims derive from already bind them.
    fn absorb_input_claims<C>(
        &mut self,
        input_claims: &[F],
        channel: &mut C,
    ) -> Result<(), Self::Error>
    where
        C: PrivateChannel<Error = TranscriptError>;

    /// Record one round polynomial of degree bound `degree`, compressed, and
    /// draw the round challenge.
    fn absorb_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, Self::Error>;

    /// Record one full-width round polynomial (every coefficient, as a
    /// uni-skip round sends) of degree bound `degree` and draw the round
    /// challenge.
    fn absorb_full_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, Self::Error>;

    /// Record the flattened output-claim values (canonical order).
    fn finish<H: Sponge>(
        self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<Self::Witness, Self::Error>;
}

/// The clear recorder: binds input claims publicly and sends compressed
/// round polynomials and output claims, exactly what the clear verifier reads
/// back.
pub struct ClearSumcheckRecorder<F> {
    _field: PhantomData<F>,
}

impl<F> Default for ClearSumcheckRecorder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ClearSumcheckRecorder<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: Field + CanonicalEncoding> SumcheckRecorder<F> for ClearSumcheckRecorder<F> {
    type Witness = ();
    type Error = SumcheckError<F>;

    fn absorb_input_claims<C>(
        &mut self,
        input_claims: &[F],
        channel: &mut C,
    ) -> Result<(), SumcheckError<F>>
    where
        C: PrivateChannel<Error = TranscriptError>,
    {
        for claim in input_claims {
            channel.bind_derived_base(claim)?;
        }
        Ok(())
    }

    fn absorb_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, SumcheckError<F>> {
        send_compressed_round(round_poly, degree, transcript)?;
        Ok(transcript.challenge_small())
    }

    fn absorb_full_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, SumcheckError<F>> {
        send_full_round(round_poly, degree, transcript)?;
        Ok(transcript.challenge_small())
    }

    fn finish<H: Sponge>(
        self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<F>> {
        let _: PadSlots = transcript.exchange_private_base(&mut output_claim_values.to_vec())?;
        Ok(())
    }
}
