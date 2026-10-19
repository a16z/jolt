//! The clear/ZK recording seam for sumcheck proving.
//!
//! A batched sumcheck's round loop is identical in clear and ZK mode; only the
//! per-round message differs (cleartext round polynomials vs. vector
//! commitments) and whether input/output claims enter the transcript in the
//! clear. [`SumcheckRecorder`] captures exactly that difference so the engine
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

use core::convert::Infallible;
use std::marker::PhantomData;

#[cfg(feature = "committed")]
use jolt_crypto::VectorCommitment;
#[cfg(feature = "committed")]
use jolt_field::JoltField;
use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_poly::UnivariatePoly;
use jolt_transcript::{
    Channel, PadSlots, PrivateChannel, ProverTranscript, Sponge, TranscriptError,
};
#[cfg(feature = "committed")]
use rand_core::RngCore;

#[cfg(feature = "committed")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
use crate::error::SumcheckError;
use crate::round_proof::send_compressed_round;

/// Records one sumcheck's proof material into the prover transcript,
/// abstracting over clear vs. committed (ZK) recording: `absorb_input_claims`
/// once (from `begin_batch`), `absorb_round` per round (returning the
/// Fiat-Shamir challenge), then `finish` with the flattened output-claim values.
///
/// The clear recorder is written against the private-message seam
/// ([`PrivateChannel`]): input claims are derived claims, round coefficients
/// and output claims are private messages.
pub trait SumcheckRecorder<F: Field> {
    /// What the prover retains after recording: nothing for a clear recorder,
    /// the openings of every commitment sent for a committed one.
    type Witness;

    /// The recorder's own failure, such as an exhausted pad schedule, reported
    /// as [`SumcheckError::Kernel`]. A driver whose kernel fails with `K`
    /// takes a recorder with `Error = K`.
    type Error;

    /// Absorb the batch's per-member input claims (present members, in
    /// declaration order). Clear: bound as derived claims. Committed: no-op,
    /// since the claims' commitments were already sent by the stage that
    /// produced them, so the transcript never sees the scalars.
    fn absorb_input_claims<C>(
        &mut self,
        input_claims: &[F],
        channel: &mut C,
    ) -> Result<(), SumcheckError<F, Self::Error>>
    where
        C: PrivateChannel<Error = TranscriptError>;

    /// Record one round polynomial of degree bound `degree` and draw the round
    /// challenge. Clear: sent compressed. Committed: its commitment is sent.
    fn absorb_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, SumcheckError<F, Self::Error>>;

    /// Record the flattened output-claim values (canonical order). Clear: each
    /// value is sent. Committed: the values are row-committed and only the
    /// commitments are sent.
    fn finish<H: Sponge>(
        self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<Self::Witness, SumcheckError<F, Self::Error>>;
}

/// The clear recorder: binds input claims publicly and sends compressed
/// round polynomials and output claims, exactly what the clear verifier reads
/// back. `K` is the error of the driver it serves; it raises none itself.
pub struct ClearSumcheckRecorder<F, K = Infallible> {
    _marker: PhantomData<fn() -> (F, K)>,
}

impl<F, K> Default for ClearSumcheckRecorder<F, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, K> ClearSumcheckRecorder<F, K> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F: Field + CanonicalEncoding, K> SumcheckRecorder<F> for ClearSumcheckRecorder<F, K> {
    type Witness = ();
    type Error = K;

    fn absorb_input_claims<C>(
        &mut self,
        input_claims: &[F],
        channel: &mut C,
    ) -> Result<(), SumcheckError<F, K>>
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
    ) -> Result<F, SumcheckError<F, K>> {
        send_compressed_round(round_poly, degree, transcript).map_err(SumcheckError::widen)?;
        Ok(transcript.challenge_small())
    }

    fn finish<H: Sponge>(
        self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<F, K>> {
        let _: PadSlots = transcript.exchange_private_base(&mut output_claim_values.to_vec())?;
        Ok(())
    }
}

/// Records one single-instance sumcheck over base field `F` and extension
/// `E` ([`prove_single`](crate::prove_single)), whose round context draws the
/// challenges: the input claim once, then each round's wire coefficients.
pub trait SingleRecorder<F, E: Field> {
    /// As [`SumcheckRecorder::Error`].
    type Error;

    /// Clear: bound as a derived claim.
    fn bind_input_claim<H: Sponge>(
        &mut self,
        claim: &E,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<E, Self::Error>>;

    /// Clear: sent as private messages.
    fn record_round<H: Sponge>(
        &mut self,
        coefficients: &mut [E],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<E, Self::Error>>;
}

impl<F, E, K> SingleRecorder<F, E> for ClearSumcheckRecorder<E, K>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
{
    type Error = K;

    fn bind_input_claim<H: Sponge>(
        &mut self,
        claim: &E,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<E, K>> {
        transcript.bind_derived::<F, E>(claim)?;
        Ok(())
    }

    fn record_round<H: Sponge>(
        &mut self,
        coefficients: &mut [E],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), SumcheckError<E, K>> {
        let _: PadSlots = transcript.exchange_private::<F, E>(coefficients)?;
        Ok(())
    }
}

/// The committed (ZK) recorder: commits each round polynomial and the
/// output-claim rows, sending only the commitments — the transcript never sees
/// a claim or coefficient scalar. Input-claim absorbs are no-ops: the claims'
/// commitments were already sent by the stage that produced them. The retained
/// witness (coefficients, rows, blindings) is returned by
/// [`finish`](SumcheckRecorder::finish) for BlindFold.
#[cfg(feature = "committed")]
pub struct CommittedSumcheckRecorder<'a, F, VC, R>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    builder: CommittedSumcheckBuilder<'a, F, VC, R>,
}

#[cfg(feature = "committed")]
impl<'a, F, VC, R> CommittedSumcheckRecorder<'a, F, VC, R>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    pub fn new(setup: &'a VC::Setup, rng: R) -> Result<Self, SumcheckError<F>> {
        Ok(Self {
            builder: CommittedSumcheckBuilder::new(setup, rng)?,
        })
    }
}

#[cfg(feature = "committed")]
impl<F, VC, R> SumcheckRecorder<F> for CommittedSumcheckRecorder<'_, F, VC, R>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    type Witness = CommittedSumcheckWitness<F>;
    type Error = Infallible;

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
        self.builder.commit_round(round_poly, degree, transcript)
    }

    fn finish<H: Sponge>(
        self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<CommittedSumcheckWitness<F>, SumcheckError<F>> {
        self.builder.finish(output_claim_values, transcript)
    }
}
